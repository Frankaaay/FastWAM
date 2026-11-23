# FastWAM with the StereoTok tokenizer

This branch replaces the frozen Wan2.2 VAE of FastWAM with the StereoTok stereo
tokenizer and adds the RoboTwin 2.0 stereo training and evaluation pipeline (`c2r`).
StereoTok keeps the Wan2.2 latent interface, so the DiT, action expert and token
budget are unchanged. The frozen StereoTok inference code is vendored in
`src/fastwam/models/wan22/stereotok/` (no StereoTok training checkout is needed);
OpenWAM and LingBot-VA import the same codec from this package.

## Checkpoints

Use a StereoTok `stereo.pth` export (Stage 2 or Stage 3). The same file serves both
input modes: `--input-mode stereo` feeds left and right views, `--input-mode mono-left`
feeds the left view only.

## Data

RoboTwin 2.0 stereo demonstrations (2,500 episodes, head and wrist cameras with their
right eyes) are prepared once:

```bash
python scripts/c2r/prepare_data.py --published-root $ROBOTWIN_STEREO --output $C2R_PREP_DIR
torchrun --nproc-per-node 8 scripts/c2r/cache_text.py --prep $C2R_PREP_DIR --cache $C2R_TEXT_CACHE
```

## Training

`configs/c2r/train.yaml` reads its paths from the environment:

| Variable | Content |
| --- | --- |
| `C2R_DATASET_ROOT` | RoboTwin 2.0 dataset root |
| `C2R_PREP_DIR` | output of `prepare_data.py` (`manifest.json`, `lr.json`, `prepared.json`) |
| `C2R_TEXT_CACHE` | output of `cache_text.py` |
| `DIFFSYNTH_MODEL_BASE_PATH` | Wan2.2-TI2V-5B weights |
| `C2R_ACTION_INIT` | action expert initialization |
| `C2R_OUTPUT_DIR`, `C2R_RUN_NAME` | run directory and W&B run name |

Recipe: 8 GPUs, global batch 256 (B8 x GA4), 5 epochs = 10,725 updates, LR 1e-4,
seed 42, BF16, 9 video frames and 32 actions on a 384x320 three-camera canvas.

```bash
# Wan baseline; also writes the shared initialization of the trainable heads.
accelerate launch --use_deepspeed --deepspeed_config_file configs/c2r/ds_zero2.json --num_processes 8 \
  scripts/c2r/run_training.py --config configs/c2r/train.yaml --variant wan --initialization $C2R_PREP_DIR/common-initialization.pt

# StereoTok (stereo or mono-left), starting from the same head initialization.
STEREOTOK_STUDENT_CKPT=stereo.pth accelerate launch --use_deepspeed --deepspeed_config_file configs/c2r/ds_zero2.json --num_processes 8 \
  scripts/c2r/run_training.py --config configs/c2r/train.yaml --variant stereotok --input-mode stereo \
  --initialization $C2R_PREP_DIR/common-initialization.pt
```

The StereoTok run refuses an initialization whose contract (seed, data manifest,
action init) differs from the Wan run, so both codecs start from identical heads.
Resume with `--resume <run>/checkpoints/state/step_N`.

## Evaluation

RoboTwin clean and randomized, 50 tasks x 100 episodes each, with frozen episode
plans: `configs/robotwin_shared_plan.json` (SHA256 `42ba67ec…`) locks the seeds,
instructions and RNG states stored in `configs/robotwin_plans/*.json.xz`. The same
plans are used for FastWAM, OpenWAM and LingBot-VA. Create an output directory with a
`manifest.json`:

```python
import json
lock = json.load(open("configs/robotwin_shared_plan.json"))
manifest = dict(model="fastwam", checkpoint="<run>/checkpoints/weights/step_010725.pt",
                training_root="<run>", episodes=100, chunk_size=10, cases=lock["cases"])
# OpenWAM / LingBot-VA: model="openwam" | "lingbot" and
# policy=dict(host=..., port=..., stereo=True[, video_guidance_scale=5, action_guidance_scale=1])
json.dump(manifest, open("<eval-dir>/manifest.json", "w"), indent=2)
```

Then run rollout workers (any number, each claims episode chunks) and aggregate:

```bash
python scripts/c2r/evaluate.py rollout --root <eval-dir>
python scripts/c2r/evaluate.py summary --root <eval-dir>
```

Restrict `cases` to one phase to evaluate only clean or only randomized scenes.
