"""Real u8000 CUDA codec and reduced-model wiring checks, never policy training.

Synthetic stereo pixels and random one-layer experts deliberately test interfaces;
this does not admit the official Wan initialization or RoboTwin training dataset.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch


def main(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from fastwam.models.wan22.stereotok_vae import StereoTokVAE
    from fastwam.models.wan22.stereotok.vae2_2 import latent_scale
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.models.wan22.mot import MoT

    root = Path(__file__).resolve().parents[1]
    report = {"source_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "input": "synthetic stereo, B1/T1/T9/384x320",
        "policy": "random one-layer experts; wiring smoke only"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Resolution only: these paths are not asserted to contain prepared datasets.
    os.environ.update(STEREOTOK_STUDENT_CKPT=str(args.student),
        ROBOTWIN_STEREO_LEROBOT_ROOT=str(args.report.parent / "not-prepared"),
        ROBOTWIN_NORM_STATS=str(args.report.parent / "not-prepared-stats.json"),
        ROBOTWIN_TEXT_CACHE=str(args.report.parent / "not-prepared-text"))
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=robotwin_stereotok_clean_1epoch"])
        resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["num_epochs"] == 1 and resolved["resume"] is None
    assert resolved["model"]["skip_dit_load_from_pretrain"] is False
    assert len(resolved["data"]["train"]["shape_meta"]["images"]) == 6
    report["hydra_compose"] = True
    torch.manual_seed(42)
    codec = StereoTokVAE(args.student, "cuda")
    report["tokenizer"] = codec.binding
    assert not codec.training and not any(p.requires_grad for p in codec.parameters())
    assert all(p.dtype == torch.float32 for p in codec.parameters())
    left = torch.rand(1, 3, 9, 384, 320, device="cuda") * 2 - 1
    right = torch.roll(left, -8, dims=-1)
    regions = torch.zeros(96, 80, dtype=torch.long, device="cuda")
    regions[64:, :40], regions[64:, 40:] = 1, 2
    mask = torch.ones(1, 2, 384, 320, dtype=torch.bool, device="cuda")
    cases = []
    for frames in (1, 9):
        x, r = left[:, :, :frames], right[:, :, :frames]
        start = time.perf_counter()
        with torch.inference_mode():
            actual = codec.encode(x, "cuda", right_videos=r)
        torch.cuda.synchronize()
        assert actual.shape == (1, 48, 1 + (frames - 1) // 4, 24, 20)
        assert torch.isfinite(actual).all()
        with torch.no_grad():
            mean, _ = codec.model.encode_posterior(x, r, mask,
                codec.model.fusion.max_disparity, optimize_encode=False, camera_regions=regions)
            shift, scale = latent_scale("cuda")
            reference = (mean - shift.view(1, 48, 1, 1, 1)) * scale.view(1, 48, 1, 1, 1)
            relative = float((actual - reference).norm() / reference.norm().clamp_min(1e-8))
            assert relative < 0.01, relative
            reconstructed = codec.decode(actual, "cuda")
            assert reconstructed.shape == x.shape and torch.isfinite(reconstructed).all()
        # Exercise inference-mode -> ordinary no-grad and A/B/A graph replay.
        same = codec.encode(x, "cuda", right_videos=r)
        changed = codec.encode(x, "cuda", right_videos=-r)
        again = codec.encode(x, "cuda", right_videos=r)
        torch.testing.assert_close(same, actual, rtol=0, atol=0)
        torch.testing.assert_close(again, same, rtol=0, atol=0)
        sensitivity = float((same - changed).abs().max())
        assert sensitivity > 1e-6, sensitivity
        cases.append(dict(frames=frames, shape=list(actual.shape), relative_l2=relative,
            max_abs=float((actual-reference).abs().max()), right_eye_sensitivity=sensitivity,
            compile_and_checks_seconds=time.perf_counter()-start))
        print(json.dumps(cases[-1]), flush=True)
    report["codec_cases"] = cases
    video_cfg = dict(resolved["model"]["video_dit_config"])
    video_cfg.update(hidden_dim=64, ffn_dim=128, num_heads=2, attn_head_dim=32,
                     num_layers=1, text_dim=64, freq_dim=32)
    action_cfg = dict(resolved["model"]["action_dit_config"])
    action_cfg.update(hidden_dim=64, ffn_dim=128, num_heads=2, attn_head_dim=32,
                      num_layers=1, text_dim=64, freq_dim=32)
    video = WanVideoDiT(**video_cfg).to(device="cuda", dtype=torch.bfloat16)
    action = ActionDiT(**action_cfg).to(device="cuda", dtype=torch.bfloat16)
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
    model = FastWAM(video, action, mot, codec, text_dim=64, proprio_dim=14,
                    device="cuda", torch_dtype=torch.bfloat16)
    sample = dict(video=left, video_right=right,
        action=torch.randn(1, 32, 14, device="cuda"),
        proprio=torch.zeros(1, 1, 14, device="cuda"),
        context=torch.zeros(1, 8, 64, device="cuda"),
        context_mask=torch.ones(1, 8, dtype=torch.bool, device="cuda"),
        image_is_pad=torch.zeros(1, 9, dtype=torch.bool, device="cuda"),
        action_is_pad=torch.zeros(1, 32, dtype=torch.bool, device="cuda"))
    model.train()
    assert not codec.training
    loss, losses = model.training_loss(sample)
    assert torch.isfinite(loss)
    loss.backward()
    for expert in (video, action):
        gradients = [p.grad for p in expert.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients)
    assert all(p.grad is None for p in codec.parameters())
    report.update(loss=float(loss.detach()), losses=losses, backward_finite=True,
                  tokenizer_frozen=True, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    main(parser.parse_args())
