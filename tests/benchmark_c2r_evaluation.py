"""Real-checkpoint inference parity and prompt-cache timing; no training updates."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from omegaconf import OmegaConf
from fastwam_c2r.policy import C2RPolicy
from fastwam_c2r.data import C2RDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = OmegaConf.load(args.training_root / 'config.yaml')
    policy = C2RPolicy(args.training_root / 'checkpoints/weights/step_010725.pt', args.training_root)
    dataset = C2RDataset(cfg.c2r.dataset.root, cfg.c2r.dataset.manifest,
                        augment=False, require_context=False, stereo=True)
    sample = dataset[0]
    model = policy.model
    kwargs = dict(input_image=sample['video'][:, 0][None].cuda(),
                  input_image_right=sample['video_right'][:, 0][None].cuda(),
                  proprio=sample['proprio'], action_horizon=32, num_inference_steps=10,
                  sigma_shift=5., text_cfg_scale=1., negative_prompt='', seed=0, rand_device='cpu', tiled=False)
    timings = {}
    outputs = {}
    with torch.no_grad():
        context, mask = model.encode_prompt(sample['prompt'])
        for mode in ('prompt', 'cached'):
            times = []
            for i in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = model.infer_action(**kwargs, **(dict(prompt=sample['prompt']) if mode == 'prompt'
                    else dict(prompt=None, context=context, context_mask=mask)))['action']
                torch.cuda.synchronize()
                if i:
                    times.append(time.perf_counter()-start)
                outputs[mode] = output.detach().float().cpu()
            timings[mode] = dict(seconds=times, median=statistics.median(times))
    diff = (outputs['prompt']-outputs['cached']).abs()
    result = dict(timings=timings, max_abs=diff.max().item(), bitwise_equal=torch.equal(outputs['prompt'],outputs['cached']),
                  finite=bool(torch.isfinite(outputs['cached']).all()), peak_gib=torch.cuda.max_memory_allocated()/2**30)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert result['bitwise_equal'] and result['finite'], result


if __name__ == '__main__':
    main()
