"""Bounded diagnostic wrapper; never use its synchronized timings as throughput.

Pass the normal scripts/c2r/run_training.py arguments. This records per-rank
phase timings and a GPU-only collective benchmark, and skips checkpoint writes.
The training implementation, batches, losses and optimizer remain unchanged.
"""
import functools
import json
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import time

import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data.dataloader import _BaseDataLoaderIter
from fastwam.models.wan22.fastwam import FastWAM
from fastwam_c2r.training import C2RTrainer


def main():
    output = Path(os.environ['C2R_OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['RANK'])
    events = []
    active = False
    backward_calls = 0

    def wrap(cls, name, label):
        original = getattr(cls, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            nonlocal backward_calls
            if not active:
                return original(*args, **kwargs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            if label == 'backward_and_deepspeed_step':
                backward_calls += 1
            if label == 'backward_and_deepspeed_step' and backward_calls == 17 and rank == 0:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profile:
                    result = original(*args, **kwargs)
                    torch.cuda.synchronize()
                averages = profile.key_averages()
                (output / 'backward-operators-cpu.txt').write_text(averages.table(
                    sort_by='self_cpu_time_total', row_limit=40))
                (output / 'backward-operators-cuda.txt').write_text(averages.table(
                    sort_by='self_device_time_total', row_limit=40))
            else:
                result = original(*args, **kwargs)
            torch.cuda.synchronize()
            events.append({'phase': label, 'seconds': time.perf_counter() - start})
            return result
        setattr(cls, name, measured)

    wrap(_BaseDataLoaderIter, '__next__', 'data_next')
    wrap(FastWAM, '_encode_video_latents', 'vae_encode')
    wrap(FastWAM, '_joint_denoise_core', 'denoise_forward')
    wrap(FastWAM, 'training_loss', 'forward_total')
    wrap(Accelerator, 'backward', 'backward_and_deepspeed_step')
    original_train = C2RTrainer.train

    def measured_train(trainer):
        nonlocal active
        world_size = dist.get_world_size()
        warmup_microbatches = 2 * trainer.gradient_accumulation_steps
        topology = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'nccl': torch.cuda.nccl.version(),
                    'device': torch.cuda.get_device_name(),
                    'peer_access': [[torch.cuda.can_device_access_peer(i, j) if i != j else True
                                     for j in range(world_size)] for i in range(world_size)]}
        if rank == 0:
            for command in (['nvidia-smi', 'topo', '-m'], ['nvidia-smi', 'nvlink', '-s']):
                topology[' '.join(command)] = subprocess.run(command, capture_output=True, text=True).stdout
            (output / 'topology.json').write_text(json.dumps(topology, indent=2))
        # No dataset/CPFS access inside the timed region. Report slowest rank.
        tensor = torch.zeros(128 * 1024 * 1024, device=trainer.accelerator.device,
                             dtype=torch.bfloat16)
        for _ in range(5):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        for _ in range(20):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = torch.tensor((time.perf_counter() - start) / 20,
                               device=trainer.accelerator.device, dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        seconds = elapsed.item()
        (output / f'collective-rank{rank}.json').write_text(json.dumps({
            'bytes': tensor.numel() * tensor.element_size(), 'seconds': seconds,
            'algorithm_GBps': 268435456 / seconds / 1e9,
            'world_size': world_size,
            'bus_GBps': 268435456 / seconds / 1e9 * 2 * (world_size - 1) / world_size}))
        del tensor
        active = True
        try:
            return original_train(trainer)
        finally:
            active = False
            # Nested forward/encode entries overlap; do not sum them together.
            report = {}
            for phase in sorted({event['phase'] for event in events}):
                values = [event['seconds'] for event in events if event['phase'] == phase]
                steady = values[warmup_microbatches:]
                report[phase] = {'count': len(values), 'warmup_microbatches': warmup_microbatches,
                                 'median_after_warmup':
                                 statistics.median(steady) if steady else None}
            (output / f'phase-rank{rank}.json').write_text(json.dumps({
                'synchronized_diagnostic': True, 'checkpoint_writes_skipped': True,
                'summary': report, 'events': events}, indent=2))

    C2RTrainer.train = measured_train
    C2RTrainer.save_checkpoint = lambda self: {'weights_path': 'SKIPPED_DIAGNOSTIC',
                                             'state_path': 'SKIPPED_DIAGNOSTIC'}
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/c2r/run_training.py'),
                   run_name='__main__')


if __name__ == '__main__':
    main()
