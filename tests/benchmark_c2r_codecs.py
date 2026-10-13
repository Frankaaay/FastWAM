"""H800 paired codec benchmark; no policy training or checkpoint mutation."""
import argparse
import ast
import gc
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch

from fastwam_c2r.data import C2RDataset
from fastwam.models.wan22.helpers.loader import _load_registered_model
from fastwam.models.wan22.stereotok_vae import StereoTokVAE
from fastwam.models.wan22.stereotok.vae2_2 import latent_scale


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    random.seed(42)
    base = Path('/mnt/workdir/frank')
    repo = Path(__file__).resolve().parents[1]
    report = {'sha': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
              'shape': [8, 3, 9, 384, 320], 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(), 'cases': {}}
    dataset = C2RDataset(base/'datasets/RoboTwin2.0',
        base/'runs/c2r-prep-2500-20261013/prepared/manifest.json',
        augment=True, require_context=False, stereo=True)
    indices = [i * (len(dataset) - 1) // 15 for i in range(16)]
    samples = [dataset[i] for i in indices]
    batches = [(torch.stack([s['video'] for s in samples[k:k+8]]).cuda(),
                torch.stack([s['video_right'] for s in samples[k:k+8]]).cuda()) for k in (0, 8)]
    report['indices'] = indices

    def save():
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')

    def measure(name, fn):
        print('WARMUP', name, flush=True)
        for left, right in batches:
            fn(left, right)
        torch.cuda.synchronize()
        counters_before = dict(torch._dynamo.utils.counters['stats'])
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(3):
            for left, right in batches:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = fn(left, right)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
                del result
        counters_after = dict(torch._dynamo.utils.counters['stats'])
        assert counters_after == counters_before, (name, 'recompiled during timing')
        report['cases'][name] = {'seconds': times, 'median_seconds': statistics.median(times),
            'peak_gib': torch.cuda.max_memory_allocated()/2**30,
            'compile_counters_stable': True}
        save()
        print(name, report['cases'][name], flush=True)

    wan = _load_registered_model(str(base/'checkpoints/fastwam-c2r-shared-20261013/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors'),
                                'wan_video_vae', torch_dtype=torch.bfloat16, device='cuda')
    wan.eval().requires_grad_(False)
    compiled = torch.compile(wan.model.encode, backend='cudagraphs', fullgraph=True)
    measure('wan_b8', lambda left, right: compiled(left.to(torch.bfloat16), wan.scale).clone())
    del compiled, wan
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()

    stereo = StereoTokVAE(base/'checkpoints/stereotok/student-u8000.pt', 'cuda')
    # Execute the exact pre-optimization method on the same frozen model/inputs.
    source = subprocess.check_output(['git', '-C', str(repo), 'show',
        'f7d384da674878d99577223c4d65dc89a0b6efb5:src/fastwam/models/wan22/stereotok_vae.py'], text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'encode')
    scope = {'torch': torch, 'latent_scale': latent_scale}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<original-encode>', 'exec'), scope)
    old = lambda left, right: scope['encode'](stereo, left, 'cuda', right_videos=right)
    measure('stereo_b1_loop', old)
    references = [old(left, right).cpu() for left, right in batches]
    candidate = lambda left, right: stereo.encode(left, 'cuda', right_videos=right)
    measure('stereo_b8', candidate)
    parity = []
    for (left, right), reference in zip(batches, references):
        result = candidate(left, right).cpu()
        error = (result-reference).double()
        relative = (error.norm()/reference.double().norm().clamp_min(1e-12)).item()
        maximum = error.abs().max().item()
        row = {'shape': list(result.shape), 'finite': bool(torch.isfinite(result).all()),
               'relative_l2': relative, 'max_abs': maximum}
        parity.append(row)
        report['parity'] = parity
        save()
        assert result.shape == reference.shape and row['finite']
        assert relative <= 5e-4 and maximum <= .01, row
    report['passed'] = True
    save()
    print('PASSED', report, flush=True)


if __name__ == '__main__':
    main()
