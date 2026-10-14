"""Launch independent, persistent C2R evaluation workers on allocated GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['prepare', 'rollout', 'pipeline'], required=True)
    parser.add_argument('--gpus', type=int, required=True)
    parser.add_argument('--workers-per-gpu', type=int, required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if sha != args.source_sha or subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip():
        raise RuntimeError('Evaluation requires the published clean SHA')
    base = Path('/mnt/workdir/frank')
    libs = json.loads((base / 'runs/fastwam-h800-smoke-20261008/system-libs-prepared.json').read_text())
    py = base / 'envs/fastwam-stereotok-eval-cu128-20261009/bin/python'
    job = os.environ.get('DLC_JOB_ID', f'dsw-{os.getpid()}')
    logs = args.root / 'workers' / job
    logs.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(repo / 'src'), PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               LD_LIBRARY_PATH=libs['ld_library_path'] + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
               PATH=str(py.parent) + ':' + libs['prefix'] + '/usr/bin:/usr/local/cuda/bin:' + os.environ['PATH'],
               VK_ICD_FILENAMES=libs['vulkan_icd'], CUDA_HOME='/usr/local/cuda',
               __EGL_VENDOR_LIBRARY_FILENAMES=str(base / 'envs/system-libs/nvidia-egl-vendor.json'),
               WANDB_MODE='disabled', DIFFSYNTH_SKIP_DOWNLOAD='true')
    started = time.time()
    children = []
    for gpu in range(args.gpus):
        for worker in range(args.workers_per_gpu):
            slot = f'{gpu}-{worker}'
            child_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu),
                             TORCHINDUCTOR_CACHE_DIR=str(logs / slot / 'inductor'),
                             TRITON_CACHE_DIR=str(logs / slot / 'triton'))
            log = (logs / (slot + '.log')).open('x')
            cmd = [str(py), str(repo / 'scripts/c2r/evaluate.py'), args.mode, '--root', str(args.root)]
            child = subprocess.Popen(cmd, env=child_env, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log, slot))
    while any(child.poll() is None for child, _, _ in children):
        probe = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True)
        with (logs / 'gpu.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), gpu=probe.stdout.strip())) + '\n')
        time.sleep(10)
    codes = {slot: child.wait() for child, _, slot in children}
    for _, log, _ in children:
        log.close()
    (logs / 'exit.json').write_text(json.dumps(dict(codes=codes, seconds=time.time()-started, source_sha=sha)))
    if args.mode in ('rollout', 'pipeline'):
        with (logs / 'summary.log').open('x') as log:
            subprocess.run([str(py), str(repo / 'scripts/c2r/evaluate.py'), 'summary', '--root', str(args.root)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if any(codes.values()):
        raise RuntimeError(codes)


if __name__ == '__main__':
    main()
