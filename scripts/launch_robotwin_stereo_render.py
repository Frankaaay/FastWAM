"""Render remaining RoboTwin episodes in forty isolated single-GPU workers."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


BASE = Path('/mnt/workdir/frank')
RENDER_SHA = 'a8810a04caf7bbc9347b1751b96305cc16c4b945'


def selected_rows(manifest, worker):
    if not 0 <= worker < 40:
        raise ValueError('Expected worker 0..39')
    pending = manifest['pending']
    indices = [row['canonical_episode'] for name in ('pending', 'accepted', 'candidates', 'excluded')
               for row in manifest[name]]
    if len(indices) != 2500 or set(indices) != set(range(2500)):
        raise ValueError('Migration manifest must account for all 2500 source episodes')
    if any(row['replan'] for row in pending):
        raise ValueError('Missing trajectories were excluded by the user')
    return [row for row in pending if row['canonical_episode'] % 40 == worker]


def publish(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    os.replace(temporary, path)


def main(args):
    manifest = json.loads(Path(args.manifest).read_text())
    rows = selected_rows(manifest, args.worker)
    if args.plan:
        print(json.dumps(dict(worker=args.worker, pending=len(rows))))
        return
    run = Path(args.run_dir).resolve()
    output = Path(args.output_root).resolve()
    if BASE not in run.parents or BASE / 'datasets' not in output.parents:
        raise ValueError('Outputs must remain in the personal workspace')
    worker_dir = run / f'worker-{args.worker}'
    worker_dir.mkdir(parents=True, exist_ok=False)
    repo = BASE / 'projects/FastWAM-render-a8810a0'
    runtime = BASE / 'projects/RoboTwin-render-runtime-a8810a0'
    py = BASE / 'envs/robotwin-render-py310/bin/python'
    prep = BASE / 'runs/fastwam-h800-smoke-20261008'
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == RENDER_SHA
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    launcher = Path(__file__).resolve().parents[1]
    launcher_sha = subprocess.check_output(['git', '-C', str(launcher), 'rev-parse', 'HEAD'], text=True).strip()
    assert not subprocess.check_output(['git', '-C', str(launcher), 'status', '--porcelain'], text=True).strip()
    token = os.environ.get('DLC_JOB_ID', args.instance_id)
    if not token or not token.startswith(('dlc', 'dsw-')):
        raise ValueError('An exact personal job or instance ID is required')
    temporary = Path('/tmp/frank-fastwam-render40') / token / f'worker-{args.worker}'
    temporary.mkdir(parents=True, mode=0o700, exist_ok=False)
    libs = BASE / 'envs/system-libs'
    env = dict(os.environ, LD_LIBRARY_PATH=json.loads((prep / 'system-libs-prepared.json').read_text())['ld_library_path'],
               PATH=str(libs / 'prefix/usr/bin') + ':' + str(py.parent) + ':/usr/local/cuda/bin:' + os.environ.get('PATH', ''),
               CUDA_HOME='/usr/local/cuda', TORCH_CUDA_ARCH_LIST='9.0', MAX_JOBS='2',
               VK_ICD_FILENAMES=str(libs / 'nvidia-icd.json'),
               __EGL_VENDOR_LIBRARY_FILENAMES=str(libs / 'nvidia-egl-vendor.json'),
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TMPDIR=str(temporary))
    for key, name in [('XDG_CACHE_HOME', 'cache'), ('TORCH_EXTENSIONS_DIR', 'torch-extensions-renderer'),
                      ('TRITON_CACHE_DIR', 'triton-cache-renderer'), ('CUDA_CACHE_PATH', 'cuda-cache-renderer'),
                      ('__GL_SHADER_DISK_CACHE_PATH', 'nvidia-shader-cache-renderer')]:
        env[key] = str(BASE / 'envs' / name)
    devices = json.loads(subprocess.check_output([str(py), '-c',
        'import torch,json;print(json.dumps([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))'], env=env, text=True))
    assert len(devices) == 1, devices
    publish(worker_dir / 'provenance.json', dict(worker=args.worker, resource_id=token,
        launcher_sha=launcher_sha, render_sha=RENDER_SHA, devices=devices, selected=len(rows),
        manifest=args.manifest, manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        strict_pixel_mae=1, joint_atol=1e-5, pixel_candidates_retained=True,
        started=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    records = []
    start = time.time()
    for row in rows:
        index, task, episode = row['canonical_episode'], row['task'], row['episode']
        target = output / task / 'demo_clean' / f'episode{episode}'
        log_path = worker_dir / f'episode-{index:04d}.log'
        command = [str(py), str(repo / 'scripts/render_robotwin_stereo.py'), '--robotwin-root', str(runtime),
            '--source-root', str(BASE / 'datasets/RoboTwin2.0/canonical-replay-inputs'),
            '--output-root', str(output), '--task-config-path', str(runtime / 'task_config/demo_clean.yml'),
            '--tasks', task, '--num-shards', '50', '--shard-index', str(episode), '--replan-missing',
            '--canonical-archives-root', '/mnt/data/data/RoboTwin2.0/dataset']
        with log_path.open('x') as log:
            try:
                code = subprocess.run(command, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600).returncode
            except subprocess.TimeoutExpired:
                code = 124
        record = dict(canonical_episode=index, task=task, episode=episode, seed=row['seed'],
            exit_code=code, verified=code == 0 and (target / 'verified.json').exists(), output=str(target))
        if not record['verified']:
            record['exception_tail'] = '\n'.join(log_path.read_text().replace('\r', '\n').splitlines()[-18:])
        record['candidate'] = not record['verified'] and code == 1 and 'pixel MAE=' in record.get('exception_tail', '')
        records.append(record)
        progress = dict(worker=args.worker, resource_id=token, selected=len(rows), processed=len(records),
            verified=sum(item['verified'] for item in records), records=records,
            candidates=sum(item['candidate'] for item in records),
            failed=[item for item in records if not item['verified']], seconds=time.time() - start)
        publish(worker_dir / 'progress.json', progress)
        print(json.dumps({key: value for key, value in record.items() if key != 'exception_tail'}), flush=True)
    publish(worker_dir / 'completed.json', dict(worker=args.worker, records=records, seconds=time.time() - start))
    print('Rendering finished; pixel-rejected candidates are retained pending the pairing decision.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--worker', required=True, type=int)
    parser.add_argument('--run-dir')
    parser.add_argument('--output-root')
    parser.add_argument('--instance-id')
    parser.add_argument('--plan', action='store_true')
    main(parser.parse_args())
