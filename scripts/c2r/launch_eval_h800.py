"""Launch independent, persistent C2R evaluation workers on allocated GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import base64
import inspect
import uuid


def retire_terminal_claims(root, states):
    """Archive only positively identified owners with authoritative terminal jobs."""
    retired = []
    for owner in (root / 'claims').glob('*/active/owner.json'):
        try:
            value = json.loads(owner.read_text())
            state = states.get(value.get('job'), {})
            if (not value.get('token') or state.get('JobId') != value.get('job')
                    or state.get('Status') not in ('Failed', 'Stopped', 'Succeeded')):
                continue
            if json.loads(owner.read_text()) != value:
                continue
            active = owner.parent
            target = active.parent / ('retired-' + uuid.uuid4().hex)
            active.rename(target)
            (target / 'retirement.json').write_text(json.dumps(
                dict(time=time.time(), owner=value, verified_job=state)))
            retired.append(active.parent.name)
        except FileNotFoundError:
            continue
    return retired


def supervise(registry, cli):
    """Host-side controller; credentials stay in the existing local CLI profile.

    A small CPU job performs each storage check. Its receipt is persisted before
    waiting, so restarting the controller never duplicates an unfinished check.
    """
    directory = registry.parent
    state_path = directory / (registry.stem + '-supervisor.json')
    lock = state_path.with_suffix('.lock')
    # An interrupted controller leaves evidence; an operator must verify its PID
    # is dead before archiving this lock and starting another controller.
    with lock.open('x') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), started=time.time())))
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def save():
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(state_path)

    def api(*arguments):
        run = subprocess.run([cli, 'pai-dlc', *arguments, '--profile', 'h800',
                              '--region', 'cn-shenzhen'], capture_output=True, text=True,
                             timeout=90, check=True)
        return json.loads(run.stdout)

    def submit(request, name):
        path = directory / (name + '.json')
        path.write_text(json.dumps(request))
        # Persist intent BEFORE sending. Unknown create outcomes must be resolved
        # by DisplayName; never automatically replay a possibly accepted POST.
        state['create_intent'] = str(path)
        save()
        receipt = api('CreateJob', '--body-file', str(path))
        (directory / (name + '-receipt.json')).write_text(json.dumps(receipt))
        state.pop('create_intent')
        return receipt['JobId']

    while True:
        try:
            if state.get('create_intent'):
                raise RuntimeError('Unresolved create intent; reconcile receipt before restarting')
            rows = json.loads(registry.read_text())
            assert sum(row['gpus'] for row in rows if row['kind'] == 'fixed') == 16
            assert sum(row['gpus'] for row in rows if row['kind'] == 'idle') <= 59
            identities = {row['JobId'] for row in rows}
            identities.update(row['old'] for row in state.get('replaced', []))
            states = {job: api('GetJob', '--JobId', job) for job in identities}
            compact = {j: {k: d.get(k) for k in ('JobId', 'Status', 'ReasonCode', 'ReasonMessage')}
                       for j, d in states.items()}
            state.update(checked_at=time.time(), jobs=compact)
            if not state.get('probe'):
                request = json.loads((directory / 'readonly-cpu.json').read_text(encoding='utf-8-sig'))
                source = inspect.getsource(retire_terminal_claims)
                script = ('import json,time,uuid,importlib.util\nfrom pathlib import Path\n' + source
                          + '\nr=Path(' + repr(state['root']) + ')\n'
                          + 'retired=retire_terminal_claims(r,' + repr(compact) + ')\n'
                          + 'spec=importlib.util.spec_from_file_location("evaluation",'
                          + repr(state['clone'] + '/scripts/c2r/evaluate.py') + ')\n'
                          + 'm=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\n'
                          + 's=m.summarize(r,json.loads((r/"manifest.json").read_text()))\n'
                          + 'print("SUPERVISOR "+json.dumps(dict(time=time.time(),retired=retired,'
                          + 'episodes=s["completed_episodes"],complete=s["complete"],'
                          + 'overall=s["overall"],errors=s["errors"])),flush=True)\n')
                request['UserCommand'] = ('python3 -c "import base64;exec(base64.b64decode(\''
                                          + base64.b64encode(script.encode()).decode() + '\'))"')
                name = 'c2r-watch-' + uuid.uuid4().hex[:12]
                request['DisplayName'] = name
                request['Settings']['OversoldType'] = 'ForbiddenQuotaOverSold'
                state['probe'] = submit(request, name)
                save()
            probe = api('GetJob', '--JobId', state['probe'])
            if probe['Status'] == 'Succeeded':
                logs = api('get-pod-logs', '--job-id', state['probe'], '--pod-id',
                           state['probe'] + '-master-0', '--containers', 'pytorch', '--max-lines', '100')
                reports = [json.loads(line[len('SUPERVISOR '):]) for line in logs.get('Logs', [])
                           if line.startswith('SUPERVISOR ')]
                if not reports:
                    raise RuntimeError('Completed probe has no readable report')
                state['progress'] = reports[-1]
                state.pop('probe')
                print(json.dumps(state['progress']), flush=True)
                if state['progress']['complete']:
                    save()
                    return
                if state['progress']['errors']:
                    raise RuntimeError('Evaluation errors require diagnosis; preserved in progress')
                # Replace fixed slots only after infrastructure preemption. Other
                # failures require diagnosis instead of an unbounded crash loop.
                for row in rows:
                    job = states[row['JobId']]
                    if job['Status'] != 'Failed':
                        continue
                    if job.get('ReasonCode') != 'JobEvicted':
                        state.setdefault('needs_diagnosis', {})[row['JobId']] = compact[row['JobId']]
                        continue
                    request = json.loads(Path(row['request_path']).read_text())
                    name = 'c2r-' + row['kind'] + '-replace-' + uuid.uuid4().hex[:12]
                    request['DisplayName'] = name
                    expected = 'ForbiddenQuotaOverSold' if row['kind'] == 'fixed' else 'ForceQuotaOverSold'
                    assert request['Settings']['OversoldType'] == expected
                    new_id = submit(request, name)
                    state.setdefault('replaced', []).append(dict(old=row['JobId'], new=new_id))
                    row['JobId'] = new_id
                    registry.write_text(json.dumps(rows, indent=2))
                    save()
            elif probe['Status'] in ('Failed', 'Stopped'):
                state['failed_probe'] = state.pop('probe')
            state.pop('error', None)
            save()
        except Exception as error:
            state.update(error=str(error), error_at=time.time())
            save()
            print('SUPERVISOR_ERROR', str(error), flush=True)
        time.sleep(60)


def main():
    if '--supervise' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--supervise', type=Path, required=True)
        parser.add_argument('--cli', required=True)
        args = parser.parse_args()
        supervise(args.supervise, args.cli)
        return
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
            cache = logs / slot
            for name in ('kernels', 'matplotlib'):
                (cache / name).mkdir(parents=True, exist_ok=True)
            child_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu),
                             PYTORCH_KERNEL_CACHE_PATH=str(cache / 'kernels'),
                             MPLCONFIGDIR=str(cache / 'matplotlib'),
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
