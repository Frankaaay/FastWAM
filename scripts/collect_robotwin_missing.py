"""Bounded collection of new stereo demonstrations for an exact missing-ID plan.

Uses the pinned RoboTwin runtime without editing it. Every attempt has a fresh
directory; unsuccessful attempts and caches are retained. Never deletes data.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


BASE = Path('/mnt/workdir/frank')
RUNTIME = BASE / 'projects/RoboTwin-render-runtime-a8810a0'
PYTHON = BASE / 'envs/robotwin-render-py310/bin/python'
DATA = BASE / 'datasets/RoboTwin2.0'
EXISTING = DATA / 'stereo-clean-fresh-v21-parallel-20261009-99f7b8c'
MIGRATION = BASE / 'runs/fastwam-render40-20261009-e2636ae/migration-manifest.json'
PAIRS = (('head_camera', .06), ('left_camera', .02), ('right_camera', .02))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    # Exclusive publication: neither existing data nor earlier evidence is replaced.
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def missing_rows(provenance, migration):
    ids = provenance['source_episode_indices']
    if len(ids) != 2461 or len(set(ids)) != 2461 or not set(ids) <= set(range(2500)):
        raise ValueError('Expected exactly 2461 unique original episode IDs')
    rows = [row for group in ('pending', 'accepted', 'candidates', 'excluded')
            for row in migration[group]]
    if len(rows) != 2500 or {r['canonical_episode'] for r in rows} != set(range(2500)):
        raise ValueError('Migration ledger must cover every canonical ID once')
    missing = sorted((r for r in rows if r['canonical_episode'] not in set(ids)),
                     key=lambda r: r['canonical_episode'])
    return [{k: r[k] for k in ('canonical_episode', 'task', 'episode', 'seed', 'replan')}
            for r in missing]


def plan(args):
    provenance = EXISTING / 'meta/stereo_provenance.json'
    rows = missing_rows(json.loads(provenance.read_text()), json.loads(MIGRATION.read_text()))
    output = Path(args.output).resolve()
    if DATA not in output.parents:
        raise ValueError('Collection output must be in the personal dataset directory')
    output.mkdir(exist_ok=False)
    hashes = {}
    for name in ('envs/_base_task.py', 'envs/utils/create_actor.py', 'envs/camera/__init__.py',
                 'envs/camera/camera.py', 'task_config/demo_clean.yml'):
        p = RUNTIME / name
        if p.is_file():
            hashes[name] = digest(p)
    for r in rows:
        task_file = f"envs/{r['task']}.py"
        hashes[task_file] = digest(RUNTIME / task_file)
    write_json(output / 'plan.json', dict(rows=rows, existing=str(EXISTING),
        provenance_sha256=digest(provenance), migration_sha256=digest(MIGRATION),
        runtime=str(RUNTIME), runtime_hashes=hashes, output=str(output),
        source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        max_attempts=12, attempt_timeout_seconds=900, seed_base=1000, seed_stride=32,
        fps=15, pairing='fresh-left+fresh-right', recovery_kind='new_demonstration',
        baselines={'head_baseline_m': .06, 'left_wrist_baseline_m': .02, 'right_wrist_baseline_m': .02}))
    print(json.dumps(dict(plan=str(output / 'plan.json'), missing=len(rows))))


def environment(run):
    libs = BASE / 'envs/system-libs'
    prepared = BASE / 'runs/fastwam-h800-smoke-20261008/system-libs-prepared.json'
    env = dict(os.environ, LD_LIBRARY_PATH=json.loads(prepared.read_text())['ld_library_path'],
        PATH=str(libs / 'prefix/usr/bin') + ':' + str(PYTHON.parent) + ':/usr/local/cuda/bin:' + os.environ['PATH'],
        CUDA_HOME='/usr/local/cuda', TORCH_CUDA_ARCH_LIST='9.0', MAX_JOBS='2',
        VK_ICD_FILENAMES=str(libs / 'nvidia-icd.json'),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(libs / 'nvidia-egl-vendor.json'),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR',
                'CUDA_CACHE_PATH', '__GL_SHADER_DISK_CACHE_PATH'):
        target = run / key.lower()
        target.mkdir(exist_ok=False)
        env[key] = str(target)
    return env


def verify_raw(path):
    import cv2
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as f:
        q = f['joint_action/vector'][:]
        n = len(q)
        if q.shape != (n, 14) or n < 2 or not np.isfinite(q).all():
            raise ValueError('Invalid qpos')
        for side in ('left', 'right'):
            pose = f[f'endpose/{side}_endpose'][:]
            grip = f[f'endpose/{side}_gripper'][:].reshape(-1)
            if pose.shape != (n, 7) or grip.shape != (n,) or not np.isfinite(pose).all() or not np.isfinite(grip).all():
                raise ValueError('Invalid EEF/gripper data')
            np.testing.assert_allclose(np.linalg.norm(pose[:, 3:], axis=1), 1, atol=1e-4, rtol=0)
        for left, baseline in PAIRS:
            right = left + '_right'
            kl, kr = (f[f'observation/{c}/intrinsic_cv'][:] for c in (left, right))
            el, er = (f[f'observation/{c}/extrinsic_cv'][:] for c in (left, right))
            if kl.shape != (n, 3, 3) or kr.shape != kl.shape or el.shape != (n, 3, 4) or er.shape != el.shape:
                raise ValueError('Invalid calibration shape')
            if not all(np.isfinite(x).all() for x in (kl, kr, el, er)):
                raise ValueError('Nonfinite calibration')
            np.testing.assert_allclose(kl, kr, atol=1e-6, rtol=0)
            rot = el[:, :3, :3]
            np.testing.assert_allclose(rot, er[:, :3, :3], atol=1e-5, rtol=0)
            np.testing.assert_allclose(rot @ rot.transpose(0, 2, 1), np.broadcast_to(np.eye(3), rot.shape), atol=1e-5, rtol=0)
            np.testing.assert_allclose(np.linalg.det(rot), 1, atol=1e-5, rtol=0)
            centers = [-np.einsum('nji,nj->ni', e[:, :3, :3], e[:, :3, 3]) for e in (el, er)]
            offset = np.einsum('nij,nj->ni', rot, centers[1] - centers[0])
            expected = np.zeros_like(offset); expected[:, 0] = baseline
            np.testing.assert_allclose(offset, expected, atol=1e-5, rtol=0)
            images = [f[f'observation/{c}/rgb'] for c in (left, right)]
            if any(len(x) != n for x in images):
                raise ValueError('Image count mismatch')
            changed = False
            for index in range(n):
                decoded = [cv2.imdecode(np.frombuffer(bytes(x[index]), np.uint8), cv2.IMREAD_COLOR) for x in images]
                if any(x is None or x.shape != (240, 320, 3) for x in decoded):
                    raise ValueError('Image decode/shape failure')
                changed |= not np.array_equal(*decoded)
            if not changed:
                raise ValueError('Duplicated stereo pair')
    return n


def actors(env):
    return [dict(name=a.get_name(), position=a.get_pose().p.tolist(), quaternion_wxyz=a.get_pose().q.tolist())
            for a in env.scene.get_all_actors()]


def attempt(args):
    import numpy as np
    import yaml
    plan_data = json.loads(Path(args.plan).read_text())
    row = next(r for r in plan_data['rows'] if r['canonical_episode'] == args.canonical)
    if not 0 <= args.attempt < plan_data['max_attempts']:
        raise ValueError('Attempt outside the finite budget')
    for name, expected in plan_data['runtime_hashes'].items():
        if digest(RUNTIME / name) != expected:
            raise ValueError(f'Runtime changed: {name}')
    seed = plan_data['seed_base'] + args.canonical * plan_data['seed_stride'] + args.attempt
    original_seeds = [int(x) for x in (DATA / 'canonical-replay-inputs' / row['task'] / 'demo_clean/seed.txt').read_text().split()]
    if seed in original_seeds:
        raise ValueError('New seed collides with an original demonstration')
    destination = Path(plan_data['output']) / f"canonical-{args.canonical:04d}" / f"attempt-{args.attempt:02d}"
    destination.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(RUNTIME)); os.chdir(RUNTIME)
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH
    config = yaml.safe_load((RUNTIME / 'task_config/demo_clean.yml').read_text())
    if config['embodiment'] != ['aloha-agilex'] or config['save_freq'] != 15:
        raise ValueError('Unexpected embodiment or frame cadence')
    embodiments = yaml.safe_load((Path(CONFIGS_PATH) / '_embodiment_config.yml').read_text())
    robot_file = embodiments['aloha-agilex']['file_path']
    embodiment = yaml.safe_load((Path(robot_file) / 'config.yml').read_text())
    options = dict(config, task_name=row['task'], task_config='demo_clean', save_path=str(destination),
        left_robot_file=robot_file, right_robot_file=robot_file, left_embodiment_config=embodiment,
        right_embodiment_config=embodiment, dual_arm_embodied=True, embodiment_name='aloha-agilex',
        need_plan=True, render_freq=0, save_data=True, stereo=dict(enabled=True, **plan_data['baselines']))
    random.seed(seed)
    env = getattr(importlib.import_module('envs.' + row['task']), row['task'])()
    env.setup_demo(now_ep_num=0, seed=seed, **options)
    initial_actors = actors(env)
    scene = env.play_once()
    if not env.plan_success or not env.check_success():
        raise RuntimeError('New demonstration planning/task success check failed')
    env.save_traj_data(0)
    final_actors = actors(env)
    env.merge_pkl_to_hdf5_video()
    # Preserve cache and all outputs. No remove_data_cache call.
    raw = destination / 'data/episode0.hdf5'
    n = verify_raw(raw)
    language = importlib.import_module('description.utils.generate_episode_instructions')
    random.seed(seed)
    instructions = language.generate_episode_descriptions(row['task'], [scene['info']], max_descriptions=100)
    if len(instructions) != 1 or not instructions[0]['seen']:
        raise ValueError('No scene-matched seen instructions')
    source = RUNTIME / 'description/task_instruction' / (row['task'] + '.json')
    write_json(destination / 'instructions.json', instructions[0])
    write_json(destination / 'scene_info.json', dict(scene=scene, initial_actors=initial_actors, final_actors=final_actors))
    # Explicit action/time contract alongside untouched simulator HDF5.
    import h5py
    with h5py.File(raw, 'r') as f:
        q = f['joint_action/vector'][:]
    np.savez(destination / 'frame_action_index.npz', frame_index=np.arange(n), timestamp=np.arange(n)/15,
             qpos=q, action_next_qpos=q[1:], action_frame_index=np.arange(n-1), target_frame_index=np.arange(1,n))
    marker = dict(canonical_episode=args.canonical, original=row, task=row['task'], episode=0, seed=seed,
        recovery_kind='new_demonstration', path=str(raw), length=n, windows=n-1,
        seen_instructions=instructions[0]['seen'], baselines=plan_data['baselines'],
        instruction_source=str(source), instruction_source_sha256=digest(source),
        plan_sha256=digest(args.plan), source_sha=plan_data['source_sha'],
        sha256=digest(raw), trajectory_sha256=digest(destination/'_traj_data/episode0.pkl'),
        success=True, full_new_episode_qa=True, fps=15)
    write_json(destination / 'accepted.json', marker)
    print(json.dumps(marker), flush=True)
    env.close_env(clear_cache=True)


def worker(args):
    p = Path(args.plan).resolve(); data = json.loads(p.read_text())
    if not 0 <= args.worker < 8:
        raise ValueError('Expected worker 0..7')
    run = Path(args.run).resolve()
    if BASE / 'runs' not in run.parents:
        raise ValueError('Run path must be in personal runs')
    run.mkdir(parents=True, exist_ok=False)
    env = environment(run)
    devices = subprocess.check_output([str(PYTHON), '-c', 'import torch; print(torch.cuda.device_count())'], env=env, text=True).strip()
    if devices != '1':
        raise ValueError(f'Expected exactly one GPU, found {devices}')
    rows = [r for i,r in enumerate(data['rows']) if i % 8 == args.worker]
    if args.only is not None:
        rows = [r for r in rows if r['canonical_episode'] == args.only]
        if len(rows) != 1:
            raise ValueError('Requested smoke ID not in this worker')
    write_json(run/'provenance.json',dict(plan=str(p),plan_sha256=digest(p),rows=rows,
        resource_id=args.resource_id or os.environ.get('DLC_JOB_ID'),started=time.time()))
    for row in rows:
        canonical = row['canonical_episode']
        slot = Path(data['output']) / f'canonical-{canonical:04d}'
        accepted = list(slot.glob('attempt-*/accepted.json')) if slot.exists() else []
        if accepted:
            if len(accepted) != 1 or json.loads(accepted[0].read_text())['plan_sha256'] != digest(p):
                raise ValueError('Ambiguous or foreign accepted episode')
            continue
        for number in range(data['max_attempts']):
            destination = slot / f'attempt-{number:02d}'
            if destination.exists():
                continue
            command = [str(PYTHON), str(Path(__file__).resolve()), 'attempt', '--plan', str(p),
                       '--canonical', str(canonical), '--attempt', str(number)]
            start = time.time()
            with (run / f'{canonical:04d}-{number:02d}.log').open('x') as log:
                try:
                    code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=data['attempt_timeout_seconds']).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            record = dict(canonical_episode=canonical, attempt=number, exit_code=code, seconds=time.time()-start)
            write_json(run / f'{canonical:04d}-{number:02d}.json', record)
            print(json.dumps(record), flush=True)
            if code == 0 and (destination / 'accepted.json').is_file():
                break
            tail = (run / f'{canonical:04d}-{number:02d}.log').read_text(errors='replace')[-12000:]
            if 'No space left on device' in tail or 'Disk quota exceeded' in tail:
                raise RuntimeError('Storage exhausted; stopping without deleting partial outputs')
            if not any(x in tail for x in ('New demonstration planning/task success check failed', 'UnStableError')):
                raise RuntimeError(f'Unexpected failure (not a new-seed retry): {canonical}/{number}')
        else:
            raise RuntimeError(f'Finite attempt budget exhausted: {canonical}')
    write_json(run/'completed.json',dict(worker=args.worker,completed_ids=[r['canonical_episode'] for r in rows]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('plan'); p.add_argument('--output', required=True)
    p = subs.add_parser('attempt'); p.add_argument('--plan', required=True)
    p.add_argument('--canonical', type=int, required=True); p.add_argument('--attempt', type=int, required=True)
    p = subs.add_parser('worker'); p.add_argument('--plan', required=True)
    p.add_argument('--worker', type=int, required=True); p.add_argument('--run', required=True)
    p.add_argument('--resource-id'); p.add_argument('--only', type=int)
    args = parser.parse_args()
    {'plan': plan, 'attempt': attempt, 'worker': worker}[args.command](args)
