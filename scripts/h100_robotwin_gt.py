"""Replay the accepted H800 2500 ledger with same-capture native GT on H100.

Export reads only the published personal H800 inputs. Preparation and rendering
write a fresh personal H100 output. No deletion, replacement, or replanning.
"""
import argparse
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tarfile
import time

ROOT = Path('/gpfs/jiuquyun/projects/Frank/robotwin-stereo-gt-20261014')
REPO = Path(__file__).resolve().parents[1]
PAIRS = (('head_camera', .06), ('left_camera', .02), ('right_camera', .02))


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def export_h800():
    import h5py
    import numpy as np
    import zipfile
    base = Path('/mnt/workdir/frank/datasets/RoboTwin2.0')
    publication = base / 'stereo-fill39-20261013/published'
    manifest_bytes = (publication / 'manifest.json').read_bytes()
    expected = 'ea55afb8f616b860633d97185fc2eb390d82bbbfdb458bfeefa4cc2fb22ee90e'
    assert hashlib.sha256(manifest_bytes).hexdigest() == expected
    manifest = json.loads(manifest_bytes)
    ready = json.loads((publication / 'reader-accepted.json').read_text())
    assert ready['passed'] and ready['manifest_sha256'] == expected
    rows = manifest['episodes']
    assert len(rows) == 2500 and {r['source_episode_index'] for r in rows} == set(range(2500))
    assert set(collections.Counter(r['task'] for r in rows).values()) == {50}
    scenes = {}
    packed = []
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
        def add(name, data):
            info = tarfile.TarInfo(name); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for row in rows:
            index, task = row['source_episode_index'], row['task']
            raw = base / row['path']; folder = raw.parent.parent
            replacement = row['recovery_kind'] == 'new_demonstration'
            marker = json.loads((folder / ('accepted.json' if replacement else 'verified.json')).read_text())
            seed = marker['seed']; episode = marker['episode']
            if replacement:
                trajectory = folder / '_traj_data/episode0.pkl'
                scene = json.loads((folder / 'scene_info.json').read_text())['scene']
            else:
                trajectory = base / 'canonical-replay-inputs' / task / 'demo_clean/_traj_data' / f'episode{episode}.pkl'
                if task not in scenes:
                    with zipfile.ZipFile('/mnt/data/data/RoboTwin2.0/dataset/' + task + '/demo_clean.zip') as z:
                        names = [n for n in z.namelist() if n.endswith('/scene_info.json')]
                        assert len(names) == 1
                        scenes[task] = json.loads(z.read(names[0]))
                scene = scenes[task][f'episode_{episode}']
            trajectory_bytes = trajectory.read_bytes()
            with h5py.File(raw, 'r') as f:
                q = f['joint_action/vector'][:]
            assert q.shape == (row['length'], 14) and np.isfinite(q).all()
            payload = io.BytesIO(); np.save(payload, q, allow_pickle=False)
            add(f'{index:04d}/trajectory.pkl', trajectory_bytes)
            add(f'{index:04d}/qpos.npy', payload.getvalue())
            models = {}
            for value in scene['info'].values():
                match = re.fullmatch(r'(.+)/base(\d+)', str(value))
                if match:
                    models.setdefault(match[1], []).append(int(match[2]))
            packed.append(dict(row, seed=seed, episode=episode, scene_info=scene,
                               recorded_actor_models=models, trajectory_sha256=hashlib.sha256(trajectory_bytes).hexdigest()))
            if len(packed) % 100 == 0:
                print(f'Exported {len(packed)}/2500', file=sys.stderr, flush=True)
        add('plan.json', json.dumps(dict(source_manifest_sha256=expected, rows=packed,
            fps=15, resolution=[240, 320], pairing='fresh-left+fresh-right',
            expected_frames=551534, shader='rt', spp=32, path_depth=8, denoiser='oidn')).encode())


def prepare():
    import zipfile
    ROOT.mkdir(exist_ok=True)
    inputs = ROOT / 'inputs'; inputs.mkdir(exist_ok=False)
    with tarfile.open(ROOT / 'h800-inputs.tar.gz') as tar:
        for item in tar:
            target = (inputs / item.name).resolve()
            if not target.is_relative_to(inputs) or not item.isfile():
                raise ValueError('Unsafe input member')
            target.parent.mkdir(exist_ok=True)
            with target.open('xb') as f:
                f.write(tar.extractfile(item).read())
    runtime = ROOT / 'runtime'; runtime.mkdir()
    for name in ('envs', 'script', 'description'):
        (runtime / name).symlink_to(REPO / 'third_party/RoboTwin' / name, target_is_directory=True)
    (runtime / 'task_config').symlink_to('/gpfs/jiuquyun/projects/Frank/RoboTwin-stereo-reference/task_config', target_is_directory=True)
    assets = runtime / 'assets'; assets.mkdir()
    for name in ('embodiments.zip', 'objects.zip'):
        with zipfile.ZipFile('/gpfs/jiuquyun/datasets/robotwin/asset_archives/' + name) as z:
            for item in z.infolist():
                if name == 'embodiments.zip' and not item.filename.startswith('embodiments/aloha-agilex/'):
                    continue
                dest = (assets / item.filename).resolve()
                if not dest.is_relative_to(assets) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unsafe asset member')
                if item.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with dest.open('xb') as f, z.open(item) as source:
                        import shutil
                        shutil.copyfileobj(source, f)
    subprocess.run([sys.executable, str(runtime / 'script/update_embodiment_config_path.py')], cwd=runtime, check=True)
    binary = Path('/gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/train/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
    (ROOT / 'bin').mkdir(); (ROOT / 'bin/ffmpeg').symlink_to(binary)
    plan = json.loads((inputs / 'plan.json').read_text())
    assert len(plan['rows']) == 2500
    for row in plan['rows']:
        p = inputs / f"{row['source_episode_index']:04d}/trajectory.pkl"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['trajectory_sha256']
    write_json(ROOT / 'prepared.json', dict(episodes=2500, source_sha=source_sha(),
        input_sha256=hashlib.sha256((ROOT / 'h800-inputs.tar.gz').read_bytes()).hexdigest()))


def source_sha():
    return subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()


def native_gt(camera, baseline, focal):
    import numpy as np
    position = camera.get_picture('Position')
    z = -position[..., 2]
    valid = np.isfinite(position[..., :3]).all(-1) & (z > 0)
    depth = np.where(valid, z, 0).astype(np.float32)
    if not valid.any():
        raise ValueError('Camera has no finite positive depth')
    labels = camera.get_picture('Segmentation')
    disparity = np.zeros_like(depth)
    np.divide(focal * baseline, depth, out=disparity, where=valid)
    return dict(depth_m=depth, valid=valid.astype(np.uint8), disparity_px=disparity,
                mesh_id=labels[..., 0].astype(np.uint32), actor_id=labels[..., 1].astype(np.uint32))


def stereo_visibility(left, right):
    """Nearest-pixel cross-eye depth agreement; outside-FOV has a separate mask."""
    import numpy as np
    h, w = left['depth_m'].shape
    yy, xx = np.indices((h, w))
    for a, b, sign in ((left, right, -1), (right, left, 1)):
        projected = np.rint(xx + sign * a['disparity_px']).astype(np.int32)
        inside = (projected >= 0) & (projected < w) & (a['valid'] > 0)
        sampled = b['depth_m'][yy, np.clip(projected, 0, w - 1)]
        tolerance = np.maximum(.002, .005 * a['depth_m'])
        # Label only positively observed occluders; missing target depth is unknown.
        a['occluded'] = (inside & (sampled > 0) & (sampled + tolerance < a['depth_m'])).astype(np.uint8)
        a['outside_right_or_left_fov'] = ((a['valid'] > 0) & ~inside).astype(np.uint8)


class Capture:
    def __init__(self, destination, expected):
        import h5py
        self.file = h5py.File(destination / 'episode.hdf5', 'x')
        self.expected = expected
        self.index = 0
        self.encoders = {}
        self.destination = destination

    def append(self, env):
        import cv2
        import numpy as np
        obs = env.get_obs()
        cameras = dict(zip(env.cameras.static_camera_name, env.cameras.static_camera_list))
        cameras.update(left_camera=env.cameras.left_camera, right_camera=env.cameras.right_camera)
        cameras.update({name: camera for name, _, camera, _ in env.cameras.stereo_pairs})
        for name, baseline in PAIRS:
            names = (name, name + '_right')
            gt = [native_gt(cameras[c], baseline, obs['observation'][c]['intrinsic_cv'][0, 0]) for c in names]
            stereo_visibility(*gt)
            for c, value in zip(names, gt):
                obs['observation'][c].update(value)
        np.testing.assert_allclose(obs['joint_action']['vector'], self.expected[self.index], atol=1e-5, rtol=0)
        def append_group(group, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    append_group(group.require_group(key), value)
                    continue
                array = np.asarray(value)
                if array.size == 0:
                    continue
                if key == 'rgb':
                    camera = group.name.rsplit('/', 1)[-1]
                    if camera not in self.encoders:
                        cmd = [str(ROOT / 'bin/ffmpeg'), '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo',
                               '-pix_fmt', 'rgb24', '-s', '320x240', '-r', '15', '-i', '-', '-an',
                               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0', '-pix_fmt', 'yuv444p',
                               '-threads', '1', str(self.destination / (camera + '.mp4'))]
                        self.encoders[camera] = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                    self.encoders[camera].stdin.write(array.tobytes())
                    ok, encoded = cv2.imencode('.jpg', array)
                    assert ok
                    import h5py
                    if key not in group:
                        group.create_dataset(key, (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
                    ds = group[key]; ds.resize(self.index + 1, axis=0); ds[self.index] = encoded.reshape(-1)
                else:
                    if key not in group:
                        group.create_dataset(key, (0, *array.shape), maxshape=(None, *array.shape),
                                             dtype=array.dtype, chunks=(1, *array.shape), compression='gzip',
                                             compression_opts=1, shuffle=True)
                    ds = group[key]; ds.resize(self.index + 1, axis=0); ds[self.index] = array
        append_group(self.file, obs)
        self.index += 1
        if self.index % 30 == 0:
            self.file.flush(); print('FRAMES', self.index, flush=True)

    def close(self):
        import numpy as np
        self.file.create_dataset('frame_index', data=np.arange(self.index))
        self.file.create_dataset('timestamp', data=np.arange(self.index) / 15)
        self.file.attrs['depth_units'] = 'meters'
        self.file.attrs['depth_definition'] = 'camera_z'
        self.file.close()
        errors = []
        for name, encoder in self.encoders.items():
            encoder.stdin.close()
            if encoder.wait(timeout=120) != 0:
                errors.append(name)
        if errors:
            raise RuntimeError(f'Video encoding failed: {errors}')


def episode(index):
    import importlib
    import numpy as np
    import random
    import sapien
    import yaml
    plan = json.loads((ROOT / 'inputs/plan.json').read_text())
    row = next(r for r in plan['rows'] if r['source_episode_index'] == index)
    episode_root = ROOT / 'episodes' / f'{index:04d}'
    episode_root.mkdir(parents=True, exist_ok=True)
    if (episode_root / 'accepted.json').exists():
        raise FileExistsError('Episode is already accepted')
    attempt_number = 0
    while (episode_root / f'attempt-{attempt_number:02d}').exists():
        attempt_number += 1
    destination = episode_root / f'attempt-{attempt_number:02d}'
    destination.mkdir()
    runtime = ROOT / 'runtime'
    os.chdir(runtime); sys.path.insert(0, str(runtime))
    config = yaml.safe_load((runtime / 'task_config/demo_clean.yml').read_text())
    assert config['save_freq'] == 15 and config['embodiment'] == ['aloha-agilex']
    assert not any(config['domain_randomization'][k] for k in ('random_background', 'cluttered_table', 'random_light'))
    robot_file = str(runtime / 'assets/embodiments/aloha-agilex')
    robot_config = yaml.safe_load((Path(robot_file) / 'config.yml').read_text())
    inputs = ROOT / 'inputs' / f'{index:04d}'
    qpos = np.load(inputs / 'qpos.npy', allow_pickle=False)
    trajectory = pickle.loads((inputs / 'trajectory.pkl').read_bytes())
    capture = Capture(destination, qpos)
    task_cls = getattr(importlib.import_module('envs.' + row['task']), row['task'])
    class Task(task_cls):
        def _take_picture(self):
            if self.save_data:
                capture.append(self)
                self.FRAME_IDX += 1
    options = dict(config, task_name=row['task'], task_config='demo_clean', save_path=str(destination),
        left_robot_file=robot_file, right_robot_file=robot_file, left_embodiment_config=robot_config,
        right_embodiment_config=robot_config, dual_arm_embodied=True, embodiment_name='aloha-agilex',
        need_plan=False, render_freq=0, save_data=True, render_device=sapien.Device('cuda:0'),
        recorded_actor_models=row['recorded_actor_models'],
        stereo=dict(enabled=True, head_baseline_m=.06, left_wrist_baseline_m=.02, right_wrist_baseline_m=.02),
        **trajectory)
    random.seed(row['seed'])
    env = Task(); start = time.monotonic()
    try:
        env.setup_demo(now_ep_num=row['episode'], seed=row['seed'], **options)
        env.set_path_lst(options)
        scene = env.play_once()
        assert env.check_success(), 'Task success check failed'
        assert capture.index == len(qpos), (capture.index, len(qpos))
    finally:
        capture.close()
        env.close_env(clear_cache=True)
    from collect_robotwin_missing import verify_raw
    # JPEG vlen arrays are decoded from their byte buffers by the existing validator.
    assert verify_raw(destination / 'episode.hdf5') == len(qpos)
    for camera in capture.encoders:
        checked = subprocess.run([str(ROOT / 'bin/ffmpeg'), '-v', 'error', '-xerror',
            '-i', str(destination / (camera + '.mp4')), '-progress', 'pipe:1', '-f', 'null', '-'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        counts = re.findall(r'^frame=(\d+)$', checked.stdout, re.MULTILINE)
        assert counts and int(counts[-1]) == len(qpos), (camera, counts[-1:])
    write_json(episode_root / 'accepted.json', dict(row, path=str(destination / 'episode.hdf5'),
        source_sha=source_sha(), seconds=time.monotonic()-start, scene_info=scene,
        gt=dict(depth='camera Z in meters float32; invalid=0', segmentation='raw uint32 simulator IDs',
                disparity='positive fx*baseline/Z in pixels',
                occlusion='cross-eye nearest-pixel depth test; tolerance=max(2mm,0.5%Z)',
                timestamps='frame_index/15', fps=15)))


def pool(worker, limit):
    import fcntl
    plan = json.loads((ROOT / 'inputs/plan.json').read_text())
    rows = sorted(plan['rows'], key=lambda r: (-r['length'], r['source_episode_index']))
    if limit:
        rows = [next(r for r in rows if r['source_episode_index'] == i) for i in (0, 16)]
    claims = ROOT / 'claims'; claims.mkdir(exist_ok=True)
    logs = ROOT / 'logs'; logs.mkdir(exist_ok=True)
    failures = []
    for row in rows:
        index = row['source_episode_index']
        if (ROOT / 'episodes' / f'{index:04d}/accepted.json').exists():
            continue
        with (claims / f'{index:04d}.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if (ROOT / 'episodes' / f'{index:04d}/accepted.json').exists():
                continue
            log_path = logs / f'{index:04d}-{os.environ["SLURM_JOB_ID"]}.log'
            with log_path.open('x') as log:
                try:
                    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'episode', '--index', str(index)],
                        stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            print(json.dumps(dict(index=index, worker=worker, exit_code=code)), flush=True)
            if code:
                failures.append(index)
                # Fail early on infrastructure or systematic mismatch, preserving all outputs.
                raise RuntimeError(f'Episode {index} failed; see {log_path}')
    write_json(ROOT / f'worker-{worker}-{os.environ["SLURM_JOB_ID"]}.json', dict(failed=failures))


def finalize():
    plan = json.loads((ROOT / 'inputs/plan.json').read_text())
    accepted = [json.loads((ROOT / 'episodes' / f"{r['source_episode_index']:04d}/accepted.json").read_text())
                for r in plan['rows']]
    assert len(accepted) == 2500
    assert {r['source_episode_index'] for r in accepted} == set(range(2500))
    assert set(collections.Counter(r['task'] for r in accepted).values()) == {50}
    assert sum(r['length'] for r in accepted) == plan['expected_frames']
    write_json(ROOT / 'manifest.json', dict(episodes=accepted, source_manifest_sha256=plan['source_manifest_sha256'],
        frames=sum(r['length'] for r in accepted), fps=15, resolution=[240, 320]))
    write_json(ROOT / 'ready.json', dict(episodes=2500, tasks=50, frames=plan['expected_frames'],
        source_sha=source_sha(), manifest_sha256=hashlib.sha256((ROOT / 'manifest.json').read_bytes()).hexdigest(),
        validation='same-source qpos, task success, six-camera calibration, full JPEG and MP4 decode',
        gt='same-capture depth_m, valid, disparity_px, mesh_id, actor_id, occluded, outside_right_or_left_fov'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['export', 'prepare', 'episode', 'pool', 'finalize'])
    parser.add_argument('--index', type=int)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.mode == 'export': export_h800()
    elif args.mode == 'prepare': prepare()
    elif args.mode == 'episode': episode(args.index)
    elif args.mode == 'finalize': finalize()
    else: pool(args.worker, args.smoke)
