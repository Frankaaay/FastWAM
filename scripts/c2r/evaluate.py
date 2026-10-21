"""Frozen RoboTwin seed plans and dynamically claimed C2R rollout chunks."""
import argparse
import hashlib
import contextlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import uuid
import subprocess
from copy import deepcopy
from functools import wraps
from inspect import signature, Parameter

SHARED_PLAN_SHA256 = '42ba67ecad191a5910a0c783569509eeaede422819db5cb6e792915e6743f84e'


def shared_plan_lock():
    path = Path(__file__).resolve().parents[2] / 'configs/robotwin_shared_plan.json'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHARED_PLAN_SHA256:
        raise ValueError('Shared RoboTwin plan lock was changed')
    return json.loads(data)


def load_shared_plans(manifest):
    lock = shared_plan_lock()
    if manifest['episodes'] != lock['episodes']:
        raise ValueError('Shared plan requires 100 episodes per task')
    expected = {(c['task'], c['phase']): c for c in lock['cases']}
    selected = [(c['task'], c['phase']) for c in manifest['cases']]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Empty or duplicate evaluation cases')
    phases = {phase for _, phase in selected}
    if set(selected) != {key for key in expected if key[1] in phases}:
        raise ValueError('Every selected phase must cover all 50 frozen tasks')
    plans = {}
    for case in manifest['cases']:
        if case != expected.get((case['task'], case['phase'])):
            raise ValueError('Task, phase or step limit differs from frozen plan')
        key = case['task'] + '--' + case['phase']
        data = (Path(lock['plan_root']) / (key + '.json')).read_bytes()
        if hashlib.sha256(data).hexdigest() != lock['files'][key + '.json']:
            raise ValueError('Frozen episode plan hash mismatch: ' + key)
        plan = json.loads(data)
        if (plan['task'], plan['phase']) != (case['task'], case['phase']):
            raise ValueError('Frozen plan identity mismatch: ' + key)
        if [e['ordinal'] for e in plan['episodes']] != list(range(100)):
            raise ValueError('Frozen plan coverage mismatch: ' + key)
        plans[key] = plan
    return plans


def validate_shared_results(root, plans, run_sha256):
    seen = set()
    for path in (root / 'results').glob('*.json'):
        record = json.loads(path.read_text())
        if record.get('shared_plan_sha256') != SHARED_PLAN_SHA256:
            raise ValueError('Result belongs to another evaluation protocol: ' + str(path))
        if record.get('run_sha256') != run_sha256:
            raise ValueError('Result belongs to another model/checkpoint configuration')
        key = record['task'] + '--' + record['phase']
        if key not in plans:
            raise ValueError('Unexpected result case: ' + key)
        for episode in record['episodes']:
            ordinal = episode['ordinal']
            if type(ordinal) is not int or not 0 <= ordinal < 100 or (key, ordinal) in seen:
                raise ValueError('Invalid or duplicate result ordinal')
            seen.add((key, ordinal))
            expected = plans[key]['episodes'][ordinal]
            if any(episode[k] != expected[k] for k in ('seed', 'instruction')):
                raise ValueError('Result seed/instruction differs from shared plan')
            if not episode.get('initial_scene_sha256') or type(episode['success']) is not bool:
                raise ValueError('Result lacks scene identity or boolean success')


def make_policy(manifest):
    kind = manifest['model']
    if kind == 'fastwam':
        from fastwam_c2r.policy import C2RPolicy
        return C2RPolicy(manifest['checkpoint'], manifest['training_root'])
    if kind == 'openwam':
        from benchmarks.robotwin.openwam2robotwin_interface import SharedPlanPolicy
    elif kind == 'lingbot':
        from XPolicyLab.policy.LingBot_VA.lingbot_va.evaluation.robotwin.shared_plan_policy import SharedPlanPolicy
    else:
        raise ValueError('Unknown shared-plan model: ' + kind)
    return SharedPlanPolicy(**manifest['policy'])


class EnvironmentRng:
    """Keep scene Python/NumPy RNG independent of policy/client implementation."""
    def __init__(self, env):
        import numpy as np
        self._env = env
        self._python = random.getstate()
        self._numpy = np.random.get_state()

    def __getattr__(self, name):
        value = getattr(self._env, name)
        if not callable(value):
            return value
        def call(*args, **kwargs):
            import numpy as np
            outside = random.getstate(), np.random.get_state()
            random.setstate(self._python)
            np.random.set_state(self._numpy)
            try:
                return value(*args, **kwargs)
            finally:
                self._python, self._numpy = random.getstate(), np.random.get_state()
                random.setstate(outside[0])
                np.random.set_state(outside[1])
        return call


def initial_scene_digest(env):
    actors = []
    for actor in env.scene.get_all_actors():
        pose = actor.get_pose()
        actors.append(dict(name=actor.get_name(), p=pose.p.tolist(), q=pose.q.tolist()))
    actors.sort(key=lambda x: (x['name'], x['p'], x['q']))
    value = dict(actors=actors, table_z_bias=float(env.table_z_bias),
                 step_limit=int(env.step_lim), instruction=env.get_instruction())
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


@contextlib.contextmanager
def claim(root, key):
    """Server-side mkdir exclusion across independently mounted DLC containers.

    Abruptly killed owners remain claimed until the job supervisor confirms
    termination and archives active; never steal a live owner's work by timeout.
    """
    path = root / 'claims' / key
    path.mkdir(parents=True, exist_ok=True)
    active = path / 'active'
    try:
        active.mkdir()
    except FileExistsError:
        yield False
        return
    token = uuid.uuid4().hex
    try:
        write_json(active / 'owner.json',
                   dict(pid=os.getpid(), job=os.environ.get('DLC_JOB_ID'), at=time.time(), token=token))
        yield True
    finally:
        active.rename(path / ('released-' + token))


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, list) else value


def ready_cases(root, cases, episodes=None, prepare=False):
    """Rescan unfinished work so released leases are reclaimed after preemption."""
    pending = list(cases)
    blocked = []
    deadline = time.monotonic() + 24 * 3600
    while pending:
        ready = []
        for case in list(pending):
            key = case['task'] + '--' + case['phase']
            completed_path = root / 'completed' / (key + '.json')
            has_plan = (root / 'plans' / (key + '.json')).exists()
            if not has_plan and (root / 'errors' / ('prepare--' + key + '.json')).exists():
                blocked.append(key)
                pending.remove(case)
                continue
            if episodes is not None:
                rows = [json.loads(p.read_text()) for p in (root / 'results').glob(key + '--*.json')]
                if any(row.get('error') for row in rows):
                    blocked.append(key)
                    pending.remove(case)
                    continue
                done = {ep['ordinal'] for row in rows for ep in row.get('episodes', [])}
                if done == set(range(episodes)):
                    write_json(completed_path, dict(episodes=episodes))
                    pending.remove(case)
                    continue
            if has_plan or prepare:
                ready.append(case)
        for case in ready:
            if episodes is None:
                pending.remove(case)
            yield case
        if pending:
            if time.monotonic() > deadline:
                raise TimeoutError('Evaluation did not finish within 24 hours')
            time.sleep(10)
    if blocked:
        raise RuntimeError(f'Rollout failures require review: {blocked}')


def isolate_mutable_defaults(method):
    # Ported from ruhu evaluation/aligned20/scene_defaults.py. Keep explicit
    # arguments explicit, and give each scene fresh copies of native defaults.
    parameters = signature(method)

    @wraps(method)
    def call(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        for name, parameter in parameters.parameters.items():
            default = parameter.default
            if (name not in bound.arguments and default is not Parameter.empty
                    and isinstance(default, (list, dict, set))):
                bound.arguments[name] = deepcopy(default)
        return method(*bound.args, **bound.kwargs)

    return call


def isolate_scene_defaults(env):
    method = env.get_cluttered_table
    parameters = signature(method).parameters
    expected = {'xlim': [-.59, .59], 'ylim': [-.34, .34], 'zlim': [.741]}
    for name, value in expected.items():
        if parameters[name].default != value:
            raise RuntimeError('Scene defaults already mutated; use a fresh worker')
    env.get_cluttered_table = isolate_mutable_defaults(method)


def setup_runtime():
    repo = Path(__file__).resolve().parents[2]
    tree = subprocess.check_output(['git', '-C', str(repo), 'rev-parse',
                                    'HEAD:third_party/RoboTwin'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain',
                                     '--', 'third_party/RoboTwin'], text=True).strip()
    if tree != shared_plan_lock()['robotwin_git_tree'] or dirty:
        raise ValueError('RoboTwin environment differs from the frozen FastWAM baseline')
    import sapien
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    repo = Path(__file__).resolve().parents[2]
    robotwin = repo / 'third_party/RoboTwin'
    os.chdir(robotwin)
    sys.path[:0] = [str(robotwin), str(robotwin / 'description/utils'), str(robotwin / 'script')]
    import eval_policy as official
    device = sapien.Device('cuda:0')
    return official, device


def environment(official, device, task, phase):
    import yaml
    args = yaml.safe_load(Path(f'task_config/demo_{phase}.yml').read_text())
    embodiments = yaml.safe_load(Path('task_config/_embodiment_config.yml').read_text())
    camera = official.get_camera_config(args['camera']['head_camera_type'])
    names = args['embodiment']
    if len(names) != 1:
        raise ValueError('Expected the trained dual-arm Aloha embodiment')
    robot_file = embodiments[names[0]]['file_path']
    embodiment = official.get_embodiment_config(robot_file)
    args.update(task_name=task, task_config='demo_' + phase, policy_name='fastwam_c2r.policy',
                left_robot_file=robot_file, right_robot_file=robot_file,
                left_embodiment_config=embodiment, right_embodiment_config=embodiment,
                dual_arm_embodied=True, head_camera_h=camera['h'], head_camera_w=camera['w'],
                eval_mode=True, render_freq=0, save_data=False, eval_video_log=False,
                render_device=device, stereo=dict(enabled=True, head_baseline_m=.06,
                left_wrist_baseline_m=.02, right_wrist_baseline_m=.02))
    env = official.class_decorator(task)
    isolate_scene_defaults(env)
    return env, args


def prepare_case(official, device, case, count):
    """Select first N expert-valid seeds, recording both scene and instruction RNG."""
    import numpy as np
    env, config = environment(official, device, case['task'], case['phase'])
    records, rejected = [], []
    seed = 100000
    random.seed(0)
    started = time.time()
    while len(records) < count:
        if seed >= 110000:
            raise RuntimeError(f'Could not prepare {count} valid episodes after 10000 candidates: {case}')
        ordinal = len(records)
        before = random.getstate()
        try:
            env.setup_demo(now_ep_num=ordinal, seed=seed, is_test=True, **config)
            info = env.play_once()
            valid = bool(env.plan_success and env.check_success())
            env.close_env()
            if not valid:
                rejected.append(dict(seed=seed, reason='expert_unsuccessful'))
                seed += 1
                continue
            before_rollout = random.getstate()
            env.setup_demo(now_ep_num=ordinal, seed=seed, is_test=True, **config)
            descriptions = official.generate_episode_descriptions(case['task'], [info['info']], 100)
            instruction = str(np.random.choice(descriptions[0]['unseen']))
            npstate = np.random.get_state()
            records.append(dict(ordinal=ordinal, seed=seed, instruction=instruction,
                                python_before_scene=before_rollout, python_after_instruction=random.getstate(),
                                numpy_after_instruction=[npstate[0], npstate[1].tolist(), *npstate[2:]]))
            env.close_env(clear_cache=((ordinal + 2) % config['clear_cache_freq'] == 0))
            print('PLAN', case['task'], case['phase'], ordinal, seed, flush=True)
        except official.UnStableError as exc:
            env.close_env()
            rejected.append(dict(seed=seed, reason=repr(exc), python_before=before))
        except Exception as exc:
            # Match official expert/setup rejection, never catch rollout failures.
            env.close_env()
            rejected.append(dict(seed=seed, reason=repr(exc), traceback=traceback.format_exc(), python_before=before))
            print('REJECT', case, seed, traceback.format_exc(), flush=True)
        seed += 1
    return dict(**case, episodes=records, rejected=rejected, seconds=time.time()-started)


def initialize_task_state(task, env):
    # Same scene-derived fields as the pinned RoboTwin experts/OpenWAM helper.
    if task not in ('open_laptop', 'place_object_scale', 'put_object_cabinet'):
        return
    module = sys.modules[type(env).__module__]
    if task == 'open_laptop':
        face = module.get_face_prod(env.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        env.arm_tag = module.ArmTag('left' if face > 0 else 'right')
    else:
        position = env.object.get_pose().p
        env.arm_tag = module.ArmTag('right' if position[0] > 0 else 'left')
        if task == 'put_object_cabinet':
            env.origin_z = position[2]


def rollout_case(official, device, model, case, episodes, on_episode=None):
    import numpy as np
    env, config = environment(official, device, case['task'], case['phase'])
    results = []
    for episode in episodes:
        start = time.time()
        random.setstate(tuples(episode['python_before_scene']))
        env.setup_demo(now_ep_num=episode['ordinal'], seed=episode['seed'], is_test=True, **config)
        initialize_task_state(case['task'], env)
        random.setstate(tuples(episode['python_after_instruction']))
        state = episode['numpy_after_instruction']
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
        env.set_instruction(episode['instruction'])
        if env.step_lim != case['step_limit']:
            raise ValueError('Environment step limit differs from frozen plan')
        scene_digest = initial_scene_digest(env)
        scoped_env = EnvironmentRng(env)
        model.reset()
        observations = 0
        initial_observation_sha256 = None
        setup_seconds = time.time() - start
        observation_seconds = 0.
        action_seconds = 0.
        while env.take_action_cnt < env.step_lim:
            tick = time.perf_counter()
            obs = scoped_env.get_obs() if model.should_request_observation() else None
            if initial_observation_sha256 is None:
                if obs is None:
                    raise ValueError('Every policy must request the frozen initial observation')
                digest = hashlib.sha256()
                cameras = [(name, value['rgb']) for name, value in sorted(obs['observation'].items()) if 'rgb' in value]
                if not cameras:
                    raise ValueError('Initial observation contains no camera images')
                for name, rgb in cameras:
                    digest.update(json.dumps([name, list(rgb.shape), str(rgb.dtype)]).encode())
                    digest.update(np.ascontiguousarray(rgb).tobytes())
                initial_observation_sha256 = digest.hexdigest()
            observation_seconds += time.perf_counter() - tick
            observations += int(obs is not None)
            tick = time.perf_counter()
            model.step(scoped_env, obs)
            if scoped_env.step_lim != case['step_limit']:
                raise ValueError('Policy attempted to change the shared step limit')
            action_seconds += time.perf_counter() - tick
            if env.eval_success:
                break
        result = dict(ordinal=episode['ordinal'], seed=episode['seed'], instruction=episode['instruction'],
                      success=bool(env.eval_success), actions=env.take_action_cnt, observations=observations,
                      initial_scene_sha256=scene_digest,
                      initial_observation_sha256=initial_observation_sha256,
                      inference_seconds=sum(model.inference_seconds), setup_seconds=setup_seconds,
                      observation_seconds=observation_seconds, action_seconds=action_seconds, seconds=time.time()-start)
        scoped_env.close_env(clear_cache=((episode['ordinal'] + 2) % config['clear_cache_freq'] == 0))
        results.append(result)
        if on_episode is not None:
            on_episode(result)
        print('EPISODE', json.dumps(result), flush=True)
    return results


def summarize(root, manifest):
    rows, errors, seen = [], [], set()
    for path in sorted((root / 'results').glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('error'):
            errors.append(record)
        for ep in record.get('episodes', []):
            key = (record['task'], record['phase'], ep['ordinal'])
            if key in seen:
                raise ValueError(f'Duplicate episode: {key}')
            seen.add(key)
            rows.append(dict(task=record['task'], phase=record['phase'], **ep))
    per_case = []
    for case in manifest['cases']:
        values = [r for r in rows if (r['task'], r['phase']) == (case['task'], case['phase'])]
        expected = {}
        if values:
            plan_root = Path(shared_plan_lock()['plan_root']) if manifest.get('shared_plan_sha256') else root / 'plans'
            plan = json.loads((plan_root / (case['task'] + '--' + case['phase'] + '.json')).read_text())
            expected = {ep['ordinal']: ep for ep in plan['episodes']}
        for value in values:
            episode = expected[value['ordinal']]
            if (value['seed'], value['instruction']) != (episode['seed'], episode['instruction']):
                raise ValueError(f'Seed/instruction mismatch: {case}, {value["ordinal"]}')
        complete = {v['ordinal'] for v in values} == set(range(manifest['episodes']))
        per_case.append(dict(**case, completed=len(values), successes=sum(v['success'] for v in values),
                             success_rate=sum(v['success'] for v in values)/len(values) if complete else None))
    overall = {}
    for phase in ('clean', 'randomized'):
        values = [r for r in per_case if r['phase'] == phase]
        overall[phase] = (sum(v['success_rate'] for v in values)/len(values)
                          if values and all(v['success_rate'] is not None for v in values) else None)
    result = dict(completed_episodes=len(rows), required_episodes=len(manifest['cases'])*manifest['episodes'],
                  complete=all(v['success_rate'] is not None for v in per_case) and not errors,
                  overall=overall, per_case=per_case, errors=errors)
    if manifest.get('shared_plan_sha256'):
        result['shared_plan_sha256'] = manifest['shared_plan_sha256']
    write_json(root / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['rollout', 'summary'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('model') not in ('fastwam', 'openwam', 'lingbot'):
        raise ValueError('manifest.json must explicitly name fastwam, openwam or lingbot')
    plans = load_shared_plans(manifest)
    if manifest.get('shared_plan_sha256') not in (None, SHARED_PLAN_SHA256):
        raise ValueError('Output manifest uses another shared plan')
    manifest['shared_plan_sha256'] = SHARED_PLAN_SHA256
    if type(manifest['chunk_size']) is not int or not 1 <= manifest['chunk_size'] <= 100:
        raise ValueError('Invalid evaluation chunk size')
    run_sha256 = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
    binding = dict(shared_plan_sha256=SHARED_PLAN_SHA256, run_sha256=run_sha256, manifest=manifest)
    binding_path = root / 'shared-plan-binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('Output directory is already bound to another model/configuration')
    validate_shared_results(root, plans, run_sha256)
    write_json(binding_path, binding)
    if args.mode == 'summary':
        import fcntl
        with (root / 'aggregate.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            print(json.dumps(summarize(root, manifest), indent=2))
        return
    official, device = setup_runtime()
    model = None
    errors = []
    cases = ready_cases(root, manifest['cases'], manifest['episodes'], prepare=True)
    for case in cases:
        key = case['task'] + '--' + case['phase']
        plan = plans[key]
        for offset in range(0, manifest['episodes'], manifest['chunk_size']):
            chunkkey = key + f'--{offset:03d}'
            resultpath = root / 'results' / (chunkkey + '.json')
            chunk = plan['episodes'][offset:offset+manifest['chunk_size']]
            if resultpath.exists():
                existing = json.loads(resultpath.read_text())
                if not existing.get('error') and {e['ordinal'] for e in existing.get('episodes', [])} == {e['ordinal'] for e in chunk}:
                    continue
            with claim(root, 'rollout--' + chunkkey) as acquired:
                if not acquired:
                    continue
                record = json.loads(resultpath.read_text()) if resultpath.exists() else dict(
                    **case, offset=offset, episodes=[], shared_plan_sha256=SHARED_PLAN_SHA256,
                    run_sha256=run_sha256)
                if record.get('error'):
                    continue
                chunk = plan['episodes'][offset:offset+manifest['chunk_size']]
                finished = {ep['ordinal'] for ep in record['episodes']}
                remaining = [ep for ep in chunk if ep['ordinal'] not in finished]
                if not remaining:
                    continue
                started = time.time()
                record.update(job=os.environ.get('DLC_JOB_ID'), pid=os.getpid())
                logpath = root / 'logs' / (f'rollout--{chunkkey}--{time.time_ns()}.log')
                logpath.parent.mkdir(parents=True, exist_ok=True)
                def save_episode(ep):
                    record['episodes'].append(ep)
                    write_json(resultpath, record)
                try:
                    with logpath.open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        if model is None:
                            model = make_policy(manifest)
                        rollout_case(official, device, model, case, remaining, on_episode=save_episode)
                except Exception:
                    record['error'] = traceback.format_exc()
                    errors.append(record['error'])
                    print(record['error'], flush=True)
                record['seconds'] = record.get('seconds', 0.) + time.time() - started
                write_json(resultpath, record)
                print('DONE', chunkkey, record['seconds'], bool(record.get('error')), flush=True)
                if record.get('error'):
                    break
        if errors:
            break
    if errors:
        raise RuntimeError(errors[0])


if __name__ == '__main__':
    main()
