"""Frozen RoboTwin seed plans and dynamically claimed C2R rollout chunks."""
import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import subprocess
import time
import traceback


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def claim(root, key):
    path = root / 'claims' / key
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return False
    write_json(path / 'owner.json', dict(pid=os.getpid(), job=os.environ.get('DLC_JOB_ID'), at=time.time()))
    return True


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, list) else value


def ready_cases(root, cases):
    """Consume completed plans while other workers finish their expert checks."""
    pending = list(cases)
    deadline = time.monotonic() + 6 * 3600
    while pending:
        ready = [case for case in pending if (root / 'plans' / (case['task'] + '--' + case['phase'] + '.json')).exists()]
        for case in ready:
            pending.remove(case)
            yield case
        if pending and not ready:
            failures = list((root / 'errors').glob('prepare--*.json'))
            if failures:
                raise RuntimeError(f'Expert preparation failed: {failures}')
            if time.monotonic() > deadline:
                raise TimeoutError('Expert preparation did not finish within six hours')
            time.sleep(10)


def setup_runtime():
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
    return official.class_decorator(task), args


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


def rollout_case(official, device, model, case, episodes):
    import numpy as np
    env, config = environment(official, device, case['task'], case['phase'])
    results = []
    for episode in episodes:
        start = time.time()
        random.setstate(tuples(episode['python_before_scene']))
        env.setup_demo(now_ep_num=episode['ordinal'], seed=episode['seed'], is_test=True, **config)
        random.setstate(tuples(episode['python_after_instruction']))
        state = episode['numpy_after_instruction']
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
        env.set_instruction(episode['instruction'])
        model.reset()
        observations = 0
        setup_seconds = time.time() - start
        observation_seconds = 0.
        action_seconds = 0.
        while env.take_action_cnt < env.step_lim:
            tick = time.perf_counter()
            obs = env.get_obs() if model.should_request_observation() else None
            observation_seconds += time.perf_counter() - tick
            observations += int(obs is not None)
            tick = time.perf_counter()
            model.step(env, obs)
            action_seconds += time.perf_counter() - tick
            if env.eval_success:
                break
        result = dict(ordinal=episode['ordinal'], seed=episode['seed'], instruction=episode['instruction'],
                      success=bool(env.eval_success), actions=env.take_action_cnt, observations=observations,
                      inference_seconds=sum(model.inference_seconds), setup_seconds=setup_seconds,
                      observation_seconds=observation_seconds, action_seconds=action_seconds, seconds=time.time()-start)
        env.close_env(clear_cache=((episode['ordinal'] + 2) % config['clear_cache_freq'] == 0))
        results.append(result)
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
            plan = json.loads((root / 'plans' / (case['task'] + '--' + case['phase'] + '.json')).read_text())
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
    write_json(root / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'rollout', 'pipeline', 'summary'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if args.mode == 'pipeline':
        subprocess.run([sys.executable, __file__, 'prepare', '--root', str(root)], check=True)
        args.mode = 'rollout'
    if args.mode == 'summary':
        import fcntl
        with (root / 'aggregate.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            print(json.dumps(summarize(root, manifest), indent=2))
        return
    official, device = setup_runtime()
    model = None
    errors = []
    cases = manifest['cases'] if args.mode == 'prepare' else ready_cases(root, manifest['cases'])
    for case in cases:
        key = case['task'] + '--' + case['phase']
        if args.mode == 'prepare':
            if not claim(root, 'prepare--' + key):
                continue
            logpath = root / 'logs' / ('prepare--' + key + '.log')
            logpath.parent.mkdir(parents=True, exist_ok=True)
            try:
                with logpath.open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    plan = prepare_case(official, device, case, manifest['episodes'])
                write_json(root / 'plans' / (key + '.json'), plan)
            except Exception:
                error = traceback.format_exc()
                write_json(root / 'errors' / ('prepare--' + key + '.json'), dict(error=error))
                errors.append(error)
                print(error, flush=True)
            continue
        plan = json.loads((root / 'plans' / (key + '.json')).read_text())
        for offset in range(0, manifest['episodes'], manifest['chunk_size']):
            chunkkey = key + f'--{offset:03d}'
            if not claim(root, 'rollout--' + chunkkey):
                continue
            started = time.time()
            record = dict(**case, offset=offset, job=os.environ.get('DLC_JOB_ID'), pid=os.getpid())
            logpath = root / 'logs' / ('rollout--' + chunkkey + '.log')
            logpath.parent.mkdir(parents=True, exist_ok=True)
            try:
                with logpath.open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    if model is None:
                        from fastwam_c2r.policy import C2RPolicy
                        model = C2RPolicy(manifest['checkpoint'], manifest['training_root'])
                    record['episodes'] = rollout_case(official, device, model, case,
                        plan['episodes'][offset:offset+manifest['chunk_size']])
            except Exception:
                record['error'] = traceback.format_exc()
                errors.append(record['error'])
                print(record['error'], flush=True)
            record['seconds'] = time.time() - started
            write_json(root / 'results' / (chunkkey + '.json'), record)
            print('DONE', chunkkey, record['seconds'], bool(record.get('error')), flush=True)
            if record.get('error'):
                break
        if errors:
            break
    if errors:
        raise RuntimeError(errors[0])


if __name__ == '__main__':
    main()
