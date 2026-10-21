"""GPU acceptance only: two frozen scenes and bounded native policy calls, no score."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shared_evaluation', REPO / 'scripts/c2r/evaluate.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initialize-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    plans = evaluation.load_shared_plans(manifest)
    official, device = evaluation.setup_runtime()
    model = None if args.initialize_only else evaluation.make_policy(manifest)
    records = []
    for phase in ('clean', 'randomized'):
        case = next(c for c in manifest['cases'] if c['task'] == 'adjust_bottle' and c['phase'] == phase)
        episode = plans['adjust_bottle--' + phase]['episodes'][0]
        env, config = evaluation.environment(official, device, case['task'], phase)
        random.setstate(evaluation.tuples(episode['python_before_scene']))
        env.setup_demo(now_ep_num=0, seed=episode['seed'], is_test=True, **config)
        try:
            evaluation.initialize_task_state(case['task'], env)
            random.setstate(evaluation.tuples(episode['python_after_instruction']))
            state = episode['numpy_after_instruction']
            np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
            env.set_instruction(episode['instruction'])
            assert env.step_lim == case['step_limit']
            scoped = evaluation.EnvironmentRng(env)
            if model is not None:
                model.reset()
                assert model.should_request_observation()
            observation = scoped.get_obs()
            digest = hashlib.sha256()
            cameras = {}
            for name, value in sorted(observation['observation'].items()):
                if 'rgb' not in value:
                    continue
                rgb = value['rgb']
                digest.update(json.dumps([name, list(rgb.shape), str(rgb.dtype)]).encode())
                digest.update(np.ascontiguousarray(rgb).tobytes())
                cameras[name] = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
            assert len(cameras) == 6, cameras
            record = dict(model=manifest['model'], phase=phase, seed=episode['seed'],
                          instruction=episode['instruction'], scene=evaluation.initial_scene_digest(env),
                          observation=digest.hexdigest(), cameras=cameras, step_limit=env.step_lim)
            print('INITIAL_SCENE', json.dumps(record), flush=True)
            if model is not None:
                native_take_action = env.take_action
                actions = []
                def checked_action(action, *positional, **kwargs):
                    value = np.asarray(action)
                    assert value.shape == (16,) and np.isfinite(value).all(), value
                    assert env.take_action_cnt < env.step_lim
                    actions.append(value.tolist())
                    return native_take_action(action, *positional, **kwargs)
                env.take_action = checked_action
                for index in range(2):
                    if index:
                        observation = scoped.get_obs() if model.should_request_observation() else None
                    model.step(scoped, observation)
                    assert env.step_lim == case['step_limit']
                    if env.eval_success or env.take_action_cnt >= env.step_lim:
                        break
                assert actions
                record.update(actions=len(actions), first_action=actions[0],
                              inference_seconds=model.inference_seconds,
                              complete_episode=False)
            records.append(record)
        finally:
            env.close_env(clear_cache=True)
    args.output.write_text(json.dumps(dict(shared_plan_sha256=evaluation.SHARED_PLAN_SHA256,
                                         initialize_only=args.initialize_only, records=records), indent=2))
    print('SHARED_RUNTIME_ACCEPTANCE_COMPLETE', str(args.output), flush=True)


if __name__ == '__main__':
    main()
