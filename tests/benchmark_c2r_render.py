"""Compare eager/deferred rendering with identical closed-loop seeds and policy."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / 'scripts/c2r'))
import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default='click_bell')
    parser.add_argument('--episodes', type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'manifest.json').read_text())
    official, device = evaluate.setup_runtime()
    from fastwam_c2r.policy import C2RPolicy
    model = C2RPolicy(manifest['checkpoint'], manifest['training_root'])
    factory = official.class_decorator
    all_results = []
    for phase in ('clean', 'randomized'):
        case = dict(task=args.task, phase=phase)
        plan = json.loads((args.root / 'plans' / f'{args.task}--{phase}.json').read_text())
        runs = {}
        # Reverse the order in the second condition to expose warm-cache bias.
        for mode in (('eager', 'deferred') if phase == 'clean' else ('deferred', 'eager')):
            trace = dict(pixels=[], actions=[], states=[])
            def create(task):
                env = factory(task)
                update = env._update_render
                if mode == 'eager':
                    env._update_render = lambda sync_scene=True: update(sync_scene=True)
                get_obs = env.get_obs
                def observe():
                    obs = get_obs()
                    trace['pixels'].append({key: hashlib.sha256(value['rgb'].tobytes()).hexdigest()
                        for key, value in obs['observation'].items() if 'rgb' in value})
                    return obs
                env.get_obs = observe
                take_action = env.take_action
                def act(action, action_type):
                    trace['actions'].append(np.asarray(action).copy())
                    take_action(action, action_type=action_type)
                    trace['states'].append(np.concatenate([env.robot.get_left_ee_pose(), env.robot.get_right_ee_pose()]))
                env.take_action = act
                return env
            official.class_decorator = create
            policy_step = model.step
            if mode == 'eager':
                def uncached_step(env, obs):
                    model._prompt = None
                    return policy_step(env, obs)
                model.step = uncached_step
            result = evaluate.rollout_case(official, device, model, case, plan['episodes'][:args.episodes])
            model.step = policy_step
            runs[mode] = dict(result=result, trace=trace)
        official.class_decorator = factory
        a, b = runs['eager'], runs['deferred']
        same_actions = len(a['trace']['actions']) == len(b['trace']['actions']) and np.array_equal(a['trace']['actions'], b['trace']['actions'])
        same_states = len(a['trace']['states']) == len(b['trace']['states']) and np.array_equal(a['trace']['states'], b['trace']['states'])
        record = dict(phase=phase, eager=a['result'], deferred=b['result'],
                      same_pixels=a['trace']['pixels'] == b['trace']['pixels'], same_actions=same_actions, same_states=same_states)
        all_results.append(record)
        evaluate.write_json(args.output, all_results)
        print(json.dumps(record), flush=True)
        assert record['same_pixels'] and same_actions and same_states, record


if __name__ == '__main__':
    main()
