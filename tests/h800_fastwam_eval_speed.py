"""Single-GPU frozen FastWAM rollout profiling and exact trace comparison."""
import argparse
import cProfile
import faulthandler
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import pstats
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
BASE = Path('/mnt/workdir/frank')
MANIFEST = BASE/'runs/fastwam-robotwin-stage3-u8000-mono-clean-eval-h100-20261102/manifest.json'


def digest(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return hashlib.sha256(str((value.shape, str(value.dtype))).encode()+value.tobytes()).hexdigest()
    if isinstance(value, dict):
        return {k: digest(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [digest(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(args):
    faulthandler.dump_traceback_later(120, repeat=True)
    logging.basicConfig(level=logging.INFO)
    import numpy as np
    spec = importlib.util.spec_from_file_location('frozen_eval', REPO/'scripts/c2r/evaluate.py')
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    official, device = e.setup_runtime()
    import torch
    from fastwam_c2r.policy import C2RPolicy
    manifest = json.loads(MANIFEST.read_text())
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,pci.device_id,compute_cap,memory.total', '--format=csv,noheader'], text=True)
    assert len(gpu.strip().splitlines()) == 1, gpu
    contract = dict(gpu=gpu, checkpoint=manifest['checkpoint'], training_root=manifest['training_root'],
                    sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(), variant=args.variant)
    (root/'contract.json').write_text(json.dumps(contract, indent=2))
    print('MODEL_LOADING', flush=True)
    policy = C2RPolicy(manifest['checkpoint'], manifest['training_root'])
    print('MODEL_READY', flush=True)
    lock = e.shared_plan_lock()
    plans = e.load_shared_plans(dict(episodes=100, cases=lock['cases']))
    original_environment = e.environment
    rows = {}

    def environment(*a):
        env, config = original_environment(*a)
        get_obs, take = env.get_obs, env.take_action
        def observe():
            start = time.perf_counter()
            value = get_obs()
            elapsed = time.perf_counter()-start
            rows['observations'].append(dict(value=digest(value), seconds=elapsed, dynamic_light=bool(env.crazy_random_light)))
            return value
        def action(value, **kw):
            start = time.perf_counter()
            result = take(value, **kw)
            elapsed = time.perf_counter()-start
            rows['actions'].append(dict(value=digest(np.asarray(value)), seconds=elapsed,
                state=digest([env.robot.left_entity.get_qpos(), env.robot.right_entity.get_qpos(),
                              env.robot.left_entity.get_qvel(), env.robot.right_entity.get_qvel()]),
                rng=digest([random.getstate(), np.random.get_state()])))
            return result
        env.get_obs, env.take_action = observe, action
        return env, config
    e.environment = environment
    cases = [('clean','click_bell',0), ('randomized','click_bell',0), ('randomized','click_bell',19)]
    if args.long:
        cases = [('clean','open_microwave',99), ('randomized','open_microwave',1)]
    for phase, task, ordinal in cases:
        key = f'{phase}-{task}-{ordinal}'
        rows = dict(observations=[], actions=[])
        case = next(c for c in lock['cases'] if c['phase']==phase and c['task']==task)
        episode = plans[task+'--'+phase]['episodes'][ordinal]
        profile = cProfile.Profile()
        torch.cuda.synchronize()
        if args.variant == 'profile': profile.enable()
        result = e.rollout_case(official, device, policy, case, [episode])[0]
        if args.variant == 'profile':
            profile.disable()
            profile.dump_stats(str(root/(key+'.prof')))
            with (root/(key+'-profile.txt')).open('w') as f:
                pstats.Stats(profile, stream=f).sort_stats('cumulative').print_stats(65)
        (root/(key+'.json')).write_text(json.dumps(dict(result=result, **rows)))
        print('COMPLETE', key, json.dumps(result), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--variant', choices=['profile','baseline'], default='profile')
    parser.add_argument('--long', action='store_true')
    args = parser.parse_args()
    main(args)
