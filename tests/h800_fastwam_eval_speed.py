"""Single-GPU frozen FastWAM rollout profiling and exact trace comparison."""
import argparse
import cProfile
import faulthandler
import gc
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
    if hasattr(value, 'detach'):
        import torch
        a = value.detach().contiguous().cpu()
        return hashlib.sha256(str((tuple(a.shape), str(a.dtype))).encode()+a.view(torch.uint8).numpy().tobytes()).hexdigest()
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

    def timed(name, fn):
        def call(*a, **kw):
            start = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                row = rows['timings'].setdefault(name, [0, 0.])
                row[0] += 1
                row[1] += time.perf_counter()-start
        return call

    infer = policy.model.infer_action
    def infer_traced(*a, **kw):
        inputs = digest(kw)
        result = infer(*a, **kw)
        rows['inference'].append(dict(inputs=inputs, outputs=digest(result)))
        return result
    policy.model.infer_action = infer_traced

    def environment(*a):
        env, config = original_environment(*a)
        get_obs, take = env.get_obs, env.take_action
        setup = env.setup_demo
        def setup_traced(*a, **kw):
            result = setup(*a, **kw)
            for name in ['_entity_qf', 'left_plan_path', 'right_plan_path', 'set_gripper', 'set_arm_joints']:
                setattr(env.robot, name, timed(name, getattr(env.robot, name)))
            return result
        env.setup_demo = setup_traced
        env._update_render = timed('_update_render', env._update_render)
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
    collect = gc.collect
    for index, (phase, task, ordinal) in enumerate(cases):
        key = f'{phase}-{task}-{ordinal}'
        case = next(c for c in lock['cases'] if c['phase']==phase and c['task']==task)
        episode = plans[task+'--'+phase]['episodes'][ordinal]
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate(), np.random.get_state())
        variants = ['warmup', 'baseline', 'candidate'] if index == 0 else ['baseline','candidate']
        if args.variant != 'paired': variants = [args.variant]
        reference = None
        for variant in variants:
            rows = dict(observations=[], actions=[], inference=[], timings={})
            torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
            random.setstate(rng[2]); np.random.set_state(rng[3])
            policy._prompt = policy._context = None
            gc.unfreeze()
            collect()
            if variant == 'candidate': gc.freeze()
            gc.collect = timed('gc.collect', collect)
            profile = cProfile.Profile()
            torch.cuda.synchronize()
            try:
                if variant == 'profile': profile.enable()
                result = e.rollout_case(official, device, policy, case, [episode])[0]
            finally:
                if variant == 'profile': profile.disable()
                gc.collect = collect
                gc.unfreeze()
            if variant == 'profile':
                profile.dump_stats(str(root/(key+'.prof')))
                with (root/(key+'-profile.txt')).open('w') as f:
                    pstats.Stats(profile, stream=f).sort_stats('cumulative').print_stats(65)
            record = dict(result=result, memory_allocated=torch.cuda.memory_allocated(), **rows)
            (root/(key+'-'+variant+'.json')).write_text(json.dumps(record))
            print('COMPLETE', key, variant, json.dumps(result), json.dumps(rows['timings']), flush=True)
            if variant == 'baseline': reference = record
            if variant == 'candidate':
                for field in ['seed','instruction','success','actions','observations','initial_scene_sha256','initial_observation_sha256']:
                    assert reference['result'][field] == result[field], field
                for field in ['actions','observations','inference']:
                    a, b = [[{k:v for k,v in row.items() if k != 'seconds'} for row in rec[field]] for rec in [reference, record]]
                    assert a == b, field
                summary = dict(passed=True, case=key, baseline=reference['result'], candidate=result,
                               baseline_timings=reference['timings'], candidate_timings=rows['timings'],
                               speedup=reference['result']['seconds']/result['seconds'])
                (root/(key+'-comparison.json')).write_text(json.dumps(summary, indent=2))
                print('PARITY_PASS', json.dumps(summary), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--variant', choices=['profile','baseline','paired'], default='profile')
    parser.add_argument('--long', action='store_true')
    args = parser.parse_args()
    main(args)
