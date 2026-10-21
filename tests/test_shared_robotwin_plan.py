"""Fail-closed plan identity, resume validation and policy-independent scene RNG."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

spec = importlib.util.spec_from_file_location('shared_eval', Path(__file__).resolve().parents[1] / 'scripts/c2r/evaluate.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class SharedPlanTests(unittest.TestCase):
    def setUp(self):
        # Files retained for diagnosis; no recursive cleanup.
        self.root = Path(tempfile.mkdtemp(prefix='shared-robotwin-test-'))
        self.case = dict(task='task', phase='randomized', step_limit=400)
        self.plan = dict(**self.case, episodes=[dict(ordinal=i, seed=100000+i, instruction=f'fixed {i}') for i in range(100)])
        self.path = self.root / 'task--randomized.json'
        self.path.write_text(json.dumps(self.plan))
        self.lock = dict(plan_root=str(self.root), episodes=100, cases=[self.case],
                         files={self.path.name: hashlib.sha256(self.path.read_bytes()).hexdigest()})
        self.manifest = dict(episodes=100, cases=[self.case])

    def load(self):
        with patch.object(evaluation, 'shared_plan_lock', return_value=self.lock):
            return evaluation.load_shared_plans(self.manifest)

    def test_real_lock_has_100_unique_cases_and_both_phases(self):
        lock = evaluation.shared_plan_lock()
        self.assertEqual(len(lock['files']), 100)
        self.assertEqual({c['phase'] for c in lock['cases']}, {'clean', 'randomized'})
        self.assertEqual(len({(c['task'], c['phase']) for c in lock['cases']}), 100)

    def test_exact_plan_is_reused(self):
        self.assertEqual(self.load()['task--randomized'], self.plan)

    def test_entry_requires_explicit_model_before_loading_simulator(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        with patch.object(sys, 'argv', ['evaluate.py', 'rollout', '--root', str(self.root)]), \
                self.assertRaisesRegex(ValueError, 'explicitly name'):
            evaluation.main()

    def test_missing_plan_does_not_generate(self):
        self.path.rename(self.root / 'missing.json')
        with self.assertRaises(FileNotFoundError):
            self.load()
        self.assertFalse(self.path.exists())

    def test_modified_instruction_is_rejected(self):
        self.plan['episodes'][0]['instruction'] = 'another instruction'
        self.path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.load()

    def test_budget_and_case_changes_are_rejected(self):
        self.manifest['episodes'] = 99
        with self.assertRaises(ValueError): self.load()
        self.manifest['episodes'] = 100
        self.manifest['cases'] = [dict(self.case, step_limit=401)]
        with self.assertRaises(ValueError): self.load()
        self.manifest['cases'] = [self.case, self.case]
        with self.assertRaises(ValueError): self.load()

    def test_results_cannot_mix_protocols_checkpoints_or_instructions(self):
        plans = self.load()
        (self.root / 'results').mkdir()
        episode = dict(self.plan['episodes'][0], success=True, initial_scene_sha256='scene')
        row = dict(**self.case, shared_plan_sha256=evaluation.SHARED_PLAN_SHA256,
                   run_sha256='run', episodes=[episode])
        path = self.root / 'results' / 'part.json'
        path.write_text(json.dumps(row))
        evaluation.validate_shared_results(self.root, plans, 'run')
        for key, bad in [('shared_plan_sha256', 'old-plan'), ('run_sha256', 'other-model')]:
            path.write_text(json.dumps(dict(row, **{key: bad})))
            with self.assertRaises(ValueError): evaluation.validate_shared_results(self.root, plans, 'run')
        row['episodes'][0]['instruction'] = 'wrong'
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'seed/instruction'):
            evaluation.validate_shared_results(self.root, plans, 'run')

    def test_policy_rng_consumption_cannot_change_environment(self):
        class Scene:
            def draw(self): return random.random(), np.random.random()
        sequences = []
        for noise in (0, 1, 100):
            random.seed(17); np.random.seed(29)
            env = evaluation.EnvironmentRng(Scene())
            result = []
            for _ in range(5):
                for _ in range(noise): random.random(); np.random.random()
                result.append(env.draw())
            sequences.append(result)
        self.assertEqual(sequences[0], sequences[1])
        self.assertEqual(sequences[0], sequences[2])

    def test_common_rollout_is_identical_despite_three_policy_rng_patterns(self):
        class Scene:
            step_lim = 2
            table_z_bias = 0.
            scene = SimpleNamespace(get_all_actors=lambda: [])
            def setup_demo(self, **kwargs):
                np.random.seed(kwargs['seed'])
                self.take_action_cnt = 0
                self.eval_success = False
                self.draws = []
            def set_instruction(self, instruction): self.instruction = instruction
            def get_instruction(self): return self.instruction
            def get_obs(self):
                return {'observation': {'head_camera': {'rgb': np.random.randint(0, 256, (2, 2, 3), dtype=np.uint8)}}}
            def take_action(self):
                self.draws.append((random.random(), np.random.random()))
                self.take_action_cnt += 1
            def close_env(self, **kwargs): pass
        class Policy:
            inference_seconds = []
            def __init__(self, noise): self.noise = noise
            def reset(self):
                for _ in range(self.noise): random.random(); np.random.random()
            def should_request_observation(self): return True
            def step(self, env, obs):
                for _ in range(self.noise): random.random(); np.random.random()
                env.take_action()
        random.seed(17); np.random.seed(29)
        state = np.random.get_state()
        episode = dict(ordinal=0, seed=100000, instruction='same frozen text',
                       python_before_scene=random.getstate(), python_after_instruction=random.getstate(),
                       numpy_after_instruction=[state[0], state[1].tolist(), *state[2:]])
        fingerprints, draws = [], []
        for noise in (0, 1, 100):
            env = Scene()
            with patch.object(evaluation, 'environment', return_value=(env, {'clear_cache_freq': 1})):
                result = evaluation.rollout_case(None, None, Policy(noise), dict(self.case, step_limit=2), [episode])[0]
            fingerprints.append((result['initial_scene_sha256'], result['initial_observation_sha256']))
            draws.append(env.draws)
        self.assertEqual(fingerprints, [fingerprints[0]] * 3)
        self.assertEqual(draws, [draws[0]] * 3)


if __name__ == '__main__':
    unittest.main()
