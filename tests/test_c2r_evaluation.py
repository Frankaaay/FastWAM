"""Evaluation coverage, duplicate detection, and fine-grained claim contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import threading

spec = importlib.util.spec_from_file_location('evaluation', Path(__file__).resolve().parents[1] / 'scripts/c2r/evaluate.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='c2r-eval-test-'))
        (self.root / 'results').mkdir()
        self.manifest = dict(episodes=2, cases=[dict(task='a', phase='clean')])
        evaluation.write_json(self.root / 'plans/a--clean.json', dict(episodes=[
            dict(ordinal=i, seed=100000+i, instruction='test') for i in range(2)]))

    def test_claim_has_single_owner_and_releases(self):
        with evaluation.claim(self.root, 'a--0') as owned:
            self.assertTrue(owned)
            with evaluation.claim(self.root, 'a--0') as other:
                self.assertFalse(other)
        with evaluation.claim(self.root, 'a--0') as reclaimed:
            self.assertTrue(reclaimed)

    def test_killed_owner_requires_confirmed_termination_before_reclaim(self):
        child = subprocess.Popen([sys.executable, '-c',
            "import importlib.util,sys,time,pathlib; s=importlib.util.spec_from_file_location('e',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); c=m.claim(pathlib.Path(sys.argv[2]),'killed'); assert c.__enter__(); print('locked',flush=True); time.sleep(60)",
            str(Path(evaluation.__file__)), str(self.root)], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'locked')
            with evaluation.claim(self.root, 'killed') as other:
                self.assertFalse(other)
        finally:
            child.kill()
            child.wait()
            child.stdout.close()
        with evaluation.claim(self.root, 'killed') as other:
            self.assertFalse(other)
        # The supervisor has observed process termination; retain its evidence.
        active = self.root / 'claims/killed/active'
        self.assertEqual(json.loads((active / 'owner.json').read_text())['pid'], child.pid)
        active.rename(active.with_name('retired-confirmed-dead'))
        with evaluation.claim(self.root, 'killed') as reclaimed:
            self.assertTrue(reclaimed)

    def test_task_state_uses_initial_scene_and_refreshes_each_episode(self):
        env = SimpleNamespace(object=SimpleNamespace(get_pose=lambda: SimpleNamespace(p=[.2, 0., .8])))
        module = SimpleNamespace(ArmTag=str)
        with patch.dict(sys.modules, {type(env).__module__: module}):
            evaluation.initialize_task_state('put_object_cabinet', env)
            self.assertEqual((env.arm_tag, env.origin_z), ('right', .8))
            env.object.get_pose = lambda: SimpleNamespace(p=[-.2, 0., .9])
            evaluation.initialize_task_state('put_object_cabinet', env)
            self.assertEqual((env.arm_tag, env.origin_z), ('left', .9))

    def test_scheduler_revisits_incomplete_work(self):
        cases = [dict(task='a', phase='clean')]
        iterator = evaluation.ready_cases(self.root, cases, episodes=2)
        self.assertEqual(next(iterator), cases[0])
        with patch.object(evaluation.time, 'sleep'):
            self.assertEqual(next(iterator), cases[0])
            evaluation.write_json(self.root / 'results/a--clean--000.json',
                                  dict(episodes=[dict(ordinal=0), dict(ordinal=1)]))
            with self.assertRaises(StopIteration):
                next(iterator)

    def test_atomic_writes_with_equal_pids_have_distinct_temporary_files(self):
        barrier = threading.Barrier(8)
        original = Path.write_text
        original_replace = Path.replace
        replace_lock = threading.Lock()
        def synchronized(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            barrier.wait(timeout=10)
            return result
        def replace(path, target):
            # Windows denies simultaneous replacements of an open destination.
            # Serialize only rename there; all temporary writes still collide
            # under the old PID-only implementation before this barrier.
            if sys.platform == 'win32':
                with replace_lock:
                    return original_replace(path, target)
            return original_replace(path, target)
        destination = self.root / 'complete.json'
        with patch.object(Path, 'write_text', synchronized), patch.object(Path, 'replace', replace):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda i: evaluation.write_json(destination, dict(value=i)), range(8)))
        self.assertIn(json.loads(destination.read_text())['value'], range(8))

    def test_ready_cases_do_not_wait_for_first_case(self):
        cases = [dict(task='slow', phase='clean'), dict(task='a', phase='clean')]
        iterator = evaluation.ready_cases(self.root, cases)
        self.assertEqual(next(iterator)['task'], 'a')
        evaluation.write_json(self.root / 'plans/slow--clean.json', dict(episodes=[]))
        self.assertEqual(next(iterator)['task'], 'slow')
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_no_partial_score_and_exact_coverage(self):
        def save(name, ordinal, success):
            evaluation.write_json(self.root / 'results' / name,
                dict(task='a', phase='clean', episodes=[dict(ordinal=ordinal, seed=100000+ordinal,
                     instruction='test', success=success)]))
        save('0.json', 0, True)
        self.assertIsNone(evaluation.summarize(self.root, self.manifest)['overall']['clean'])
        save('1.json', 1, False)
        result = evaluation.summarize(self.root, self.manifest)
        self.assertTrue(result['complete'])
        self.assertEqual(result['overall']['clean'], .5)
        save('duplicate.json', 1, False)
        with self.assertRaises(ValueError):
            evaluation.summarize(self.root, self.manifest)


if __name__ == '__main__':
    unittest.main()
