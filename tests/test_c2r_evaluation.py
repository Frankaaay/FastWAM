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

    @unittest.skipIf(sys.platform == 'win32', 'Production leases use Linux flock')
    def test_claim_has_single_owner_and_releases(self):
        with evaluation.claim(self.root, 'a--0') as owned:
            self.assertTrue(owned)
            with evaluation.claim(self.root, 'a--0') as other:
                self.assertFalse(other)
        with evaluation.claim(self.root, 'a--0') as reclaimed:
            self.assertTrue(reclaimed)

    @unittest.skipIf(sys.platform == 'win32', 'Production leases use Linux flock')
    def test_claim_releases_after_worker_killed(self):
        lockdir = self.root / 'claims/killed'
        lockdir.mkdir(parents=True)
        child = subprocess.Popen([sys.executable, '-c',
            "import fcntl,sys,time; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); time.sleep(60)",
            str(lockdir / 'lock')], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'locked')
            with evaluation.claim(self.root, 'killed') as other:
                self.assertFalse(other)
        finally:
            child.kill()
            child.wait()
            child.stdout.close()
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
