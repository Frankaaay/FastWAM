import importlib.util
import os
import sys
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('collector', Path(__file__).parents[1] / 'scripts/collect_robotwin_missing.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from publish_robotwin_missing import merge_manifest


class MissingCoverage(unittest.TestCase):
    def setUp(self):
        self.ids = list(range(39, 2500))
        self.rows = [dict(canonical_episode=i, task=f'task{i//50}', episode=i%50, seed=i, replan=i<37)
                     for i in range(2500)]
        self.ledger = dict(pending=self.rows, accepted=[], candidates=[], excluded=[])

    def test_final_provenance_not_intermediate_exclusions(self):
        rows = collector.missing_rows({'source_episode_indices': self.ids}, self.ledger)
        self.assertEqual([r['canonical_episode'] for r in rows], list(range(39)))
        self.assertEqual(sum(r['replan'] for r in rows), 37)

    def test_duplicate_existing_id_fails(self):
        self.ids[-1] = self.ids[0]
        with self.assertRaises(ValueError):
            collector.missing_rows({'source_episode_indices': self.ids}, self.ledger)

    def test_duplicate_ledger_id_fails(self):
        self.rows[-1] = self.rows[0]
        with self.assertRaises(ValueError):
            collector.missing_rows({'source_episode_indices': self.ids}, self.ledger)

    def test_evidence_is_never_overwritten(self):
        fd, name = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        p = Path(name)
        try:
            p.write_text('{"original": true}')
            with self.assertRaises(FileExistsError):
                collector.write_json(p, {'original': False})
            self.assertIn('true', p.read_text())
        finally:
            p.unlink()


class PublicationCoverage(unittest.TestCase):
    def setUp(self):
        episodes = [dict(source_episode_index=i, task=f'task{i//50}', length=3,
                         path=f'episode{i}.hdf5', seen_instructions=['instruction']) for i in range(2500)]
        self.existing = dict(episodes=episodes[39:], frames=2461*3, windows=2461*2)
        self.replacements = episodes[:39]

    def test_union_preserves_old_rows_and_recounts_windows(self):
        result = merge_manifest(self.existing, self.replacements)
        self.assertEqual(result['windows'], 5000)
        self.assertEqual(result['frames'], 7500)
        self.assertEqual(result['episodes'][39]['path'], 'episode39.hdf5')
        self.assertNotIn('recovery_kind', self.existing['episodes'][0])

    def test_overlap_rejected(self):
        self.replacements[0]['source_episode_index'] = 39
        with self.assertRaises(ValueError):
            merge_manifest(self.existing, self.replacements)

    def test_task_substitution_rejected(self):
        self.replacements[0]['task'] = 'task1'
        with self.assertRaises(ValueError):
            merge_manifest(self.existing, self.replacements)


if __name__ == '__main__':
    unittest.main()
