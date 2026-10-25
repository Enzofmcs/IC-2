import math
from pathlib import Path
import tempfile
import unittest
from ic2.data import split_metadata, validate_manifest, attach_paths
from ic2.metrics import binary_metrics, select_threshold


def fixture():
    return [dict(image_id=f'{label}_{group}_{image}', lesion_id=f'{label}_{group}', dx=label)
            for label in ('mel','nv') for group in range(10) for image in range(2)]


class DataTests(unittest.TestCase):
    def test_split_is_reproducible_and_grouped(self):
        rows = split_metadata(fixture())
        self.assertEqual(rows, split_metadata(list(reversed(fixture()))))
        self.assertEqual(len(rows), 40)
        for lesion in {r['lesion_id'] for r in rows}:
            self.assertEqual(len({r['split'] for r in rows if r['lesion_id']==lesion}), 1)

    def test_rejects_leakage_duplicates_and_unknown_diagnosis(self):
        rows = split_metadata(fixture())
        changed = [dict(r) for r in rows]
        lesion = changed[0]['lesion_id']
        same = [r for r in changed if r['lesion_id']==lesion]
        same[0]['split'] = 'train' if same[1]['split']!='train' else 'test'
        with self.assertRaises(ValueError):validate_manifest(changed)
        with self.assertRaises(ValueError):split_metadata(fixture()+[fixture()[0]])
        bad = fixture(); bad[0]['dx']='unknown'
        with self.assertRaises(ValueError):split_metadata(bad)

    def test_missing_groups_or_inconsistent_labels(self):
        with self.assertRaises(ValueError):split_metadata(fixture()[:3])
        bad = fixture(); bad[1]['dx']='nv'
        with self.assertRaises(ValueError):split_metadata(bad)

    def test_paths_are_resolved_without_altering_sources(self):
        rows = split_metadata(fixture())
        with tempfile.TemporaryDirectory() as folder:
            for row in rows:(Path(folder)/f'{row["image_id"]}.jpg').write_bytes(b'synthetic fixture')
            attached = attach_paths(rows,folder)
            self.assertEqual(len(attached),40)
            self.assertEqual(len(list(Path(folder).glob('*.jpg'))),40)
            (Path(folder)/attached[0]['relative_path']).unlink()
            with self.assertRaises(ValueError):attach_paths(rows,folder)


class MetricsTests(unittest.TestCase):
    def test_known_confusion_matrix_and_auc(self):
        m=binary_metrics([0,0,1,1],[0.1,0.8,0.4,0.9])
        self.assertEqual(m['confusion_matrix'],[[1,1],[1,1]])
        self.assertEqual(m['accuracy'],0.5)
        self.assertEqual(m['f1'],0.5)
        self.assertEqual(m['roc_auc'],0.75)

    def test_auc_ties_and_single_class(self):
        self.assertEqual(binary_metrics([0,1],[0.5,0.5])['roc_auc'],0.5)
        self.assertIsNone(binary_metrics([0],[0.1])['roc_auc'])
        self.assertIsNone(binary_metrics([0],[0.1])['sensitivity'])

    def test_single_sample_shape_and_global_aggregation(self):
        self.assertEqual(binary_metrics([1],[0.8])['accuracy'],1)
        self.assertAlmostEqual(binary_metrics([0,0,1],[0.1,0.2,0.1])['accuracy'],2/3)

    def test_threshold_uses_validation_inputs_only(self):
        threshold=select_threshold([0,0,1,1],[0.1,0.2,0.3,0.4])
        self.assertEqual(threshold,0.3)
        self.assertEqual(binary_metrics([0,1],[0.25,0.35],threshold)['accuracy'],1)

    def test_rejects_bad_probabilities_and_labels(self):
        for labels, probabilities in [([],[]),([1],[math.nan]),([2],[0.3]),([1,0],[0.3]),([1],[1.1])]:
            with self.assertRaises(ValueError):binary_metrics(labels,probabilities)
        with self.assertRaises(ValueError):select_threshold([1],[0.5])


if __name__ == '__main__':
    unittest.main()
