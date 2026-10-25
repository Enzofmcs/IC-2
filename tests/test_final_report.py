import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('report_checks',ROOT/'scripts/check_reported_results.py')
checks=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class ReportEvidenceTests(unittest.TestCase):
    def test_reported_percentages_match_confusion_counts(self):
        rows=checks.verify(ROOT/'results/final_report_results.json')
        self.assertEqual(len(rows),2)
        vgg=rows[0]['computed_from_reported_counts']
        self.assertEqual(vgg['positive_support'],889)
        self.assertEqual(vgg['negative_support'],899)
        self.assertEqual(vgg['n'],1788)
        self.assertAlmostEqual(vgg['f1'],1618/1793)

    def test_counts_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):checks.calculate_from_counts(-1,0,0,1)
        with self.assertRaises(ValueError):checks.calculate_from_counts(1,0,0,0)


if __name__=='__main__':
    unittest.main()
