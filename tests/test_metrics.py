import unittest
from metrics import (
    compute_coverage_metrics,
    compute_expression_level_metrics,
    compute_structure_accuracy,
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.gt_edges = [("1", "2"), ("2", "3")]

    def test_coverage_perfect(self):
        metrics = compute_coverage_metrics(self.gt_edges, self.gt_edges)
        self.assertAlmostEqual(metrics['coverage'], 1.0)
        self.assertAlmostEqual(metrics['redundancy'], 0.0)
        self.assertAlmostEqual(metrics['precision'], 1.0)
        self.assertAlmostEqual(metrics['recall'], 1.0)
        self.assertAlmostEqual(metrics['f1_score'], 1.0)

    def test_coverage_partial(self):
        pred_edges = [("1", "2"), ("2", "4")]
        metrics = compute_coverage_metrics(pred_edges, self.gt_edges)
        self.assertAlmostEqual(metrics['coverage'], 0.5)
        self.assertAlmostEqual(metrics['redundancy'], 0.5)
        self.assertAlmostEqual(metrics['precision'], 0.5)
        self.assertAlmostEqual(metrics['recall'], 0.5)

    def test_expression_level_metrics(self):
        preds = [self.gt_edges, [("1", "2")]]
        gts = [self.gt_edges, self.gt_edges]
        result = compute_expression_level_metrics(preds, gts)
        self.assertAlmostEqual(result['precision'], 1.0)
        self.assertAlmostEqual(result['recall'], 0.75)
        self.assertAlmostEqual(result['f1_score'], (1.0 + 2 * 1 * 0.5 / (1 + 0.5)) / 2)

    def test_structure_accuracy(self):
        preds = [self.gt_edges, [("1", "2")]]
        gts = [self.gt_edges, self.gt_edges]
        acc = compute_structure_accuracy(preds, gts)
        self.assertAlmostEqual(acc, 0.5)


if __name__ == '__main__':
    unittest.main()
