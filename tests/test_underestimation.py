"""Saved evidence and rendering contracts for high-concentration diagnostics."""
import unittest
import numpy as np
from dashboard import underestimation as diagnostic


class UnderestimationTests(unittest.TestCase):
    def test_saved_cohorts_and_training_threshold(self):
        for target in ['CO', 'NOX']:
            self.assertEqual(len(diagnostic.read(target, 'OOF')), 27570)
            self.assertEqual(len(diagnostic.read(target, 'prueba')), 7411)
            values = diagnostic.summary(target)
            threshold = diagnostic.read(target, 'OOF')[target].quantile(.95)
            np.testing.assert_allclose(values.umbral, threshold)
            self.assertTrue(values.porcentaje_subestimado.between(0, 100).all())
            self.assertEqual(values.n_altos.iloc[0], 1379)

    def test_complete_points_common_axes_and_training_bins(self):
        for target in ['CO', 'NOX']:
            figures = [diagnostic.figure(target, phase) for phase in ['OOF', 'prueba']]
            self.assertEqual(len(figures[0].data[0].x), 27570)
            self.assertEqual(len(figures[1].data[0].x), 7411)
            self.assertEqual(figures[0].layout.xaxis.range, figures[1].layout.xaxis.range)
            self.assertEqual(figures[0].layout.yaxis.range, figures[1].layout.yaxis.range)
            self.assertEqual(figures[0].layout.yaxis.scaleanchor, 'x')
            for phase in ['OOF', 'prueba']:
                self.assertEqual(diagnostic.trend(target, phase).n.sum(), len(diagnostic.read(target, phase)))

    def test_clear_conclusion_and_invalid_inputs(self):
        self.assertIn('subestima el 73 %', diagnostic.conclusion('CO'))
        self.assertIn('subestima el 80 %', diagnostic.conclusion('NOX'))
        self.assertIsNotNone(diagnostic.panel('CO'))
        with self.assertRaises(ValueError):
            diagnostic.read('CO', 'train_in_sample')


if __name__ == '__main__':
    unittest.main()
