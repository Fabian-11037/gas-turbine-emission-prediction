"""Pruebas del contraste armonico y conservacion de posiciones originales."""
import unittest

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from seasonality_analysis import analyze, harmonic_test


class AnalysisTests(unittest.TestCase):
    def test_known_cycle(self):
        t = np.arange(2400)
        y = .005*t+3*np.sin(2*np.pi*t/24)+np.random.default_rng(42).normal(size=len(t))
        row, residual = harmonic_test(y,t,24)
        self.assertLess(row['p'],.001)
        self.assertGreater(row['R2_parcial'],.8)
        self.assertAlmostEqual(row['amplitud'],3,delta=.15)
        self.assertEqual(len(residual),len(t))

    def test_HAC_matches_native_on_complete_sequence(self):
        t = np.arange(1500)
        y = np.random.default_rng(3).normal(size=len(t))+.2*np.cos(2*np.pi*t/24)
        x = np.column_stack([np.ones(len(t)),(t-t.mean())/(t.max()-t.min()),
            np.sin(2*np.pi*t/24),np.cos(2*np.pi*t/24)])
        fit = OLS(y,x).fit()
        covariance = cov_hac(fit,nlags=336,use_correction=True)
        test = fit.wald_test(np.eye(4)[2:],cov_p=covariance,use_f=False,scalar=True)
        actual,_ = harmonic_test(y,t,24)
        self.assertAlmostEqual(float(test.pvalue),actual['p'],places=12)

    def test_holes_preserved_and_holdout_not_used_in_development(self):
        t = np.arange(1600)
        rng = np.random.default_rng(42)
        data = pd.DataFrame({'anio':2011,'orden_registro':t,
            'particion':np.where((t>=500)&(t<700),'prueba_bloques','entrenamiento_CV'),
            'CO':np.sin(2*np.pi*t/24)+rng.normal(size=len(t)),
            'NOX':rng.normal(size=len(t))})
        first,_ = analyze(data)
        changed = data.copy()
        changed.loc[changed.particion == 'prueba_bloques',['CO','NOX']] += 100
        second,_ = analyze(changed)
        pd.testing.assert_frame_equal(first.loc[first.poblacion == 'desarrollo'].reset_index(drop=True),
            second.loc[second.poblacion == 'desarrollo'].reset_index(drop=True))
        kept = data.particion.eq('entrenamiento_CV')
        original,_ = harmonic_test(data.loc[kept,'CO'],t[kept],24)
        compressed,_ = harmonic_test(data.loc[kept,'CO'],np.arange(kept.sum()),24)
        self.assertNotAlmostEqual(original['amplitud'],compressed['amplitud'],places=3)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            harmonic_test(np.ones(1000),np.arange(1000),24)
        with self.assertRaises(ValueError):
            harmonic_test(np.arange(1000),np.zeros(1000),24)


if __name__ == '__main__':
    unittest.main()
