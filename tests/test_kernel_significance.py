"""Seleccion declarada, orientacion de contrastes y evidencia portable."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from dashboard import kernel_view, model_view
from dashboard.models import registry
from dashboard.repository import results
from tools.export_kernel_significance import holm, paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'book'/'informe_resultados'/'kernel_operativo'


class SignificanceTests(unittest.TestCase):
    def test_selected_model_is_cv_winner_for_both_targets(self):
        self.assertEqual(model_view.default_model(), 'kernel_operativo')
        summary = results()
        self.assertEqual(len(summary), 16)
        for target in ['CO', 'NOX']:
            rows = summary.loc[summary.objetivo.eq(target)]
            self.assertEqual(rows.loc[rows.MSE_CV_externo.idxmin(), 'modelo'], 'Kernel operativo')
            self.assertEqual(rows.loc[rows.seleccionado_por_CV, 'modelo'].tolist(), ['Kernel operativo'])
            self.assertTrue(registry()['kernel_operativo']['respuestas'][target]['seleccionado_por_cv'])

    def test_orientation_and_exact_source_p_values(self):
        table = pd.read_csv(ROOT/'data'/'kernel_significancia.csv')
        self.assertEqual(len(table), 2)
        np.testing.assert_allclose(table.delta_R2, table.R2_kernel-table.R2_referencia, atol=1e-12)
        co = table.loc[table.objetivo.eq('CO')].iloc[0]
        self.assertEqual(co.referencia, 'Random Forest')
        self.assertGreater(co.delta_R2, 0)
        self.assertAlmostEqual(co.p_prueba_nominal, .5091745412729364)
        self.assertGreater(co.p_prueba_Holm2, .05)
        self.assertLess(co.p_CV_Holm2, .05)
        self.assertLess(co.IC95_inf, 0)
        self.assertGreater(co.IC95_sup, 0)
        xgb = table.loc[table.objetivo.eq('NOX')].iloc[0]
        self.assertEqual(xgb.referencia, 'XGBoost')
        self.assertAlmostEqual(xgb.p_prueba_nominal, 4/20001)
        self.assertAlmostEqual(xgb.p_CV_nominal, .0009058732928651)
        self.assertLess(xgb.p_prueba_Holm2, .05)
        self.assertLess(xgb.p_CV_Holm2, .05)
        self.assertTrue(table.p_prueba_Holm2.ge(table.p_prueba_nominal).all())
        np.testing.assert_allclose(table.p_prueba_Holm2, holm(table.p_prueba_nominal))
        np.testing.assert_allclose(table.p_CV_Holm2, holm(table.p_CV_nominal))
        self.assertTrue(set(table.referencia).issubset({m['nombre'] for m in registry().values()}))
        np.testing.assert_allclose(holm([.03, .001, .3]), [.06, .003, .3])

    def test_source_integrity_and_sensitivity(self):
        metadata = json.loads((ROOT/'data'/'kernel_significancia_metodologia.json').read_text(encoding='utf-8'))
        for name, expected in metadata['fuentes_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected)
        blocks = pd.read_csv(ROOT/'data'/'kernel_significancia_bloques.csv')
        self.assertEqual(len(blocks), 6)
        self.assertEqual(set(blocks.longitud_bloque), {168,336,672})
        self.assertTrue(blocks.n.eq(7411).all())
        self.assertTrue(blocks.loc[blocks.longitud_bloque.eq(168), 'n_unidades'].eq(45).all())
        self.assertTrue(metadata['prueba_reutilizada'])
        self.assertEqual(metadata['modelos_entrenados'], 0)
        folds = pd.read_csv(EVIDENCE/'significancia_dashboard_folds.csv')
        self.assertEqual(len(folds), 10)
        self.assertTrue((folds.n_train+folds.n_val < 27570).all())

    def test_bootstrap_identical_predictions_and_direction(self):
        frame = pd.read_csv(ROOT/'data'/'predicciones_prueba.csv.gz')
        observations = pd.read_csv(ROOT/'data'/'observaciones.csv.gz')
        starts = observations.groupby('anio').orden_registro.min().to_dict()
        frame['referencia'] = frame.CO_Random_Forest
        frame['nuevo'] = frame.referencia
        equal = paired_bootstrap(frame, 'CO', starts, 168, replicates=200)
        self.assertEqual(equal['p_prueba_nominal'], 1)
        self.assertEqual(equal['delta_R2'], 0)
        frame['nuevo'] = frame.CO_Kernel_operativo
        forward = paired_bootstrap(frame, 'CO', starts, 168, replicates=200)
        frame[['nuevo','referencia']] = frame[['referencia','nuevo']].to_numpy()
        reverse = paired_bootstrap(frame, 'CO', starts, 168, replicates=200)
        self.assertAlmostEqual(forward['delta_R2'], -reverse['delta_R2'])
        self.assertAlmostEqual(forward['IC95_inf'], -reverse['IC95_sup'])
        self.assertEqual(forward['p_prueba_nominal'], reverse['p_prueba_nominal'])

    def test_evidence_panel_and_executed_book_tables(self):
        for target in ['CO', 'NOX']:
            style, content = kernel_view.render('kernel_operativo', target, 'evidencia')
            self.assertEqual(style, {})
            self.assertIsNotNone(content)
        notebook = json.loads((ROOT/'book'/'gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        text = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'markdown')
        self.assertIn('Modelo elegido: kernel operativo para CO y NOx', text)
        self.assertIn('4.10.5 Selección y significancia', text)
        self.assertIn('no demuestra que los modelos sean equivalentes', text)
        cell = next(c for c in notebook['cells'] if c.get('id') == 'kernel-statistics-code')
        self.assertEqual(len(cell['outputs']), 3)
        self.assertIn('kernel_significancia.csv', ''.join(cell['source']))
        output = '\n'.join(o['data']['text/plain'] for o in cell['outputs'])
        self.assertIn('CO', output)
        self.assertIn('NOX', output)
        self.assertIn('Random Forest', output)
        self.assertIn('XGBoost', output)
        for old in ['ET-SVR ajustado', 'ET–SVR ajustado', 'ET-SVR ponderado', 'ET–SVR ponderado',
                    'referencia_ETSVR', 'Holm3', 'Holm14', 'Holm15']:
            self.assertNotIn(old, json.dumps(notebook, ensure_ascii=False))
        summary_cell = next(c for c in notebook['cells'] if c.get('id') == 'kernel-results-code')
        summary_output = '\n'.join(o['data']['text/plain'] for o in summary_cell['outputs'])
        self.assertIn('Random Forest', summary_output)
        self.assertIn('XGBoost', summary_output)


if __name__ == '__main__':
    unittest.main()
