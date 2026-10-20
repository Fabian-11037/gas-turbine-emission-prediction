"""Contratos de representacion, publicacion y cross-fitting del kernel."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from dashboard import kernel_view
from dashboard.models import registry, predictor, predict, StreamingPipelinePredictor
from dashboard.repository import sample_frame
from emission_models.kernel_operativo import operational_features, weight_operational_features, PREDICTORS

ROOT = Path(__file__).resolve().parents[1]


class KernelTests(unittest.TestCase):
    def test_tree_explanation_matches_book(self):
        from dashboard.kernel_content import TREE_MODELS
        for phrase in ['árbol de regresión','Extra Tree','200','Árbol de decisión',
                       'Random Forest','reemplazo','sin remuestreo','XGBoost','sobreajuste']:
            self.assertIn(phrase,TREE_MODELS)
        notebook = json.loads((ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        intro = next(c for c in notebook['cells'] if c.get('id')=='kernel-intro')
        self.assertIn(TREE_MODELS.replace('### ','#### '),''.join(intro['source']))

    def test_operational_interpretation_matches_book(self):
        from dashboard.kernel_content import MATHEMATICS
        for phrase in ['Posible interpretación operativa', 'no necesariamente mayor eficiencia',
                       'MWh·°C', 'cambiar según las demás condiciones', 'ablación', 'NASA Glenn', 'diccionario UCI']:
            self.assertIn(phrase, MATHEMATICS)
        notebook = json.loads((ROOT/'book'/'gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        cell = next(c for c in notebook['cells'] if c.get('id') == 'kernel-math')
        self.assertEqual(''.join(cell['source']),
            '#### 4.10.1 Fundamentos matemáticos\n\n'+MATHEMATICS.replace('### ', '##### '))

    def test_features_and_group_weights(self):
        x = np.arange(18, dtype=float).reshape(2, 9)
        phi = operational_features(x)
        np.testing.assert_array_equal(phi[:, :9], x)
        np.testing.assert_array_equal(phi[:, 9], x[:, 5]-x[:, 6])
        np.testing.assert_array_equal(phi[:, 10], x[:, 7]*(x[:, 5]-x[:, 6]))
        np.testing.assert_array_equal(weight_operational_features(np.ones((1, 11)), .5, 2, 3),
            [[.5, .5, .5, 3, 3, 2, 2, 1, 3, 2, 1]])
        with self.assertRaises(ValueError):
            operational_features(x[:, :8])

    def test_panel_only_for_kernel(self):
        self.assertEqual(kernel_view.render('ridge', 'CO', 'arquitectura'), ({'display': 'none'}, None))
        for target in ['CO', 'NOX']:
            for section in ['arquitectura', 'matematicas', 'aporte', 'evidencia']:
                style, content = kernel_view.render('kernel_operativo', target, section)
                self.assertEqual(style, {})
                self.assertIsNotNone(content)
        self.assertIn('kernel_arquitectura.svg', (ROOT/'dashboard'/'kernel_view.py').read_text())
        summary = pd.read_csv(ROOT/'data'/'resultados.csv').query("modelo == 'Kernel operativo'")
        self.assertTrue(np.isfinite(summary[['RMSE_CV_std', 'MAE_CV_media', 'MAE_CV_std']]).all().all())

    def test_chunked_forest_metadata_and_invalid_weights(self):
        from dashboard.models import artifact_path
        for target, response in registry()['kernel_operativo']['respuestas'].items():
            parts = response['fragmentos']
            self.assertEqual(parts['n_arboles'], 200)
            self.assertEqual([p['n_arboles'] for p in parts['arboles']], [50]*4)
            for component in parts['arboles']+[parts['svr'], parts['meta']]:
                self.assertTrue(artifact_path(component).is_file())
                self.assertLess(artifact_path(component).stat().st_size, 30*2**20)
        with self.assertRaises(ValueError):
            StreamingPipelinePredictor._predict_parts({'n_arboles':200,'arboles':[]}, sample_frame())

    def test_streaming_inference_and_original_unit_equation(self):
        from threadpoolctl import threadpool_limits
        self.assertIsInstance(predictor('kernel_operativo'), StreamingPipelinePredictor)
        self.assertFalse(hasattr(predictor('kernel_operativo'), 'estimators'))
        x = sample_frame()
        expected = predict('kernel_operativo', x)
        parameters = kernel_view.architecture()
        with threadpool_limits(limits=1):
            for target, definition in registry()['kernel_operativo']['respuestas'].items():
                model = joblib.load(ROOT/'data'/definition['ruta'])['estimador']
                p = parameters[target]
                values = np.column_stack([b.predict(x[PREDICTORS]) for b in model.models])
                combined = p['intercepto']+values@np.asarray(p['coeficientes'])
                np.testing.assert_allclose(combined, expected[target], rtol=1e-12, atol=1e-12)
                self.assertEqual(model.models[1].regressor_[2].kw_args, p['pesos_svr'])
                del model

    def test_offline_acceleration_equals_native(self):
        from tools.export_kernel import ImportancePredictor
        from threadpoolctl import threadpool_limits
        frame = pd.read_csv(ROOT/'data'/'observaciones.csv.gz').query("particion == 'prueba_bloques'").iloc[:80]
        with threadpool_limits(limits=1):
            for target, definition in registry()['kernel_operativo']['respuestas'].items():
                model = joblib.load(ROOT/'data'/definition['ruta'])['estimador']
                x = frame[PREDICTORS].copy()
                x['TIT'] = np.random.default_rng(42).permutation(x.TIT)
                np.testing.assert_allclose(ImportancePredictor(model).predict(x), model.predict(x), rtol=1e-10, atol=1e-10)
                del model

    def test_reproduction_folds_and_honest_meta_training(self):
        sys.path.insert(0, str(ROOT/'book'/'training'))
        import kernel_operativo_training as train
        data = pd.read_csv(ROOT/'data'/'observaciones.csv.gz')
        development, test, outer, _, _ = train.particiones(data, 168, 24)
        self.assertEqual((len(development), len(test)), (27570, 7411))
        configurations = train.candidates()
        self.assertEqual(len(configurations), 24)
        self.assertEqual(configurations[4], {'members': [0, 3], 'lambda': .01})
        self.assertEqual(configurations[6], {'members': [0, 4], 'lambda': .001})
        # Baratas y deterministas: cualquier cruce indebido de indices hace fallar fit/predict.
        frame = data.iloc[development].copy()
        folds = train.inner_splits(frame)
        records = []
        class Spy(DummyRegressor):
            def fit(self, X, y):
                self.ids = set(X.index)
                records.append(self.ids)
                return super().fit(X, y)
            def predict(self, X):
                if len(self.ids) < len(frame):
                    assert not self.ids.intersection(X.index)
                return super().predict(X)
        train.Suite().fit(frame, 'CO', factory=lambda: [Spy() for _ in range(6)])
        self.assertEqual(len(records), 24)
        for k, (tr, va) in enumerate(folds):
            self.assertTrue(all(r == set(frame.iloc[tr].index) for r in records[k*6:k*6+6]))
            self.assertFalse(set(frame.iloc[tr].bloque)&set(frame.iloc[va].bloque))
            np.testing.assert_array_equal(train.purge(frame, tr, va), tr)
        for a, b in outer:
            sub = data.iloc[development[a]]
            for tr, va in train.inner_splits(sub):
                self.assertFalse(set(sub.iloc[tr].index)&set(data.iloc[test].index))
                np.testing.assert_array_equal(train.purge(sub, tr, va), tr)

    def test_notebook_evidence_and_no_provisional_imports(self):
        notebook = json.loads((ROOT/'book'/'gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        cells = [c for c in notebook['cells'] if c.get('id', '').startswith('kernel-')]
        self.assertEqual(len(cells), 8)
        for cell in cells:
            if cell['cell_type'] == 'code':
                self.assertIsNotNone(cell['execution_count'])
                self.assertTrue(cell['outputs'])
                self.assertFalse(any(o['output_type'] == 'error' for o in cell['outputs']))
        for folder in ['dashboard', 'emission_models']:
            for path in (ROOT/folder).glob('*.py'):
                self.assertNotIn('modelos_kernel_regimen', path.read_text(encoding='utf-8'))
                self.assertNotIn('Pruebas provisionales', path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
