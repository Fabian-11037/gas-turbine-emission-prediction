"""Frozen splits, no-leakage construction and seed-analysis contracts."""
import json
from pathlib import Path
import unittest

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from tools import validate_stability_ablation as validation

ROOT = Path(__file__).resolve().parents[1]


class StabilityTests(unittest.TestCase):
    def test_saved_outer_and_inner_splits(self):
        dev, test, splits = validation.load_frames()
        self.assertEqual((len(dev), len(test)), (27570, 7411))
        self.assertFalse(set(dev.bloque)&set(test.bloque))
        self.assertEqual(len(splits), 5)
        for train, heldout in splits:
            self.assertTrue(np.array_equal(validation.purge(dev,train,heldout),train))
            inner_frame = dev.iloc[train]
            first, second = validation.inner_splits(inner_frame), validation.inner_splits(inner_frame)
            seen = np.zeros(len(inner_frame))
            for (a,b),(c,d) in zip(first,second):
                np.testing.assert_array_equal(a,c)
                np.testing.assert_array_equal(b,d)
                self.assertFalse(set(inner_frame.iloc[a].bloque)&set(inner_frame.iloc[b].bloque))
                np.testing.assert_array_equal(validation.purge(inner_frame,a,b),a)
                seen[b] += 1
            np.testing.assert_array_equal(seen,1)

    def test_ablation_only_changes_requested_components(self):
        config = dict(members=[6,9],weights=[.5,.5,2.],**{'lambda':.01})
        original = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),SVR(C=20.,gamma=.2)),
                                               transformer=StandardScaler())
        x = np.ones((3,9))
        for derived in [True,False]:
            for use_weights in [True,False]:
                model = validation.branch_template(original,'svr',config,derived,use_weights,123)
                self.assertEqual(model.regressor.steps[-1][1].C,20.)
                self.assertEqual(model.regressor.steps[-1][1].gamma,.2)
                weights = model.regressor.steps[-2][1]
                values = validation.operational_features(x) if derived else x
                transformed = weights.transform(values)
                self.assertEqual(transformed.shape[1],11 if derived else 9)
                if not use_weights:
                    np.testing.assert_array_equal(transformed,values)
        original = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),
            ExtraTreesRegressor(n_estimators=200,max_features=.75)),transformer=StandardScaler())
        model = validation.branch_template(original,'tree',config,True,True,123)
        tree = model.regressor.steps[-1][1]
        self.assertEqual((tree.random_state,tree.n_estimators,tree.max_depth,tree.min_samples_leaf),(123,200,20,2))
        self.assertEqual(tree.max_features,.75)

    def test_result_integrity_when_complete(self):
        path = ROOT/'data/stability_ablation/metodologia.json'
        if not path.exists() or json.loads(path.read_text(encoding='utf-8'))['estado']!='Completado':
            self.skipTest('Training not complete yet')
        import pandas as pd
        meta = json.loads(path.read_text(encoding='utf-8'))
        self.assertFalse(meta['validacion_modificada'])
        self.assertFalse(meta['modelos_publicados_sustituidos'])
        self.assertEqual(meta['huellas_protegidas'],validation.frozen_hashes())
        rows = pd.read_csv(path.parent/'resultados.csv')
        self.assertEqual(len(rows),180)
        self.assertTrue(np.isfinite(rows.R2).all())
        for _, group in rows.loc[rows.variante.eq('completo')&rows.fase.eq('OOF')].groupby(['objetivo','modelo_id']):
            self.assertEqual(sorted(group.semilla),sorted(validation.SEEDS))
            if group.determinista.all():
                self.assertTrue(group.reutilizado.all())
                self.assertEqual(group.R2.nunique(),1)
        contrast = pd.read_csv(path.parent/'ablacion_contrastes.csv')
        self.assertEqual(len(contrast),30)
        self.assertTrue(contrast.p_Holm.between(0,1).all())
        self.assertTrue((contrast.p_Holm>=contrast.p_bootstrap).all())

    def test_runtime_and_book_reports_when_complete(self):
        path = ROOT/'data/stability_ablation/sensibilidad_semillas.csv'
        if not path.exists():
            self.skipTest('Training not complete yet')
        from dashboard import stability
        from dashboard.stability_content import METHOD, ablation_conclusion
        for target in ['CO','NOX']:
            for model_id in validation.registry():
                self.assertIsNotNone(stability.view(model_id,target))
                self.assertEqual(len(stability.seed_figure(model_id,target).data[0].x),5)
            self.assertEqual(len(stability.ablation_table(target)),6)
        explanation = ablation_conclusion(stability.read('ablacion_contrastes'),'CO',stability.read('resultados'))
        self.assertIn('algunas variantes sin adaptaciones superan',explanation)
        notebook = json.loads((ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        cells = [c for c in notebook['cells'] if c.get('id','').startswith('stability-')]
        self.assertIn(METHOD.replace('### ','#### '),'\n'.join(''.join(c['source']) for c in cells))
        code = [c for c in cells if c['cell_type']=='code']
        self.assertEqual(len(code),2)
        for cell in code:
            self.assertNotIn('.fit(',''.join(cell['source']))
            self.assertEqual(sum('image/png' in o.get('data',{}) for o in cell['outputs']),2)
        co = next(c for c in code if c['id']=='stability-results-CO')
        self.assertIn(explanation,[o.get('data',{}).get('text/markdown','') for o in co['outputs']])
