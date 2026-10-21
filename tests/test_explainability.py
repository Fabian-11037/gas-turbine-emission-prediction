"""Frozen predictions, complete cohorts and comparable explanation evidence."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

from app import app
from dashboard import explainability as xai
from dashboard.explainability_content import INTRODUCTION, METHOD

ROOT = Path(__file__).resolve().parents[1]


class ExplainabilityTests(unittest.TestCase):
    def test_frozen_artifacts_and_training_only_background(self):
        metadata = xai.methodology()
        self.assertEqual(metadata['modelos_reentrenados'],0)
        self.assertFalse(metadata['validacion_modificada'])
        for relative,expected in metadata['huellas_protegidas'].items():
            with (ROOT/relative).open('rb') as source:
                self.assertEqual(hashlib.file_digest(source,'sha256').hexdigest(),expected,relative)
        observations = pd.read_csv(ROOT/'data/observaciones.csv.gz')
        development = observations.loc[observations.particion.eq('entrenamiento_CV')]
        background = pd.read_csv(xai.DIRECTORY/'referencia_desarrollo.csv')
        self.assertEqual(background.groupby('anio').size().tolist(),[2]*5)
        keys = ['anio','orden_registro']
        self.assertEqual(len(background.merge(development[keys],on=keys)),len(background))
        self.assertTrue(background.particion.eq('entrenamiento_CV').all())

    def test_global_coverage_additivity_and_original_predictions(self):
        original = pd.read_csv(ROOT/'data/predicciones_prueba.csv.gz')
        keys = ['anio','orden_registro']
        for mid,family in [('kernel_operativo','Kernel_operativo'),('xgboost','XGBoost')]:
            for target in ['CO','NOX']:
                frame = xai.global_frame(mid,target)
                self.assertEqual(len(frame),7411)
                self.assertFalse(frame.duplicated(keys).any())
                pd.testing.assert_frame_equal(frame[keys],original[keys])
                values = frame[['shap_'+f for f in xai.FEATURES]].to_numpy()
                self.assertTrue(np.isfinite(values).all())
                np.testing.assert_allclose(frame.base+values.sum(axis=1),frame.prediccion,rtol=1e-8,atol=1e-7)
                np.testing.assert_allclose(frame.prediccion,original[f'{target}_{family}'],rtol=1e-8,atol=1e-7)
                summary = xai.selected('resumen_global',mid,target).set_index('sensor').loc[xai.FEATURES]
                np.testing.assert_allclose(summary.media_abs_SHAP,np.mean(abs(values),axis=0))

    def test_common_cases_exact_local_and_lime_reconstruction(self):
        cases = xai.evidence()['casos']
        self.assertEqual(len(cases),8)
        self.assertFalse(cases.duplicated(['modelo_id','objetivo','caso']).any())
        for (_,case),group in cases.groupby(['objetivo','caso']):
            self.assertEqual(group[['anio','orden_registro']].drop_duplicates().shape[0],1)
        for row in cases.itertuples():
            _,frame = xai.case_data(row.modelo_id,row.objetivo,row.caso)
            self.assertEqual(set(frame.sensor),set(xai.FEATURES))
            self.assertAlmostEqual(row.base_SHAP+frame.SHAP.sum(),row.prediccion,places=8)
            self.assertAlmostEqual(row.intercepto_LIME+frame.LIME.sum(),row.prediccion_LIME,places=8)
            self.assertAlmostEqual(abs(row.prediccion-row.observado),row.error_absoluto,places=9)
            self.assertTrue(frame.condicion_LIME.notna().all())
            self.assertGreaterEqual(row.R2_local_LIME,0.)
            self.assertLessEqual(row.R2_local_LIME,1.)

    def test_figures_views_and_runtime_dependencies(self):
        for mid in xai.NAMES:
            for target in ['CO','NOX']:
                for view in ['global','local','comparacion']:
                    for case in xai.CASE_NAMES:
                        visible,content,control = xai.render(mid,target,view,case)
                        self.assertEqual(visible,{})
                        self.assertIsNotNone(content)
                fig = xai.global_summary(mid,target)
                self.assertEqual(len(fig.data),9)
                self.assertTrue(all(len(trace.x)==7411 for trace in fig.data))
                self.assertEqual(len(xai.waterfall(mid,target,'error_alto').data[0].measure),11)
        self.assertEqual(xai.render('ridge','CO','global','error_bajo'),
                         ({'display':'none'},None,{'display':'none'}))
        dependencies = app.server.test_client().get('/_dash-dependencies').json
        callback = next(c for c in dependencies if 'explain-content' in c['output'])
        self.assertEqual({i['id'] for i in callback['inputs']},
                         {'model-select','model-target','explain-view','explain-case'})
        self.assertNotIn('shap',sys.modules)
        self.assertNotIn('lime',sys.modules)
        requirements = (ROOT/'requirements.txt').read_text()
        self.assertNotIn('shap',requirements.lower())
        self.assertNotIn('lime',requirements.lower())

    def test_report_has_outputs_and_shared_professional_text(self):
        notebook = json.loads((ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        new = [c for c in notebook['cells'] if c.get('id','').startswith('xai-')]
        self.assertTrue(new)
        introduction = next(c for c in new if c['id']=='xai-method')
        source = ''.join(introduction['source'])
        self.assertIn(INTRODUCTION,source)
        self.assertIn(METHOD.replace('### ','#### '),source)
        code = [c for c in new if c['cell_type']=='code']
        self.assertEqual(len(code),4)
        for cell in code:
            images = [o for o in cell['outputs'] if 'image/png' in o.get('data',{})]
            self.assertEqual(len(images),3)
            self.assertNotIn('.fit(',''.join(cell['source']))
        coverage = ''.join(next(c['source'] for c in notebook['cells'] if c.get('id')=='1b49e614'))
        self.assertNotIn('SHAP/LIME, semillas adicionales',coverage)
