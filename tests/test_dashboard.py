"""Integridad de datos, equivalencia de prediccion y contratos de callbacks."""
import base64
import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pandas as pd
from app import TITLE,app
from dashboard import eda,model_view
from dashboard.models import artifact_path,feature_importance,predict,predictor,registry,_cached_predictor
from dashboard.repository import importances,observations,predictions,results,sample_frame,sensors,optimization_results


class DashboardTest(unittest.TestCase):
    def test_http_and_layout(self):
        client = app.server.test_client()
        for path in ['/','/healthz','/_dash-layout','/_dash-dependencies','/assets/dashboard.css']:
            response = client.get(path)
            self.assertEqual(response.status_code,200,path)
            response.close()
        self.assertEqual(client.get('/healthz').json['dataset'],36733)
        self.assertEqual(app.title,TITLE)
        layout = client.get('/_dash-layout').json
        self.assertEqual(layout['props']['children'][0]['props']['children'][0]['props']['children'][1]['props']['children'],TITLE)

    def test_registry_and_no_training_dependencies(self):
        import sys
        self.assertEqual(set(registry()),{'knn','ridge','lasso','decision_tree','random_forest','xgboost','svr_lineal'})
        self.assertNotIn('block_model',sys.modules)
        self.assertNotIn('regression_entregable2',sys.modules)
        self.assertNotIn('deap',sys.modules)
        self.assertEqual(len(results()),14)
        self.assertEqual(len(optimization_results()),56)
        selected=results().loc[lambda d:d.seleccionado_por_CV].set_index('objetivo').modelo.to_dict()
        self.assertEqual(selected,{'CO':'Random Forest','NOX':'XGBoost'})

    def test_equivalence_with_saved_test_predictions(self):
        df = observations()
        holdout = predictions()
        joined = df.merge(holdout,on=['anio','orden_registro'],suffixes=('','_expected'),validate='one_to_one')
        self.assertEqual(len(joined),7411)
        valid = joined.loc[joined.AH.between(0,100)]
        for mid,spec in registry().items():
            y = predict(mid,valid)
            for target in ['CO','NOX']:
                np.testing.assert_allclose(y[target],valid[f"{target}_{spec['nombre'].replace(' ','_')}"],rtol=1e-10,atol=1e-10)
                self.assertLess(spec['error_max_exportacion'][target],1e-10)

    def test_inference_validation(self):
        sample = sample_frame()
        np.testing.assert_allclose(predict('ridge',sample),predict('ridge',sample[sample.columns[::-1]]))
        for mid in registry():
            for column,value in [('AT',np.nan),('AP',-1),('AH',101)]:
                bad = sample.copy()
                bad[column] = value
                with self.assertRaises(ValueError):
                    predict(mid,bad)
        with self.assertRaises(ValueError):
            predict('ridge',sample.drop(columns='AT'))

    def test_eda_filters_and_all_views(self):
        for target in ['CO','NOX']:
            for view in ['distribuciones','relaciones','temporal','correlaciones']:
                output = eda.render(target,[2014,2015],'desarrollo',view,'AT')
                self.assertEqual(len(output),7)
                self.assertTrue(output[0].data)
                self.assertTrue(output[1].data)
        self.assertEqual(len(eda.render('CO',[],'desarrollo','relaciones','AT')),7)
        self.assertFalse(eda.render('CO',[],'desarrollo','relaciones','AT')[0].data)

    def test_results_for_every_registered_model(self):
        for mid in registry():
            for target in ['CO','NOX']:
                out = model_view.model_results(mid,target)
                self.assertEqual(len(out),10)
                self.assertEqual(len(out[1]),5)
                self.assertEqual(len(out[5].children.children[1].children),7)
                self.assertEqual(len(out[8].children.children[1].children),2)
                self.assertEqual(len(out[9].children.children[1].children),28)

    def test_batch_and_predict_form(self):
        csv = sample_frame().to_csv(index=False)
        content = 'data:text/csv;base64,'+base64.b64encode(csv.encode()).decode()
        for mid in registry():
            out,warnings,negative = model_view.parse_batch(content,mid)
            self.assertEqual(len(out),1)
            self.assertEqual(warnings,[])
            np.testing.assert_allclose(out[['CO_predicho','NOX_predicho']],predict(mid,sample_frame()))
        with self.assertRaises(ValueError):
            model_view.parse_batch('data:text/csv;base64,bad!','ridge')
        values = [s['ejemplo'] for s in sensors()]
        self.assertEqual(len(model_view.estimate('ridge',values)[0]),2)

    def test_importance_for_every_model_and_target(self):
        for mid in registry():
            for target in ['CO','NOX']:
                importance = feature_importance(mid,target)
                self.assertEqual(set(importance.sensor),set(s['id'] for s in sensors()))
                self.assertTrue(np.isfinite(importance[['aumento_MSE','desviacion']]).all().all())
                self.assertTrue(importance.aumento_MSE.is_monotonic_decreasing)
                output = model_view.importance_results(mid,target)
                self.assertEqual(len(output),3)
                self.assertEqual(len(output[0].data[0].y),9)
        self.assertEqual(len(importances()),126)
        self.assertTrue(importances().n.eq(7411).all())
        self.assertTrue(importances().repeticiones.eq(10).all())

    def test_precomputed_permutation_equals_native_prediction(self):
        frame = observations().loc[lambda d:d.particion.eq('prueba_bloques')]
        sensor = registry()['ridge']['predictores'][0]
        for target in ['CO','NOX']:
            estimator=predictor('ridge').estimators[target]
            baseline = np.mean((estimator.predict(frame[registry()['ridge']['predictores']])-frame[target])**2)
            rng = np.random.default_rng(42)
            changes = []
            for _ in range(10):
                shuffled = frame.copy()
                shuffled[sensor] = rng.permutation(frame[sensor].to_numpy())
                changes.append(np.mean((estimator.predict(shuffled[registry()['ridge']['predictores']])-frame[target])**2)-baseline)
            importance = feature_importance('ridge',target)
            value = importance.set_index('sensor').loc[sensor]
            self.assertAlmostEqual(value.aumento_MSE,np.mean(changes),places=10)
            self.assertAlmostEqual(value.desviacion,np.std(changes,ddof=1),places=10)
            with patch('dashboard.models.predictor',side_effect=AssertionError('Debe leer la evidencia precalculada.')):
                pd.testing.assert_frame_equal(importance,feature_importance('ridge',target))

    def test_artifact_integrity_and_path_validation(self):
        with TemporaryDirectory(prefix='artifact-',dir=Path(__file__).resolve().parent) as temp:
            root=Path(temp)
            (root/'models').mkdir()
            path=root/'models'/'test.joblib'
            path.write_bytes(b'test')
            with patch('dashboard.models.DATA',root):
                with self.assertRaises(ValueError):
                    artifact_path({'ruta':'models/test.joblib','sha256':'0'*64})
                with self.assertRaises(ValueError):
                    artifact_path({'ruta':'../test.joblib','sha256':'0'*64})
                self.assertEqual(artifact_path({'ruta':'models/test.joblib','sha256':hashlib.sha256(b'test').hexdigest()}),path)

    def test_inference_never_fits_and_cache_is_bounded(self):
        from sklearn.pipeline import Pipeline
        with patch.object(Pipeline,'fit',side_effect=AssertionError('No se permite entrenamiento.')):
            for mid in registry():
                self.assertEqual(predict(mid,sample_frame()).shape,(1,2))
        self.assertLessEqual(_cached_predictor.cache_info().currsize,3)


if __name__ == '__main__':
    unittest.main()
