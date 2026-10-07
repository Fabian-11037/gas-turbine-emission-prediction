"""Integridad de datos, equivalencia de prediccion y contratos de callbacks."""
import base64
import unittest
import numpy as np
import pandas as pd
from app import app
from dashboard import eda,model_view
from dashboard.models import predict,registry
from dashboard.repository import observations,predictions,sample_frame,sensors


class DashboardTest(unittest.TestCase):
    def test_http_and_layout(self):
        client = app.server.test_client()
        for path in ['/','/healthz','/_dash-layout','/_dash-dependencies','/assets/dashboard.css']:
            response = client.get(path)
            self.assertEqual(response.status_code,200,path)
            response.close()
        self.assertEqual(client.get('/healthz').json['dataset'],36733)

    def test_registry_and_no_training_dependencies(self):
        import sys
        self.assertEqual(set(registry()),{'ridge','svr_lineal','media'})
        self.assertNotIn('sklearn',sys.modules)
        self.assertNotIn('joblib',sys.modules)
        self.assertNotIn('block_model',sys.modules)

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
        for column,value in [('AT',np.nan),('AP',-1),('AH',101)]:
            bad = sample.copy()
            bad[column] = value
            with self.assertRaises(ValueError):
                predict('ridge',bad)
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
                self.assertEqual(len(out),8)
                self.assertEqual(len(out[1]),5)

    def test_batch_and_predict_form(self):
        csv = sample_frame().to_csv(index=False)
        content = 'data:text/csv;base64,'+base64.b64encode(csv.encode()).decode()
        out,warnings,negative = model_view.parse_batch(content,'svr_lineal')
        self.assertEqual(len(out),1)
        self.assertEqual(warnings,[])
        np.testing.assert_allclose(out[['CO_predicho','NOX_predicho']],predict('svr_lineal',sample_frame()))
        with self.assertRaises(ValueError):
            model_view.parse_batch('data:text/csv;base64,bad!','ridge')
        values = [s['ejemplo'] for s in sensors()]
        self.assertEqual(len(model_view.estimate('ridge',values)[0]),2)


if __name__ == '__main__':
    unittest.main()
