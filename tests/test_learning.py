"""Contratos de curvas precalculadas y correspondencia con la CV publicada."""
import json
import unittest
import numpy as np
from app import app
from dashboard import learning
from dashboard.models import registry
from dashboard.repository import DATA, learning_curves, results


class LearningCurveTest(unittest.TestCase):
    def test_complete_curves_and_purged_groups(self):
        frame = learning_curves()
        self.assertEqual(len(frame),400)
        self.assertEqual(set(frame.modelo_id),set(registry()))
        self.assertEqual(set(frame.objetivo),{'CO','NOX'})
        self.assertEqual(set(frame.fraccion),{.2,.4,.6,.8,1.})
        self.assertFalse(frame[['modelo_id','objetivo','fold','fraccion']].duplicated().any())
        self.assertTrue(frame.min_separacion.gt(24).all())
        self.assertTrue(np.isfinite(frame[['R2_train','R2_validacion','RMSE_train','RMSE_validacion']]).all().all())
        for _,group in frame.groupby(['modelo_id','objetivo','fold']):
            group = group.sort_values('fraccion')
            self.assertTrue(group.n_train.is_monotonic_increasing)
            self.assertEqual(group.n_validacion.nunique(),1)
            self.assertEqual(group.parametros.nunique(),1)
            self.assertEqual(group.iloc[-1].n_train,group.iloc[-1].n_train_disponible)
        # Los mismos registros alimentan las curvas de todas las familias y emisiones.
        for _,group in frame.groupby(['fold','fraccion']):
            self.assertEqual(group.huella_subconjunto.nunique(),1)
        self.assertTrue(frame.loc[frame.fraccion.eq(1),'origen'].eq('validacion_externa_guardada').all())

    def test_endpoint_matches_current_external_validation(self):
        endpoint = learning_curves().loc[lambda d:d.fraccion.eq(1)].groupby(['modelo','objetivo']).R2_validacion.mean()
        for row in results().itertuples():
            self.assertAlmostEqual(endpoint.loc[row.modelo,row.objetivo],row.R2_CV_promedio,places=12)
        metadata = json.loads((DATA/'curvas_aprendizaje_metodologia.json').read_text(encoding='utf-8'))
        self.assertFalse(metadata['usa_prueba'])
        self.assertTrue(metadata['modelos_publicados_sin_cambios'])
        self.assertEqual(metadata['pliegues'],5)

    def test_figures_tables_and_callback_for_all_models(self):
        for model in registry():
            for target in ['CO','NOX']:
                for metric in ['R2','RMSE']:
                    figure,note,table = learning.render(model,target,metric)
                    means = [trace for trace in figure.data if trace.showlegend is not False]
                    self.assertEqual([trace.name for trace in means],['Entrenamiento','Validación'])
                    self.assertTrue(all(len(trace.x)==5 for trace in means))
                    self.assertIn('no un intervalo de confianza',note)
                    self.assertEqual(len(table.children.children[1].children),5)
        with self.assertRaises(ValueError):
            learning.figure(learning.summary('ridge','CO'),'accuracy')
        response = app.server.test_client().get('/_dash-dependencies')
        callback = next(c for c in response.json if 'model-learning-chart' in c['output'])
        self.assertEqual({i['id'] for i in callback['inputs']},{'model-select','model-target','learning-metric'})


if __name__=='__main__':
    unittest.main()
