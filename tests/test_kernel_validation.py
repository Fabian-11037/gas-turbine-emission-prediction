"""Comparacion pareada, interpretacion prudente y presentacion sin entrenamiento."""
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image
from dashboard import kernel_validation as view, kernel_validation_content as text, kernel_view
from dashboard.models import registry
from tools.export_kernel_significance import holm

ROOT=Path(__file__).resolve().parents[1]


class KernelValidationTests(unittest.TestCase):
    def test_evidence_hashes_cohorts_and_no_retraining(self):
        metadata=json.loads((view.LOCATION/'metodologia.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['modelos_reentrenados_al_publicar'],0)
        self.assertFalse(metadata['prueba_usada_en_seleccion'])
        self.assertEqual((metadata['n_desarrollo'],metadata['n_prueba']),(27570,7411))
        for path,digest in metadata['archivos'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest)
        for target in ['CO','NOX']:
            for phase,n in [('OOF',27570),('prueba',7411)]:
                new=pd.read_csv(ROOT/f'book/informe_resultados/kernel_validation/aleatoria_{phase}_{target}.csv.gz')
                old=pd.read_csv(ROOT/f'book/informe_resultados/kernel_operativo/predicciones_{phase}_{target}.csv.gz')
                pd.testing.assert_frame_equal(new[['anio','orden_registro','bloque',target]],
                                            old[['anio','orden_registro','bloque',target]])
                self.assertEqual(len(new),n)

    def test_statistics_are_adjusted_and_orientation_is_explicit(self):
        results,cv,test=view.evidence()
        self.assertEqual(len(results),4)
        self.assertEqual(set(results.estrategia),{'bloques','aleatoria'})
        np.testing.assert_allclose(cv.p_Holm,holm(cv.p_CV_nominal))
        for _,group in test.groupby('longitud_bloque'):
            np.testing.assert_allclose(group.p_Holm,holm(group.p_prueba_nominal))
        np.testing.assert_allclose(test.delta_R2,test.R2_prueba_referencia-test.R2_prueba_nuevo,atol=1e-12)
        co=test.loc[test.objetivo.eq('CO')]
        nox=test.loc[test.objetivo.eq('NOX')]
        self.assertTrue((co.p_Holm>.05).all())
        self.assertTrue(((co.IC95_inf<0)&(co.IC95_sup>0)).all())
        self.assertTrue((nox.p_Holm<.05).all())
        self.assertTrue((nox.IC95_inf>0).all())
        self.assertAlmostEqual(cv.set_index('objetivo').loc['NOX','p_Holm'],.008825276569201797)

    def test_cv_correction_matches_saved_folds(self):
        _,cv,_=view.evidence()
        differences=pd.read_csv(view.LOCATION/'diferencias_pliegues.csv')
        stages=pd.read_csv(view.LOCATION/'etapas.csv')
        for row in cv.itertuples():
            delta=differences.loc[differences.objetivo.eq(row.objetivo)].sort_values('fold').delta_MSE.to_numpy()
            folds=stages.loc[stages.objetivo.eq(row.objetivo)&stages.etapa.ne('final')]
            factor=1/5+(folds.n_val/folds.n_train).mean()
            se=np.sqrt(factor*np.var(delta,ddof=1))
            p=2*stats.t.sf(abs(delta.mean()/se),4)
            self.assertAlmostEqual(row.p_CV_nominal,p)

    def test_interpretation_and_views_do_not_claim_a_co_advantage(self):
        self.assertIn('no demuestra superioridad',text.CONCLUSION)
        self.assertIn('objetivo',text.CONCLUSION)
        self.assertIn('reduce la similitud',text.CONCLUSION)
        self.assertIn('no es significativa',view.interpretation('CO'))
        self.assertIn('La ventaja es significativa',view.interpretation('NOX'))
        self.assertEqual(len(registry()),8)
        for target in ['CO','NOX']:
            style,panel=kernel_view.render('kernel_operativo',target,'validacion')
            self.assertEqual(style,{})
            self.assertEqual(panel.id,'kernel-validation-analysis')
            chart=view.results_figure(target)
            self.assertEqual(len(chart.data),2)
            self.assertEqual(tuple(chart.layout.yaxis.range),(0,1))
        self.assertEqual(kernel_view.render('ridge','CO','validacion'),({'display':'none'},None))
        intervals=view.difference_figure()
        self.assertEqual(len(intervals.data),2)
        self.assertEqual(intervals.layout.shapes[0].x0,0)

    def test_book_shares_text_and_keeps_executed_figures(self):
        notebook=json.loads((ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        cells={c['id']:c for c in notebook['cells']}
        self.assertIn(text.INTRO,''.join(cells['kernel-validation-method']['source']))
        conclusion=''.join(cells['kernel-validation-conclusion']['source'])
        self.assertIn(text.CONCLUSION,conclusion)
        self.assertIn(view.interpretation('CO'),conclusion)
        self.assertIn(text.CONCLUSION,''.join(cells['fe9bdcbf']['source']))
        output=cells['kernel-validation-results']['outputs']
        self.assertEqual(len(output),4)
        for entry,name in zip(output,['r2','ci']):
            data=base64.b64decode(entry['data']['image/png'])
            self.assertEqual(data,(ROOT/f'assets/kernel_validation_{name}.png').read_bytes())
            image=Image.open(BytesIO(data))
            self.assertGreater(image.width,1000)
            self.assertGreater(np.asarray(image).std(),10)


if __name__=='__main__':
    unittest.main()
