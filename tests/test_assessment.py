"""Frozen-data hierarchy, bootstrap boundaries and computational evidence."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from app import app
from dashboard import assessment
from tools.export_hierarchical_statistics import StationaryOriginalBlocks,confidence_set,directed

ROOT = Path(__file__).resolve().parents[1]


def toy_frame():
    rng = np.random.default_rng(7)
    rows = []
    for year in [2011,2012]:
        for block in range(4):
            for index in range(4):
                rows.append(dict(anio=year,bloque=year*10+block,orden_registro=year*1000+block*4+index,
                    fold_validacion=1+block//2,TEY=rng.normal(),CO=rng.normal(),
                    good=rng.uniform(.1,1),bad=rng.uniform(1,3)))
    return pd.DataFrame(rows)


class AssessmentTests(unittest.TestCase):
    def test_checks_are_grouped_and_explanations_precede_evidence(self):
        from dash import html
        from dashboard.assessment_content import SENSITIVITY_HELP,RANKING_HELP,STATISTICS,EFFICIENCY
        for target in ['CO','NOX']:
            disclosures = {child.children[0].children:child.children[1].children
                for child in assessment.statistical_view(target).children if isinstance(child,html.Details)}
            self.assertEqual(set(disclosures),{'Método y comprobaciones estadísticas',
                'Los 29 procedimientos y controles coincidentes'})
            tabs = disclosures['Método y comprobaciones estadísticas'][0]
            self.assertEqual(tabs.value,'metodo')
            panels = {tab.value:tab.children for tab in tabs.children}
            self.assertEqual(panels['metodo'][0].children,STATISTICS)
            self.assertEqual(panels['sensibilidad'][0].children,SENSITIVITY_HELP)
            self.assertEqual(panels['rangos'][0].children,RANKING_HELP)
            for model_id in ['ridge','kernel_operativo']:
                cost = {child.children[0].children:child.children[1].children
                    for child in assessment.efficiency_view(model_id,target).children if isinstance(child,html.Details)}
                self.assertNotIn('Detalle de ejecución · modelo seleccionado',cost)
                self.assertNotIn('Condiciones de medición',cost)
                self.assertEqual(cost['Método y detalle de ejecución'][0].children,EFFICIENCY)
                self.assertEqual(cost['Método y detalle de ejecución'][2].className,'table-scroll')

    def test_protected_models_and_validation(self):
        for folder in ['statistical_hierarchy','efficiency']:
            metadata = assessment.metadata(folder)
            self.assertEqual(metadata['modelos_reentrenados'],0)
            self.assertFalse(metadata['validacion_modificada'])
            for path,expected in metadata['huellas_protegidas'].items():
                with (ROOT/path).open('rb') as source:
                    self.assertEqual(hashlib.file_digest(source,'sha256').hexdigest(),expected,path)

    def test_all_procedures_and_external_only_losses(self):
        metadata = assessment.metadata('statistical_hierarchy')
        self.assertEqual(metadata['procedimientos_por_emision'],29)
        candidates = assessment.read('statistical_hierarchy','procedimientos')
        self.assertEqual(len(candidates),58)
        data = pd.read_csv(ROOT/'data/observaciones.csv.gz')
        expected = data.loc[data.particion.eq('entrenamiento_CV'),['anio','orden_registro','bloque']].reset_index(drop=True)
        original = pd.read_csv(ROOT/'book/informe_resultados/entregable2_anidada/predicciones_OOF.csv.gz')
        for target in ['CO','NOX']:
            frame = pd.read_csv(ROOT/f'data/statistical_hierarchy/perdidas_OOF_{target}.csv.gz')
            pd.testing.assert_frame_equal(frame[['anio','orden_registro','bloque']],expected)
            labels = candidates.loc[candidates.objetivo.eq(target),'procedimiento']
            self.assertEqual(len(frame),27570)
            self.assertTrue(np.isfinite(frame[labels]).all().all())
            for row in candidates.loc[candidates.objetivo.eq(target)&candidates.publicado].itertuples():
                if row.modelo=='Kernel operativo':
                    kernel = pd.read_csv(ROOT/f'book/informe_resultados/kernel_operativo/predicciones_OOF_{target}.csv.gz')
                    p = kernel.prediccion.to_numpy()
                else:
                    p = original[target+'_'+row.modelo.replace(' ','_')].to_numpy()
                np.testing.assert_allclose(frame[row.procedimiento],(frame[target].to_numpy()-p)**2,rtol=1e-8,atol=1e-8)
            for length in [1,2,4]:
                rows = assessment.mcs_table(target,length,False)
                self.assertEqual(len(rows),29)
                self.assertTrue(rows.p_MCS.between(0,1).all())
                np.testing.assert_array_equal(rows.incluido_005,rows.p_MCS>.05)

    def test_bootstrap_boundaries_and_native_mcs(self):
        from arch.bootstrap import MCS
        frame = toy_frame()
        bootstrap = StationaryOriginalBlocks(frame,2)
        for a,b in enumerate(bootstrap.successor):
            first,last = frame.iloc[bootstrap.blocks[a]],frame.iloc[bootstrap.blocks[b]]
            self.assertEqual(first.anio.iloc[0],last.anio.iloc[0])
            self.assertEqual(first.fold_validacion.iloc[0],last.fold_validacion.iloc[0])
        for args,_ in bootstrap.bootstrap(5):
            indices = args[0]
            self.assertEqual(len(indices),len(frame))
            self.assertEqual(frame.iloc[indices].groupby('anio').size().tolist(),[16,16])
        native = MCS(frame[['good','bad']],size=.05,reps=200,method='R',seed=42)
        native.bootstrap = StationaryOriginalBlocks(frame,2)
        native.compute()
        actual = confidence_set(frame,['good','bad'],2,reps=200).set_index('procedimiento')
        for name in ['good','bad']:
            self.assertAlmostEqual(actual.loc[name,'p_MCS'],native.pvalues.loc[name].iloc[0])
        frame['same'] = frame.good
        collapsed = confidence_set(frame,['good','same'],1,reps=100)
        self.assertTrue(collapsed.p_MCS.eq(1).all())
        null = directed(frame,'good','same',reps=100)
        self.assertEqual(null['p_GW'],1.)
        self.assertEqual(null['p_DM'],1.)
        self.assertEqual(null['Cohen_dz_bloques'],0.)

    def test_efficiency_repetitions_equivalence_and_pareto(self):
        measurements = assessment.read('efficiency','mediciones')
        costs = assessment.read('efficiency','resumen')
        self.assertEqual(len(costs),20)
        self.assertEqual(len(measurements),300)
        self.assertTrue(measurements.segundos.gt(0).all())
        self.assertTrue(costs.error_maximo.lt(1e-7).all())
        self.assertTrue(costs.RSS_pico_MB.ge(costs.RSS_base_MB).all())
        self.assertTrue(costs.memoria_entrenamiento_MB.isna().all())
        kernel = costs.loc[costs.modelo_id.eq('kernel_operativo')&costs.modo.eq('residente')]
        self.assertTrue(kernel.segundos_ajuste_historico.isna().all())
        self.assertEqual(kernel.n_soporte.astype(int).tolist(),[16718,17805])
        pareto = assessment.read('efficiency','optimizacion_pareto')
        self.assertEqual(len(pareto),56)
        for _,group in pareto.groupby(['objetivo','modelo']):
            self.assertTrue(group.Pareto.any())
            for row in group.itertuples():
                dominated = ((group.MSE_CV_externo<=row.MSE_CV_externo)&(group.segundos_total<=row.segundos_total)&
                    ((group.MSE_CV_externo<row.MSE_CV_externo)|(group.segundos_total<row.segundos_total))).any()
                self.assertEqual(row.Pareto,not dominated)

    def test_all_views_callbacks_and_no_production_dependencies(self):
        for target in ['CO','NOX']:
            self.assertIsNotNone(assessment.statistical_view(target))
            self.assertEqual(len(assessment.critical_difference(target).data[0].x),8)
            for mid in ['knn','ridge','lasso','decision_tree','random_forest','xgboost','svr_lineal','kernel_operativo']:
                self.assertIsNotNone(assessment.efficiency_view(mid,target))
                self.assertTrue(all(len(t.x)==3 for t in assessment.latency(mid,target).data))
        callbacks = app.server.test_client().get('/_dash-dependencies').json
        self.assertTrue(any('assessment-statistics' in row['output'] for row in callbacks))
        self.assertTrue(any('assessment-efficiency' in row['output'] for row in callbacks))
        requirements = (ROOT/'requirements.txt').read_text()
        self.assertNotIn('arch==',requirements)
        self.assertNotIn('psutil',requirements)

    def test_book_outputs_and_shared_scope(self):
        from dashboard.assessment_content import STATISTICS,COMPLEXITY,EFFICIENCY
        book = json.loads((ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        cells = [c for c in book['cells'] if c.get('id','').startswith('assessment-')]
        text = '\n'.join(''.join(c['source']) for c in cells)
        for source in [STATISTICS,COMPLEXITY,EFFICIENCY]:
            self.assertIn(source.replace('### ','#### '),text)
        code = [c for c in cells if c['cell_type']=='code']
        self.assertEqual(len(code),4)
        for cell in code:
            self.assertNotIn('.fit(',''.join(cell['source']))
            self.assertEqual(sum('image/png' in o.get('data',{}) for o in cell['outputs']),1)

    def test_clear_conclusions_shared_with_book(self):
        book = json.loads((ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        for target,expected in [('CO','no hay evidencia suficientemente estable'),
                                ('NOX','hay evidencia estadística consistente')]:
            conclusion = assessment.statistical_conclusion(target)
            self.assertIn(expected,conclusion)
            cell = next(c for c in book['cells'] if c.get('id')=='assessment-stat-'+target)
            interpretations = [o.get('data',{}).get('text/markdown','') for o in cell['outputs']]
            self.assertIn(conclusion,interpretations)
        self.assertIn('no demuestra que los modelos sean iguales',assessment.statistical_conclusion('CO'))
