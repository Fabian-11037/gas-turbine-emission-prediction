"""Training-only EDA reference, correct loading semantics and read-only filters."""
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app import app
from dashboard import eda,multivariate as view
from dashboard.repository import observations

ROOT = Path(__file__).resolve().parents[1]


class MultivariateTests(unittest.TestCase):
    def test_reference_and_protected_artifacts(self):
        scores,loads,variance,meta = view.evidence()
        self.assertEqual((len(scores),len(loads),len(variance)),(36733,9,9))
        self.assertTrue(meta['ajuste_solo_desarrollo'])
        self.assertFalse(meta['objetivos_usados_en_ajuste'])
        for name,expected in meta['archivos'].items():
            self.assertEqual(hashlib.sha256((view.LOCATION/name).read_bytes()).hexdigest(),expected)
        for name,expected in meta['huellas_protegidas'].items():
            with (ROOT/name).open('rb') as source:
                self.assertEqual(hashlib.file_digest(source,'sha256').hexdigest(),expected)
        self.assertAlmostEqual(variance.varianza.iloc[0],.54465388,places=7)
        self.assertAlmostEqual(variance.varianza.iloc[1],.19661911,places=7)
        self.assertAlmostEqual(variance.acumulada.iloc[-1],1.)

    def test_scores_and_loading_correlations(self):
        scores,loads,_,meta = view.evidence()
        data = observations().sort_values('orden_registro').reset_index(drop=True)
        pd.testing.assert_frame_equal(scores[['orden_registro','anio','particion']],data[['orden_registro','anio','particion']])
        train = data.particion.eq('entrenamiento_CV')
        np.testing.assert_allclose(data.loc[train,meta['sensores']].mean(),meta['escala']['media'])
        np.testing.assert_allclose(data.loc[train,meta['sensores']].std(ddof=0),meta['escala']['desviacion'])
        z = (data[meta['sensores']].to_numpy()-meta['escala']['media'])/meta['escala']['desviacion']
        np.testing.assert_allclose(z @ loads[['coef_PC1','coef_PC2']].to_numpy(),scores[['PC1','PC2']],atol=1e-10)
        correlation = np.corrcoef(z[train].T,scores.loc[train,['PC1','PC2']].to_numpy().T)[:9,9:]
        np.testing.assert_allclose(correlation,loads[['PC1','PC2']],atol=1e-10)
        self.assertAlmostEqual(loads.set_index('sensor').loc['CDP','PC1'],.991097,places=5)
        self.assertAlmostEqual(loads.set_index('sensor').loc['AT','PC2'],.916903,places=5)

    def test_groups_and_unusual_flags(self):
        scores,_,_,meta = view.evidence()
        train = scores.loc[scores.particion.eq('entrenamiento_CV')]
        self.assertEqual(set(scores.grupo),{1,2,3})
        self.assertTrue(pd.api.types.is_bool_dtype(scores.inusual))
        self.assertEqual(train.inusual.sum(),meta['isolation_forest']['n_inusuales_desarrollo'])
        self.assertLess(abs(train.inusual.mean()-.02),1/len(train))
        self.assertTrue(np.array_equal(scores.inusual.to_numpy(),scores.puntaje_anomalia.lt(0).to_numpy()))
        self.assertEqual(meta['kmeans']['muestra_silhouette'],2000)

    def test_views_filters_and_no_runtime_fitting(self):
        with patch.object(PCA,'fit',side_effect=AssertionError('Runtime fit')), \
             patch.object(KMeans,'fit',side_effect=AssertionError('Runtime fit')), \
             patch.object(IsolationForest,'fit',side_effect=AssertionError('Runtime fit')), \
             patch.object(StandardScaler,'fit',side_effect=AssertionError('Runtime fit')):
            self.assertIsNotNone(view.layout())
            for population in ['desarrollo','todos']:
                for target in ['CO','NOX']:
                    for color in ['anio','grupo','emision','inusual']:
                        result = view.render([2014,2015],population,color,target)
                        self.assertEqual(len(result),4)
                        self.assertEqual(sum(len(trace.x) for trace in result[1].data),2200)
                        self.assertEqual(set(view.selected([2014,2015],population).anio),{2014,2015})
            empty = view.render([],'desarrollo','anio','CO')
            self.assertFalse(empty[1].data)
        self.assertEqual(len(view.loadings_figure().data[0].z),9)
        self.assertEqual(len(view.variance_figure().data),2)
        self.assertEqual(eda.view_visibility('multivariado')[2],{'display':'none'})
        self.assertEqual(eda.view_content('multivariado')[-2],{})
        self.assertEqual(eda.view_content('multivariado')[-1],{'display':'none'})
        dependencies = app.server.test_client().get('/_dash-dependencies').json
        self.assertTrue(any('multivariate-content.children' in row['output'] for row in dependencies))

    def test_only_selected_analysis_is_rendered(self):
        def ids(component):
            if isinstance(component,(list,tuple)):
                return set().union(*(ids(child) for child in component))
            if not hasattr(component,'to_plotly_json'):
                return set()
            own = {component.id} if getattr(component,'id',None) else set()
            return own | ids(getattr(component,'children',None))

        panels = {'pca':'multivariate-pca-panel','kmeans':'multivariate-kmeans-panel',
                  'isolation':'multivariate-isolation-panel'}
        for method,panel_id in panels.items():
            for years in [[],[2011,2012,2013,2014,2015]]:
                panel,color_style = view.render_content(years,'desarrollo','anio','CO',method)
                identifiers = ids(panel)
                self.assertIn(panel_id,identifiers)
                self.assertEqual(identifiers & set(panels.values()),{panel_id})
                self.assertEqual(color_style,{} if method=='pca' else {'display':'none'})
        frame = view.selected(list(range(2011,2016)),'desarrollo')
        self.assertIn('grupo 2',view.group_interpretation(frame,'CO'))
        self.assertEqual(view.group_profiles(frame).data[0].z.shape,(9,3))
        histogram = view.isolation_distribution(frame)
        self.assertEqual(sum(histogram.data[0].y),27570)
        self.assertEqual(histogram.layout.shapes[0].x0,0)
        self.assertIn('no es una probabilidad de falla',str(view.isolation_panel(frame,'CO')))
