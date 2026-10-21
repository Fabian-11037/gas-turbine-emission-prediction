"""Published TEY evidence, statistical interpretation and read-only views."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy import stats
from dashboard import tey_ablation as view, tey_content as text, stability
from dashboard.models import registry
from tools.export_kernel_significance import holm

ROOT = Path(__file__).resolve().parents[1]


def walk(component):
    yield component
    children = getattr(component, 'children', None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None:
            yield from walk(child)


class TEYEvidenceTests(unittest.TestCase):
    def test_provenance_hashes_and_unchanged_model_scope(self):
        meta = json.loads((view.LOCATION/'metodologia.json').read_text(encoding='utf-8'))
        self.assertEqual((meta['n_desarrollo'], meta['n_prueba']), (27570, 7411))
        self.assertEqual((meta['bloque'], meta['purga'], meta['crossfit']), (168, 24, 3))
        self.assertEqual(meta['retiradas'], ['TEY', 'TEY*(TIT-TAT)'])
        self.assertFalse(meta['prueba_usada_en_seleccion'])
        self.assertFalse(meta['modelos_publicados_sustituidos'])
        self.assertEqual(meta['modelos_reentrenados_al_incorporar'], 0)
        self.assertEqual(len(registry()), 8)
        self.assertNotIn('C:\\Users', json.dumps(meta))
        for name, digest in meta['archivos'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), digest)

    def test_original_rows_and_metric_values(self):
        results, _, _ = view.evidence()
        assignment = pd.read_csv(ROOT/'book/informe_resultados/entregable2_anidada/asignacion_particiones.csv.gz')
        dev = assignment.loc[assignment.particion.eq('entrenamiento_CV')].reset_index(drop=True)
        for target in ['CO', 'NOX']:
            row = results.loc[results.objetivo.eq(target)&~results.modelo.eq('Kernel completo')].iloc[0]
            for phase, n in [('OOF', 27570), ('prueba', 7411)]:
                new = pd.read_csv(ROOT/f'book/informe_resultados/tey_ablation/sin_TEY_{phase}_{target}.csv.gz')
                old = pd.read_csv(ROOT/f'book/informe_resultados/kernel_operativo/predicciones_{phase}_{target}.csv.gz')
                pd.testing.assert_frame_equal(new[['anio', 'orden_registro', 'bloque', target]],
                                            old[['anio', 'orden_registro', 'bloque', target]])
                self.assertEqual(len(new), n)
                y = new[target].to_numpy()
                r2 = 1-((y-new.prediccion)**2).sum()/((y-y.mean())**2).sum()
                self.assertAlmostEqual(r2, row['R2_OOF' if phase == 'OOF' else 'R2_prueba'])
                if phase == 'OOF':
                    scores = []
                    for k in range(1, 6):
                        frame = new.loc[dev.fold_validacion.eq(k)]
                        y = frame[target]
                        scores.append(1-((y-frame.prediccion)**2).sum()/((y-y.mean())**2).sum())
                    self.assertAlmostEqual(np.mean(scores), row.R2_CV)

    def test_significance_and_direction_are_consistent(self):
        results, cv, tests = view.evidence()
        details = pd.read_csv(view.LOCATION/'diferencias_por_fold.csv')
        np.testing.assert_allclose(cv.p_Holm, holm(cv.p_CV_nominal))
        for row in cv.itertuples():
            group = details.loc[details.objetivo.eq(row.objetivo)]
            delta = group.delta_MSE.to_numpy()
            se = np.sqrt((1/5+(group.n_val/group.n_train).mean())*delta.var(ddof=1))
            self.assertAlmostEqual(2*stats.t.sf(abs(delta.mean()/se), 4), row.p_CV_nominal)
        for _, group in tests.groupby('longitud_bloque'):
            np.testing.assert_allclose(group.p_Holm, holm(group.p_prueba_nominal))
        for target in ['CO', 'NOX']:
            rows = results.loc[results.objetivo.eq(target)]
            delta = rows.loc[~rows.modelo.eq('Kernel completo'), 'R2_prueba'].iloc[0]-rows.loc[rows.modelo.eq('Kernel completo'), 'R2_prueba'].iloc[0]
            np.testing.assert_allclose(tests.loc[tests.objetivo.eq(target), 'delta_R2'], delta)
        self.assertTrue((tests.loc[tests.objetivo.eq('NOX'), 'p_Holm'] < .05).all())
        self.assertTrue((tests.loc[tests.objetivo.eq('CO'), 'p_Holm'] > .05).all())

    def test_views_share_clear_language_and_only_show_for_kernel(self):
        self.assertIn('después del momento de predicción', text.AVAILABILITY)
        self.assertIn('calculara a partir de las emisiones objetivo', text.AVAILABILITY)
        self.assertIn('sin usar CO ni NOx como entradas', text.AVAILABILITY)
        self.assertIn('no presupone disponibilidad instantánea', text.AVAILABILITY)
        self.assertIn('estimación contemporánea', text.AVAILABILITY)
        self.assertIn('Para sustentar su permanencia', text.DESIGN)
        self.assertIn('no son significativas', text.CONCLUSIONS['CO'])
        self.assertIn('significativa en ambas', text.CONCLUSIONS['NOX'])
        for target in ['CO', 'NOX']:
            combined = view.comparison_table(target)
            self.assertEqual(len(combined), 2)
            self.assertEqual(combined['Evaluación'].tolist(), ['CV','Prueba'])
            np.testing.assert_allclose(combined['Aporte ΔR²'],
                                       combined['R² completo']-combined['R² sin TEY e interacción'])
            panel = stability.view('kernel_operativo', target)
            self.assertEqual(sum(getattr(c, 'id', None) == 'tey-ablation-analysis' for c in walk(panel)), 1)
            section = next(c for c in walk(panel) if getattr(c, 'id', None) == 'tey-ablation-analysis')
            self.assertEqual(sum(type(c).__name__ == 'Table' for c in walk(section)), 1)
            self.assertFalse(any(type(c).__name__ == 'Details' for c in walk(section)))
            base = stability.view('ridge', target)
            self.assertFalse(any(getattr(c, 'id', None) == 'tey-ablation-analysis' for c in walk(base)))

    def test_book_has_saved_tables_and_consistent_interpretation(self):
        notebook = json.loads((ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        cells = {c['id']:c for c in notebook['cells']}
        method = ''.join(cells['tey-ablation-method']['source'])
        conclusion = ''.join(cells['tey-ablation-conclusion']['source'])
        self.assertIn(text.AVAILABILITY, method)
        self.assertIn(text.DESIGN, method)
        for value in text.CONCLUSIONS.values():
            self.assertIn(value, conclusion)
        self.assertIn(text.READING, conclusion)
        code = cells['tey-ablation-results']
        self.assertNotIn('.fit(', ''.join(code['source']))
        self.assertEqual(len(code['outputs']), 1)
        self.assertIn('p ajustado · Holm', code['outputs'][0]['data']['text/html'])
        self.assertTrue(all('text/html' in o['data'] for o in code['outputs']))
        ids = [c['id'] for c in notebook['cells']]
        self.assertLess(ids.index('stability-results-NOX'), ids.index('tey-ablation-method'))


if __name__ == '__main__':
    unittest.main()
