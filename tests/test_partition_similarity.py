"""Evidence, editorial consistency, and isolated EDA view."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from dashboard import eda, partition_content as text, partition_similarity as view

ROOT = Path(__file__).resolve().parents[1]


class PartitionSimilarityTests(unittest.TestCase):
    def test_bundle_matches_provenance_and_primary_statistics(self):
        metadata = json.loads((view.LOCATION/'metodologia.json').read_text(encoding='utf-8'))
        for name, expected in metadata['archivos'].items():
            self.assertEqual(hashlib.sha256((view.LOCATION/name).read_bytes()).hexdigest(), expected)
        self.assertFalse(metadata['objetivos_usados'])
        self.assertEqual(metadata['modelos_entrenados'], 0)
        results, queries = view.evidence()
        self.assertEqual(len(results), 30)
        self.assertEqual(len(queries), 1540)
        original = view.principal().set_index('control').loc['original']
        self.assertAlmostEqual(original.distancia_aleatoria, queries.d_aleatoria.mean())
        self.assertAlmostEqual(original.distancia_bloques, queries.d_bloques.mean())
        self.assertAlmostEqual(original.aumento_pct, 61.405619, places=5)
        matched = view.principal().set_index('control').loc['igualado']
        self.assertEqual(matched.n_train_aleatorio, matched.n_train_bloques)
        self.assertTrue((results.IC95_delta_inf > 0).all())
        self.assertTrue((view.principal().p_Holm2_semilla42 < .001).all())

    def test_figures_and_filter_visibility(self):
        cdf, differences = view.cdf_figure(), view.difference_figure()
        self.assertEqual(len(cdf.data), 2)
        for trace in cdf.data:
            self.assertEqual(len(trace.x), 1540)
            self.assertTrue(np.all(np.diff(trace.x) >= 0))
            self.assertEqual(trace.y[-1], 1.)
        self.assertEqual(len(differences.data), 2)
        self.assertEqual(len(differences.data[0].error_x.array), 5)
        self.assertEqual(eda.view_visibility('particiones')[0], {'display':'none'})
        self.assertEqual(eda.view_visibility('particiones')[2], {'display':'none'})
        self.assertEqual(eda.view_visibility('distribuciones')[3], {'display':'none'})
        self.assertEqual(len(eda.render('CO', [], 'desarrollo', 'particiones', 'AT')), 7)
        self.assertIsNotNone(view.layout())

    def test_book_and_dashboard_share_questions_and_conclusions(self):
        notebook = json.loads((ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        cells = {c['id']:c for c in notebook['cells']}
        method = ''.join(cells['partition-similarity-method']['source'])
        conclusion = ''.join(cells['partition-similarity-conclusion']['source'])
        for content in [text.INTRO, text.RANDOM_QUESTION, text.BLOCK_QUESTION, text.METHOD, text.UNCERTAINTY, view.result_text()]:
            self.assertIn(content, method)
        self.assertIn(text.CONCLUSION, conclusion)
        self.assertEqual(len(cells['partition-similarity-results']['outputs']), 2)
        section = next(i for i,c in enumerate(notebook['cells']) if c['id']=='partition-similarity-method')
        training = next(i for i,c in enumerate(notebook['cells']) if ''.join(c['source']).startswith('## 4.'))
        self.assertLess(section, training)
        self.assertNotIn('evidencia empírica más directa', method+conclusion)


if __name__ == '__main__':
    unittest.main()
