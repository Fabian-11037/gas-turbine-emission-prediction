"""Coherencia de la redaccion y de la guia con las evidencias publicadas."""
import json
from pathlib import Path
import unittest

from dashboard import model_view
from dashboard.kernel_content import ARCHITECTURE, CONTRIBUTION, SELECTION
from dashboard.models import registry

ROOT = Path(__file__).resolve().parents[1]


class EditorialTests(unittest.TestCase):
    def test_plain_model_explanations_cover_registry(self):
        self.assertEqual(set(model_view.MODEL_EXPLANATIONS), set(registry()))
        for explanation in model_view.MODEL_EXPLANATIONS.values():
            self.assertLess(len(explanation), 160)
        notebook = json.loads((ROOT/'book'/'gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        cells = {c['id']:''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'markdown'}
        self.assertIn(ARCHITECTURE, cells['kernel-intro'])
        self.assertIn(CONTRIBUTION.replace('### ', '##### '), cells['kernel-contribution'])
        self.assertIn(SELECTION.replace('### ', '##### '), cells['kernel-contribution'])
        intro = cells['7aa9d234']
        self.assertIn('planta eléctrica', intro)
        self.assertIn('monitoreo predictivo', intro)
        self.assertIn('36.733', intro)

    def test_presentation_guide_has_story_results_and_glossary(self):
        guide = (ROOT/'docs'/'guia_sustentacion.md').read_text(encoding='utf-8')
        for concept in ['Negocio de origen', 'EDA', '27.570', '7.411', '0,7491', '0,8242',
                        '0,5092', '0,0004', 'Spearman', 'PCA', 'KMeans', 'Wald', 'HAC',
                        'Holm', 'LOYO', 'Cross-fitting', 'MAE', 'MSE', 'RMSE', 'R²',
                        'SVR', 'Ridge', 'Extra Trees', 'hipótesis nula', 'No significancia']:
            self.assertIn(concept, guide)
        self.assertGreater(sum(line.startswith('- ') for line in guide.splitlines()), 100)
        self.assertNotIn('ET–SVR ajustado', guide)
        self.assertNotIn('ET–SVR ponderado', guide)


if __name__ == '__main__':
    unittest.main()
