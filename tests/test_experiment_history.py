"""Portable evidence and read-only historical view contracts."""
import json
from pathlib import Path
import unittest

import pandas as pd

from dashboard import experiment_history, kernel_view
from dashboard.models import registry
from tools.export_experiment_history import EXPERIMENTS, consolidate

ROOT = Path(__file__).resolve().parents[1]


class HistoryTests(unittest.TestCase):
    def test_all_sources_and_missing_final_results(self):
        frame = experiment_history.results()
        metadata = json.loads((ROOT/'data/historial_modelos.json').read_text(encoding='utf-8'))
        self.assertEqual(len(metadata['fuentes']), len(EXPERIMENTS)+1)
        self.assertEqual(len(frame), metadata['n_variantes'])
        for folder in EXPERIMENTS:
            self.assertTrue(frame.fuentes.str.contains(folder, regex=False).any(), folder)
        bart = frame.loc[frame.modelo.eq('BART')]
        self.assertEqual(bart.estado.tolist(), ['Interrumpido'])
        self.assertTrue(bart[['R2_train_CO','R2_train_NOX']].isna().all().all())
        self.assertEqual(len(registry()), 8)

    def test_authoritative_kernel_and_cleaning_scores(self):
        frame = experiment_history.results()
        official = pd.read_csv(ROOT/'data/resultados.csv')
        kernel = frame.loc[frame.modelo.eq('Kernel operativo')]
        self.assertEqual(len(kernel), 1)
        for target in ['CO', 'NOX']:
            expected = official.loc[official.modelo.eq('Kernel operativo') & official.objetivo.eq(target), 'R2_train'].iloc[0]
            self.assertAlmostEqual(kernel[f'R2_train_{target}'].iloc[0], expected)
        cleaned = frame.loc[frame.modelo.eq('KNN') & frame.politica.eq('Limpieza de entrenamiento')]
        self.assertAlmostEqual(cleaned.R2_train_CO.iloc[0], 0.9566550533733676)

    def test_only_kernel_and_literal_search(self):
        self.assertEqual(kernel_view.render('ridge', 'CO', 'historial'), ({'display':'none'}, None))
        self.assertIsNotNone(kernel_view.render('kernel_operativo', 'CO', 'historial')[1])
        content = experiment_history.render('[').to_plotly_json()
        self.assertIn('No hay coincidencias.', str(content))
        for target in ['CO','NOX']:
            content = str(experiment_history.render('',target).to_plotly_json())
            self.assertIn('R² entrenamiento',content)
            self.assertIn('R² CV',content)
            self.assertIn('R² prueba',content)
            self.assertIn('Tipo de R² CV',content)
        for target in ['CO','NOX']:
            published = pd.read_csv(ROOT/'data/resultados.csv')
            rows = experiment_history.results()
            kernel = rows.loc[rows.modelo.eq('Kernel operativo')].iloc[0]
            expected = published.loc[published.objetivo.eq(target)&published.modelo.eq('Kernel operativo')].iloc[0]
            self.assertAlmostEqual(kernel[f'R2_CV_promedio_{target}'],expected.R2_CV_promedio)
            self.assertAlmostEqual(kernel[f'R2_test_{target}'],expected.R2_test)

    def test_export_reproducible_when_original_evidence_present(self):
        source = ROOT.parent/'Pruebas provisionales CO NOx'
        if not source.exists():
            self.skipTest('Source experiments are not needed for dashboard deployment')
        rebuilt, metadata = consolidate(source)
        stored = experiment_history.results()
        pd.testing.assert_frame_equal(stored.reset_index(drop=True), rebuilt.reset_index(drop=True), check_dtype=False)
        for folder in EXPERIMENTS:
            original = pd.read_csv(source/folder/'resultados_provisionales.csv')
            candidates = stored.loc[stored.fuentes.str.contains(folder, regex=False)]
            for _, row in original.iterrows():
                values = candidates[f'R2_train_{row.objetivo}'].dropna()
                self.assertTrue(any(abs(values-row.R2_train)<1e-12), (folder,row.modelo,row.objetivo))

    def test_cover_available_with_attribution(self):
        from PIL import Image
        with Image.open(ROOT/'assets/turbina_portada.jpg') as photo:
            self.assertGreater(photo.width, 1000)
        credits = (ROOT/'assets/IMAGE_CREDITS.md').read_text(encoding='utf-8')
        self.assertIn('Pro-Per Energy Services', credits)
        self.assertIn('by-sa/4.0', credits)
