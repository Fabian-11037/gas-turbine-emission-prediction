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
        self.assertEqual(len(metadata['fuentes']), len(EXPERIMENTS)+2)
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
        kernel = frame.loc[frame.modelo.eq('Kernel operativo') & frame.estado.eq('Dashboard')]
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
            kernel = rows.loc[rows.modelo.eq('Kernel operativo') & rows.estado.eq('Dashboard')].iloc[0]
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

    def test_recent_protocols_are_separate_and_missing_scores_not_invented(self):
        rows = experiment_history.results()
        self.assertEqual(set(rows.grupo_validacion), set(experiment_history.PROTOCOLS))
        kernel = rows.loc[rows.modelo.eq('Kernel operativo')]
        self.assertEqual(set(kernel.grupo_validacion), {'bloques','aleatorio','estratificado'})
        reference = kernel.loc[kernel.grupo_validacion.eq('aleatorio')].iloc[0]
        self.assertAlmostEqual(reference.R2_test_CO, .7969468894058673)
        self.assertAlmostEqual(reference.R2_CV_promedio_NOX, .8836527015391826)
        controlled = rows.loc[rows.grupo_validacion.eq('comun_aleatoria')]
        self.assertEqual(len(controlled), 1)
        self.assertAlmostEqual(controlled.R2_test_CO.iloc[0], .7665214953202879)
        self.assertTrue(controlled.R2_train_CO.isna().all())
        no_tey = rows.loc[rows.modelo.eq('Kernel sin TEY ni su interacción')].iloc[0]
        self.assertAlmostEqual(no_tey.R2_CV_promedio_NOX, .7698201975183518)
        self.assertTrue(pd.isna(no_tey.R2_train_NOX))
        paused = rows.loc[rows.estado.eq('Pausado')]
        self.assertEqual(len(paused), 1)
        self.assertTrue(paused[['R2_train_CO','R2_train_NOX','R2_CV_promedio_NOX','R2_test_NOX']].isna().all().all())
        cat = rows.loc[rows.modelo.eq('Kernel operativo + CatBoost → Ridge') & rows.grupo_validacion.eq('aleatorio')].iloc[0]
        self.assertAlmostEqual(cat.R2_train_NOX, .9660981931830903)
        self.assertAlmostEqual(cat.R2_test_NOX, .88983970805914)
        self.assertIn('80/20 aleatorio', str(experiment_history.render('CatBoost','NOX','aleatorio')))
        self.assertIn('No hay coincidencias', str(experiment_history.render('CatBoost','NOX','comun_aleatoria')))
