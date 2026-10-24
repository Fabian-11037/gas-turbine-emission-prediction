"""Executable book cells and readable sensor meanings without changing model evidence."""
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR

from dashboard import explainability, figures, multivariate
from dashboard.kernel_code import RUN_CODE, training_source
from dashboard.kernel_content import MATHEMATICS
from dashboard.repository import observations, sensors
from dashboard.sensor_labels import sensor_label

ROOT = Path(__file__).resolve().parents[1]


class TrainingPresentationTests(unittest.TestCase):
    def test_book_source_executes_from_root_and_book_without_training(self):
        notebook = json.loads((ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        cells = {c['id']: c for c in notebook['cells']}
        source = ''.join(cells['kernel-training-code']['source'])
        self.assertTrue(source.startswith(training_source()))
        self.assertEqual(''.join(cells['kernel-training-run']['source']), RUN_CODE)
        previous = Path.cwd()
        try:
            for cwd in [ROOT, ROOT/'book']:
                os.chdir(cwd)
                namespace = {'__name__': '__main__'}
                with patch.object(ExtraTreesRegressor, 'fit', side_effect=AssertionError('Unexpected training')), \
                     patch.object(SVR, 'fit', side_effect=AssertionError('Unexpected training')), redirect_stdout(StringIO()):
                    exec(compile(source, 'kernel-training-code', 'exec'), namespace)
                    exec(compile(RUN_CODE, 'kernel-training-run', 'exec'), namespace)
                self.assertEqual(namespace['ROOT'], ROOT)
                self.assertEqual(len(namespace['candidates']()), 24)
                self.assertEqual(len(namespace['estimators']()), 6)
                self.assertEqual(len(namespace['datos_kernel']), 36733)
                self.assertFalse(namespace['EJECUTAR_ENTRENAMIENTO'])
                train, test, _, assignment, _ = namespace['particiones'](namespace['datos_kernel'], 168, 24)
                data = namespace['datos_kernel']
                np.testing.assert_array_equal(train, np.flatnonzero(data.particion.eq('entrenamiento_CV')))
                np.testing.assert_array_equal(test, np.flatnonzero(data.particion.eq('prueba_bloques')))
                np.testing.assert_array_equal(assignment.bloque, data.bloque)
        finally:
            os.chdir(previous)

    def test_extra_trees_formulas_and_configuration(self):
        for phrase in ['H(t)', '\\Delta H', 'T_b(x)', 'B=200', '75 %', 'mínimo 2', 'sin límite']:
            self.assertIn(phrase, MATHEMATICS)

    def test_sensor_meanings_in_interactive_figures(self):
        data = observations().query("particion == 'entrenamiento_CV'")
        features = [s['id'] for s in sensors()]
        for fig in [figures.associations(data, 'CO', features), figures.correlation(data, features),
                    multivariate.loadings_figure(), multivariate.group_profiles(multivariate.selected([2011], 'desarrollo')),
                    explainability.global_importance('kernel_operativo', 'CO'),
                    explainability.global_summary('kernel_operativo', 'CO'),
                    explainability.waterfall('kernel_operativo', 'CO', 'error_alto'),
                    explainability.lime_figure('xgboost', 'NOX', 'error_bajo')]:
            labels = list(fig.layout.yaxis.ticktext)
            for sensor in features:
                self.assertIn(sensor_label(sensor, width=24, html=True), labels)
            self.assertGreaterEqual(fig.layout.height, 440)
        relation = figures.relationship(data, 'CO', 'TEY')
        self.assertIn('rendimiento energético', relation.layout.xaxis.title.text)


if __name__ == '__main__':
    unittest.main()
