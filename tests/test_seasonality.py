"""Evidencias estacionales, filtros y ausencia de calculo inferencial en Dash."""
import json
import unittest
from unittest.mock import patch

import numpy as np
from dashboard import eda, seasonality
from dashboard.repository import DATA, observations


class SeasonalityTests(unittest.TestCase):
    def test_fixed_tests_and_counts(self):
        rows, profiles = seasonality.evidence()
        self.assertEqual(len(rows), 80)
        self.assertEqual(set(rows.HAC_rezagos), {168, 336})
        self.assertEqual(set(rows.periodo), {24, 168})
        self.assertFalse(rows.duplicated(['poblacion','anio','objetivo','periodo','HAC_rezagos']).any())
        self.assertTrue(np.isfinite(rows[['p','p_Holm','amplitud','R2_parcial']]).all().all())
        self.assertTrue(rows.p_Holm.between(0, 1).all())
        self.assertTrue((rows.p_Holm >= rows.p).all())
        for population, count in [('todos', 36733), ('desarrollo', 27570)]:
            for target in ['CO','NOX']:
                for period in [24,168]:
                    selected = rows.loc[(rows.poblacion == population) & (rows.objetivo == target) &
                        (rows.periodo == period) & (rows.HAC_rezagos == 336)]
                    self.assertEqual(selected.n.sum(), count)
                    profile = profiles.loc[(profiles.poblacion == population) &
                        (profiles.objetivo == target) & (profiles.periodo == period)]
                    self.assertEqual(profile['count'].sum(), count)

    def test_filters_empty_and_both_periods(self):
        for target in ['CO','NOX']:
            for population in ['todos','desarrollo']:
                for period in [24,168]:
                    out = eda.render(target,[2014,2015],population,'estacionalidad','AT',period)
                    self.assertEqual(len(out),7)
                    self.assertEqual(len(out[0].data),2)
                    self.assertEqual(len(out[0].data[0].x),period)
                    self.assertEqual(len(out[1].data[0].y),2)
        self.assertFalse(eda.render('CO',[],'todos','estacionalidad','AT')[0].data)

    def test_no_fit_and_data_provenance(self):
        from sklearn.pipeline import Pipeline
        with patch.object(Pipeline,'fit',side_effect=AssertionError('No ajustar modelos.')):
            self.assertTrue(eda.render('CO',[2011],'desarrollo','estacionalidad','AT')[0].data)
        import hashlib
        metadata = json.loads((DATA/'estacionalidad_metodologia.json').read_text())
        self.assertFalse(metadata['fecha_calendario_disponible'])
        self.assertEqual(metadata['contrastes_Holm_por_poblacion'],40)
        self.assertEqual(metadata['datos_sha256'],hashlib.sha256((DATA/'observaciones.csv.gz').read_bytes()).hexdigest())
        self.assertEqual(len(observations()),36733)


if __name__ == '__main__':
    unittest.main()
