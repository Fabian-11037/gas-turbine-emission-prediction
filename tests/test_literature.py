"""Consistencia editorial entre referencias, protocolo y presentacion."""
import json
from pathlib import Path
import unittest
from urllib.parse import urlparse

from dashboard.context import layout
from dashboard.repository import DATA, results

ROOT = Path(__file__).resolve().parents[1]


class LiteratureTest(unittest.TestCase):
    def test_study_sources_and_required_fields(self):
        studies = json.loads((DATA / 'literatura.json').read_text(encoding='utf-8'))
        self.assertEqual(len(studies), 9)
        self.assertEqual(len({s['estudio'] for s in studies}), len(studies))
        for study in studies:
            self.assertEqual(urlparse(study['enlace']).scheme, 'https')
            self.assertTrue(all(study[field] for field in ['estudio', 'modelos', 'resultado', 'validacion']))
        sinap = next(s for s in studies if s['estudio'] == 'Sinap (2025)')
        for value in ['Raw', 'IQR', '0,8026', '0,8942', '0,8556', '0,9194']:
            self.assertIn(value, sinap['resultado'])
        kochueva = next(s for s in studies if s['estudio'].startswith('Kochueva'))
        for value in ['2011–2013', '2014–2015', '30%']:
            self.assertIn(value, kochueva['validacion'])

    def test_methodology_caveats_and_dimensions(self):
        comparison = json.loads((DATA / 'literatura_metodologia.json').read_text(encoding='utf-8'))
        self.assertEqual(len(comparison['dimensiones']), 4)
        self.assertEqual(comparison['revision'], '2026-10-20')
        text = json.dumps(comparison, ensure_ascii=False)
        for phrase in ['168 registros', '24 vecinos', '5 × 3',
                       'nueve sensores', 'posibles explicaciones', 'mismos datos']:
            self.assertIn(phrase, text)
        for item in comparison['dimensiones']:
            self.assertEqual(urlparse(item['enlace']).scheme, 'https')

    def test_context_disclosures_and_reference(self):
        component = layout()
        found = {}
        def visit(node):
            if isinstance(node, (list, tuple)):
                for child in node:
                    visit(child)
            elif hasattr(node, 'to_plotly_json'):
                props = node.to_plotly_json()['props']
                if props.get('id') in ['literature-methods', 'literature-studies']:
                    found[props['id']] = node
                visit(props.get('children'))
        visit(component)
        self.assertEqual(set(found), {'literature-methods', 'literature-studies'})
        self.assertEqual(list(found), ['literature-studies', 'literature-methods'])
        self.assertFalse(found['literature-methods'].open)
        table = found['literature-methods'].children[1].children[1]
        self.assertEqual(len(table.children.children[1].children), 2)
        studies = found['literature-studies'].children[1].children[0]
        self.assertEqual(len(studies.children.children[1].children), 9)
        selected = results().loc[lambda d:d.seleccionado_por_CV]
        self.assertEqual(set(selected.modelo), {'Kernel operativo'})

    @unittest.skipUnless((ROOT / 'book' / 'gas_turbine_eda_hilo_conductor_redaccion.ipynb').exists(),
                         'El despliegue minimo no incluye el JBook')
    def test_notebook_uses_same_explanations(self):
        notebook = json.loads((ROOT / 'book' / 'gas_turbine_eda_hilo_conductor_redaccion.ipynb').read_text(encoding='utf-8'))
        text = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'markdown')
        comparison = json.loads((DATA / 'literatura_metodologia.json').read_text(encoding='utf-8'))
        for item in comparison['dimensiones']:
            self.assertIn(item['titulo'], text)
            self.assertIn(item['diferencia'], text)
            self.assertIn(item['interpretacion'], text)
        self.assertIn('### 8.4 Comparabilidad', text)
        self.assertIn('validación anidada con cinco pliegues externos y tres internos', text)
        self.assertIn('0,9194', text)
        self.assertNotIn('Subconjunto y orden de validación de esas cifras pendientes', text)


if __name__ == '__main__':
    unittest.main()
