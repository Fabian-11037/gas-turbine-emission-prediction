"""Insert a saved-prediction diagnostic without executing previous notebook cells."""
import base64
import copy
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'book/training'))
import matplotlib
matplotlib.use('Agg')
from dashboard.underestimation import METHOD, conclusion, summary
from underestimation_plots import observed_figure

CODE = '''from pathlib import Path
import sys
from IPython.display import display, Markdown
_rutas_diagnostico = [globals().get('REPO_KERNEL', Path.cwd()), Path.cwd(), Path.cwd().parent]
REPO_DIAGNOSTICO = next(p for p in _rutas_diagnostico if (p/'data/stability_ablation').exists())
sys.path.insert(0, str(REPO_DIAGNOSTICO))
sys.path.insert(0, str(REPO_DIAGNOSTICO/'book/training'))
from dashboard.underestimation import summary, conclusion
from underestimation_plots import observed_figure
objetivo_diagnostico = {target!r}
display(observed_figure(objetivo_diagnostico))
display(Markdown(conclusion(objetivo_diagnostico)))
display(summary(objetivo_diagnostico).rename(columns={{'conjunto':'Conjunto', 'umbral':'Umbral (mg/m³)',
    'n_altos':'Registros sobre el umbral', 'porcentaje_subestimado':'Casos subestimados (%)',
    'diferencia_media':'Diferencia media: real − estimado (mg/m³)'}}).round(2))
'''


def main():
    path = ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    prior = [copy.deepcopy(c) for c in notebook['cells'] if c['cell_type']=='code' and not c.get('id','').startswith('underestimation-')]
    notebook['cells'] = [c for c in notebook['cells'] if not c.get('id','').startswith('underestimation-')]
    new = [dict(cell_type='markdown', id='underestimation-method', metadata={},
        source=('### 4.15 Subestimación en concentraciones altas\n\n'+METHOD).splitlines(keepends=True))]
    count = max(c.get('execution_count') or 0 for c in prior)
    for i, target in enumerate(['CO', 'NOX'], 1):
        new.append(dict(cell_type='markdown', id='underestimation-heading-'+target, metadata={},
            source=[f'#### 4.15.{i} '+('NOx' if target=='NOX' else target)]))
        fig = observed_figure(target)
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=145, bbox_inches='tight')
        text = conclusion(target)
        table = summary(target).rename(columns={'conjunto':'Conjunto', 'umbral':'Umbral (mg/m³)',
            'n_altos':'Registros sobre el umbral', 'porcentaje_subestimado':'Casos subestimados (%)',
            'diferencia_media':'Diferencia media: real − estimado (mg/m³)'}).round(2)
        outputs = [dict(output_type='display_data', metadata={}, data={'image/png':base64.b64encode(buffer.getvalue()).decode('ascii'),
                    'text/plain':'<Figure observed versus estimated>'}),
            dict(output_type='display_data', metadata={}, data={'text/markdown':text, 'text/plain':text}),
            dict(output_type='display_data', metadata={}, data={'text/html':table.to_html(index=False), 'text/plain':table.to_string(index=False)})]
        new.append(dict(cell_type='code', id='underestimation-results-'+target, metadata={'saved_predictions_only':True},
            source=CODE.format(target=target).splitlines(keepends=True), execution_count=count+i, outputs=outputs))
    insertion = next(i for i, c in enumerate(notebook['cells']) if ''.join(c['source']).startswith('## 5.'))
    notebook['cells'][insertion:insertion] = new
    after = [c for c in notebook['cells'] if c['cell_type']=='code' and not c.get('id','').startswith('underestimation-')]
    assert after == prior, 'Previous code or outputs changed.'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print('Sección 4.15 añadida; celdas previas y modelos intactos.')


if __name__ == '__main__':
    main()
