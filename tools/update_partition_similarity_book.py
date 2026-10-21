"""Add a compact EDA section from completed evidence, preserving existing cells."""
import base64
import copy
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dashboard import partition_content as text
from dashboard.partition_similarity import evidence, distance_table, result_text, CONTROLS

CODE = '''from pathlib import Path
import sys
from IPython.display import display
_rutas_similitud = [globals().get('REPO_KERNEL', Path.cwd()), Path.cwd(), Path.cwd().parent]
REPO_SIMILITUD = next(p for p in _rutas_similitud if (p/'data/partition_similarity').exists())
sys.path.insert(0, str(REPO_SIMILITUD))
from tools.update_partition_similarity_book import figure
from dashboard.partition_similarity import distance_table
display(figure())
display(distance_table().rename(columns={'comparacion':'Comparación',
    'distancia_aleatoria':'Distancia aleatoria', 'distancia_bloques':'Distancia por bloques',
    'delta_distancia':'Diferencia media', 'IC95':'IC 95 % de la diferencia',
    'p_ajustado':'p ajustado'}).round(4))
'''


def figure():
    results, queries = evidence()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for name, label, color in [('aleatoria','80/20 aleatorio','#157a7c'), ('bloques','Por bloques','#b65346')]:
        values = np.sort(queries['d_'+name])
        axes[0].plot(values, np.arange(1,len(values)+1)/len(values), label=label, color=color, lw=2)
    axes[0].set(xlabel='Distancia estandarizada al vecino más cercano', ylabel='Proporción acumulada',
        title='Similitud con el entrenamiento')
    axes[0].legend(frameon=False)
    for control, color, offset in [('original','#b65346',-.12), ('igualado','#157a7c',.12)]:
        group = results.loc[results.control.eq(control)&results.longitud_segmento.eq(168)]
        axes[1].errorbar(group.delta_distancia, np.arange(5)+offset,
            xerr=np.vstack([group.delta_distancia-group.IC95_delta_inf, group.IC95_delta_sup-group.delta_distancia]),
            fmt='o', color=color, label=CONTROLS[control], capsize=3)
    axes[1].axvline(0, color='#8b949c', linestyle='--', lw=1)
    axes[1].set(yticks=range(5), yticklabels=['42','7','21','84','123'], ylabel='Semilla de partición',
        xlabel='Diferencia de distancia: bloques − aleatoria', title='Diferencia media e intervalo del 95 %')
    axes[1].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    return fig


def main():
    path = ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    prefix = 'partition-similarity-'
    prior = copy.deepcopy([c for c in notebook['cells'] if not c.get('id','').startswith(prefix)])
    notebook['cells'] = copy.deepcopy(prior)
    paragraphs = ['### 3.9 Partición y similitud entre entrenamiento y evaluación', text.INTRO,
        '**80/20 aleatorio.** '+text.RANDOM_QUESTION+' '+text.RANDOM_DESCRIPTION,
        '**Por bloques.** '+text.BLOCK_QUESTION+' '+text.BLOCK_DESCRIPTION,
        '#### Evidencia estadística', result_text(), text.METHOD, text.UNCERTAINTY,
        'La figura compara las distancias de los mismos registros de prueba y muestra la diferencia media '
        'con su intervalo del 95 % para cinco semillas. Una distancia mayor indica menor similitud.']
    count = max(c.get('execution_count') or 0 for c in prior if c['cell_type']=='code')+1
    fig = figure()
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=145)
    plt.close(fig)
    table = distance_table().rename(columns={'comparacion':'Comparación', 'distancia_aleatoria':'Distancia aleatoria',
        'distancia_bloques':'Distancia por bloques', 'delta_distancia':'Diferencia media',
        'IC95':'IC 95 % de la diferencia', 'p_ajustado':'p ajustado'}).round(4)
    cells = [dict(cell_type='markdown', id=prefix+'method', metadata={}, source='\n\n'.join(paragraphs).splitlines(keepends=True)),
        dict(cell_type='code', id=prefix+'results', metadata={'saved_evidence_only':True}, source=CODE.splitlines(keepends=True),
            execution_count=count, outputs=[dict(output_type='display_data', metadata={},
                data={'image/png':base64.b64encode(buffer.getvalue()).decode('ascii'), 'text/plain':'<Figure sensor-distance comparison>'}),
                dict(output_type='display_data', metadata={}, data={'text/html':table.to_html(index=False), 'text/plain':table.to_string(index=False)})]),
        dict(cell_type='markdown', id=prefix+'conclusion', metadata={},
            source=(text.ROBUSTNESS+'\n\n'+text.CONCLUSION+'\n\n'
            'El diagnóstico es común a CO y NOx porque ambos usan las mismas particiones y sensores. '
            'Se conserva el protocolo por bloques del proyecto; esta comparación no cambia los modelos ni sus resultados.').splitlines(keepends=True))]
    index = next(i for i,c in enumerate(notebook['cells']) if ''.join(c['source']).startswith('## 4.'))
    notebook['cells'][index:index] = cells
    assert [c for c in notebook['cells'] if not c.get('id','').startswith(prefix)] == prior
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print('Sección 3.9 añadida; todas las celdas previas y sus resultados se conservaron.')


if __name__ == '__main__':
    main()
