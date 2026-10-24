"""Synchronize executable kernel cells and relabel saved figures without fitting models."""
import base64
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'book/training'))
from dashboard.kernel_code import INTRODUCTION, RUN_CODE, training_source
from dashboard.kernel_content import MATHEMATICS
from dashboard.sensor_labels import sensor_label
from explanation_plots import global_figure, local_figure


def image_output(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def replace_images(cell, figures):
    outputs = [o for o in cell['outputs'] if 'image/png' in o.get('data', {})]
    assert len(outputs) == len(figures), (cell['id'], len(outputs), len(figures))
    for output, fig in zip(outputs, figures):
        output['data']['image/png'] = image_output(fig)
        output['metadata'] = {}


def code_cell(identifier, source, count, namespace):
    output = StringIO()
    with redirect_stdout(output):
        exec(compile(source, identifier, 'exec'), namespace)
    return dict(cell_type='code', id=identifier, metadata={}, source=source.splitlines(keepends=True),
        execution_count=count, outputs=[dict(output_type='stream', name='stdout', text=output.getvalue().splitlines(keepends=True))])


def update():
    path = ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    cells = {c['id']: c for c in notebook['cells']}
    cells['kernel-math']['source'] = ('#### 4.10.1 Fundamentos matemáticos\n\n'+
        MATHEMATICS.replace('### ', '##### ')).splitlines(keepends=True)
    new_ids = {'kernel-training-text', 'kernel-training-code', 'kernel-training-run'}
    notebook['cells'] = [c for c in notebook['cells'] if c['id'] not in new_ids]
    count = max(c.get('execution_count') or 0 for c in notebook['cells'])
    namespace = {'__name__': '__main__'}
    training = training_source()+'\nprint("Pipeline definido: 24 configuraciones, 5 pliegues externos y cross-fitting de 3 pliegues.")\n'
    new_cells = [dict(cell_type='markdown', id='kernel-training-text', metadata={}, source=(
        '##### Pipeline de entrenamiento en Python\n\n'+INTRODUCTION+'\n\n'
        'Las dos celdas siguientes son ejecutables desde la raíz del repositorio o desde `book/`. '
        'Utilizan `data/observaciones.csv.gz` y los módulos incluidos en `book/training` y `emission_models`. '
        'El entorno requiere las dependencias del proyecto, declaradas en `requirements.txt`. '
        'La primera celda define el pipeline; en la segunda, cambiar `EJECUTAR_ENTRENAMIENTO` a `True` '
        'inicia el ajuste completo de CO y NOx. Los modelos y las predicciones OOF se guardan en una carpeta nueva.\n'
        ).splitlines(keepends=True)),
        code_cell('kernel-training-code', training, count+1, namespace),
        code_cell('kernel-training-run', RUN_CODE, count+2, namespace)]
    insertion = next(i for i, c in enumerate(notebook['cells']) if c['id'] == 'kernel-math')+1
    notebook['cells'][insertion:insertion] = new_cells

    setup = ''.join(cells['224e720b']['source'])
    marker = '# Etiquetas compartidas con el dashboard.'
    if marker not in setup:
        setup += ('\n'+marker+'\n'
            '_rutas_graficas = [RAIZ, RAIZ.parent, RAIZ.parent/"CO_NOx_Dashboard"]\n'
            'REPO_GRAFICAS = next(p for p in _rutas_graficas if (p/"dashboard/sensor_labels.py").exists())\n'
            'sys.path.insert(0, str(REPO_GRAFICAS))\n'
            'from dashboard.sensor_labels import sensor_label\n')
    cells['224e720b']['source'] = setup.splitlines(keepends=True)

    replacements = {
        '41ca1848': [('figsize=(10,3)', 'figsize=(10,5.5)'), ('yticklabels=columnas', 'yticklabels=[sensor_label(v, width=28) for v in columnas]')],
        '7de3a74c': [("title=f'Distribución de {v}'", "title=f'Distribución de {sensor_label(v)}'"),
                     ("title=f'Dispersión de {v}'", "title=f'Dispersión de {sensor_label(v)}'")],
        'd0ef09bd': [('figsize=(16,6)', "figsize=(20,8),layout='constrained'"),
                     ('yticklabels=columnas', 'yticklabels=[sensor_label(v, width=28) for v in columnas]'),
                     ('figsize=(22,6)', 'figsize=(24,8)'), ('xlabel=v,ylabel=t', 'xlabel=sensor_label(v, width=20),ylabel=t')],
        '1567811a': [('figsize=(12,5)', 'figsize=(14,6)'),
                     ("ax.barh(tabla.sensor,tabla.R2_OOF,color=color)",
                      "ax.barh(tabla.sensor,tabla.R2_OOF,color=color)\n    ax.set_yticks(range(len(tabla)),labels=[sensor_label(v, width=28) for v in tabla.sensor],fontsize=8)")]
    }
    for identifier, pairs in replacements.items():
        source = ''.join(cells[identifier]['source'])
        for before, after in pairs:
            if after not in source:
                source = source.replace(before, after)
        cells[identifier]['source'] = source.splitlines(keepends=True)

    data = pd.read_csv(ROOT/'data/observaciones.csv.gz')
    df = data.loc[data.particion.eq('entrenamiento_CV')]
    predictors = ['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP']
    columns = predictors+['CO','NOX']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    # Refresh only drawing segments; numeric tables and fitted estimators are untouched.
    context = dict(df=df, PREDICTORES=predictors, OBJETIVOS=['CO','NOX'], columnas=columns,
        sensor_label=sensor_label, np=np, pd=pd, plt=plt,
        pearson=df[columns].corr(method='pearson'), spearman=df[columns].corr(method='spearman'))
    for identifier, start, end in [
        ('41ca1848', 'fig, ax =', 'dup ='),
        ('7de3a74c', 'fig, axes =', 'q1,q3 ='),
        ('d0ef09bd', 'fig,axes=', 'vif=[]')]:
        source = ''.join(cells[identifier]['source'])
        segment = source[source.index(start):source.index(end)]
        with plt.ioff():
            exec(segment.replace('plt.show()', ''), context)
        figures = [context['fig']]
        if identifier == 'd0ef09bd':
            extra = source[source.index('fig,axes=plt.subplots(2,9'):source.index('for t in OBJETIVOS:\n    top=')]
            exec(extra.replace('plt.show()', ''), context)
            figures.append(context['fig'])
        replace_images(cells[identifier], figures)

    audit = pd.read_csv(ROOT/'data/auditoria_predictiva_univariada.csv')
    fig, axes = plt.subplots(1, 2, figsize=(14,6), layout='constrained')
    for ax, target, color in zip(axes, ['CO','NOX'], ['#297c79','#a44b6f']):
        table = audit.loc[audit.objetivo.eq(target)].sort_values('R2_OOF')
        ax.barh(table.sensor, table.R2_OOF, color=color)
        ax.set_yticks(range(len(table)), labels=[sensor_label(v, width=28) for v in table.sensor], fontsize=8)
        ax.axvline(0, color='grey', lw=.8)
        ax.set(title=f'{target}: Ridge con un solo sensor', xlabel='R² OOF en desarrollo', ylabel='Sensor')
    replace_images(cells['1567811a'], [fig])

    annual = data.groupby('anio')[columns].mean()
    shifts = ((annual-df[columns].mean())/df[columns].std(ddof=1)).T
    fig, ax = plt.subplots(figsize=(10,6), layout='constrained')
    im = ax.imshow(shifts, vmin=-2, vmax=2, cmap='coolwarm', aspect='auto')
    ax.set(xticks=range(5), xticklabels=annual.index, yticks=range(len(columns)),
        yticklabels=[sensor_label(v, width=28) for v in columns],
        title='Cambio de medias respecto a 2011-2015 (SD de desarrollo)')
    for i in range(len(columns)):
        for j in range(5):
            ax.text(j, i, f'{shifts.iloc[i,j]:.2f}', ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, label='Desviaciones estándar; color saturado fuera de [-2,2]')
    temporal_images = [o for o in cells['3fc6405d']['outputs'] if 'image/png' in o.get('data', {})]
    assert len(temporal_images) == 2
    temporal_images[1]['data']['image/png'] = image_output(fig)

    for model in ['kernel_operativo','xgboost']:
        for target in ['CO','NOX']:
            replace_images(cells[f'xai-results-{model}-{target}'], [global_figure(model,target),
                local_figure(model,target,'error_bajo'),local_figure(model,target,'error_alto')])
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print('Celdas del pipeline ejecutadas sin entrenar; etiquetas y figuras sincronizadas.')


if __name__ == '__main__':
    update()
