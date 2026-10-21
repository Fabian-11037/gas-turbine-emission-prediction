"""Read-only EDA view of packaged sensor-distance evidence."""
from functools import lru_cache
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html
from . import figures as f
from . import partition_content as text
from .components import disclosure, graph, metric, notice, number, table
from .repository import DATA

LOCATION = DATA/'partition_similarity'
CONTROLS = {'original':'Entrenamientos originales', 'igualado':'Igual número por año'}


@lru_cache(maxsize=1)
def evidence():
    return pd.read_csv(LOCATION/'resumen.csv'), pd.read_csv(LOCATION/'distancias.csv.gz')


def principal():
    results, _ = evidence()
    return results.loc[results.semilla.eq(42)&results.longitud_segmento.eq(168)].copy()


def result_text():
    rows = principal().set_index('control')
    original, matched = rows.loc['original'], rows.loc['igualado']
    return (f'Con bloques, la distancia media aumenta {number(original.aumento_pct,1)} % frente al '
        f'80/20 aleatorio. Al igualar el número de registros de entrenamiento por año, aumenta '
        f'{number(matched.aumento_pct,1)} %. Las dos diferencias son significativas '
        '(p ajustado ≈ 0,0001).')


def distance_table():
    frame = principal()
    frame['comparacion'] = frame.control.map(CONTROLS)
    frame['IC95'] = frame.apply(lambda r:f'[{number(r.IC95_delta_inf,4)}; {number(r.IC95_delta_sup,4)}]', axis=1)
    frame['p_ajustado'] = frame.p_Holm2_semilla42.map(lambda p:number(p,4))
    return frame[['comparacion', 'distancia_aleatoria', 'distancia_bloques', 'delta_distancia', 'IC95', 'p_ajustado']]


def cdf_figure():
    _, queries = evidence()
    fig = go.Figure()
    for name, label, color in [('aleatoria','80/20 aleatorio',f.TEAL), ('bloques','Por bloques',f.CORAL)]:
        x = np.sort(queries['d_'+name])
        fig.add_trace(go.Scatter(x=x, y=np.arange(1,len(x)+1)/len(x), name=label, mode='lines',
            line={'color':color, 'width':2}, hovertemplate='Distancia: %{x:.3f}<br>Proporción: %{y:.1%}<extra>%{fullData.name}</extra>'))
    f.style(fig, 'Similitud con el entrenamiento', 'Distancia estandarizada al vecino más cercano', 'Proporción acumulada')
    fig.update_layout(title={'font':{'size':15}}, margin={'l':55,'r':20,'t':82,'b':60},
        legend={'orientation':'h','y':1.02,'x':0,'yanchor':'bottom'})
    fig.update_xaxes(range=[0, float(queries[['d_aleatoria','d_bloques']].max().max())*1.02])
    fig.update_yaxes(tickformat='.0%')
    return fig


def difference_figure():
    rows, _ = evidence()
    fig = go.Figure()
    for control, color, offset in [('original',f.CORAL,-.12), ('igualado',f.TEAL,.12)]:
        group = rows.loc[rows.control.eq(control)&rows.longitud_segmento.eq(168)]
        fig.add_trace(go.Scatter(x=group.delta_distancia, y=np.arange(len(group))+offset,
            customdata=group.semilla, mode='markers',
            name=CONTROLS[control], marker={'size':8,'color':color},
            error_x={'type':'data','symmetric':False,'array':group.IC95_delta_sup-group.delta_distancia,
                     'arrayminus':group.delta_distancia-group.IC95_delta_inf,'color':color},
            hovertemplate='Semilla %{customdata}<br>Diferencia: %{x:.4f}<extra>%{fullData.name}</extra>'))
    f.style(fig, 'Menor similitud con bloques', 'Diferencia de distancia: bloques − aleatoria', 'Semilla de partición')
    fig.update_layout(title={'font':{'size':15}}, margin={'l':55,'r':20,'t':94,'b':60},
        legend={'orientation':'h','y':1.02,'x':0,'yanchor':'bottom','font':{'size':10}})
    fig.update_yaxes(tickmode='array', tickvals=list(range(5)), ticktext=['42','7','21','84','123'], autorange='reversed')
    fig.add_vline(x=0, line_dash='dash', line_color=f.GRAY)
    return fig


def layout():
    rows, _ = evidence()
    data = principal().set_index('control')
    sensitivity = rows[['semilla', 'control', 'longitud_segmento', 'delta_distancia', 'IC95_delta_inf', 'IC95_delta_sup']].copy()
    sensitivity.control = sensitivity.control.map(CONTROLS)
    metadata = json.loads((LOCATION/'metodologia.json').read_text(encoding='utf-8'))
    return html.Section([
        html.H3('Qué pregunta responde cada partición'), html.P(text.INTRO),
        html.Div([
            html.Section([html.H4('80/20 aleatorio'), html.P(text.RANDOM_QUESTION), html.P(text.RANDOM_DESCRIPTION,className='caption')]),
            html.Section([html.H4('Por bloques'), html.P(text.BLOCK_QUESTION), html.P(text.BLOCK_DESCRIPTION,className='caption')])
        ],className='two-columns context-columns'),
        html.H3('Evidencia de similitud entre sensores'),
        html.Div([metric('Distancia con bloques', '+'+number(data.loc['original'].aumento_pct,1)+' %', 'entrenamientos originales'),
                  metric('Con igual tamaño por año', '+'+number(data.loc['igualado'].aumento_pct,1)+' %', 'control de cantidad de datos'),
                  metric('Registros de prueba comunes', '1.540', 'mismos registros en ambos diseños')],className='metric-strip compact'),
        html.P(result_text()),
        html.Div([graph('partition-similarity-cdf', cdf_figure()), graph('partition-similarity-difference', difference_figure())],className='chart-grid'),
        html.P('Una curva más a la derecha indica menor similitud. En la segunda figura, cada punto es una diferencia '
            'media y la barra su intervalo del 95 %; quedar a la derecha de cero favorece una mayor separación por bloques.',className='caption'),
        notice(text.CONCLUSION),
        disclosure('Resultados e intervalos de confianza', [table(distance_table(),
            {'comparacion':'Comparación', 'distancia_aleatoria':'Distancia aleatoria', 'distancia_bloques':'Distancia por bloques',
             'delta_distancia':'Diferencia media', 'IC95':'IC 95 % de la diferencia', 'p_ajustado':'p ajustado'}, digits=4),
             html.P(text.ROBUSTNESS,className='caption'), table(sensitivity,
             {'semilla':'Semilla','control':'Comparación','longitud_segmento':'Segmento de remuestreo',
              'delta_distancia':'Diferencia media','IC95_delta_inf':'IC 95 % inferior','IC95_delta_sup':'IC 95 % superior'}, digits=4)]),
        disclosure('Cómo se hizo la comparación', [html.P(text.METHOD), html.P(text.UNCERTAINTY),
            html.P('Sensores: '+', '.join(metadata['sensores'])+'. El análisis es común a CO y NOx. '
                'Se conserva el protocolo por bloques del proyecto; las particiones adicionales solo se usan en este diagnóstico.',className='caption')])
    ])
