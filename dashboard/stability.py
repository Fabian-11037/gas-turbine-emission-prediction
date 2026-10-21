"""Read-only fixed-fold stability and ablation results."""
from functools import lru_cache
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from .repository import DATA
from .components import disclosure, table, graph, number
from .figures import style, TEAL, CORAL
from .stability_content import (METHOD,NAMES,SEED_HELP,ABLATION_HELP,ABLATION_SENSITIVITY_HELP,
                                seed_conclusion,ablation_conclusion)
from . import tey_ablation


@lru_cache(maxsize=4)
def read(name):
    return pd.read_csv(DATA/'stability_ablation'/(name+'.csv'))


def seed_figure(model_id,target):
    rows = read('resultados')
    rows = rows.loc[rows.modelo_id.eq(model_id)&rows.objetivo.eq(target)&rows.variante.eq('completo')&rows.fase.eq('OOF')]
    rows = rows.sort_values('semilla')
    fig = go.Figure(go.Scatter(x=rows.semilla.astype(str),y=rows.R2,mode='markers',
        marker={'size':10,'color':TEAL},hovertemplate='Semilla %{x}<br>R² OOF: %{y:.5f}<extra></extra>'))
    style(fig,'Sensibilidad a semillas · modelo seleccionado','Semilla del entrenamiento','R² OOF')
    fig.update_layout(height=310)
    fig.update_yaxes(range=[0,1])
    return fig


def ablation_table(target):
    rows = read('resultados')
    rows = rows.loc[rows.modelo_id.eq('kernel_operativo')&rows.objetivo.eq(target)&rows.semilla.eq(42)]
    wide = rows.pivot(index='variante',columns='fase',values='R2').reindex(NAMES)
    wide['Variante'] = [NAMES[v] for v in wide.index]
    return wide[['Variante','OOF','prueba']].reset_index(drop=True)


def view(model_id,target):
    if not (DATA/'stability_ablation/sensibilidad_semillas.csv').exists():
        return html.P('Validación de estabilidad en preparación.')
    summary = read('sensibilidad_semillas')
    selected = summary.loc[summary.objetivo.eq(target)].copy()
    selected['ajuste'] = selected.determinista.map({True:'Sin aleatoriedad activa',False:'Cinco semillas de ajuste'})
    for column in ['std','rango']:
        selected[column] = selected[column].map(lambda value:
            f'{value:.2e}'.replace('.',',') if 0<value<.00001 else number(value,5))
    comparisons = read('ablacion_contrastes')
    primary = comparisons.loc[comparisons.objetivo.eq(target)&comparisons.bloques_medios.eq(1)].copy()
    primary['Variante retirada / individual'] = primary.variante.map(NAMES)
    primary['IC95'] = [f'[{number(a,4)}; {number(b,4)}]' for a,b in zip(primary.IC95_inf,primary.IC95_sup)]
    primary['p_Holm'] = primary.p_Holm.map(lambda p: '<0,0001' if p<.0001 else number(p,4))
    sensitivity = comparisons.loc[comparisons.objetivo.eq(target)].copy()
    sensitivity['variante'] = sensitivity.variante.map(NAMES)
    return html.Div([
        disclosure('Método de estabilidad y ablación',dcc.Markdown(METHOD,mathjax=True,className='kernel-markdown')),
        html.H3('Estabilidad del entrenamiento · '+('NOx' if target=='NOX' else target)),
        html.P(SEED_HELP),
        table(selected[['modelo','mean','std','min','max','rango','ajuste']],
              {'modelo':'Modelo','mean':'R² OOF medio','std':'Desv. estándar','min':'Mínimo',
               'max':'Máximo','rango':'Rango','ajuste':'Aleatoriedad'},digits=5),
        html.P(seed_conclusion(summary,target)),
        graph('stability-seeds-figure',seed_figure(model_id,target)),
        html.P('Los métodos sin aleatoriedad activa reutilizan el resultado original; no son cinco ajustes independientes.',className='caption'),
        html.Div([
            html.H3('Ablación del kernel operativo'),
            html.P(ABLATION_HELP),
            table(ablation_table(target),{'OOF':'R² OOF','prueba':'R² prueba'},digits=4),
            table(primary[['Variante retirada / individual','delta_R2','IC95','p_Holm']],
                {'delta_R2':'ΔR² a favor del kernel completo','IC95':'IC95%','p_Holm':'p ajustado · Holm'},digits=4),
            html.P(ablation_conclusion(comparisons,target,read('resultados'))),
            disclosure('Sensibilidad de las ablaciones a la longitud de bloques',[
                html.P(ABLATION_SENSITIVITY_HELP),table(
                sensitivity[['variante','bloques_medios','delta_R2','IC95_inf','IC95_sup','p_Holm']],
                {'variante':'Variante','bloques_medios':'Bloques medios','delta_R2':'ΔR²',
                 'IC95_inf':'IC95% inferior','IC95_sup':'IC95% superior','p_Holm':'p Holm'},digits=5)]),
            tey_ablation.panel(target)
        ]) if model_id=='kernel_operativo' else None])


def layout():
    return disclosure('Sensibilidad a semillas y ablación',html.Div(id='stability-results'))


def register(app):
    app.callback(Output('stability-results','children'),Input('model-select','value'),Input('model-target','value'))(view)
