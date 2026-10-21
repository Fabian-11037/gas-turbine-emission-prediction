"""Explicacion del modelo operativo, visible solo al seleccionarlo."""
import json
from functools import lru_cache
from dash import Input, Output, dcc, html
from .components import disclosure, table
from .repository import DATA
from . import experiment_history, underestimation
from .kernel_content import (ARCHITECTURE, TREE_MODELS, MATHEMATICS, CONTRIBUTION, SELECTION,
    STATISTICAL_METHOD, STATISTICAL_INTERPRETATION)


@lru_cache(maxsize=1)
def architecture():
    return json.loads((DATA/'kernel_operativo.json').read_text(encoding='utf-8'))


def layout():
    return html.Div([disclosure('Kernel operativo · arquitectura, fundamentos y aporte', [
        dcc.Tabs(id='kernel-section', value='arquitectura', children=[
            dcc.Tab(label='Arquitectura', value='arquitectura'),
            dcc.Tab(label='Fundamentos', value='matematicas'),
            dcc.Tab(label='Aporte y literatura', value='aporte'),
            dcc.Tab(label='Evidencia', value='evidencia'),
            dcc.Tab(label='Modelos probados', value='historial')],
            className='kernel-tabs', mobile_breakpoint=0),
        html.Div(id='kernel-detail', className='kernel-detail')]),
        disclosure('Concentraciones altas · valores observados y estimados',
            html.Div(id='kernel-underestimation'), open=True)],
        id='kernel-panel', style={'display': 'none'})


def render(model_id, target, section):
    if model_id != 'kernel_operativo':
        return {'display': 'none'}, None
    if section == 'arquitectura':
        return {}, html.Div([html.Div([
            html.Figure([html.Img(src='/assets/kernel_arquitectura.svg',
                alt='Sensores y dos variables derivadas alimentan Extra Trees y SVR ponderado; Ridge combina ambas predicciones.'),
                html.Figcaption('Misma arquitectura; un modelo independiente por contaminante.', className='caption')]),
            dcc.Markdown(ARCHITECTURE, mathjax=True)], className='kernel-architecture'),
            dcc.Markdown(TREE_MODELS,className='kernel-markdown')])
    if section == 'aporte':
        return {}, dcc.Markdown(CONTRIBUTION, mathjax=True, className='kernel-markdown')
    if section == 'evidencia':
        return {}, evidence(target)
    if section == 'historial':
        return {}, experiment_history.layout()
    import pandas as pd
    parameters = architecture()[target]
    w = parameters['pesos_svr']
    weights = pd.DataFrame([
        ['Ambiente', 'AT, AP, AH', w['environmental']],
        ['Temperaturas', 'TIT, TAT, ΔT', w['temperature']],
        ['Presiones', 'AFDP, GTEP, CDP', w['pressure']],
        ['Carga e interacción', 'TEY, TEY × ΔT', 1.]], columns=['Grupo', 'Variables', 'Peso'])
    a, b = parameters['coeficientes']
    intercept = parameters['intercepto']
    equation = rf'$$\widehat{{{target}}}={intercept:.4f}+{a:.4f}\,\widehat y_{{ET}}+{b:.4f}\,\widehat y_{{SVR}}$$'
    return {}, html.Div([
        dcc.Markdown(MATHEMATICS, mathjax=True, className='kernel-markdown'),
        html.H4('Configuración final · '+target), table(weights),
        dcc.Markdown(equation, mathjax=True, className='kernel-equation'),
        html.P('Coeficientes expresados en la escala original de concentración. '
               f"λ Ridge = {parameters['lambda_ridge']}; α = {parameters['alpha_ridge']:.2f}. "
               'Son pesos de combinación aprendidos, no efectos causales.', className='caption')])


@lru_cache(maxsize=1)
def statistical_tables():
    import pandas as pd
    return (pd.read_csv(DATA/'kernel_significancia.csv'),
            pd.read_csv(DATA/'kernel_significancia_bloques.csv'))


def evidence(target):
    from .components import number
    summary, blocks = statistical_tables()
    selected = summary.loc[summary.objetivo.eq(target)].copy()
    selected['IC95'] = ['['+number(a,4)+'; '+number(b,4)+']'
                       for a,b in zip(selected.IC95_inf, selected.IC95_sup)]
    labels = {'referencia':'Referencia', 'delta_R2':'ΔR² prueba', 'IC95':'IC95% de ΔR²',
              'p_prueba_Holm2':'p prueba · Holm2', 'p_CV_Holm2':'p CV · Holm2'}
    details = selected[['referencia', 'p_prueba_nominal', 'p_prueba_Holm2',
                        'p_CV_nominal', 'p_CV_Holm2']]
    sensitivity = blocks.loc[blocks.objetivo.eq(target),
        ['referencia','longitud_bloque','n_unidades','delta_R2','IC95_inf','IC95_sup','p_prueba_nominal','p_prueba_Holm2']]
    return html.Div([
        dcc.Markdown(SELECTION, className='kernel-markdown'),
        html.H3('Evidencia estadística · '+('NOx' if target == 'NOX' else target)),
        table(selected[list(labels)], labels, digits=4),
        html.P('ΔR² positivo favorece al kernel. IC95% es el intervalo de incertidumbre de cada diferencia. '
               'Los valores p se ajustan con Holm para las dos emisiones.', className='caption'),
        html.P(STATISTICAL_INTERPRETATION[target]),
        disclosure('Ajustes múltiples y sensibilidad',[
            table(details, {'referencia':'Referencia','p_prueba_nominal':'p prueba nominal',
                'p_prueba_Holm2':'p prueba Holm2','p_CV_nominal':'p CV nominal',
                'p_CV_Holm2':'p CV Holm2'}, digits=4),
            html.H4('Sensibilidad al tamaño del bloque'),
            table(sensitivity, {'referencia':'Referencia','longitud_bloque':'Bloque','n_unidades':'Unidades',
                'delta_R2':'ΔR²','IC95_inf':'IC95% inferior','IC95_sup':'IC95% superior',
                'p_prueba_nominal':'p nominal','p_prueba_Holm2':'p Holm2'}, digits=4)]),
        dcc.Markdown(STATISTICAL_METHOD, className='kernel-markdown')])


def register(app):
    experiment_history.register(app)
    app.callback(Output('kernel-underestimation', 'children'),
        Input('model-select', 'value'), Input('model-target', 'value'))(
            lambda model_id, target: underestimation.panel(target) if model_id == 'kernel_operativo' else None)
    app.callback(Output('kernel-panel', 'style'), Output('kernel-detail', 'children'),
        Input('model-select', 'value'), Input('model-target', 'value'),
        Input('kernel-section', 'value'))(render)
