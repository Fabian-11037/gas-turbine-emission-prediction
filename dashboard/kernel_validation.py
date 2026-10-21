"""Resultados ya calculados; no se ajustan modelos ni se repiten contrastes."""
from functools import lru_cache
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from . import figures as f, kernel_validation_content as text
from .components import disclosure, graph, number, table
from .repository import DATA

LOCATION = DATA/'kernel_validation'
STRATEGIES = {'bloques':'Bloques', 'aleatoria':'Aleatoria'}


@lru_cache(maxsize=1)
def evidence():
    return (pd.read_csv(LOCATION/'resultados.csv'), pd.read_csv(LOCATION/'cv.csv'),
            pd.read_csv(LOCATION/'prueba.csv'))


def result_table():
    frame = evidence()[0].copy()
    frame['objetivo'] = frame.objetivo.replace({'NOX':'NOx'})
    frame['estrategia'] = frame.estrategia.map(STRATEGIES)
    return frame[['objetivo','estrategia','R2_CV','R2_OOF','R2_prueba']].rename(columns={
        'objetivo':'Emisión','estrategia':'Preparación interna','R2_CV':'R² CV medio',
        'R2_OOF':'R² OOF','R2_prueba':'R² prueba'})


def significance_table():
    _, cv, test = evidence()
    frame = test.loc[test.longitud_bloque.eq(168)].merge(
        cv[['objetivo','p_Holm']],on='objetivo',suffixes=('_prueba','_CV'),validate='one_to_one')
    frame['IC 95 %'] = ['['+number(a,4)+'; '+number(b,4)+']' for a,b in zip(frame.IC95_inf,frame.IC95_sup)]
    frame['objetivo'] = frame.objetivo.replace({'NOX':'NOx'})
    return frame[['objetivo','delta_R2','IC 95 %','p_Holm_CV','p_Holm_prueba']].rename(columns={
        'objetivo':'Emisión','delta_R2':'ΔR² prueba','p_Holm_CV':'p CV ajustado','p_Holm_prueba':'p prueba ajustado'})


def interpretation(target):
    _, cv, test = evidence()
    validation = cv.set_index('objetivo').loc[target]
    holdout = test.loc[test.longitud_bloque.eq(168)].set_index('objetivo').loc[target]
    if target == 'CO':
        return (f'CO: la preparación aleatoria mejora el R² de prueba en {number(-holdout.delta_R2,4)}, '
            f'pero la diferencia no es significativa (p ajustado = {number(holdout.p_Holm,4)}). '
            f'Tampoco se distingue una ventaja en CV (p = {number(validation.p_Holm,4)}). '
            'No hay evidencia suficiente para afirmar que una estrategia sea superior.')
    return (f'NOx: la preparación por bloques mejora el R² de prueba en {number(holdout.delta_R2,4)} '
        f'(p ajustado = {number(holdout.p_Holm,4)}) y también mejora CV '
        f'(p = {number(validation.p_Holm,4)}). La ventaja es significativa y se mantiene '
        'con las tres longitudes de remuestreo.')


def results_figure(target):
    selected = evidence()[0].loc[lambda d:d.objetivo.eq(target)].set_index('estrategia')
    fig = go.Figure()
    for strategy, color in [('bloques',f.TEAL),('aleatoria',f.CORAL)]:
        values = selected.loc[strategy,['R2_CV','R2_prueba']].astype(float).to_list()
        fig.add_trace(go.Bar(name=STRATEGIES[strategy],x=['CV externa','Prueba'],y=values,
            marker_color=color,text=[number(v,3) for v in values],textposition='outside',
            cliponaxis=False,hovertemplate='%{x}<br>R²: %{y:.4f}<extra>%{fullData.name}</extra>'))
    f.style(fig,('NOx' if target=='NOX' else target)+' · misma evaluación externa','Evaluación','R²')
    fig.update_yaxes(range=[0,1],dtick=.25)
    fig.update_layout(barmode='group',legend={'orientation':'h','y':-.30,'x':0},
        margin={'l':52,'r':16,'t':50,'b':94})
    return fig


def difference_figure():
    selected = evidence()[2].loc[lambda d:d.longitud_bloque.eq(168)].set_index('objetivo')
    fig = go.Figure()
    for target, color in [('CO',f.CORAL),('NOX',f.TEAL)]:
        row = selected.loc[target]
        fig.add_trace(go.Scatter(x=[row.delta_R2],y=['NOx' if target=='NOX' else target],
            mode='markers',name=target,marker={'color':color,'size':9},showlegend=False,
            error_x={'type':'data','symmetric':False,'array':[row.IC95_sup-row.delta_R2],
                     'arrayminus':[row.delta_R2-row.IC95_inf],'color':color,'thickness':2},
            customdata=[[row.IC95_inf,row.IC95_sup,row.p_Holm]],
            hovertemplate='ΔR²: %{x:.4f}<br>IC 95 %: [%{customdata[0]:.4f}; %{customdata[1]:.4f}]'
                          '<br>p ajustado: %{customdata[2]:.4f}<extra>%{y}</extra>'))
    fig.add_vline(x=0,line_dash='dash',line_color=f.GRAY,line_width=1)
    f.style(fig,'Diferencia e incertidumbre en prueba','ΔR²: bloques − aleatoria','Emisión')
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(height=285,margin={'l':52,'r':20,'t':50,'b':64})
    return fig


def panel(target):
    return html.Div([
        html.H3(text.TITLE),html.P(text.INTRO),
        disclosure('Diseño de la comparación',html.P(text.DESIGN)),
        graph('kernel-validation-r2',results_figure(target)),
        html.P(interpretation(target)),
        graph('kernel-validation-uncertainty',difference_figure()),
        html.P(text.READING,className='caption'),
        html.P(text.CONCLUSION),
        disclosure('R² de ambas emisiones',table(result_table(),digits=4)),
        disclosure('Significancia y método',[
            table(significance_table(),digits=4),html.P(text.METHOD),
            html.P('ΔR² = R² por bloques − R² aleatoria.',className='caption')])],
        id='kernel-validation-analysis')
