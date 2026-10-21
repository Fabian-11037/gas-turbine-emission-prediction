"""Plots and text from precomputed explanations; no SHAP/LIME runtime dependency."""
from functools import lru_cache
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from .components import disclosure, graph, metric, number, table
from .figures import style, TEAL, CORAL
from .repository import DATA
from .explainability_content import INTRODUCTION, METHOD

DIRECTORY = DATA/'explainability'
FEATURES = ['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP']
NAMES = {'kernel_operativo':'Kernel operativo','xgboost':'XGBoost'}
CASE_NAMES = {'error_bajo':'Error bajo','error_alto':'Error alto'}


@lru_cache(maxsize=1)
def evidence():
    return {name:pd.read_csv(DIRECTORY/(name+'.csv')) for name in ['resumen_global','casos','local']}


@lru_cache(maxsize=1)
def methodology():
    return json.loads((DIRECTORY/'metodologia.json').read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def global_frame(model_id,target):
    if model_id not in NAMES or target not in ['CO','NOX']:
        raise ValueError('Modelo o emisión sin explicaciones publicadas.')
    return pd.read_csv(DIRECTORY/f'global_{model_id}_{target}.csv.gz')


def selected(name, model_id, target):
    frame = evidence()[name]
    return frame.loc[frame.modelo_id.eq(model_id) & frame.objetivo.eq(target)].copy()


def case_data(model_id,target,case):
    cases = selected('casos',model_id,target)
    rows = selected('local',model_id,target)
    return cases.loc[cases.caso.eq(case)].iloc[0], rows.loc[rows.caso.eq(case)].copy()


def global_importance(model_id,target):
    frame = selected('resumen_global',model_id,target).sort_values('media_abs_SHAP')
    fig = go.Figure(go.Bar(x=frame.media_abs_SHAP,y=frame.sensor,orientation='h',marker_color=TEAL,
        hovertemplate='%{y}<br>Contribución absoluta media: %{x:.3f} mg/m³<extra></extra>'))
    return style(fig,'Importancia global · SHAP','Contribución absoluta media (mg/m³)','Sensor')


def global_summary(model_id,target):
    frame = global_frame(model_id,target)
    importance = selected('resumen_global',model_id,target).sort_values('media_abs_SHAP')
    fig = go.Figure()
    rng = np.random.default_rng(42)
    for index,sensor in enumerate(importance.sensor):
        relative = frame[sensor].rank(pct=True).to_numpy()
        custom = np.column_stack([frame[sensor],frame.anio,frame.orden_registro])
        fig.add_trace(go.Scattergl(x=frame['shap_'+sensor],y=index+rng.uniform(-.27,.27,len(frame)),
            mode='markers',showlegend=False,name=sensor,
            marker={'size':3,'opacity':.45,'color':relative,'cmin':0,'cmax':1,
                'colorscale':[[0,TEAL],[.5,'#d7dddd'],[1,CORAL]],'showscale':index==0,
                'colorbar':{'title':{'text':'Valor'},'tickvals':[0,1],'ticktext':['Bajo','Alto'],
                            'thickness':9,'len':.65}},customdata=custom,
            hovertemplate=sensor+': %{customdata[0]:.3f}<br>SHAP: %{x:.3f} mg/m³<br>Año: %{customdata[1]:.0f}<br>Registro: %{customdata[2]:.0f}<extra></extra>'))
    style(fig,'Distribución de contribuciones · SHAP','Contribución a la estimación (mg/m³)','')
    fig.update_yaxes(tickvals=list(range(9)),ticktext=importance.sensor.tolist())
    fig.add_vline(x=0,line_color='#808b8c',line_dash='dot',line_width=1)
    fig.update_layout(height=410,margin={'l':58,'r':75,'t':56,'b':60})
    return fig


def waterfall(model_id,target,case):
    summary,frame = case_data(model_id,target,case)
    frame = frame.assign(magnitude=frame.SHAP.abs()).sort_values('magnitude',ascending=False)
    names = ['Referencia']+frame.sensor.tolist()+['Estimación']
    values = [summary.base_SHAP]+frame.SHAP.tolist()+[0]
    fig = go.Figure(go.Waterfall(orientation='h',y=names,x=values,
        measure=['absolute']+['relative']*9+['total'],
        increasing={'marker':{'color':CORAL}},decreasing={'marker':{'color':TEAL}},
        totals={'marker':{'color':'#556365'}},connector={'line':{'color':'#b8c2c2'}},
        hovertemplate='%{y}<br>%{x:.3f} mg/m³<extra></extra>'))
    style(fig,'Cómo se forma la estimación · SHAP','Concentración y contribuciones (mg/m³)','')
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(height=460,showlegend=False,margin={'l':90,'r':25,'t':55,'b':55})
    return fig


def lime_figure(model_id,target,case):
    _,frame = case_data(model_id,target,case)
    frame = frame.assign(magnitude=frame.LIME.abs()).sort_values('magnitude')
    fig = go.Figure(go.Bar(x=frame.LIME,y=frame.sensor,orientation='h',
        customdata=frame[['condicion_LIME']].to_numpy(),
        marker_color=[CORAL if value>0 else TEAL for value in frame.LIME],
        hovertemplate='%{y}<br>%{customdata[0]}<br>Peso LIME: %{x:.3f} mg/m³<extra></extra>'))
    style(fig,'Aproximación del caso · LIME','Peso del intervalo del sensor (mg/m³)','Sensor')
    fig.add_vline(x=0,line_color='#808b8c',line_width=1)
    fig.update_layout(height=460)
    return fig


def local_discussion(model_id,target,case):
    summary,frame = case_data(model_id,target,case)
    positive = frame.loc[frame.SHAP.gt(0)].sort_values('SHAP',ascending=False)
    negative = frame.loc[frame.SHAP.lt(0)].sort_values('SHAP')
    effects = []
    if len(positive):
        effects.append(f"{positive.iloc[0].sensor} aporta el mayor aumento")
    if len(negative):
        effects.append(f"{negative.iloc[0].sensor} aporta la mayor reducción")
    return (f"La referencia es {number(summary.base_SHAP)} mg/m³ y la estimación es "
            f"{number(summary.prediccion)} mg/m³. "+'; '.join(effects)+'. '
            f"El valor observado es {number(summary.observado)} mg/m³; el error absoluto es "
            f"{number(summary.error_absoluto)} mg/m³. Las contribuciones explican la estimación, no la causa del error.")


def comparison_discussion(model_id,target,case):
    summary,frame = case_data(model_id,target,case)
    return (f"SHAP y LIME coinciden en {int(summary.coincidencias_top3)} de los tres sensores de mayor magnitud. "
        f"La aproximación LIME estima {number(summary.prediccion_LIME)} mg/m³, frente a "
        f"{number(summary.prediccion)} mg/m³ del modelo; su R² local es {number(summary.R2_local_LIME)}. "
        'Un R² local bajo indica que LIME reproduce poco fielmente al modelo alrededor de este caso. '
        'SHAP y LIME usan referencias y aproximaciones distintas, por lo que sus contribuciones no tienen que coincidir.')


def layout():
    return html.Div(disclosure('Interpretación de predicciones · SHAP y LIME',[
        html.P(INTRODUCTION),
        dcc.RadioItems(id='explain-view',value='global',options=[
            {'label':'Global','value':'global'},{'label':'Individual','value':'local'},
            {'label':'SHAP frente a LIME','value':'comparacion'}],className='segmented explain-modes'),
        html.Div([html.Label('Caso de prueba',htmlFor='explain-case'),
            dcc.RadioItems(id='explain-case',value='error_bajo',options=[
                {'label':label,'value':value} for value,label in CASE_NAMES.items()],className='segmented')],
            id='explain-case-wrap',className='explain-case',style={'display':'none'}),
        html.Div(id='explain-content'),
        disclosure('Método y alcance de las explicaciones',dcc.Markdown(METHOD,mathjax=True,className='kernel-markdown'))]),
        id='explain-panel',style={'display':'none'})


def render(model_id,target,view,case):
    if model_id not in NAMES:
        return {'display':'none'},None,{'display':'none'}
    name = NAMES[model_id]+' · '+('NOx' if target=='NOX' else target)
    if view == 'global':
        frame = selected('resumen_global',model_id,target).sort_values('media_abs_SHAP',ascending=False)
        diagnostic = next(r['precision'] for r in methodology()['resultados']
                          if r['modelo_id']==model_id and r['objetivo']==target)
        return {},html.Div([html.H3(name),html.P('Mayor contribución absoluta media: '+', '.join(frame.sensor.head(3))+'.'),
            html.Div([graph('explain-global-summary',global_summary(model_id,target)),
                      graph('explain-global-importance',global_importance(model_id,target))],className='chart-grid'),
            html.P('Cada punto representa un registro de prueba. Hacia la derecha aumenta la estimación; '
                   'hacia la izquierda la reduce. El color indica si el valor del sensor es bajo o alto dentro de la prueba.',className='caption'),
            disclosure('Valores globales y precisión de la aproximación',[
                table(frame[['sensor','media_abs_SHAP']],{'sensor':'Sensor','media_abs_SHAP':'Media |SHAP| (mg/m³)'},digits=4),
                html.P(f"Al ampliar de dos a diez recorridos en 64 registros, la correlación entre los órdenes de importancia es "
                    f"{number(diagnostic['rho_rangos'])}; coinciden {diagnostic['coincidencias_top3']} de los tres sensores principales.",className='caption')])]),{'display':'none'}
    summary,frame = case_data(model_id,target,case)
    header = [html.H3(name+' · '+CASE_NAMES[case].lower()),
        html.P(f"Año {int(summary.anio)} · registro {int(summary.orden_registro)}",className='caption'),
        html.Div([metric('Observado',number(summary.observado),'mg/m³'),
                  metric('Estimado',number(summary.prediccion),'mg/m³'),
                  metric('Error absoluto',number(summary.error_absoluto),'mg/m³')],className='metric-strip')]
    if view == 'local':
        content = [graph('explain-waterfall',waterfall(model_id,target,case)),
            html.P(local_discussion(model_id,target,case)),
            disclosure('Sensores y contribuciones del caso',table(frame[['sensor','valor','SHAP']],
                {'sensor':'Sensor','valor':'Valor observado','SHAP':'Contribución SHAP (mg/m³)'},digits=4))]
    else:
        content = [html.Div([graph('explain-shap-comparison',waterfall(model_id,target,case)),
                            graph('explain-lime',lime_figure(model_id,target,case))],className='chart-grid'),
            html.P(comparison_discussion(model_id,target,case)),
            html.P('Rojo: aumento; verde: reducción. SHAP y LIME usan referencias distintas: '
                   'sus magnitudes no son intercambiables. R² local mide la fidelidad de LIME al modelo, no el desempeño de prueba.',className='caption'),
            disclosure('Comparación por sensor e intervalo LIME',table(frame[['sensor','SHAP','LIME','condicion_LIME']],
                {'sensor':'Sensor','SHAP':'SHAP (mg/m³)','LIME':'Peso LIME (mg/m³)','condicion_LIME':'Intervalo del sensor'},digits=4))]
    return {},html.Div(header+content),{}


def register(app):
    app.callback(Output('explain-panel','style'),Output('explain-content','children'),
        Output('explain-case-wrap','style'),Input('model-select','value'),Input('model-target','value'),
        Input('explain-view','value'),Input('explain-case','value'))(render)
