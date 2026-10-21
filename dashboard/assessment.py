"""Read-only statistical hierarchy and computational cost evidence."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input,Output,dcc,html

from .assessment_content import (STATISTICS,COMPLEXITY,EFFICIENCY,SENSITIVITY_HELP,RANKING_HELP,
                                 statistical_interpretation,computational_interpretation)
from .components import disclosure,graph,number,table
from .figures import style,TEAL,CORAL,GRAY
from .repository import DATA


@lru_cache(maxsize=8)
def read(folder,name):
    return pd.read_csv(DATA/folder/(name+'.csv'))


@lru_cache(maxsize=2)
def metadata(folder):
    return json.loads((DATA/folder/'metodologia.json').read_text(encoding='utf-8'))


def mcs_table(target,length=1.,published_only=True):
    candidates = read('statistical_hierarchy','procedimientos')
    results = read('statistical_hierarchy','mcs')
    frame = candidates.loc[candidates.objetivo.eq(target)]
    if published_only:
        frame = frame.loc[frame.publicado]
    frame = frame.merge(results.loc[results.objetivo.eq(target)&results.bloques_medios.eq(length)],
                        on=['objetivo','procedimiento'],validate='one_to_one')
    return frame.sort_values('MSE_OOF')


def statistical_conclusion(target):
    rows = read('statistical_hierarchy','mcs').query('objetivo == @target')
    sizes = [int(rows.loc[rows.bloques_medios.eq(length),'incluido_005'].sum()) for length in [1,2,4]]
    return statistical_interpretation(target,sizes)


def p_columns(frame):
    frame = frame.copy()
    resolution = 1/metadata('statistical_hierarchy')['replicas']
    for column in frame:
        if column.startswith('p_'):
            frame[column] = [('<'+number(resolution,4) if value==0 else
                f'{value:.2e}'.replace('.',',') if value<.0001 else number(value,4)) for value in frame[column]]
    return frame


def critical_difference(target):
    ranking = read('statistical_hierarchy','rangos').query('objetivo == @target').sort_values('rango_medio')
    details = next(x for x in metadata('statistical_hierarchy')['auxiliar'] if x['objetivo']==target)
    k = len(ranking)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ranking.rango_medio,y=ranking.procedimiento.str.split(' · ').str[0],
        mode='markers',marker={'size':9,'color':TEAL},showlegend=False,
        hovertemplate='%{y}<br>Rango: %{x:.2f}<extra></extra>'))
    style(fig,'Rangos y diferencia crítica · análisis auxiliar','Rango medio: menor es mejor','')
    fig.update_yaxes(autorange='reversed')
    if details['posthoc_aplicado']:
        cd = details['CD']
        fig.add_shape(type='line',x0=1,x1=1+cd,y0=1.08,y1=1.08,xref='x',yref='paper',line={'color':CORAL,'width':3})
        fig.add_annotation(x=1+cd/2,y=1.15,xref='x',yref='paper',text=f'CD = {number(cd,2)}',showarrow=False)
        # Equal-length intervals overlap exactly when rank distance is at most CD.
        fig.data[0].error_x={'type':'constant','value':cd/2,'color':'#a5b6b6','thickness':2,'width':3}
    fig.update_layout(margin={'l':112,'r':24,'t':95,'b':55},height=430)
    return fig


def statistical_view(target):
    current = mcs_table(target)
    display = current[['modelo','optimizador','MSE_OOF','R2_OOF','p_MCS','incluido_005']].copy()
    display['incluido_005'] = display.incluido_005.map({True:'Retenido',False:'Excluido'})
    directed = read('statistical_hierarchy','dirigidas')
    main = directed.loc[directed.objetivo.eq(target)&directed.bloques_medios.eq(1)].copy()
    main['IC95_delta_R2'] = [f'[{number(a,4)}; {number(b,4)}]' for a,b in zip(main.IC95_inf,main.IC95_sup)]
    sensitivity = directed.loc[directed.objetivo.eq(target),
        ['bloques_medios','p_GW_Holm2','p_DM_Holm2','IC95_inf','IC95_sup']]
    counts = []
    for length in [1,2,4]:
        frame = mcs_table(target,length,False)
        counts.append({'Bloques medios':length,'Retenidos α=0,05':int(frame.incluido_005.sum()),
                       'Retenidos α=0,10':int(frame.incluido_010.sum())})
    all_rows = mcs_table(target,1,False)[['procedimiento','MSE_OOF','p_MCS','incluido_005','representante']].copy()
    all_rows.incluido_005 = all_rows.incluido_005.map({True:'Retenido',False:'Excluido'})
    aux = next(row for row in metadata('statistical_hierarchy')['auxiliar'] if row['objetivo']==target)
    return html.Div([
        html.P('Validación externa OOF · 29 procedimientos por emisión · modelos y particiones sin cambios.',className='caption'),
        html.H3('Comparación conjunta · '+('NOx' if target=='NOX' else target)),
        table(p_columns(display),{'modelo':'Modelo publicado','optimizador':'Optimizador','MSE_OOF':'MSE OOF',
            'R2_OOF':'R² OOF','p_MCS':'p MCS','incluido_005':'MCS · α=0,05'},digits=4),
        html.P(statistical_conclusion(target)),
        html.H4('Contraste dirigido · kernel frente a la referencia de CV'),
        table(p_columns(main[['referencia','delta_R2','IC95_delta_R2','p_GW_Holm2','p_DM_Holm2','Cohen_dz_bloques']]),
            {'referencia':'Referencia','delta_R2':'ΔR² OOF','IC95_delta_R2':'IC95% de ΔR²',
             'p_GW_Holm2':'p Wald condicional · Holm','p_DM_Holm2':'p DM-HLN · Holm','Cohen_dz_bloques':'Cohen dz · bloques'},digits=4),
        html.P('Diferencia positiva: menor error del kernel. Son contrastes adaptados sobre OOF; '
               'los resultados del conjunto de prueba permanecen separados.',className='caption'),
        disclosure('Sensibilidad y valores p sin ajuste',[
            html.P(SENSITIVITY_HELP),
            table(pd.DataFrame(counts)),table(p_columns(sensitivity),{'bloques_medios':'Bloques medios',
                'p_GW_Holm2':'p Wald · Holm','p_DM_Holm2':'p DM-HLN · Holm','IC95_inf':'IC95% inferior','IC95_sup':'IC95% superior'},digits=4),
            table(p_columns(main[['p_GW','p_GW_Holm2','p_DM','p_DM_Holm2']]),
                {'p_GW':'p Wald nominal','p_GW_Holm2':'p Wald Holm','p_DM':'p DM nominal','p_DM_Holm2':'p DM Holm'},digits=6)]),
        disclosure('Los 29 procedimientos y controles coincidentes',table(p_columns(all_rows),
            {'procedimiento':'Procedimiento','MSE_OOF':'MSE OOF','p_MCS':'p MCS','incluido_005':'MCS',
             'representante':'Contraste compartido'},digits=4)),
        disclosure('Friedman–Nemenyi · comprobación auxiliar',[
            html.P(RANKING_HELP),
            html.P(f"Friedman: p = {aux['p_Friedman']:.2e}; cinco pliegues con entrenamiento compartido."),
            graph('assessment-critical-difference',critical_difference(target)),
            html.P('Los intervalos se superponen cuando la distancia entre rangos no supera la diferencia crítica de Nemenyi. '
                   'Esta comprobación auxiliar no sustituye el análisis MCS por bloques.',className='caption')]),
        disclosure('Método de comparación',dcc.Markdown(STATISTICS,mathjax=True,className='kernel-markdown'))])


def latency(model_id,target):
    frame = read('efficiency','mediciones')
    frame = frame.loc[frame.modelo_id.eq(model_id)&frame.objetivo.eq(target)]
    names = {'residente':'Residente','fragmentado':'Fragmentado','rbf_vectorizado':'RBF vectorizado'}
    fig = go.Figure()
    for color,(mode,group) in zip([GRAY,CORAL,TEAL],frame.groupby('modo',sort=False)):
        summary = group.groupby('n_lote').ms_por_registro.median()
        fig.add_trace(go.Scatter(x=summary.index,y=summary.values,mode='lines+markers',
            name=names[mode],line={'color':color},hovertemplate='Lote: %{x}<br>%{y:.4f} ms/registro<extra>%{fullData.name}</extra>'))
    style(fig,'Costo de predicción · modelo seleccionado','Registros por lote','Milisegundos por registro')
    fig.update_xaxes(type='log',tickvals=[1,100,1000])
    fig.update_yaxes(type='log')
    return fig


def efficiency_view(model_id,target):
    frame = read('efficiency','resumen')
    native = frame.loc[frame.objetivo.eq(target)&frame.modo.eq('residente')].copy()
    native['ms_por_registro_1000'] = native.segundos_lote_1000
    details = frame.loc[frame.objetivo.eq(target)&frame.modelo_id.eq(model_id)].copy()
    details.error_maximo = details.error_maximo.map(lambda value:f'{value:.2e}'.replace('.',','))
    details.modo = details.modo.map({'residente':'Residente','fragmentado':'Fragmentado','rbf_vectorizado':'RBF vectorizado'})
    return html.Div([
        html.H3('Complejidad y eficiencia · '+('NOx' if target=='NOX' else target)),
        html.P('Referencia común: modelos cargados completos en memoria. El despliegue del kernel utiliza fragmentos.',className='caption'),
        table(native[['modelo','segundos_ajuste_historico','ms_por_registro_1000','RSS_pico_MB','aceleracion_lote_100','R2_prueba_original']],
            {'modelo':'Modelo','segundos_ajuste_historico':'Ajuste histórico (s)',
             'ms_por_registro_1000':'ms/registro · lote 1.000','RSS_pico_MB':'Pico RSS (MiB)',
             'aceleracion_lote_100':'Aceleración lote vs individual · 100','R2_prueba_original':'R² prueba original'},digits=4),
        html.P('—: medición histórica no disponible. Los tiempos de entrenamiento son registros originales; '
               'los tiempos y la memoria de inferencia se midieron ahora con los mismos estimadores.',className='caption'),
        graph('assessment-latency',latency(model_id,target)),
        html.P(computational_interpretation(frame.loc[frame.objetivo.eq(target)&frame.modelo_id.eq(model_id)]))
            if model_id=='kernel_operativo' else None,
        disclosure('Detalle de ejecución · modelo seleccionado',[
            table(details[['modo','segundos_carga','segundos_lote_1','segundos_lote_100','segundos_lote_1000','RSS_pico_MB','error_maximo']],
                {'modo':'Ejecución','segundos_carga':'Carga inicial (s)','segundos_lote_1':'Lote 1 (s)',
                 'segundos_lote_100':'Lote 100 (s)','segundos_lote_1000':'Lote 1.000 (s)',
                 'RSS_pico_MB':'Pico RSS (MiB)','error_maximo':'Máxima diferencia de predicción'},digits=6),
            html.P('Fragmentar reduce árboles residentes y añade lecturas. Vectorizar el RBF cambia su evaluación numérica, '
                   'no el SVR ajustado. El adaptador publicado no se modifica.',className='caption')
                if model_id=='kernel_operativo' else None]),
        disclosure('Complejidad teórica',dcc.Markdown(COMPLEXITY,mathjax=True,className='kernel-markdown')),
        disclosure('Presupuesto y eficiencia de los optimizadores',table(
            read('efficiency','optimizacion_pareto').loc[lambda d:d.objetivo.eq(target),
                ['modelo','optimizador','MSE_CV_externo','segundos_total','segundos_evaluacion_media','evaluaciones_unicas','Pareto']],
            {'modelo':'Modelo','optimizador':'Optimizador','MSE_CV_externo':'MSE externo','segundos_total':'Tiempo total histórico (s)',
             'segundos_evaluacion_media':'Media por evaluación sin caché (s)','evaluaciones_unicas':'Evaluaciones únicas','Pareto':'En frente de Pareto'},digits=4)),
        disclosure('Condiciones de medición',dcc.Markdown(EFFICIENCY,className='kernel-markdown'))])


def layout():
    return html.Div([
        disclosure('Comparación estadística jerárquica · validación OOF',html.Div(id='assessment-statistics')),
        disclosure('Complejidad y eficiencia computacional',html.Div(id='assessment-efficiency'))])


def register(app):
    app.callback(Output('assessment-statistics','children'),Input('model-target','value'))(statistical_view)
    app.callback(Output('assessment-efficiency','children'),Input('model-select','value'),Input('model-target','value'))(efficiency_view)
