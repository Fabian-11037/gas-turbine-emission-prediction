"""EDA filtrable, con dos figuras por vista y tablas plegables."""
import pandas as pd
from dash import Input,Output,dcc,html
from . import figures as f
from .components import disclosure,graph,metric,notice,number,table
from .repository import filtered,sensors,sensor_audit


def layout():
    return html.Div([
        html.Aside([html.H3('Filtros'),html.Label('Emisión'),
                    dcc.RadioItems(id='eda-target',options=[{'label':'CO','value':'CO'},{'label':'NOx','value':'NOX'}],value='CO',className='segmented'),
                    html.Label('Años'),dcc.Dropdown(id='eda-years',options=[{'label':str(y),'value':y} for y in range(2011,2016)],
                                                   value=list(range(2011,2016)),multi=True,clearable=True),
                    html.Label('Población'),dcc.Dropdown(id='eda-population',options=[
                        {'label':'Desarrollo','value':'desarrollo'},{'label':'Fuente completa · descriptivo','value':'todos'}],
                        value='desarrollo',clearable=False),
                    html.Label('Sensor'),dcc.Dropdown(id='eda-sensor',options=[{'label':s['id']+' · '+s['nombre'],'value':s['id']} for s in sensors()],
                                                      value='TIT',clearable=False),
                    html.Div('Fuente completa incluye prueba y purga. No se utiliza esta exploración para seleccionar modelos.',className='sidebar-note')],className='filter-sidebar'),
        html.Div([html.P('02 / EDA',className='eyebrow'),html.H2('Exploración de los datos'),
                  dcc.Tabs(id='eda-view',value='distribuciones',children=[
                      dcc.Tab(label='Distribuciones',value='distribuciones'),dcc.Tab(label='Relaciones',value='relaciones'),
                      dcc.Tab(label='Cambios anuales',value='temporal'),dcc.Tab(label='Correlaciones',value='correlaciones')],
                           className='local-tabs',parent_className='local-tabs-parent',mobile_breakpoint=0),
                  html.Div(id='eda-summary',className='metric-strip compact'),
                  dcc.Loading(html.Div([graph('eda-chart-one'),graph('eda-chart-two')],className='chart-grid'),type='circle',color=f.TEAL),
                  notice('',identifier='eda-insight'),
                  disclosure('Estadísticas descriptivas y resumen anual',html.Div(id='eda-table')),
                  disclosure('Calidad y valores extremos',html.Div(id='eda-quality')),
                  disclosure('Capacidad predictiva de cada sensor · CV',html.Div(id='eda-sensor-audit'))],className='workspace')
    ],className='eda-layout')


def render(target, years, population, view, sensor):
    df = filtered(years,population)
    if df.empty:
        return f.empty(),f.empty(),[metric('Registros','0')],'Seleccione al menos un año con observaciones.',html.P('Sin datos.'),html.P('Sin datos.'),audit_table(target)
    features = [s['id'] for s in sensors()]
    stats = df[target].describe(percentiles=[.25,.5,.75,.95,.99])
    annual = df.groupby('anio')[target].agg(['count','mean','median','std','min','max']).reset_index()
    summaries = [metric('Registros',number(len(df),0)),metric(f'Mediana {target}',number(df[target].median(),2),'mg/m³'),
                 metric('Población','Desarrollo' if population == 'desarrollo' else 'Fuente completa')]
    tables = [table(stats.rename_axis('estadistico').reset_index(name='valor'),{'estadistico':'Estadístico','valor':f'{target} (mg/m³)'}),
              html.H4('Por año'),table(annual,{'anio':'Año','count':'Registros','mean':'Media','median':'Mediana','std':'Desv. estándar','min':'Mínimo','max':'Máximo'})]
    q1,q3 = df[target].quantile([.25,.75])
    extreme = int(((df[target] < q1-1.5*(q3-q1)) | (df[target] > q3+1.5*(q3-q1))).sum())
    quality = pd.DataFrame([{'Control':'Celdas faltantes (sensores y emisiones)','Registros':int(df[features+['CO','NOX']].isna().sum().sum())},
                            {'Control':'Filas duplicadas en sensores y emisiones','Registros':int(df[features+['CO','NOX']].duplicated().sum())},
                            {'Control':'Humedad superior a 100 %','Registros':int(df.AH.gt(100).sum())},
                            {'Control':f'{target}: extremos por 1,5×IQR','Registros':extreme},
                            {'Control':'Emisiones negativas','Registros':int(df[['CO','NOX']].lt(0).any(axis=1).sum())}])
    quality_ui = [table(quality),html.P('Los conteos describen los filtros actuales. IQR señala extremos estadísticos, no errores físicos; no se eliminan ni se usan para reajustar modelos.',className='caption')]
    if view == 'distribuciones':
        figs = f.distribution(df,target),f.annual_boxes(df,target)
        insight = f'Asimetría de {target}: {number(df[target].skew(),2)}. El histograma incluye toda la cola; las cajas omiten los puntos extremos, no sus registros. Diferencias anuales no implican causalidad.'
    elif view == 'relaciones':
        rho = df[[sensor,target]].rank().corr().iloc[0,1]
        figs = f.relationship(df,target,sensor),f.associations(df,target,features)
        insight = f'{sensor} y {target}: ρ de Spearman = {number(rho)} en todos los registros filtrados. La dispersión muestra una muestra reproducible de hasta 2.200 puntos. Asociación no equivale a efecto causal.'
    elif view == 'temporal':
        figs = f.block_evolution(df,target),f.annual_summary(df,target)
        insight = 'La posición es el orden original del registro, no una fecha. Las líneas se separan por año; cada media utiliza los registros disponibles del bloque de origen. Los cambios pueden reflejar ambiente, operación o composición de la muestra.'
    else:
        figs = f.correlation(df,features),f.associations(df,target,features)
        insight = 'Correlaciones de Spearman sobre la muestra completa filtrada. Sensores relacionados pueden compartir información: el ranking no mide aporte incremental dentro de un modelo ni demuestra ausencia de fuga.'
    return *figs,summaries,insight,tables,quality_ui,audit_table(target)


def audit_table(target):
    audit = sensor_audit().loc[sensor_audit().objetivo.eq(target),
        ['sensor','R2_OOF','RMSE_OOF','MAE_OOF','reduccion_RMSE_vs_Dummy_pct']].sort_values('R2_OOF',ascending=False)
    return [table(audit,{'sensor':'Sensor','R2_OOF':'R² OOF','RMSE_OOF':'RMSE OOF','MAE_OOF':'MAE OOF',
                         'reduccion_RMSE_vs_Dummy_pct':'Reducción RMSE vs media (%)'}),
            html.P('Ridge individual con α=1 y los cinco pliegues del notebook. Esta auditoría usa todo el desarrollo de 2011–2015 y no cambia con los filtros de años. No mide aporte incremental al modelo completo ni garantiza ausencia de fuga.',className='caption')]


def register(app):
    app.callback(Output('eda-chart-one','figure'),Output('eda-chart-two','figure'),
                 Output('eda-summary','children'),Output('eda-insight','children'),
                 Output('eda-table','children'),Output('eda-quality','children'),Output('eda-sensor-audit','children'),
                 Input('eda-target','value'),Input('eda-years','value'),Input('eda-population','value'),
                 Input('eda-view','value'),Input('eda-sensor','value'))(render)
