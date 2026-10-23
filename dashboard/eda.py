"""EDA filtrable, con dos figuras por vista y tablas plegables."""
import pandas as pd
from dash import Input,Output,dcc,html
from . import figures as f
from .components import disclosure,graph,metric,notice,number,table
from .repository import filtered,sensors,sensor_audit
from . import seasonality, partition_similarity,multivariate


def layout():
    return html.Div([
        html.Aside([html.H3('Filtros'),html.Label('Emisión'),
                    dcc.RadioItems(id='eda-target',options=[{'label':'CO','value':'CO'},{'label':'NOx','value':'NOX'}],value='CO',className='segmented'),
                    html.Label('Años'),dcc.Dropdown(id='eda-years',options=[{'label':str(y),'value':y} for y in range(2011,2016)],
                                                   value=list(range(2011,2016)),multi=True,clearable=True),
                    html.Label('Población'),dcc.Dropdown(id='eda-population',options=[
                        {'label':'Desarrollo','value':'desarrollo'},{'label':'Fuente completa · descriptivo','value':'todos'}],
                        value='desarrollo',clearable=False),
                    html.Div([html.Label('Período'),dcc.Dropdown(id='eda-period',options=[
                        {'label':'24 registros','value':24},{'label':'168 registros','value':168}],value=24,clearable=False)],
                        id='eda-period-controls',style={'display':'none'}),
                    html.Div([html.Label('Sensor'),dcc.Dropdown(id='eda-sensor',options=[{'label':s['id']+' · '+s['nombre'],'value':s['id']} for s in sensors()],
                                                      value='TIT',clearable=False)],id='eda-sensor-controls'),
                    html.Div('Desarrollo: datos usados para entrenar. Fuente completa: todos los registros disponibles.',className='sidebar-note')],className='filter-sidebar',id='eda-filters'),
        html.Div([html.P('02 / EDA',className='eyebrow'),html.H2('Exploración de los datos'),
                  dcc.Tabs(id='eda-view',value='distribuciones',children=[
                      dcc.Tab(label='Distribuciones',value='distribuciones'),dcc.Tab(label='Relaciones',value='relaciones'),
                      dcc.Tab(label='Multivariado',value='multivariado'),
                      dcc.Tab(label='Cambios anuales',value='temporal'),dcc.Tab(label='Estacionalidad',value='estacionalidad'),
                      dcc.Tab(label='Correlaciones',value='correlaciones'),
                      dcc.Tab(label='Particiones y similitud',value='particiones')],
                           className='local-tabs',parent_className='local-tabs-parent',mobile_breakpoint=0),
                  html.Div([
                  html.Div(id='eda-summary',className='metric-strip compact'),
                  dcc.Loading(html.Div([graph('eda-chart-one'),graph('eda-chart-two')],className='chart-grid'),type='circle',color=f.TEAL),
                  notice('',identifier='eda-insight'),
                  disclosure('Tablas del análisis seleccionado',html.Div(id='eda-table')),
                  disclosure('Calidad y valores extremos',html.Div(id='eda-quality')),
                  disclosure('Capacidad predictiva de cada sensor · CV',html.Div(id='eda-sensor-audit'))],id='eda-standard-content'),
                  html.Div(id='eda-partition-content',style={'display':'none'}),
                  html.Div(multivariate.layout(),id='eda-multivariate-content',style={'display':'none'})],className='workspace')
    ],className='eda-layout',id='eda-root')


def render(target, years, population, view, sensor, period=24):
    if view in ['particiones','multivariado']:
        return f.empty(),f.empty(),[], '', [], [], []
    if view == 'estacionalidad':
        one,two,summary,insight,tables,_ = seasonality.render(target,years,population,period)
        return one,two,summary,insight,tables,html.P('Sin exclusión de valores extremos.'),audit_table(target)
    df = filtered(years,population)
    if df.empty:
        return f.empty(),f.empty(),[metric('Registros','0')],'Seleccione al menos un año con observaciones.',html.P('Sin datos.'),html.P('Sin datos.'),audit_table(target)
    features = [s['id'] for s in sensors()]
    stats = df[target].describe(percentiles=[.25,.5,.75,.95,.99])
    annual = df.groupby('anio')[target].agg(['count','mean','median','std','min','max']).reset_index()
    summaries = [metric('Registros',number(len(df),0)),metric(f'Mediana {target}',number(df[target].median(),2),'mg/m³'),
                 metric('Población','Desarrollo' if population == 'desarrollo' else 'Fuente completa')]
    stats = stats.rename(index={'count':'Registros','mean':'Promedio','std':'Desviación estándar',
        'min':'Mínimo','25%':'Percentil 25','50%':'Mediana','75%':'Percentil 75',
        '95%':'Percentil 95','99%':'Percentil 99','max':'Máximo'})
    tables = [table(stats.rename_axis('estadistico').reset_index(name='valor'),{'estadistico':'Estadístico','valor':f'{target} (mg/m³)'}),
              html.H4('Por año'),table(annual,{'anio':'Año','count':'Registros','mean':'Promedio','median':'Mediana','std':'Desv. estándar','min':'Mínimo','max':'Máximo'})]
    q1,q3 = df[target].quantile([.25,.75])
    extreme = int(((df[target] < q1-1.5*(q3-q1)) | (df[target] > q3+1.5*(q3-q1))).sum())
    quality = pd.DataFrame([{'Control':'Celdas faltantes (sensores y emisiones)','Registros':int(df[features+['CO','NOX']].isna().sum().sum())},
                            {'Control':'Filas duplicadas en sensores y emisiones','Registros':int(df[features+['CO','NOX']].duplicated().sum())},
                            {'Control':'Humedad superior a 100 %','Registros':int(df.AH.gt(100).sum())},
                            {'Control':f'{target}: extremos por 1,5×IQR','Registros':extreme},
                            {'Control':'Emisiones negativas','Registros':int(df[['CO','NOX']].lt(0).any(axis=1).sum())}])
    quality_ui = [table(quality),html.P('IQR es el rango del 50% central de los datos. Permite señalar valores alejados de lo habitual; se conservan porque no necesariamente son errores.',className='caption')]
    if view == 'distribuciones':
        figs = f.distribution(df,target),f.annual_boxes(df,target)
        insight = f'Asimetría de {target}: {number(df[target].skew(),2)}; un valor positivo indica una cola hacia concentraciones altas. El histograma muestra frecuencias y las cajas resumen el centro y la dispersión; sus puntos extremos se ocultan solo en la figura.'
    elif view == 'relaciones':
        rho = df[[sensor,target]].rank().corr().iloc[0,1]
        figs = f.relationship(df,target,sensor),f.associations(df,target,features)
        insight = f'Asociación de Spearman entre {sensor} y {target}: {number(rho)}. Cerca de +1 tienden a aumentar juntos; cerca de −1, a moverse en sentido contrario. La asociación no prueba causalidad. La figura muestra hasta 2.200 registros.'
    elif view == 'temporal':
        figs = f.block_evolution(df,target),f.annual_summary(df,target)
        insight = 'Cada punto resume un bloque de registros próximos. El eje horizontal sigue su orden, no una fecha exacta. Los cambios entre años pueden acompañar variaciones de ambiente u operación.'
    else:
        figs = f.correlation(df,features),f.associations(df,target,features)
        insight = 'El mapa resume qué variables cambian juntas: valores próximos a +1 indican la misma dirección y a −1, direcciones opuestas. Sensores muy relacionados pueden aportar información parecida.'
    return *figs,summaries,insight,tables,quality_ui,audit_table(target)


def audit_table(target):
    audit = sensor_audit().loc[sensor_audit().objetivo.eq(target),
        ['sensor','R2_OOF','RMSE_OOF','MAE_OOF','reduccion_RMSE_vs_Dummy_pct']].sort_values('R2_OOF',ascending=False)
    return [table(audit,{'sensor':'Sensor','R2_OOF':'R² OOF','RMSE_OOF':'RMSE OOF','MAE_OOF':'MAE OOF',
                         'reduccion_RMSE_vs_Dummy_pct':'Reducción RMSE vs media (%)'}),
            html.P('Cada sensor se evalúa por separado con Ridge, un modelo lineal con control de complejidad, en cinco grupos de validación. OOF reúne las predicciones de registros no usados en su propio ajuste. Se usa todo el desarrollo de 2011–2015; este resultado no mide el aporte del sensor dentro del modelo completo.',className='caption')]


def register(app):
    app.callback(Output('eda-filters','style'),Output('eda-root','style'),
                 Output('eda-standard-content','style'),Output('eda-partition-content','style'),
                 Output('eda-partition-content','children'),Output('eda-multivariate-content','style'),
                 Output('eda-sensor-controls','style'),Input('eda-view','value'))(view_content)
    @app.callback(Output('eda-period-controls','style'),Input('eda-view','value'))
    def period_visibility(view):
        return {'display':'contents' if view == 'estacionalidad' else 'none'}

    app.callback(Output('eda-chart-one','figure'),Output('eda-chart-two','figure'),
                 Output('eda-summary','children'),Output('eda-insight','children'),
                 Output('eda-table','children'),Output('eda-quality','children'),Output('eda-sensor-audit','children'),
                 Input('eda-target','value'),Input('eda-years','value'),Input('eda-population','value'),
                 Input('eda-view','value'),Input('eda-sensor','value'),Input('eda-period','value'))(render)
    multivariate.register(app)


def view_visibility(view):
    if view == 'particiones':
        return {'display':'none'}, {'gridTemplateColumns':'minmax(0,1fr)'}, {'display':'none'}, {}
    if view == 'multivariado':
        return {}, {}, {'display':'none'}, {'display':'none'}
    return {}, {}, {}, {'display':'none'}


def view_content(view):
    return (*view_visibility(view), partition_similarity.layout() if view == 'particiones' else [],
            {} if view == 'multivariado' else {'display':'none'},
            {'display':'none'} if view == 'multivariado' else {})
