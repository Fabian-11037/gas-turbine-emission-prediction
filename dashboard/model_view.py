"""Resultados de modelos y prediccion individual o por lote sin entrenamiento."""
import base64
from io import StringIO
import binascii
import numpy as np
import pandas as pd
from dash import Input,Output,State,ctx,dcc,html,no_update
from . import figures as f
from .components import disclosure,graph,metric,notice,number,table
from .models import feature_importance,predict,registry
from .repository import intervals,optimization_results,predictions,range_warnings,results,sample_frame,sensors

MAX_ROWS = 2000
MAX_BYTES = 2*1024*1024


def layout():
    inputs = [html.Div([html.Label([html.Strong(s['id']),f" · {s['nombre']} ({s['unidad']})"],htmlFor=f"sensor-{s['id']}"),
                       dcc.Input(id=f"sensor-{s['id']}",type='number',value=s['ejemplo'],debounce=True,
                                 step='any',inputMode='decimal',className='sensor-input'),
                       html.Small(f"Desarrollo: {number(s['min'],2)}–{number(s['max'],2)} {s['unidad']}",className='input-range')],
                      className='sensor-field') for s in sensors()]
    return html.Div([
        html.Div([html.Div([html.P('03 / MODELOS',className='eyebrow'),html.H2('Estimación de emisiones'),
                            html.P('Benchmarks de regresión con validación anidada. Las estimaciones no sustituyen mediciones.',className='lead')]),
                  html.Div([html.Label('Modelo',htmlFor='model-select'),dcc.Dropdown(id='model-select',
                      options=[{'label':m['nombre'],'value':mid} for mid,m in registry().items()],
                      value='random_forest',clearable=False)],className='model-selector')],className='section-heading'),
        html.Div(id='model-description',className='model-description'),
        disclosure('Configuración entrenada · CO y NOx',html.Div(id='model-configuration')),
        html.Div([html.Section([html.H3('Condiciones de operación'),html.Div(inputs,className='sensor-grid'),
                                html.Div([html.Button('Calcular estimación',id='predict-button',n_clicks=0,className='primary-button'),
                                          html.Button('Registro de ejemplo',id='reset-inputs',n_clicks=0,className='text-button')],className='command-row'),
                                html.P('El registro inicial es una observación real de desarrollo, no una combinación artificial de medianas.',className='caption')]),
                  html.Section([html.H3('Resultado'),html.Div([metric('CO (mg/m³)','—'),metric('NOx (mg/m³)','—')],
                                  id='prediction-result',className='prediction-values'),
                                html.Div(id='prediction-feedback',role='status',className='prediction-feedback'),
                                notice('No certifica cumplimiento ambiental. Dentro del rango de cada sensor tampoco se garantiza que la combinación esté representada.',kind='warning')],
                               className='prediction-panel')],className='prediction-grid'),
        disclosure('Predicciones por lote · CSV',[
            html.Div([dcc.Upload(id='batch-upload',children=html.Button('Seleccionar CSV',className='secondary-button'),
                                 accept='.csv',multiple=False,max_size=MAX_BYTES),
                      html.Button('Descargar plantilla',id='template-button',n_clicks=0,className='text-button'),
                      html.Button('Descargar estimaciones',id='batch-download-button',n_clicks=0,disabled=True,className='secondary-button')],className='command-row'),
            html.P('Sensores: AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY y CDP. CSV UTF-8, máximo 2 MB y 2.000 filas. Los archivos no se almacenan en el servidor.',className='caption'),
            html.Div(id='batch-feedback',role='status'),dcc.Store(id='batch-store'),
            dcc.Download(id='template-download'),dcc.Download(id='batch-download')]),
        html.Div([html.Div([html.H3('Desempeño del modelo'),html.Span('Métricas históricas · protocolo fijo',className='subtle')]),
                  dcc.RadioItems(id='model-target',options=[{'label':'CO','value':'CO'},{'label':'NOx','value':'NOX'}],value='CO',className='segmented')],className='section-heading results-heading'),
        html.Div(id='model-metrics',className='metric-strip'),html.P(id='model-ci',className='caption'),
        dcc.Loading(html.Div([graph('model-stage-chart'),graph('model-prediction-chart')],className='chart-grid'),type='circle',color=f.TEAL),
        html.P('La dispersión presenta hasta 2.200 puntos; las métricas utilizan los 7.411 registros de prueba. CV externa: media ± desviación de cinco pliegues anidados, no un intervalo de confianza.',className='caption'),
        disclosure('Comparación entre modelos · entrenamiento, validación y prueba',html.Div(id='model-comparison')),
        disclosure('Optimización · 28 procedimientos por emisión',[
            html.Div(id='model-optimization'),html.P('Tiempo histórico total por procedimiento de búsqueda y evaluación, no latencia de inferencia en Render. Los presupuestos de los optimizadores son distintos; estas cifras no demuestran superioridad general de un método.',className='caption')]),
        disclosure('Importancia de variables · modelo y emisión seleccionados',[
            dcc.Loading(graph('model-importance-chart'),type='circle',color=f.TEAL),
            html.P(id='model-importance-note',className='caption'),
            disclosure('Valores de importancia por sensor',html.Div(id='model-importance-table'))]),
        disclosure('Diagnóstico de errores',[
            graph('model-residual-chart'),html.Div(id='model-residual-summary'),
            html.P('Residuo = observado − estimado. Las predicciones negativas se conservan sin recorte y se señalan: no tienen interpretación como concentración física.',className='caption')]),
        notice(selection_summary())
    ])


def selection_summary():
    selected=results().loc[lambda d:d.seleccionado_por_CV].set_index('objetivo')
    n=f"{next(iter(registry().values()))['n_ajuste']:,}".replace(',','.')
    return (f"Selección por MSE externo: {selected.loc['CO','modelo']} para CO y "
            f"{selected.loc['NOX','modelo']} para NOx. Cada familia conserva su mejor estrategia "
            f"de búsqueda y un regresor por emisión, ajustados con {n} registros. "
            'La prueba no decide la selección. Las diferencias puntuales no demuestran superioridad estadística.')


def model_results(model_id, target):
    definition = registry()[model_id]
    family = definition['nombre']
    row = results().loc[(results().objetivo == target) & (results().modelo == family)].iloc[0]
    metrics = [metric('R² entrenamiento',number(row.R2_train)),metric('R² CV externa',number(row.R2_CV_promedio)),
               metric('R² prueba',number(row.R2_test)),metric('RMSE prueba',number(row.RMSE_test),'mg/m³'),
               metric('MAE prueba',number(row.MAE_test),'mg/m³')]
    ci = intervals().loc[(intervals().objetivo == target) & (intervals().modelo == family) & (intervals().metrica == 'R2')].iloc[0]
    ci_text = f"R² de prueba · IC95 % bootstrap por bloques: [{number(ci.IC95_inf)}; {number(ci.IC95_sup)}]. Es incertidumbre de la métrica, no un intervalo para una predicción individual."
    comparison = results().loc[results().objetivo == target].sort_values('MSE_CV_externo')[
        ['modelo','optimizador','R2_train','R2_CV_promedio','R2_CV_std','R2_OOF','R2_test',
         'RMSE_CV_media','RMSE_CV_std','RMSE_test','MAE_test','seleccionado_por_CV']].copy()
    comparison['seleccionado_por_CV'] = comparison.seleccionado_por_CV.map({True:'Sí',False:'No'})
    p_text = lambda p:'<1e-300' if p==0 else f'{p:.3e}'
    diagnostics = pd.DataFrame([{'White · p nominal':p_text(row.p_White),
        'BDS · p nominal':p_text(row.p_BDS),'Ljung–Box · p nominal':p_text(row.p_Ljung_Box),
        'Bloque serial':f'{int(row.anio_bloque_serial)} / {int(row.bloque_serial)}',
        'Registros del bloque':int(row.n_serial)}])
    summary = html.Div([
        html.P(f"Sesgo de prueba (estimado − observado): {number(row.sesgo_test)} mg/m³. Predicciones negativas: {int(row.negativos_test)}."),
        table(diagnostics),html.P('Diagnósticos exploratorios: White usa toda la prueba; BDS y Ljung–Box usan un único bloque continuo, no los 7.411 registros concatenados. Los valores p no miden precisión predictiva y no están corregidos por multiplicidad.',className='caption')])
    selected = definition['respuestas'][target]['seleccionado_por_cv']
    description = [html.Strong(family),html.Span(' · '+definition['descripcion']),
                   html.Span(f'Seleccionado para {target} por CV externa' if selected else f'Comparador para {target}',className='status-badge')]
    configuration = pd.DataFrame([{'Emisión':t,'Optimizador':r['optimizador'],
        'Hiperparámetros':str(r['parametros']),'Registros de ajuste':definition['n_ajuste']}
        for t,r in definition['respuestas'].items()])
    optimization = optimization_results().loc[lambda d:d.objetivo == target,
        ['modelo','optimizador','RMSE_CV_media','RMSE_CV_std','MAE_CV_media','MAE_CV_std',
         'R2_CV_media','R2_CV_std','RMSE_test','MAE_test','segundos_total','evaluaciones','mejor_en_familia']].copy()
    optimization['mejor_en_familia'] = optimization.mejor_en_familia.map({True:'Sí',False:'No'})
    return (description,metrics,ci_text,f.stage_scores(row),f.observed_predictions(predictions(),target,family),
            table(comparison,{'modelo':'Modelo','optimizador':'Optimizador','R2_train':'R² ajuste','R2_CV_promedio':'R² CV externa','R2_CV_std':'Desv. CV externa',
                              'R2_OOF':'R² OOF','R2_test':'R² prueba','RMSE_test':'RMSE prueba','MAE_test':'MAE prueba',
                              'RMSE_CV_media':'RMSE CV externa','RMSE_CV_std':'Desv. RMSE CV','seleccionado_por_CV':'Selección CV'}),
            f.residual_histogram(predictions(),target,family),summary,table(configuration),
            table(optimization,{'modelo':'Modelo','optimizador':'Optimizador','RMSE_CV_media':'RMSE externo',
                'RMSE_CV_std':'Desv. RMSE','MAE_CV_media':'MAE externo','MAE_CV_std':'Desv. MAE',
                'R2_CV_media':'R² externo','R2_CV_std':'Desv. R²','RMSE_test':'RMSE prueba',
                'MAE_test':'MAE prueba','segundos_total':'Tiempo (s)','evaluaciones':'Evaluaciones',
                'mejor_en_familia':'Mejor estrategia en familia'}))


def estimate(model_id, values):
    frame = pd.DataFrame([dict(zip([s['id'] for s in sensors()],values))])
    y = predict(model_id,frame)
    outside = range_warnings(frame)
    notes = []
    if outside:
        notes.append(notice('Fuera del rango de desarrollo: '+', '.join(outside)+'. La estimación es extrapolación y requiere cautela.',kind='warning'))
    if y.lt(0).any(axis=None):
        notes.append(notice('Hay una estimación negativa. No se recorta; revise las condiciones y las limitaciones del modelo.',kind='warning'))
    return [metric('CO (mg/m³)',number(y.CO.iloc[0])),metric('NOx (mg/m³)',number(y.NOX.iloc[0]))],notes


def importance_results(model_id, target):
    importance = feature_importance(model_id,target)
    if importance is None:
        return f.empty('Importancia no disponible para este modelo.'),'El adaptador del modelo no proporciona esta interpretación.',''
    note = ('Importancia precalculada sobre el pipeline entrenado. Cada sensor se permuta individualmente en los 7.411 registros originales de prueba, '
            'con 10 repeticiones y semilla 42. Un mayor aumento de MSE indica mayor dependencia predictiva; '
            'valores negativos indican que la permutación redujo el error. Las barras de error muestran '
            'la desviación entre repeticiones, no un intervalo de confianza. Los sensores correlacionados '
            'pueden compartir importancia y la permutación puede generar combinaciones poco realistas. '
            'No representa causalidad ni la dirección del efecto. Es un diagnóstico posterior a la evaluación: '
            'no se utiliza para seleccionar variables ni modelos con la prueba.')
    return (f.feature_importance(importance,target),note,
            table(importance,{'sensor':'Sensor','aumento_MSE':'Aumento de MSE ((mg/m³)²)',
                              'desviacion':'Desviación entre repeticiones'}))


def parse_batch(contents, model_id):
    if not isinstance(contents,str) or len(contents) > MAX_BYTES*1.4:
        raise ValueError('El archivo supera el límite de 2 MB.')
    try:
        raw = base64.b64decode(contents.split(',',1)[1],validate=True)
        if len(raw) > MAX_BYTES:
            raise ValueError('El archivo supera el límite de 2 MB.')
        df = pd.read_csv(StringIO(raw.decode('utf-8-sig')),nrows=MAX_ROWS+1)
    except (binascii.Error,UnicodeError,IndexError,pd.errors.ParserError,pd.errors.EmptyDataError) as e:
        raise ValueError('No se pudo leer un CSV UTF-8 válido.') from e
    if len(df) > MAX_ROWS:
        raise ValueError('El archivo supera el límite de 2.000 filas.')
    df = df[[s['id'] for s in sensors()]] if set(s['id'] for s in sensors()).issubset(df.columns) else df
    estimates = predict(model_id,df)
    output = df.copy()
    for target in ['CO','NOX']:
        output[f'{target}_predicho'] = estimates[target]
    output['modelo'] = registry()[model_id]['nombre']
    warnings = range_warnings(df.apply(pd.to_numeric,errors='coerce'))
    negative = int(estimates.lt(0).any(axis=1).sum())
    return output,warnings,negative


def register(app):
    app.callback(Output('model-importance-chart','figure'),Output('model-importance-note','children'),
                 Output('model-importance-table','children'),Input('model-select','value'),
                 Input('model-target','value'))(importance_results)
    app.callback(Output('model-description','children'),Output('model-metrics','children'),Output('model-ci','children'),
                 Output('model-stage-chart','figure'),Output('model-prediction-chart','figure'),
                 Output('model-comparison','children'),Output('model-residual-chart','figure'),Output('model-residual-summary','children'),
                 Output('model-configuration','children'),Output('model-optimization','children'),
                 Input('model-select','value'),Input('model-target','value'))(model_results)

    @app.callback([Output(f"sensor-{s['id']}",'value') for s in sensors()],Input('reset-inputs','n_clicks'),prevent_initial_call=True)
    def reset(_):
        return [s['ejemplo'] for s in sensors()]

    @app.callback(Output('prediction-result','children'),Output('prediction-feedback','children'),
                  Input('predict-button','n_clicks'),Input('model-select','value'),Input('reset-inputs','n_clicks'),
                  *[Input(f"sensor-{s['id']}",'value') for s in sensors()])
    def prediction_callback(clicks,model_id,reset_count,*values):
        blank = [metric('CO (mg/m³)','—'),metric('NOx (mg/m³)','—')]
        if not clicks or ctx.triggered_id in ['model-select','reset-inputs',None]:
            return blank,html.P('Estimación pendiente.',className='subtle')
        try:
            return estimate(model_id,values)
        except (ValueError,KeyError,TypeError) as e:
            return blank,notice(str(e),kind='error')

    @app.callback(Output('batch-store','data'),Output('batch-feedback','children'),Output('batch-download-button','disabled'),
                  Input('batch-upload','contents'),Input('model-select','value'))
    def batch_callback(contents,model_id):
        if not contents:
            return None,'',True
        try:
            output,outside,negative = parse_batch(contents,model_id)
            messages = [html.P(f"{len(output):,} registros estimados con {registry()[model_id]['nombre']}.")]
            if outside:
                messages.append(notice('Extrapolación en al menos una fila: '+', '.join(outside)+'.',kind='warning'))
            if negative:
                messages.append(notice(f'{negative} registros con predicción negativa, sin recorte.',kind='warning'))
            messages.append(table(output.head(5)))
            return output.to_csv(index=False),messages,False
        except (ValueError,KeyError,TypeError) as e:
            return None,notice(str(e),kind='error'),True

    @app.callback(Output('template-download','data'),Input('template-button','n_clicks'),prevent_initial_call=True)
    def template(_):
        return dcc.send_data_frame(sample_frame().to_csv,'plantilla_sensores.csv',index=False)

    @app.callback(Output('batch-download','data'),Input('batch-download-button','n_clicks'),State('batch-store','data'),prevent_initial_call=True)
    def download(_,csv):
        return {'content':csv,'filename':'estimaciones_CO_NOx.csv','type':'text/csv'} if csv else no_update
