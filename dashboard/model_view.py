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
from . import kernel_view,learning,explainability,assessment,stability

MAX_ROWS = 2000
MAX_BYTES = 2*1024*1024
MODEL_EXPLANATIONS = {
    'knn':'Estima a partir de registros con sensores parecidos.',
    'ridge':'Combina los sensores de forma lineal y limita pesos excesivos.',
    'lasso':'Modelo lineal que puede reducir a cero el peso de algunos sensores.',
    'decision_tree':'Aplica reglas de decisión según los valores de los sensores.',
    'random_forest':'Promedia muchos árboles entrenados con muestras distintas.',
    'xgboost':'Añade árboles que corrigen los errores de los anteriores.',
    'svr_lineal':'Ajusta una relación lineal tolerando diferencias pequeñas y penalizando las mayores.',
    'kernel_operativo':'Combina Extra Trees y SVR con Ridge, usando dos variables operativas calculadas.'}


def default_model():
    return next(mid for mid, model in registry().items() if model['respuestas']['CO']['seleccionado_por_cv'])


def prediction_loading(children, component_id, message):
    return dcc.Loading(children,id=f'{component_id}-loading',delay_show=100,delay_hide=150,
        target_components={component_id:'children'},
        custom_spinner=html.Div([html.Span(className='prediction-spinner',**{'aria-hidden':'true'}),
                                 html.Span(message)],className='prediction-progress',role='status',
                                **{'aria-live':'polite'}))


def layout():
    inputs = [html.Div([html.Label([html.Strong(s['id']),f" · {s['nombre']} ({s['unidad']})"],htmlFor=f"sensor-{s['id']}"),
                       dcc.Input(id=f"sensor-{s['id']}",type='number',value=s['ejemplo'],debounce=True,
                                 step='any',inputMode='decimal',className='sensor-input'),
                       html.Small(f"Desarrollo: {number(s['min'],2)}–{number(s['max'],2)} {s['unidad']}",className='input-range')],
                      className='sensor-field') for s in sensors()]
    return html.Div([
        html.Div([html.Div([html.P('03 / MODELOS',className='eyebrow'),html.H2('Estimación de emisiones'),
                            html.P('Estimaciones de CO y NOx a partir de los sensores de operación.',className='lead')]),
                  html.Div([html.Label('Modelo',htmlFor='model-select'),dcc.Dropdown(id='model-select',
                      options=[{'label':m['nombre'],'value':mid} for mid,m in registry().items()],
                      value=default_model(),clearable=False)],className='model-selector')],className='section-heading'),
        html.Div(id='model-description',className='model-description'),
        disclosure('Configuración entrenada · CO y NOx',html.Div(id='model-configuration')),
        html.Div([html.Section([html.H3('Condiciones de operación'),html.Div(inputs,className='sensor-grid'),
                                html.Div([html.Button('Calcular estimación',id='predict-button',n_clicks=0,className='primary-button'),
                                          html.Button('Registro de ejemplo',id='reset-inputs',n_clicks=0,className='text-button')],className='command-row'),
                                html.P('El ejemplo corresponde a un registro real de desarrollo.',className='caption')]),
                  html.Section([html.H3('Resultado'),prediction_loading(html.Div([
                                html.Div([metric('CO (mg/m³)','—'),metric('NOx (mg/m³)','—')],
                                  id='prediction-result',className='prediction-values'),
                                html.Div(id='prediction-feedback',role='status',className='prediction-feedback')],
                                className='prediction-output'),'prediction-result','Calculando CO y NOx…'),
                                notice('Estimación de apoyo; no sustituye una medición certificada de emisiones.',kind='warning')],
                               className='prediction-panel')],className='prediction-grid'),
        disclosure('Predicciones por lote · CSV',[
            html.Div([dcc.Upload(id='batch-upload',children=html.Button('Seleccionar CSV',className='secondary-button'),
                                 accept='.csv',multiple=False,max_size=MAX_BYTES),
                      html.Button('Descargar plantilla',id='template-button',n_clicks=0,className='text-button'),
                      html.Button('Descargar estimaciones',id='batch-download-button',n_clicks=0,disabled=True,className='secondary-button')],className='command-row'),
            html.P('Sensores: AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY y CDP. CSV UTF-8, máximo 2 MB y 2.000 filas. Los archivos no se almacenan en el servidor.',className='caption'),
            prediction_loading(html.Div(id='batch-feedback',role='status',className='batch-output'),
                               'batch-feedback','Calculando predicciones del archivo…'),dcc.Store(id='batch-store'),
            dcc.Download(id='template-download'),dcc.Download(id='batch-download')]),
        html.Div([html.Div([html.H3('Desempeño del modelo'),html.Span('Métricas históricas · protocolo fijo',className='subtle')]),
                  dcc.RadioItems(id='model-target',options=[{'label':'CO','value':'CO'},{'label':'NOx','value':'NOX'}],value='CO',className='segmented')],className='section-heading results-heading'),
        html.Div(id='model-metrics',className='metric-strip'),html.P(id='model-ci',className='caption'),
        dcc.Loading(html.Div([graph('model-stage-chart'),graph('model-prediction-chart')],className='chart-grid'),type='circle',color=f.TEAL),
        html.P('CV externa resume cinco grupos de evaluación: se muestra el promedio y su variación. Las métricas de prueba usan 7.411 registros; la figura muestra hasta 2.200.',className='caption'),
        disclosure('Curva de aprendizaje · modelo y emisión seleccionados',[
            html.Div([html.H3('Aprendizaje con más datos'),
                dcc.RadioItems(id='learning-metric',options=[{'label':'R²','value':'R2'},
                    {'label':'RMSE','value':'RMSE'}],value='R2',className='segmented')],className='section-heading'),
            dcc.Loading(graph('model-learning-chart'),type='circle',color=f.TEAL),
            html.P(id='model-learning-note',className='caption'),
            html.P('Se añaden bloques completos de cada año, conservando el margen de 24 registros. '
                   'La configuración se mantiene fija dentro de cada pliegue; los escalados y el combinador se vuelven a ajustar solo con su entrenamiento. '
                   'La prueba no participa en estas curvas.',className='caption'),
            disclosure('Valores por tamaño de entrenamiento',html.Div(id='model-learning-table'))]),
        disclosure('Significado de los indicadores',[
            html.P('R² resume cuánto mejora el modelo frente a estimar siempre el promedio: 1 es un ajuste perfecto y un valor negativo es peor que esa referencia. No es un porcentaje de predicciones correctas.'),
            html.P('MAE es el tamaño medio del error. RMSE da más peso a los errores grandes. Ambos se expresan en mg/m³; un valor menor es mejor. MSE es el promedio de errores al cuadrado y se utiliza para elegir configuraciones.'),
            html.P('Entrenamiento mide el ajuste sobre datos usados para aprender. OOF reúne las predicciones obtenidas cuando cada registro queda fuera de su entrenamiento. El R² CV promedio calcula primero el R² de cada grupo y luego los promedia; el R² OOF se calcula una sola vez con todas esas predicciones.'),
            html.P('La prueba evalúa el modelo final con registros reservados. Una caída marcada respecto al entrenamiento puede indicar sobreajuste: aprender detalles que no se mantienen en otros datos.')]),
        kernel_view.layout(),
        disclosure('Comparación entre modelos · entrenamiento, validación y prueba',html.Div(id='model-comparison')),
        assessment.layout(),
        stability.layout(),
        disclosure('Optimización de comparadores · 28 procedimientos por emisión',[
            html.Div(id='model-optimization'),html.P('Se comparan búsquedas de configuraciones: exhaustiva (Grid), aleatoria, guiada por resultados previos (Bayesiana) y evolutiva (Genética). El tiempo corresponde al entrenamiento y la evaluación, no a una predicción.',className='caption')]),
        disclosure('Importancia de variables · modelo y emisión seleccionados',[
            dcc.Loading(graph('model-importance-chart'),type='circle',color=f.TEAL),
            html.P(id='model-importance-note',className='caption'),
            disclosure('Valores de importancia por sensor',html.Div(id='model-importance-table'))]),
        explainability.layout(),
        disclosure('Diagnóstico de errores',[
            graph('model-residual-chart'),html.Div(id='model-residual-summary'),
            html.P('Residuo = observado − estimado. Las predicciones negativas se conservan sin recorte y se señalan: no tienen interpretación como concentración física.',className='caption')]),
        notice(selection_summary())
    ])


def selection_summary():
    selected=results().loc[lambda d:d.seleccionado_por_CV].set_index('objetivo')
    n=f"{next(iter(registry().values()))['n_ajuste']:,}".replace(',','.')
    return (f"Modelo elegido: {selected.loc['CO','modelo']} para CO y "
            f"{selected.loc['NOX','modelo']} para NOx, por su menor error de validación. "
            f"Cada emisión tiene un modelo entrenado con {n} registros de desarrollo.")


def model_results(model_id, target):
    definition = registry()[model_id]
    family = definition['nombre']
    row = results().loc[(results().objetivo == target) & (results().modelo == family)].iloc[0]
    metrics = [metric('R² entrenamiento',number(row.R2_train)),metric('R² CV externa',number(row.R2_CV_promedio)),
               metric('R² prueba',number(row.R2_test)),metric('RMSE prueba',number(row.RMSE_test),'mg/m³'),
               metric('MAE prueba',number(row.MAE_test),'mg/m³')]
    ci = intervals().loc[(intervals().objetivo == target) & (intervals().modelo == family) & (intervals().metrica == 'R2')].iloc[0]
    ci_text = f"R² de prueba · intervalo al 95%: [{number(ci.IC95_inf)}; {number(ci.IC95_sup)}]. Se obtiene remuestreando bloques y resume la incertidumbre del desempeño, no de una predicción individual."
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
        table(diagnostics),html.P('White revisa si la dispersión del error cambia con los sensores; BDS y Ljung–Box buscan dependencia entre errores. Los dos últimos usan un bloque continuo. Sus valores p son exploratorios y no miden precisión predictiva.',className='caption')])
    selected = definition['respuestas'][target]['seleccionado_por_cv']
    description = [html.Strong(family),html.Span(' · '+MODEL_EXPLANATIONS[model_id]),
                   html.Span(f'Seleccionado para {target} por CV externa' if selected else f'Comparador para {target}',className='status-badge')]
    configuration = pd.DataFrame([{'Emisión':t,'Optimizador':r['optimizador'],
        'Configuración (hiperparámetros)':str(r['parametros']),'Registros de ajuste':definition['n_ajuste']}
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
    if importance is None or importance.empty:
        return f.empty('Importancia no disponible para este modelo.'),'El adaptador del modelo no proporciona esta interpretación.',''
    note = ('Se mezclan los valores de un sensor y se observa cuánto aumenta el error, en 10 repeticiones sobre la prueba. '
            'Un aumento mayor indica que el modelo depende más de ese sensor; las barras muestran la variación entre repeticiones. '
            'Sensores relacionados pueden compartir importancia. No es una medida causal ni se usó para elegir variables.')
    if model_id == 'kernel_operativo':
        note += ' En el kernel se recalculan las dos variables derivadas después de mezclar cada sensor.'
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
    @app.callback(Output('model-importance-chart','style'), Input('model-importance-chart','figure'))
    def importance_height(figure):
        return {'height': str((figure or {}).get('layout', {}).get('height', 350))+'px', 'width': '100%'}

    assessment.register(app)
    explainability.register(app)
    kernel_view.register(app)
    stability.register(app)
    app.callback(Output('model-learning-chart','figure'),Output('model-learning-note','children'),
                 Output('model-learning-table','children'),Input('model-select','value'),
                 Input('model-target','value'),Input('learning-metric','value'))(learning.render)
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
                  *[Input(f"sensor-{s['id']}",'value') for s in sensors()],
                  running=[(Output('predict-button','disabled'),True,False),
                           (Output('predict-button','children'),'Calculando…','Calcular estimación'),
                           (Output('reset-inputs','disabled'),True,False),
                           (Output('model-select','disabled'),True,False),
                           *[(Output(f"sensor-{s['id']}",'disabled'),True,False) for s in sensors()]])
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
