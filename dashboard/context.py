"""Pestana de contexto cientifico, diccionario y bibliografia."""
import json
from urllib.parse import urlparse
import pandas as pd
from dash import html
from .components import disclosure,graph,metric,notice,table
from .figures import partition_chart
from .repository import DATA,bibliography,manifest,sensors


def layout():
    diccionario = pd.DataFrame([{'Variable':s['id'],'Significado':s['nombre'],
                                'Unidad':s['unidad'],'Papel':'Predictor continuo'} for s in sensors()]
                              + [{'Variable':'CO','Significado':'Monóxido de carbono','Unidad':'mg/m³','Papel':'Objetivo'},
                                 {'Variable':'NOX','Significado':'Óxidos de nitrógeno','Unidad':'mg/m³','Papel':'Objetivo'},
                                 {'Variable':'anio / posición / bloque','Significado':'Controles de descripción y partición',
                                  'Unidad':'Año / registro','Papel':'No son predictores'}])
    literature = json.loads((DATA/'literatura.json').read_text(encoding='utf-8'))
    literature_rows = []
    for p in literature:
        literature_rows.append(html.Tr([html.Td(html.A(p['estudio'],href=p['enlace'],target='_blank',rel='noopener noreferrer')),
                                       html.Td(p['modelos']),html.Td(p['resultado']),html.Td(p['validacion'])]))
    corpus = []
    for r in bibliography().itertuples(index=False):
        enlace = str(r.enlace)
        titulo = html.A(str(r.titulo),href=enlace,target='_blank',rel='noopener noreferrer') if urlparse(enlace).scheme == 'https' else str(r.titulo)
        corpus.append(html.Tr([html.Td(str(r.anio)),html.Td(titulo),html.Td(str(r.revista))]))
    return html.Div([
        html.Div([html.Div([html.P('01 / CONTEXTO',className='eyebrow'),html.H2('El problema y su alcance'),
                            html.P('Estimación de CO y NOx a partir de condiciones ambientales y operativas de una turbina de gas.',className='lead')]),
                  html.A('Fuente original · UCI ↗',href='https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set',
                         target='_blank',rel='noopener noreferrer',className='source-link')],className='section-heading'),
        html.Div([metric('Observaciones','36.733'),metric('Sensores','9'),metric('Periodo','2011–2015')],className='metric-strip'),
        html.Div([
            html.Section([html.H3('Pregunta de investigación'),html.P('¿En qué medida los sensores permiten estimar las concentraciones de emisiones en condiciones de operación representadas?'),
                          html.H3('Relevancia industrial'),html.P('Una estimación basada en sensores puede apoyar el análisis del desempeño y la identificación de condiciones asociadas con mayores emisiones. No sustituye medición, control ambiental ni certificación.'),
                          html.H3('Unidad de observación'),html.P('Agregaciones horarias publicadas por año. La posición conserva el orden original, pero no es una fecha verificada. Concentraciones en mg/m³; nueve predictores continuos.')]),
            html.Section([html.H3('Metodología adoptada'),html.Ol([
                html.Li([html.Strong('Separar bloques'),html.Span('168 registros por bloque, sin cruzar años.')]),
                html.Li([html.Strong('Reservar prueba'),html.Span('≈20% de bloques, balanceados por año; purga de 24 registros.')]),
                html.Li([html.Strong('Seleccionar por CV'),html.Span('5 pliegues de bloques; escalado dentro de cada ajuste.')]),
                html.Li([html.Strong('Evaluar y guardar'),html.Span('Ajuste final solo en desarrollo; prueba no decide la familia.')])],className='method-list'),
                          graph('context-partition',partition_chart())])],className='two-columns context-columns'),
        notice('La partición mezcla bloques de 2011–2015: estima condiciones representadas, no generalización a años futuros.'),
        disclosure('Diccionario de datos',[
            table(diccionario),html.P('La disponibilidad de los sensores al momento de la estimación debe verificarse en operación. Los dos objetivos no se usan como entradas.',className='caption')]),
        disclosure('Validación, calidad y límites',[
            html.P('Desarrollo: 27.570 registros. Prueba: 7.411. Purga de prueba: 1.752. Semilla 42. Los vecinos excluidos nunca se incorporan al ajuste final.'),
            html.P('La purga reduce proximidad entre ajuste y evaluación, pero 24 registros no garantizan independencia. LOYO evalúa transferencia a un año no representado: es otra pregunta válida, no un diseño incorrecto por producir menor R².'),
            html.P('No se retiran extremos solamente por IQR en este proyecto. Los faltantes, duplicados y valores físicamente dudosos se auditan en el notebook. Las diferencias anuales y las emisiones extremas condicionan los errores.'),
            html.P('CV no anidada: las puntuaciones están condicionadas a hiperparámetros seleccionados. La prueba histórica ya fue inspeccionada; se requiere confirmación externa antes de un uso operativo.'),
            html.P('R² mide variación explicada. RMSE penaliza especialmente errores grandes; MAE expresa el error absoluto medio. RMSE y MAE están en mg/m³. MAPE no se prioriza para CO por valores próximos a cero.')]),
        html.Section([html.Div([html.H3('Literatura: lo que ya se ha probado'),
                               html.Span('Síntesis del notebook',className='subtle')],className='section-heading'),
                      html.P('Regresión, ensambles de árboles, modelos simbólicos e híbridos aparecen como antecedentes. Sus cifras no son comparables directamente cuando cambian datos, unidades o particiones.'),
                      html.Div([html.Div([html.Strong('Regresión y ensambles'),html.P('Comparadores consolidados, no una contribución original por sí mismos.')]),
                                html.Div([html.Strong('Desplazamiento entre años'),html.P('La validación debe responder al escenario de generalización previsto.')]),
                                html.Div([html.Strong('Originalidad por comprobar'),html.P('La propuesta propia sigue pendiente; el dashboard presenta líneas base.')])],className='literature-summary'),
                      disclosure('Estudios, modelos y desempeño reportado',[
                          html.Div(html.Table([html.Thead(html.Tr([html.Th(t) for t in ['Estudio','Modelos / enfoque','Resultado reportado','Validación / cautela']])),
                                               html.Tbody(literature_rows)]),className='table-scroll'),
                          notice('Cifras tomadas de la revisión existente. Cuando proceden del resumen bibliográfico o el protocolo no está verificado, se indica expresamente. No se han reproducido esos experimentos.')]),
                      disclosure('Catálogo bibliográfico · 50 registros',html.Div(html.Table([
                          html.Thead(html.Tr([html.Th('Año'),html.Th('Artículo'),html.Th('Publicación')])),html.Tbody(corpus)]),className='table-scroll'))],className='literature-section')
    ])
