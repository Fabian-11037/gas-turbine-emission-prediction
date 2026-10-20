"""Pestana de contexto cientifico, diccionario y bibliografia."""
import json
from urllib.parse import urlparse
import pandas as pd
from dash import html
from .components import disclosure,graph,metric,notice,table
from .figures import partition_chart
from .repository import DATA,bibliography,manifest,results,sensors


def layout():
    diccionario = pd.DataFrame([{'Variable':s['id'],'Significado':s['nombre'],
                                'Unidad':s['unidad'],'Papel':'Entrada numérica'} for s in sensors()]
                              + [{'Variable':'CO','Significado':'Monóxido de carbono','Unidad':'mg/m³','Papel':'Objetivo'},
                                 {'Variable':'NOX','Significado':'Óxidos de nitrógeno','Unidad':'mg/m³','Papel':'Objetivo'},
                                 {'Variable':'anio / posición / bloque','Significado':'Controles de descripción y partición',
                                  'Unidad':'Año / registro','Papel':'No son predictores'}])
    literature = json.loads((DATA/'literatura.json').read_text(encoding='utf-8'))
    comparison = json.loads((DATA/'literatura_metodologia.json').read_text(encoding='utf-8'))
    reference = results().loc[lambda d:d.seleccionado_por_CV,
                              ['objetivo','modelo','R2_test','RMSE_test']]
    interpretation = [html.Div([html.H4(html.A(item['titulo'],href=item['enlace'],target='_blank',rel='noopener noreferrer')),html.P(item['diferencia']),
                                html.P(item['interpretacion'])]) for item in comparison['dimensiones']]
    literature_rows = []
    for p in literature:
        literature_rows.append(html.Tr([html.Td(html.A(p['estudio'],href=p['enlace'],target='_blank',rel='noopener noreferrer')),
                                       html.Td(p['modelos']),html.Td(p['resultado']),
                                       html.Td([html.P(p['validacion']),
                                                html.P(p['comparacion'],className='caption')]
                                               if p.get('comparacion') else p['validacion'])]))
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
            html.Section([html.H3('Origen del problema'),html.P('Los datos provienen de una turbina de una planta eléctrica en Turquía. La turbina transforma la energía de la combustión en movimiento para generar electricidad; ese proceso también produce CO y NOx.'),
                          html.H3('Necesidad del negocio'),html.P('El operador necesita conocer las emisiones mientras mantiene la producción de energía. Estimarlas con sensores de operación puede complementar su seguimiento y ayudar a reconocer condiciones de mayor emisión.'),
                          html.H3('Pregunta de investigación'),html.P('¿Qué tan bien podemos estimar CO y NOx con nueve sensores ambientales y operativos? Se estima la concentración del registro actual, no una emisión futura.'),
                          html.P(['Contexto técnico: ',html.A('generación con turbinas',href='https://www.energy.gov/hgeo/how-gas-turbine-power-plants-work',target='_blank',rel='noopener noreferrer'),'.'],className='caption')]),
            html.Section([html.H3('Metodología adoptada'),html.Ol([
                html.Li([html.Strong('Agrupar registros próximos'),html.Span('Bloques de 168 observaciones, separados por año.')]),
                html.Li([html.Strong('Reservar cerca del 20% para prueba'),html.Span('Se apartan también 24 vecinos de cada lado para reducir semejanzas con el entrenamiento.')]),
                html.Li([html.Strong('Validar antes de elegir'),html.Span('Cinco grupos evalúan el desempeño; dentro de cada uno, tres grupos permiten ajustar las configuraciones.')]),
                html.Li([html.Strong('Guardar los modelos'),html.Span('Se entrenan con desarrollo y se evalúan con la prueba reservada.')])],className='method-list'),
                          graph('context-partition',partition_chart())])],className='two-columns context-columns'),
        notice('El estudio representa condiciones de 2011–2015 de una turbina. Las estimaciones complementan las mediciones; no certifican cumplimiento ambiental.'),
        disclosure('Diccionario de datos',[
            table(diccionario),html.P('Cada fila resume una hora de operación. mg/m³ significa miligramos de contaminante por metro cúbico de gas. CO y NOx son los valores que se busca estimar, no entradas del modelo.',className='caption')]),
        disclosure('Validación, calidad y límites',[
            html.P('Se usan 27.570 registros para desarrollo, 7.411 para prueba y se apartan 1.752 vecinos de la prueba. Esta separación, llamada purga, reduce la cercanía entre los datos de entrenamiento y evaluación.'),
            html.P('La validación cruzada repite el entrenamiento dejando un grupo fuera cada vez. La versión anidada separa la elección de configuraciones de su evaluación. El escalado, que pone los sensores en unidades comparables, se calcula solo con el entrenamiento de cada grupo.'),
            html.P('Se comparan siete familias conocidas y cuatro métodos para buscar configuraciones: 56 procedimientos en total. El kernel operativo añade dos variables calculadas y combina Extra Trees, SVR y Ridge; se elige para ambas emisiones por su menor error de validación.'),
            html.P('Se conservan los extremos porque pueden representar episodios reales. Los archivos mantienen el orden de los registros, pero no fechas exactas. Dejar un año completo fuera, o LOYO, mediría la capacidad de predecir un periodo desconocido; aquí se evalúan condiciones representadas en los cinco años.'),
            html.P('Antes de usarlo en otra planta o periodo se necesita una evaluación con datos nuevos. El proyecto no cuantifica ahorro económico ni reducción efectiva de emisiones.')]),
        html.Section([html.Div([html.H3('Literatura: lo que ya se ha probado'),
                               html.Span('Síntesis del notebook',className='subtle')],className='section-heading'),
                      html.P('Los estudios usan distintos datos, sensores y divisiones de entrenamiento y prueba. Sus resultados sirven como referencia, no como una clasificación directamente comparable.'),
                      html.Div([html.Div([html.Strong('Regresión y ensambles'),html.P('Comparadores consolidados, no una contribución original por sí mismos.')]),
                                html.Div([html.Strong('Protocolos distintos'),html.P('Algunos filtran extremos, separan periodos o incorporan lecturas anteriores.')]),
                                html.Div([html.Strong('Propuesta operativa'),html.P('Combina variables operativas y modelos complementarios; esa integración no se identificó en la revisión.')])],className='literature-summary'),
                      disclosure('Estudios, modelos y desempeño reportado',[
                          html.Div(html.Table([html.Thead(html.Tr([html.Th(t) for t in ['Estudio','Modelos / enfoque','Resultado reportado','Validación / cautela']])),
                                               html.Tbody(literature_rows)]),className='table-scroll'),
                          notice('Se revisaron metodología y tablas completas de Kochueva, Sinap, Jhinaoui y Aslan. Los demás casos indican cuándo la información procede de un resumen.')],identifier='literature-studies'),
                      disclosure('Diferencias metodológicas y posibles explicaciones',[
                          html.H4('Referencia vigente del proyecto: prueba por bloques'),
                          table(reference,labels={'objetivo':'Emisión','modelo':'Modelo',
                                                  'R2_test':'R² prueba','RMSE_test':'RMSE prueba · mg/m³'},digits=4),
                          html.P(comparison['alcance'],className='caption'),
                          *interpretation,
                          notice(comparison['reproduccion'])],identifier='literature-methods'),
                      disclosure('Catálogo bibliográfico · 50 registros',html.Div(html.Table([
                          html.Thead(html.Tr([html.Th('Año'),html.Th('Artículo'),html.Th('Publicación')])),html.Tbody(corpus)]),className='table-scroll'))],className='literature-section')
    ])
