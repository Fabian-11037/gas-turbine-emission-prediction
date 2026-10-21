"""Read-only, portable history of completed experiments."""
from functools import lru_cache

import pandas as pd
from dash import Input, Output, dcc, html

from .components import table, disclosure
from .repository import DATA

PROTOCOLS = {'bloques':'Bloques · protocolo principal', 'aleatorio':'80/20 aleatorio',
             'estratificado':'80/20 estratificado',
             'comun_aleatoria':'Bloques externos · selección interna aleatoria'}


@lru_cache(maxsize=1)
def results():
    return pd.read_csv(DATA/'historial_modelos.csv')


def render(search='', target='CO', protocol='todos'):
    frame = results()
    if protocol != 'todos':
        frame = frame.loc[frame.grupo_validacion.eq(protocol)]
    query = (search or '').strip().casefold()
    if query:
        text = frame[['modelo', 'variante', 'estado', 'protocolo']].fillna('').agg(' '.join, axis=1).str.casefold()
        frame = frame.loc[text.str.contains(query, regex=False)]
    label = 'NOx' if target == 'NOX' else 'CO'
    columns = ['modelo', 'variante', f'R2_train_{target}', f'R2_CV_promedio_{target}',
               f'R2_test_{target}', 'tipo_CV', 'protocolo', 'estado']
    return html.Div([
        html.P(f"{len(frame)} {'variante' if len(frame)==1 else 'variantes'} · {label}", className='caption'),
        table(frame[columns], {'modelo': 'Modelo', 'variante': 'Experimento / variante',
            f'R2_train_{target}': f'R² entrenamiento · {label}',
            f'R2_CV_promedio_{target}': f'R² CV · {label}', f'R2_test_{target}': f'R² prueba · {label}',
            'tipo_CV': 'Tipo de R² CV', 'protocolo': 'Validación',
            'estado': 'Estado'}, digits=4) if len(frame) else html.P('No hay coincidencias.')])


def layout():
    return html.Div([
        html.H3('Historial de modelos probados'),
        html.P('Entrenamiento, validación cruzada y prueba de los modelos publicados y las pruebas provisionales, '
               'para la emisión seleccionada. CV es la media del R² de los pliegues externos, no el R² OOF conjunto '
               'ni la puntuación de búsqueda de hiperparámetros. Los resultados se identifican por protocolo; '
               'un R² mayor con otra partición no demuestra por sí solo una mejora del modelo.'),
        html.Div([
            html.Div([html.Label('Buscar modelo o variante', htmlFor='history-search'),
                dcc.Input(id='history-search', type='search', value='', debounce=True,
                    placeholder='Ej.: Sparse, LightGBM, TEY…', className='history-search')]),
            html.Div([html.Label('Validación', htmlFor='history-protocol'),
                dcc.Dropdown(id='history-protocol', value='todos', clearable=False,
                    options=[{'label':'Todas las validaciones','value':'todos'}]+
                            [{'label':label,'value':key} for key,label in PROTOCOLS.items()])])],
            style={'display':'grid','gridTemplateColumns':'repeat(auto-fit,minmax(min(100%,260px),1fr))','gap':'16px'}),
        disclosure('Qué cambia entre las validaciones', [
            html.P('Bloques: evaluación de segmentos operativos separados, con margen de 24 registros.'),
            html.P('80/20 aleatorio: registros repartidos al azar; evalúa condiciones cercanas a las observadas.'),
            html.P('80/20 estratificado: además mantiene proporciones similares de intervalos de la emisión; las particiones son distintas para CO y NOx.'),
            html.P('Comparación controlada: conserva los mismos bloques externos y prueba del protocolo principal, pero cambia la selección interna y el cross-fitting a aleatorios.')]),
        html.Div(render(), id='history-results'),
        html.P('—: resultado final no disponible o emisión no evaluada. '
               'Se consolidan los controles repetidos con resultados idénticos. '
               'En las pruebas de limpieza, el R² se calcula sobre el conjunto de desarrollo original.',
               className='caption')])


def register(app):
    app.callback(Output('history-results', 'children'), Input('history-search', 'value'),
                 Input('model-target', 'value'), Input('history-protocol', 'value'))(render)
