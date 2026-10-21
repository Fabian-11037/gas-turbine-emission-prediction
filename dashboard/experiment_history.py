"""Read-only, portable history of completed experiments."""
from functools import lru_cache

import pandas as pd
from dash import Input, Output, dcc, html

from .components import table
from .repository import DATA


@lru_cache(maxsize=1)
def results():
    return pd.read_csv(DATA/'historial_modelos.csv')


def render(search='', target='CO'):
    frame = results()
    query = (search or '').strip().casefold()
    if query:
        text = frame[['modelo', 'variante', 'estado']].fillna('').agg(' '.join, axis=1).str.casefold()
        frame = frame.loc[text.str.contains(query, regex=False)]
    label = 'NOx' if target == 'NOX' else 'CO'
    columns = ['modelo', 'variante', f'R2_train_{target}', f'R2_CV_promedio_{target}',
               f'R2_test_{target}', 'tipo_CV', 'protocolo', 'estado']
    return html.Div([
        html.P(f'{len(frame)} variantes · {label}', className='caption'),
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
               'ni la puntuación de búsqueda de hiperparámetros.'),
        html.Label('Buscar modelo o variante', htmlFor='history-search'),
        dcc.Input(id='history-search', type='search', value='', debounce=True,
                  placeholder='Ej.: Sparse, LightGBM, restringida…', className='history-search'),
        html.Div(render(), id='history-results'),
        html.P('—: resultado final no disponible o emisión no evaluada. '
               'Se consolidan los controles repetidos con resultados idénticos. '
               'En las pruebas de limpieza, el R² se calcula sobre el conjunto de desarrollo original.',
               className='caption')])


def register(app):
    app.callback(Output('history-results', 'children'), Input('history-search', 'value'),
                 Input('model-target', 'value'))(render)
