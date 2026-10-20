"""Vista de periodicidad: solo consume evidencias, sin inferencia en Render."""
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import figures as f
from dash import html
from .components import metric, number, table
from .repository import DATA


@lru_cache(maxsize=1)
def evidence():
    return pd.read_csv(DATA/'estacionalidad_pruebas.csv'), pd.read_csv(DATA/'estacionalidad_perfiles.csv.gz')


def render(target, years, population, period):
    results, profiles = evidence()
    selected = results.loc[(results.objetivo == target) & results.anio.isin(years or []) &
        (results.poblacion == population) & (results.periodo == period)]
    if selected.empty:
        return f.empty(), f.empty(), [metric('Registros', '0')], 'Sin datos.', [], []
    principal = selected.loc[selected.HAC_rezagos == 336]
    profile = profiles.loc[(profiles.objetivo == target) & profiles.anio.isin(years) &
        (profiles.poblacion == population) & (profiles.periodo == period)]
    fig = go.Figure()
    phase = np.arange(period)
    for color, row in zip(f.COLORS, principal.itertuples()):
        values = row.beta_seno*np.sin(2*np.pi*phase/period)+row.beta_coseno*np.cos(2*np.pi*phase/period)
        fig.add_trace(go.Scatter(x=phase, y=values, name=str(row.anio), mode='lines',
            line={'color': color, 'width': 2}))
    f.style(fig, f'{target}: patrón cíclico estimado',
        f'Posición dentro de cada grupo de {period}', 'Cambio estimado (mg/m³)')
    fig.add_hline(y=0, line_color=f.GRAY, line_width=.7)
    grid = profile.pivot(index='anio', columns='fase', values='mean')
    heat = go.Figure(go.Heatmap(z=grid.to_numpy(), x=grid.columns, y=grid.index.astype(str),
        colorscale=[[0, f.CORAL], [.5, '#fafbfc'], [1, f.TEAL]], zmid=0,
        colorbar={'title': 'mg/m³'}, hovertemplate='Año %{y}<br>Posicion %{x}<br>Media sin tendencia: %{z:.3f}<extra></extra>'))
    f.style(heat, f'{target}: promedio por posición, sin tendencia',
        f'Posición dentro de cada grupo de {period}', 'Año')
    summaries = [metric('Registros', number(principal.n.sum(), 0)),
        metric('Mayor aporte del ciclo', number(100*principal.R2_parcial.max(),2)+' %', 'variación adicional descrita'),
        metric('Años con evidencia', f'{int(principal.evidencia_005.sum())}/{len(principal)}', 'p Holm < 0,05; HAC 336')]
    rows = selected[['anio', 'periodo', 'n', 'HAC_rezagos', 'amplitud', 'R2_parcial', 'p', 'p_Holm']].copy()
    for column in ['p', 'p_Holm']:
        rows[column] = rows[column].map(lambda value: f'{value:.3e}')
    rows['R2_parcial'] *= 100
    methods = ('Wald evalúa si hay una oscilación regular después de separar la tendencia de cada año. '
        'HAC permite considerar que registros cercanos están relacionados: se usan 336 posiciones y se compara con 168. '
        'Holm ajusta las 40 pruebas de cada población para reducir falsas alarmas. '
        'La amplitud indica cuánto se aparta el ciclo de su centro; el R² parcial mide cuánto añade a la descripción, no a la predicción.')
    tables = [html.P(methods,className='caption'),table(rows, {'anio': 'Año', 'periodo': 'Período', 'n': 'Registros',
        'HAC_rezagos': 'Rezagos HAC', 'amplitud': 'Amplitud (mg/m³)',
        'R2_parcial': 'Variación residual explicada (%)', 'p': 'p Wald', 'p_Holm': 'p Holm'}),
        table(profile[['anio', 'fase', 'mean', 'count']], {'anio': 'Año', 'fase': 'Posición en ciclo',
            'mean': 'Media sin tendencia (mg/m³)', 'count': 'Registros'})]
    conclusion = ('No se detecta evidencia suficiente' if not principal.evidencia_005.any() else 'Se detecta evidencia exploratoria')
    insight = (f'{conclusion} de un ciclo regular de {period} registros en los años seleccionados. '
        'La posición se cuenta desde 0 y vuelve a empezar cada grupo; no representa una hora del día. '
        'La curva muestra el cambio estimado alrededor de cero, no la concentración media. '
        'Sin fechas exactas no se pueden confirmar ciclos diarios o semanales, ni descartar otros patrones.')
    return fig, heat, summaries, insight, tables, []
