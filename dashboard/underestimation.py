"""Observed versus estimated emissions using only saved operational-kernel evidence."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1]/'data/stability_ablation'
METHOD = ('El R² global puede ocultar errores en concentraciones altas. Para examinarlos, se comparan '
          'las emisiones observadas con las estimadas. La diagonal representa una estimación perfecta. '
          'Los puntos por debajo indican subestimación; '
          'por encima, sobreestimación. Se muestran todos los registros: predicciones OOF de desarrollo '
          'y predicciones del modelo final en prueba. La línea de promedios resume intervalos de concentración '
          'definidos con los datos de desarrollo.')


@lru_cache(maxsize=4)
def read(target, phase):
    if target not in ['CO', 'NOX'] or phase not in ['OOF', 'prueba']:
        raise ValueError('Objetivo o conjunto no válido.')
    return pd.read_csv(DATA/f'{target}_kernel_operativo_42_completo_{phase}.csv.gz')


def bins(target):
    edges = np.unique(read(target, 'OOF')[target].quantile([0., .25, .5, .75, .9, .95, .99, 1.]).to_numpy())
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def trend(target, phase):
    frame = read(target, phase)
    return frame.groupby(pd.cut(frame[target], bins(target), include_lowest=True), observed=True).agg(
        observado=(target, 'mean'), estimado=('prediccion', 'mean'), n=(target, 'size')).reset_index(drop=True)


def summary(target):
    threshold = float(read(target, 'OOF')[target].quantile(.95))
    rows = []
    for phase in ['OOF', 'prueba']:
        high = read(target, phase).loc[lambda d:d[target] > threshold]
        rows.append(dict(conjunto='Desarrollo · OOF' if phase == 'OOF' else 'Prueba',
            umbral=threshold, n_altos=len(high),
            porcentaje_subestimado=100*float(high.prediccion.lt(high[target]).mean()),
            diferencia_media=float((high[target]-high.prediccion).mean())))
    return pd.DataFrame(rows)


def conclusion(target):
    values = summary(target)
    name = 'NOx' if target == 'NOX' else target
    threshold = f'{values.umbral.iloc[0]:.2f}'.replace('.', ',')
    a, b = [f'{value:.0f}' for value in values.porcentaje_subestimado]
    return (f'Para {name} superior a {threshold} mg/m³ —el 5 % más alto de desarrollo—, '
            f'el modelo subestima el {a} % de los casos OOF y el {b} % de prueba. '
            'El patrón aparece en ambos conjuntos; no se limita a la prueba.')


def figure(target, phase):
    import plotly.graph_objects as go
    from .figures import style
    frame = read(target, phase)
    combined = pd.concat([read(target, 'OOF'), read(target, 'prueba')])
    low = min(float(combined[target].min()), float(combined.prediccion.min()), 0.)
    high = max(float(combined[target].max()), float(combined.prediccion.max()))
    pad = (high-low)*.03
    bounds = [low-pad, high+pad]
    color = '#157a7c' if phase == 'OOF' else '#627f35'
    fig = go.Figure(go.Scattergl(x=frame[target], y=frame.prediccion, mode='markers', name='Registros',
        marker=dict(size=4, opacity=.25, color=color),
        hovertemplate='Observado: %{x:.3f}<br>Estimado: %{y:.3f} mg/m³<extra></extra>'))
    fig.add_trace(go.Scatter(x=bounds, y=bounds, mode='lines', name='Estimación perfecta',
        line=dict(color='#68747c', dash='dash', width=1.3), hoverinfo='skip'))
    averages = trend(target, phase)
    fig.add_trace(go.Scatter(x=averages.observado, y=averages.estimado, mode='lines+markers', name='Promedios por intervalo',
        line=dict(color='#b65346', width=2), marker=dict(size=6), customdata=averages.n,
        hovertemplate='Promedio observado: %{x:.3f}<br>Promedio estimado: %{y:.3f}<br>Registros: %{customdata}<extra></extra>'))
    title = 'Desarrollo · OOF' if phase == 'OOF' else 'Prueba reservada'
    name = 'NOx' if target == 'NOX' else target
    style(fig, f'{title} · {len(frame):,}'.replace(',', '.'), f'{name} observado (mg/m³)', f'{name} estimado (mg/m³)')
    fig.update_layout(height=420, margin=dict(l=60, r=15, t=55, b=95),
        legend=dict(orientation='h', y=-.22, x=0, font=dict(size=10)), showlegend=True)
    fig.update_xaxes(range=bounds)
    fig.update_yaxes(range=bounds, scaleanchor='x', scaleratio=1, constrain='domain')
    return fig


def panel(target):
    from dash import html
    from .components import disclosure, graph, table
    return html.Div([
        html.P(METHOD, className='caption'),
        html.Div([graph('kernel-observed-OOF', figure(target, 'OOF')),
                  graph('kernel-observed-test', figure(target, 'prueba'))], className='chart-grid'),
        html.P(conclusion(target)),
        disclosure('Concentraciones altas · resumen', table(summary(target),
            {'conjunto':'Conjunto', 'umbral':'Umbral (mg/m³)', 'n_altos':'Registros sobre el umbral',
             'porcentaje_subestimado':'Casos subestimados (%)', 'diferencia_media':'Diferencia media: real − estimado (mg/m³)'}, digits=2))])
