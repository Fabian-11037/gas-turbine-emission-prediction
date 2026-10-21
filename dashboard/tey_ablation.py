"""Read-only presentation of TEY ablation, without additional prediction models."""
from functools import lru_cache
import pandas as pd
from dash import dcc, html
from .repository import DATA
from .components import table
from . import tey_content as text

LOCATION = DATA / 'tey_ablation'


@lru_cache(maxsize=1)
def evidence():
    return tuple(pd.read_csv(LOCATION / name) for name in
                 ['resultados.csv', 'significancia_CV.csv', 'significancia_prueba.csv'])


def comparison_table(target=None):
    results, cv, tests = evidence()
    primary = tests.loc[tests.longitud_bloque.eq(168)]
    rows = []
    for emission in ['CO', 'NOX']:
        group = results.loc[results.objetivo.eq(emission)]
        full = group.loc[group.modelo.eq('Kernel completo')].iloc[0]
        ablated = group.loc[~group.modelo.eq('Kernel completo')].iloc[0]
        a = cv.loc[cv.objetivo.eq(emission)].iloc[0]
        b = primary.loc[primary.objetivo.eq(emission)].iloc[0]
        if target and target != emission:
            continue
        for phase, metric, p in [('CV', 'R2_CV', a.p_Holm), ('Prueba', 'R2_prueba', b.p_Holm)]:
            rows.append({'Emisión': 'NOx' if emission == 'NOX' else emission,
                         'Evaluación': phase, 'R² completo': full[metric],
                         'R² sin TEY e interacción': ablated[metric],
                         'Aporte ΔR²': full[metric]-ablated[metric], 'p ajustado · Holm': p})
    return pd.DataFrame(rows)


def panel(target):
    return html.Section([
        html.H4(text.TITLE),
        dcc.Markdown(text.AVAILABILITY, className='kernel-markdown'),
        html.P(text.DESIGN),
        table(comparison_table(target).drop(columns='Emisión'), digits=5),
        html.P(text.READING, className='caption'),
        html.P(text.CONCLUSIONS[target])
    ], id='tey-ablation-analysis')
