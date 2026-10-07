"""Lectura unica de evidencias exportadas; los callbacks no modifican estos datos."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'


@lru_cache(maxsize=1)
def manifest():
    return json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def observations():
    return pd.read_csv(DATA/'observaciones.csv.gz')


@lru_cache(maxsize=1)
def results():
    return pd.read_csv(DATA/'resultados.csv')


@lru_cache(maxsize=1)
def predictions():
    return pd.read_csv(DATA/'predicciones_prueba.csv.gz')


@lru_cache(maxsize=1)
def intervals():
    return pd.read_csv(DATA/'intervalos_bootstrap.csv')


@lru_cache(maxsize=1)
def sensor_audit():
    return pd.read_csv(DATA/'auditoria_predictiva_univariada.csv')


@lru_cache(maxsize=1)
def bibliography():
    return pd.read_csv(DATA/'bibliografia.csv').fillna('')


def filtered(years, population):
    df = observations()
    mask = df.anio.isin(years or [])
    if population == 'desarrollo':
        mask &= df.particion.eq('entrenamiento_CV')
    return df.loc[mask]


def sensors():
    return manifest()['sensores']


def sample_frame():
    return pd.DataFrame([{s['id']:s['ejemplo'] for s in sensors()}])


def validate_inputs(frame, predictors):
    faltan = set(predictors)-set(frame.columns)
    if faltan:
        raise ValueError('Faltan sensores: '+', '.join(sorted(faltan)))
    if frame.empty:
        raise ValueError('No hay registros para estimar.')
    x = frame[predictors].apply(pd.to_numeric,errors='coerce')
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('Todos los sensores deben ser números finitos, sin campos vacíos.')
    if ((x.AH < 0) | (x.AH > 100)).any():
        raise ValueError('La humedad relativa debe estar entre 0 y 100 %.')
    for c in ['AP','AFDP','GTEP','CDP','TEY']:
        if x[c].le(0).any():
            raise ValueError(f'{c} debe ser positivo.')
    return x


def range_warnings(frame):
    return [s['id'] for s in sensors() if
            ((frame[s['id']] < s['min']) | (frame[s['id']] > s['max'])).any()]
