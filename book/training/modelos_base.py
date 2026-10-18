"""Definiciones comunes de modelos, metricas e inferencia de CO y NOx."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'notebook_dependencies'))

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

PREDICTORES = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
OBJETIVOS = ['CO', 'NOX']
NOMBRES = ['Media', 'Ridge', 'SVR lineal']
SEMILLA = 42


def construir(nombre):
    if nombre == 'Media':
        return DummyRegressor(strategy='mean'), {}
    if nombre == 'Ridge':
        base = Ridge()
        grid = {'regressor__ridge__alpha': [1, 100, 1000]}
    elif nombre == 'SVR lineal':
        base = LinearSVR(dual=False, loss='squared_epsilon_insensitive',
                         random_state=SEMILLA, max_iter=30000, tol=1e-5)
        grid = {'regressor__linearsvr__C': [.01, .1, 1],
                'regressor__linearsvr__epsilon': [.05, .1]}
    else:
        raise ValueError('Familia de modelo no reconocida.')
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), base),
        transformer=StandardScaler()), grid


def metricas(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {'R2': r2_score(y, p), 'RMSE': np.sqrt(mean_squared_error(y, p)),
            'MAE': mean_absolute_error(y, p), 'sesgo': np.mean(p-y),
            'MAPE_pct': 100*np.mean(np.abs((y-p)/y)) if np.all(y != 0) else np.nan}


def verificar_escalado(modelo, n):
    assert modelo.n_features_in_ == len(PREDICTORES)
    if isinstance(modelo, TransformedTargetRegressor):
        assert modelo.regressor_[0].n_samples_seen_ == n
        assert modelo.transformer_.n_samples_seen_ == n


def predecir_final(artefacto, entradas):
    """Predice CO y NOX sin requerir las emisiones observadas como entradas."""
    if not isinstance(entradas, pd.DataFrame):
        raise TypeError('Las entradas deben ser un DataFrame con nombres de sensores.')
    faltantes = set(artefacto['predictores'])-set(entradas.columns)
    if faltantes:
        raise ValueError('Sensores ausentes: '+', '.join(sorted(faltantes)))
    x = entradas.loc[:, artefacto['predictores']]
    if not all(pd.api.types.is_numeric_dtype(x[c]) for c in x):
        raise ValueError('Los nueve sensores deben ser numericos.')
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('Las mediciones no pueden contener faltantes o infinitos.')
    return pd.DataFrame({t: artefacto['estimadores'][t].predict(x)
                         for t in artefacto['objetivos']}, index=entradas.index)


def huellas_protegidas():
    paths = [ROOT/'gas_turbine_eda_hilo_conductor_redaccion.ipynb']
    paths.extend((ROOT/'modelos_finales').glob('*'))
    paths.extend((ROOT/'data').glob('gt_*.csv'))
    paths.extend((ROOT/'informe_resultados'/'validacion_bloques_168_purga_24').glob('*'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}
