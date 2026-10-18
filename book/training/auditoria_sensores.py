"""Auditoria univariada e interpretacion de modelos lineales ya ajustados."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'notebook_dependencies'))

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def auditar_sensores(desarrollo, splits, predictores, objetivos):
    resumen, detalles = [], []
    for objetivo in objetivos:
        y = desarrollo[objetivo].to_numpy()
        dummy_oof = np.full(len(y), np.nan)
        for tr, va in splits:
            dummy_oof[va] = DummyRegressor().fit(
                desarrollo.iloc[tr][[predictores[0]]], y[tr]).predict(
                desarrollo.iloc[va][[predictores[0]]])
        rmse_dummy = np.sqrt(mean_squared_error(y, dummy_oof))
        for sensor in predictores:
            oof = np.full(len(y), np.nan)
            cobertura = np.zeros(len(y), dtype=int)
            for numero, (tr, va) in enumerate(splits, 1):
                assert not np.intersect1d(tr, va).size
                modelo = TransformedTargetRegressor(
                    regressor=make_pipeline(StandardScaler(), Ridge(alpha=1)),
                    transformer=StandardScaler())
                modelo.fit(desarrollo.iloc[tr][[sensor]], y[tr])
                assert modelo.regressor_[0].n_samples_seen_ == len(tr)
                assert modelo.transformer_.n_samples_seen_ == len(tr)
                oof[va] = modelo.predict(desarrollo.iloc[va][[sensor]])
                cobertura[va] += 1
                detalles.append({'objetivo': objetivo, 'sensor': sensor, 'fold': numero,
                    'n_ajuste': len(tr), 'n_validacion': len(va),
                    'R2': r2_score(y[va], oof[va]),
                    'RMSE': np.sqrt(mean_squared_error(y[va], oof[va])),
                    'MAE': mean_absolute_error(y[va], oof[va])})
            assert np.isfinite(oof).all() and np.all(cobertura == 1)
            folds = pd.DataFrame([f for f in detalles
                if f['objetivo'] == objetivo and f['sensor'] == sensor])
            rmse = np.sqrt(mean_squared_error(y, oof))
            resumen.append({'objetivo': objetivo, 'sensor': sensor, 'alpha': 1,
                'R2_CV_promedio': folds.R2.mean(), 'R2_CV_std': folds.R2.std(ddof=1),
                'R2_OOF': r2_score(y, oof), 'RMSE_OOF': rmse,
                'MAE_OOF': mean_absolute_error(y, oof), 'RMSE_Dummy_OOF': rmse_dummy,
                'reduccion_RMSE_vs_Dummy_pct': 100*(1-rmse/rmse_dummy),
                'n_desarrollo': len(y), 'n_folds': len(splits)})
    return pd.DataFrame(resumen), pd.DataFrame(detalles)


def interpretar_coeficientes(artefacto, entradas):
    coeficientes, interceptos = [], []
    nombres = artefacto['predictores']
    x = entradas[nombres].to_numpy()
    for objetivo in artefacto['objetivos']:
        modelo = artefacto['estimadores'][objetivo]
        escala_x, lineal = modelo.regressor_[0], modelo.regressor_[-1]
        escala_y = float(modelo.transformer_.scale_[0])
        media_y = float(modelo.transformer_.mean_[0])
        beta_std = np.asarray(lineal.coef_).reshape(-1)
        beta_original = escala_y*beta_std/escala_x.scale_
        intercepto = media_y+escala_y*float(lineal.intercept_)-beta_original@escala_x.mean_
        np.testing.assert_allclose(intercepto+x@beta_original,
                                   modelo.predict(entradas[nombres]), rtol=1e-9, atol=1e-8)
        interceptos.append({'objetivo': objetivo, 'intercepto_mg_m3': intercepto,
                            'n_ajuste': artefacto['n_ajuste']})
        for j, sensor in enumerate(nombres):
            coeficientes.append({'objetivo': objetivo, 'sensor': sensor,
                'unidad_sensor': artefacto['unidades_predictores'][sensor],
                'media_sensor_desarrollo': escala_x.mean_[j],
                'SD_sensor_desarrollo': escala_x.scale_[j],
                'coeficiente_estandarizado': beta_std[j],
                'cambio_mg_m3_por_1_SD_sensor': escala_y*beta_std[j],
                'cambio_mg_m3_por_unidad_sensor': beta_original[j]})
    return pd.DataFrame(coeficientes), pd.DataFrame(interceptos)
