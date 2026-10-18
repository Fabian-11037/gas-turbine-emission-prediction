"""Protocolo principal por bloques y artefacto final reutilizable."""
import importlib.metadata as metadata
import json
import warnings

from block_cv_comparison import entrenar, particiones
from modelos_base import ROOT, PREDICTORES, OBJETIVOS, NOMBRES, SEMILLA
from modelos_base import metricas, predecir_final
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning


def ejecutar_experimento(datos, salida):
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always', ConvergenceWarning)
        resultado = entrenar(datos, 168, 24, salida, devolver_artefactos=True)
        if any(issubclass(w.category, ConvergenceWarning) for w in avisos):
            raise RuntimeError('Un estimador no alcanzo convergencia.')
    resumen = resultado['resumen']
    seleccion = {t: resumen.loc[(resumen.objetivo == t) &
                               resumen.seleccionado_por_CV, 'modelo'].item() for t in OBJETIVOS}
    artefacto = {
        'formato': 'gas-turbine-bloques-v1', 'predictores': PREDICTORES,
        'unidades_predictores': {'AT': 'degC', 'AP': 'mbar', 'AH': '%',
            'AFDP': 'mbar', 'GTEP': 'mbar', 'TIT': 'degC', 'TAT': 'degC',
            'TEY': 'MWh', 'CDP': 'mbar'},
        'objetivos': OBJETIVOS, 'unidades_respuesta': 'mg/m3', 'familias': seleccion,
        'estimadores': {t: resultado['modelos'][t, seleccion[t]] for t in OBJETIVOS},
        'hiperparametros': {t: resultado['parametros'][t, seleccion[t]] for t in OBJETIVOS},
        'anios_ajuste': list(range(2011, 2016)), 'n_ajuste': len(resultado['train_idx']),
        'longitud_bloque': 168, 'margen_purga': 24, 'n_folds': 5,
        'criterio_seleccion': 'menor MSE medio CV por bloques, sin usar prueba',
        'versiones': {p: metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
        'semilla': SEMILLA}
    carpeta = ROOT / 'modelos_finales'
    carpeta.mkdir(exist_ok=True)
    ruta = carpeta / 'modelo_final_co_nox_bloques.joblib'
    joblib.dump(artefacto, ruta)
    (carpeta / 'modelo_final_co_nox_bloques_metadata.json').write_text(
        json.dumps({k: v for k, v in artefacto.items() if k != 'estimadores'}, indent=2), encoding='utf-8')
    recargado = joblib.load(ruta)
    prueba = datos.iloc[resultado['test_idx']]
    p = predecir_final(recargado, prueba[PREDICTORES])
    exportacion = prueba[['anio', 'orden_registro'] + OBJETIVOS].copy()
    for t in OBJETIVOS:
        np.testing.assert_allclose(p[t], resultado['predicciones_prueba'][f'{t}_{seleccion[t].replace(" ", "_")}'])
        exportacion[f'{t}_predicho'] = p[t].to_numpy()
    exportacion.to_csv(salida / 'predicciones_modelo_final.csv', index=False)
    for (t, nombre), m in resultado['modelos'].items():
        joblib.dump(m, salida / f'modelo_{t}_{nombre.replace(" ", "_")}.joblib')
    protocolo = {'anios': list(range(2011, 2016)), 'longitud_bloque': 168,
        'margen_purga': 24, 'n_folds': 5, 'semilla': SEMILLA,
        'n_desarrollo': len(resultado['train_idx']), 'n_prueba': len(resultado['test_idx']),
        'n_purga_prueba': len(datos) - len(resultado['train_idx']) - len(resultado['test_idx']),
        'fraccion_prueba': len(resultado['test_idx']) / len(datos),
        'familias_finales': seleccion, 'predictores': PREDICTORES,
        'prueba_utilizada_para_seleccion': False, 'prueba_historica_previamente_evaluada': True,
        'validacion': 'cinco folds de bloques mezclados, balanceados por anio, purga en cada ajuste',
        'refit': 'solo desarrollo elegible; no incluye prueba ni margen excluido de prueba',
        'metricas_CV': 'condicionadas a hiperparametros, no CV anidada',
        'alcance': 'estimacion contemporanea de condiciones representadas, no extrapolacion futura'}
    (salida / 'protocolo.json').write_text(json.dumps(protocolo, indent=2), encoding='utf-8')
    resultado.update(seleccion=seleccion, artefacto=recargado, ruta_modelo=ruta)
    return resultado


def bootstrap_prueba(frame, observado, predicho, replicas=2000, semilla=SEMILLA):
    """Remuestrea bloques completos de prueba por anio, sin unir huecos."""
    auxiliar = frame[['anio', 'bloque']].copy()
    y, p = np.asarray(observado), np.asarray(predicho)
    auxiliar = auxiliar.assign(n=1, y=y, y2=y ** 2,
                              error2=(y-p) ** 2, error_abs=np.abs(y-p), sesgo=p-y)
    rng = np.random.default_rng(semilla)
    acumulado = np.zeros((replicas, 6))
    for _, grupo in auxiliar.groupby('anio', sort=True):
        suma = grupo.groupby('bloque')[['n', 'y', 'y2', 'error2', 'error_abs', 'sesgo']].sum().to_numpy()
        indices = rng.integers(0, len(suma), size=(replicas, len(suma)))
        acumulado += suma[indices].sum(axis=1)
    n, y, y2, e2, ea, sesgo = acumulado.T
    var = y2 / n - (y / n) ** 2
    if np.any(var <= 0):
        raise ValueError('Varianza nula en una replica bootstrap.')
    return {'R2': 1 - (e2 / n) / var, 'RMSE': np.sqrt(e2 / n),
            'MAE': ea / n, 'sesgo': sesgo / n}


def correlacion_intra_bloque(frame, residuos, max_lag=40):
    auxiliar = frame[['anio', 'bloque']].assign(residuo=np.asarray(residuos))
    grupos = [g.residuo.to_numpy() for _, g in auxiliar.groupby(['anio', 'bloque'], sort=True)]
    valores = [1.0]
    for lag in range(1, max_lag + 1):
        izquierda = np.concatenate([r[:-lag] for r in grupos if len(r) > lag])
        derecha = np.concatenate([r[lag:] for r in grupos if len(r) > lag])
        valores.append(np.corrcoef(izquierda, derecha)[0, 1])
    return np.asarray(valores)
