"""Comparacion independiente de validacion por bloques mezclados entre anios."""
import json
import warnings

from modelos_base import ROOT, PREDICTORES, OBJETIVOS, NOMBRES, SEMILLA
from modelos_base import construir, metricas, verificar_escalado, huellas_protegidas
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, train_test_split


CONFIGURACIONES = [(48, 24), (168, 24), (336, 24), (168, 0)]


def asignar_bloques(datos, longitud):
    if longitud < 1:
        raise ValueError('La longitud debe ser positiva.')
    grupos = np.empty(len(datos), dtype=int)
    siguiente = 0
    for anio in sorted(datos.anio.unique()):
        posiciones = np.flatnonzero(datos.anio.to_numpy() == anio)
        locales = np.arange(len(posiciones)) // longitud
        grupos[posiciones] = siguiente + locales
        siguiente += int(locales.max()) + 1
    return grupos


def purgar(datos, candidatos, reservados, margen):
    """Excluye vecinos de registros reservados dentro del mismo archivo anual."""
    if margen < 0:
        raise ValueError('El margen no puede ser negativo.')
    prohibidos = np.zeros(len(datos), dtype=bool)
    prohibidos[reservados] = True
    if margen:
        for anio in sorted(datos.anio.unique()):
            posiciones = np.flatnonzero(datos.anio.to_numpy() == anio)
            prefijo = np.r_[0, np.cumsum(prohibidos[posiciones])]
            orden = np.arange(len(posiciones))
            inicio = np.maximum(0, orden - margen)
            fin = np.minimum(len(posiciones), orden + margen + 1)
            prohibidos[posiciones] = (prefijo[fin] - prefijo[inicio]) > 0
    return np.asarray(candidatos)[~prohibidos[candidatos]]


def particiones(datos, longitud, margen):
    grupos = asignar_bloques(datos, longitud)
    bloques = pd.DataFrame({'bloque': grupos, 'anio': datos.anio}).drop_duplicates()
    assert bloques.bloque.is_unique
    grupos_train, grupos_test = train_test_split(
        bloques.bloque.to_numpy(), test_size=.2, random_state=SEMILLA,
        stratify=bloques.anio.to_numpy())
    train_sin_purga = np.flatnonzero(np.isin(grupos, grupos_train))
    test = np.flatnonzero(np.isin(grupos, grupos_test))
    train = purgar(datos, train_sin_purga, test, margen)
    folds_grupo = {}
    rng = np.random.default_rng(SEMILLA)
    # Se barajan bloques dentro de cada anio, nunca observaciones individuales.
    for anio in sorted(datos.anio.unique()):
        ids = bloques.loc[(bloques.anio == anio) &
                         bloques.bloque.isin(grupos_train), 'bloque'].to_numpy(copy=True)
        rng.shuffle(ids)
        for i, grupo in enumerate(ids):
            folds_grupo[int(grupo)] = i % 5
    fold = np.array([folds_grupo.get(int(g), -1) for g in grupos[train]])
    splits, detalle = [], []
    cobertura = np.zeros(len(train), dtype=int)
    for numero in range(5):
        va = np.flatnonzero(fold == numero)
        candidatos = np.flatnonzero(fold != numero)
        permitidos = purgar(datos, train[candidatos], train[va], margen)
        tr = candidatos[np.isin(train[candidatos], permitidos)]
        assert not set(grupos[train[tr]]) & set(grupos[train[va]])
        assert set(datos.iloc[train[tr]].anio) == set(datos.iloc[train[va]].anio)
        assert len(tr) and len(va)
        cobertura[va] += 1
        splits.append((tr, va))
        detalle.append({'fold': numero + 1, 'n_train': len(tr), 'n_val': len(va),
                        'n_purgados_CV': len(candidatos) - len(tr)})
    assert np.all(cobertura == 1)
    assert not set(grupos[train]) & set(grupos[test])
    asignacion = datos[['anio', 'orden_registro']].copy()
    asignacion['bloque'] = grupos
    asignacion['particion'] = 'purga_prueba'
    asignacion.loc[test, 'particion'] = 'prueba_bloques'
    asignacion.loc[train, 'particion'] = 'entrenamiento_CV'
    asignacion['fold_validacion'] = -1
    asignacion.loc[train, 'fold_validacion'] = fold + 1
    return train, test, splits, asignacion, pd.DataFrame(detalle)


def entrenar(datos, longitud, margen, salida, devolver_artefactos=False):
    salida.mkdir(parents=True, exist_ok=True)
    train_idx, test_idx, splits, asignacion, conteos_fold = particiones(
        datos, longitud, margen)
    train, test = datos.iloc[train_idx], datos.iloc[test_idx]
    asignacion.to_csv(salida / 'asignacion_particiones.csv', index=False)
    conteos_fold.to_csv(salida / 'conteos_folds.csv', index=False)
    asignacion.groupby(['anio', 'particion']).size().unstack(fill_value=0).to_csv(
        salida / 'conteos_por_anio.csv')
    membresia = pd.DataFrame({'orden_registro': train.orden_registro.to_numpy()})
    for numero, (tr, va) in enumerate(splits, 1):
        papel = np.full(len(train), 'purga_CV', dtype=object)
        papel[tr], papel[va] = 'entrenamiento', 'validacion'
        membresia[f'fold_{numero}'] = papel
    membresia.to_csv(salida / 'membresia_folds.csv', index=False)
    oof_export = train[['anio', 'orden_registro'] + OBJETIVOS].copy()
    test_export = test[['anio', 'orden_registro'] + OBJETIVOS].copy()
    resumen, detalles, modelos, parametros = [], [], {}, {}
    for objetivo in OBJETIVOS:
        for nombre in NOMBRES:
            est, grid = construir(nombre)
            if grid:
                busqueda = GridSearchCV(est, grid, cv=splits, refit=True,
                                        scoring='neg_mean_squared_error', n_jobs=1,
                                        error_score='raise')
                busqueda.fit(train[PREDICTORES], train[objetivo])
                final, params = busqueda.best_estimator_, busqueda.best_params_
                pd.DataFrame(busqueda.cv_results_).to_csv(
                    salida / f'busqueda_{objetivo}_{nombre.replace(" ", "_")}.csv', index=False)
            else:
                final, params = est.fit(train[PREDICTORES], train[objetivo]), {}
            verificar_escalado(final, len(train))
            modelos[objetivo, nombre] = final
            parametros[objetivo, nombre] = params
            oof = np.full(len(train), np.nan)
            folds = []
            for numero, (tr, va) in enumerate(splits, 1):
                m = clone(final).fit(train.iloc[tr][PREDICTORES], train.iloc[tr][objetivo])
                verificar_escalado(m, len(tr))
                oof[va] = m.predict(train.iloc[va][PREDICTORES])
                met = metricas(train.iloc[va][objetivo], oof[va])
                detalles.append({'objetivo': objetivo, 'modelo': nombre,
                                 'fold': numero, 'n_train': len(tr), 'n_val': len(va), **met})
                folds.append(met)
            assert np.isfinite(oof).all()
            tabla = pd.DataFrame(folds)
            p_train, p_test = final.predict(train[PREDICTORES]), final.predict(test[PREDICTORES])
            fila = {'objetivo': objetivo, 'modelo': nombre, 'longitud_bloque': longitud,
                    'margen_purga': margen, 'n_train': len(train), 'n_test': len(test),
                    'fraccion_test': len(test) / len(datos),
                    'n_purgados_prueba': len(datos) - len(train) - len(test),
                    'parametros': json.dumps(params, sort_keys=True),
                    'R2_CV_promedio': tabla.R2.mean(), 'R2_CV_std': tabla.R2.std(ddof=1),
                    'MSE_CV_promedio': (tabla.RMSE ** 2).mean(),
                    'negativos_test': int((p_test < 0).sum())}
            for etiqueta, y, p in [('train', train[objetivo], p_train),
                                   ('OOF', train[objetivo], oof), ('test', test[objetivo], p_test)]:
                fila.update({f'{k}_{etiqueta}': v for k, v in metricas(y, p).items()})
            resumen.append(fila)
            columna = f'{objetivo}_{nombre.replace(" ", "_")}'
            oof_export[columna], test_export[columna] = oof, p_test
    resumen = pd.DataFrame(resumen)
    # La configuracion principal esta prefijada; no se elige por su prueba.
    resumen['seleccionado_por_CV'] = False
    for objetivo in OBJETIVOS:
        indice = resumen.loc[resumen.objetivo == objetivo].MSE_CV_promedio.idxmin()
        resumen.loc[indice, 'seleccionado_por_CV'] = True
    for nombre, tabla in {'resultados': resumen, 'validacion_por_fold': pd.DataFrame(detalles),
                          'predicciones_OOF': oof_export, 'predicciones_prueba': test_export}.items():
        tabla.to_csv(salida / f'{nombre}.csv', index=False)
    if devolver_artefactos:
        return {'resumen': resumen, 'modelos': modelos, 'parametros': parametros,
                'splits': splits, 'train_idx': train_idx, 'test_idx': test_idx,
                'asignacion': asignacion, 'conteos_folds': conteos_fold,
                'validacion_por_fold': pd.DataFrame(detalles),
                'predicciones_OOF': oof_export, 'predicciones_prueba': test_export}
    return resumen


def ejecutar():
    antes = huellas_protegidas()
    salida = ROOT / 'informe_resultados' / 'comparacion_cv_bloques_80_20'
    salida.mkdir(parents=True, exist_ok=True)
    datos = pd.concat([pd.read_csv(ROOT / 'data' / f'gt_{a}.csv').assign(anio=a)
                       for a in range(2011, 2016)], ignore_index=True)
    datos['orden_registro'] = np.arange(len(datos))
    resultados = []
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always', ConvergenceWarning)
        for longitud, margen in CONFIGURACIONES:
            tabla = entrenar(datos, longitud, margen, salida / f'bloques_{longitud}_purga_{margen}')
            resultados.append(tabla)
            print(f'Bloques {longitud}, purga {margen}: completado.', flush=True)
        conv = [str(w.message) for w in avisos if issubclass(w.category, ConvergenceWarning)]
        if conv:
            raise RuntimeError('Falta de convergencia: ' + conv[0])
    resumen = pd.concat(resultados, ignore_index=True)
    resumen.to_csv(salida / 'resultados.csv', index=False)
    assert antes == huellas_protegidas(), 'Se modificaron artefactos oficiales.'
    protocolo = {
        'experimento': 'comparacion independiente; cuaderno y modelos oficiales sin cambios',
        'anios': list(range(2011, 2016)), 'n_total': len(datos), 'predictores': PREDICTORES,
        'semilla': SEMILLA, 'configuracion_principal': {'longitud_bloque': 168, 'margen_purga': 24},
        'sensibilidad': [{'longitud_bloque': l, 'margen_purga': g} for l, g in CONFIGURACIONES],
        'particion': '20% de bloques completos, aleatoria estratificada por anio; fraccion de filas aproximada',
        'CV': 'cinco folds; bloques barajados dentro de cada anio; grupos indivisibles entre ajuste y validacion',
        'purga': 'excluir del ajuste vecinos a distancia <= margen dentro del mismo anio, tanto en prueba como CV',
        'seleccion': 'MSE medio entre cinco folds; mismas familias e hiperparametros que las comparaciones previas',
        'prueba_utilizada_para_seleccion': False,
        'metricas_CV': 'condicionadas a seleccion de hiperparametros, no CV anidada',
        'limitaciones': ['Sin fechas exactas: longitudes en registros, no dias verificados.',
            'El margen de 24 registros es una sensibilidad prefijada, no garantiza independencia.',
            'Mezcla los cinco anios; no estima desempeno en un anio futuro.',
            'Cambian las observaciones de prueba entre protocolos y longitudes: no comparacion pareada.',
            'La sensibilidad reutiliza el dataset; las pruebas no son nuevos datos externos.'],
        'artefactos_oficiales_sin_cambios': True, 'huellas_SHA256': antes}
    (salida / 'protocolo.json').write_text(json.dumps(protocolo, indent=2), encoding='utf-8')
    print(resumen.loc[resumen.modelo != 'Media', ['objetivo', 'modelo', 'longitud_bloque',
        'margen_purga', 'n_train', 'n_test', 'R2_train', 'R2_CV_promedio', 'R2_CV_std',
        'R2_test', 'RMSE_test', 'MAE_test', 'sesgo_test']].round(4).to_string(index=False))
    print('SHA256: cuaderno, datos, modelos finales y resultados vigentes sin cambios.')
    return resumen


if __name__ == '__main__':
    ejecutar()
