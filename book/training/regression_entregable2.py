"""Benchmarks de regresion: cinco folds externos, tres internos y purga doble."""
from pathlib import Path
import copy
import hashlib
import importlib.metadata as metadata
import itertools
import json
import os
import random
import sys
from time import perf_counter
import warnings

HERE = Path(__file__).resolve().parent
for directory in [HERE/'notebook_dependencies', HERE/'entregable2_dependencies']:
    if directory.exists():
        sys.path.insert(0, str(directory))

import joblib
import numpy as np
import pandas as pd
from deap import base, tools
from scipy import stats
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor
from block_cv_comparison import particiones
from modelos_base import PREDICTORES, OBJETIVOS, SEMILLA, metricas

FAMILIAS = ['KNN', 'Ridge', 'Lasso', 'Decision Tree', 'Random Forest', 'XGBoost', 'SVR lineal']
METODOS = ['Grid', 'Random', 'Bayesiano', 'Genetico']
CONFIG = {'version': 1, 'semilla': SEMILLA, 'bloque': 168, 'purga': 24,
          'folds_externos': 5, 'folds_internos': 3, 'random_evaluaciones': 4,
          'bayes_evaluaciones': 5, 'gen_poblacion': 4, 'gen_generaciones': 2,
          'gen_cruce': .7, 'gen_mutacion': .3, 'gen_mutacion_gen': .5,
          'gen_torneo': 2, 'gen_elites': 1}


def especificacion(familia):
    if familia == 'KNN':
        model = KNeighborsRegressor(algorithm='kd_tree', n_jobs=1)
        space = {'n_neighbors': [8, 24, 64], 'weights': ['uniform', 'distance']}
    elif familia == 'Ridge':
        model = Ridge(solver='svd')
        space = {'alpha': [.01, 1., 100., 1000.]}
    elif familia == 'Lasso':
        model = Lasso(max_iter=100000, tol=1e-5, random_state=SEMILLA)
        space = {'alpha': [.0001, .001, .01, .1]}
    elif familia == 'Decision Tree':
        model = DecisionTreeRegressor(random_state=SEMILLA)
        space = {'max_depth': [4, 8, 16], 'min_samples_leaf': [5, 25]}
    elif familia == 'Random Forest':
        model = RandomForestRegressor(random_state=SEMILLA, n_jobs=1, max_features=1.)
        space = {'n_estimators': [40, 80], 'max_depth': [6, 12], 'min_samples_leaf': [5, 20]}
    elif familia == 'XGBoost':
        model = XGBRegressor(objective='reg:squarederror', tree_method='hist',
                             random_state=SEMILLA, n_jobs=1, verbosity=0)
        space = {'n_estimators': [80, 160], 'max_depth': [3, 6], 'learning_rate': [.05, .1]}
    elif familia == 'SVR lineal':
        model = LinearSVR(dual=False, loss='squared_epsilon_insensitive',
                          max_iter=50000, tol=1e-5, random_state=SEMILLA)
        space = {'C': [.01, .1, 1.], 'epsilon': [.05, .1]}
    else:
        raise ValueError(familia)
    estimator = TransformedTargetRegressor(
        regressor=Pipeline([('escalado', StandardScaler()), ('modelo', model)]),
        transformer=StandardScaler())
    return estimator, space


def configurar(estimator, params):
    return clone(estimator).set_params(**{'regressor__modelo__'+k: v for k, v in params.items()})


def folds_internos(frame, cantidad=3):
    """frame ya excluye prueba y validacion externa, incluidos sus margenes."""
    ids_fold = {}
    rng = np.random.default_rng(SEMILLA)
    for _, group in frame.groupby('anio', sort=True):
        ids = np.unique(group.bloque)
        rng.shuffle(ids)
        for i, block in enumerate(ids):
            ids_fold[int(block)] = i % cantidad
    fold = frame.bloque.map(ids_fold).to_numpy()
    indices = np.arange(len(frame))
    result = []
    for i in range(cantidad):
        va = indices[fold == i]
        tr = purga_original(frame, indices[fold != i], va, 24)
        assert len(tr) and len(va)
        assert not set(frame.iloc[tr].bloque) & set(frame.iloc[va].bloque)
        result.append((tr, va))
    return result


def purga_original(frame, candidates, reserved, margin):
    keep = np.ones(len(candidates), dtype=bool)
    years = frame.anio.to_numpy()
    order = frame.orden_registro.to_numpy()
    for year in np.unique(years):
        mask = years[candidates] == year
        points = np.sort(order[reserved[years[reserved] == year]])
        if not len(points):
            continue
        positions = order[candidates[mask]]
        left = np.searchsorted(points, positions-margin, side='left')
        right = np.searchsorted(points, positions+margin, side='right')
        keep[mask] = left == right
    return np.asarray(candidates)[keep]


class FitnessMin(base.Fitness):
    weights = (-1.,)


class Individuo(list):
    def __init__(self, values):
        super().__init__(values)
        self.fitness = FitnessMin()


def buscar(familia, metodo, frame, target, splits):
    estimator, space = especificacion(familia)
    names = list(space)
    candidates = list(itertools.product(*[range(len(space[k])) for k in names]))
    x = frame[PREDICTORES]
    y = frame[target]
    cache, history = {}, []
    started = perf_counter()

    def evaluate(genes, generation=None, diversity=None):
        key = tuple(genes)
        tick = perf_counter()
        reused = key in cache
        params = {k: space[k][int(v)] for k, v in zip(names, key)}
        if not reused:
            scores = []
            for tr, va in splits:
                m = configurar(estimator, params)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    m.fit(x.iloc[tr], y.iloc[tr])
                assert m.regressor_['escalado'].n_samples_seen_ == len(tr)
                assert m.transformer_.n_samples_seen_ == len(tr)
                scores.append(mean_squared_error(y.iloc[va], m.predict(x.iloc[va])))
            cache[key] = float(np.mean(scores))
        score = cache[key]
        history.append({'evaluacion': len(history)+1, 'MSE_interno': score,
                        'mejor_MSE': min([r['MSE_interno'] for r in history]+[score]),
                        'segundos_evaluacion': perf_counter()-tick,
                        'segundos_acumulados': perf_counter()-started,
                        'cache': reused, 'parametros': json.dumps(params, sort_keys=True),
                        'generacion': generation, 'diversidad': diversity})
        return score

    rng = np.random.default_rng(SEMILLA)
    if metodo == 'Grid':
        for genes in candidates:
            evaluate(genes)
    elif metodo == 'Random':
        for i in rng.choice(len(candidates), min(CONFIG['random_evaluaciones'], len(candidates)), replace=False):
            evaluate(candidates[i])
    elif metodo == 'Bayesiano':
        budget = min(CONFIG['bayes_evaluaciones'], len(candidates))
        initial = rng.choice(len(candidates), min(2, budget), replace=False)
        for i in initial:
            evaluate(candidates[i])
        encoding = np.asarray(candidates, dtype=float)
        encoding /= np.maximum(np.array([len(space[k])-1 for k in names]), 1)
        while len(cache) < budget:
            seen = [candidates.index(k) for k in cache]
            remaining = [i for i in range(len(candidates)) if candidates[i] not in cache]
            gp = GaussianProcessRegressor(kernel=Matern(length_scale=1., nu=2.5),
                                          alpha=1e-6, normalize_y=True, optimizer=None,
                                          random_state=SEMILLA)
            gp.fit(encoding[seen], np.log(np.maximum(list(cache.values()), 1e-12)))
            mu, sd = gp.predict(encoding[remaining], return_std=True)
            improvement = np.log(min(cache.values())) - mu - .01
            z = improvement / np.maximum(sd, 1e-12)
            ei = improvement*stats.norm.cdf(z) + sd*stats.norm.pdf(z)
            evaluate(candidates[remaining[int(np.argmax(ei))]])
    elif metodo == 'Genetico':
        random.seed(SEMILLA)
        upper = [len(space[k])-1 for k in names]
        population = [Individuo([random.randint(0, v) for v in upper])
                      for _ in range(CONFIG['gen_poblacion'])]
        for generation in range(CONFIG['gen_generaciones']+1):
            diversity = len({tuple(ind) for ind in population})/len(population)
            for individual in population:
                individual.fitness.values = (evaluate(individual, generation, diversity),)
            if generation == CONFIG['gen_generaciones']:
                break
            elite = copy.deepcopy(tools.selBest(population, 1)[0])
            children = [copy.deepcopy(ind) for ind in tools.selTournament(population, len(population)-1, tournsize=2)]
            for i in range(0, len(children)-1, 2):
                if random.random() < CONFIG['gen_cruce']:
                    tools.cxUniform(children[i], children[i+1], indpb=.5)
            for child in children:
                if random.random() < CONFIG['gen_mutacion']:
                    tools.mutUniformInt(child, low=[0]*len(upper), up=upper, indpb=.5)
            population = [elite]+children
    else:
        raise ValueError(metodo)
    best = min(cache, key=cache.get)
    return ({k: space[k][int(v)] for k, v in zip(names, best)}, cache[best],
            pd.DataFrame(history), perf_counter()-started)


def evaluar_familia(datos, desarrollo, prueba, outer, familia, target, salida):
    started = perf_counter()
    rows, searches, predictions = [], [], {}
    all_params, final_models = {}, {}
    for metodo in METODOS:
        oof = np.full(len(desarrollo), np.nan)
        start_method = perf_counter()
        for fold, (tr, va) in enumerate(outer, 1):
            frame = desarrollo.iloc[tr].reset_index(drop=True)
            params, mse_inner, history, search_time = buscar(familia, metodo, frame, target, folds_internos(frame))
            model = configurar(especificacion(familia)[0], params).fit(frame[PREDICTORES], frame[target])
            oof[va] = model.predict(desarrollo.iloc[va][PREDICTORES])
            train_metrics = metricas(frame[target], model.predict(frame[PREDICTORES]))
            rows.append({'objetivo': target, 'modelo': familia, 'optimizador': metodo, 'fold': fold,
                         'n_train': len(tr), 'n_val': len(va), 'MSE_interno': mse_inner,
                         'parametros': json.dumps(params, sort_keys=True), 'segundos_busqueda': search_time,
                         **metricas(desarrollo.iloc[va][target], oof[va]),
                         **{k+'_train': v for k, v in train_metrics.items()}})
            searches.append(history.assign(objetivo=target, modelo=familia, optimizador=metodo, etapa=f'externo_{fold}'))
        assert np.isfinite(oof).all()
        params, mse_inner, history, search_time = buscar(familia, metodo, desarrollo, target, folds_internos(desarrollo))
        searches.append(history.assign(objetivo=target, modelo=familia, optimizador=metodo, etapa='final'))
        tick = perf_counter()
        final = configurar(especificacion(familia)[0], params).fit(desarrollo[PREDICTORES], desarrollo[target])
        fit_time = perf_counter()-tick
        tick = perf_counter()
        ptest = final.predict(prueba[PREDICTORES])
        infer_time = perf_counter()-tick
        ptrain = final.predict(desarrollo[PREDICTORES])
        predictions[metodo] = {'OOF': oof, 'test': ptest, 'train': ptrain,
                               'segundos_ajuste': fit_time, 'segundos_inferencia': infer_time,
                               'segundos_total': perf_counter()-start_method}
        final_models[metodo], all_params[metodo] = final, params
    folds = pd.DataFrame(rows)
    history = pd.concat(searches, ignore_index=True)
    best_method = folds.assign(MSE=lambda d:d.RMSE**2).groupby('optimizador').MSE.mean().sort_values(kind='stable').index[0]
    path = salida/'modelos'/f'{target}_{familia.replace(" ", "_")}.joblib'
    payload = {'formato':'gas-turbine-entregable2-v1', 'objetivo':target, 'familia':familia,
               'optimizador':best_method, 'predictores':PREDICTORES, 'unidad_respuesta':'mg/m3',
               'parametros':all_params[best_method], 'n_ajuste':len(desarrollo), 'protocolo':CONFIG,
               'estimador': final_models[best_method]}
    joblib.dump(payload, path, compress=3)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded['estimador'].predict(prueba[PREDICTORES]), predictions[best_method]['test'])
    metrics = []
    for method in METODOS:
        group = folds[folds.optimizador == method]
        summary = {'objetivo':target, 'modelo':familia, 'optimizador':method,
                   'parametros_finales':json.dumps(all_params[method],sort_keys=True),
                   'mejor_en_familia':method == best_method,
                   'MSE_CV_externo':float(np.mean(group.RMSE**2))}
        for metric in ['RMSE','MAE','R2']:
            summary[metric+'_CV_media'] = group[metric].mean()
            summary[metric+'_CV_std'] = group[metric].std(ddof=1)
        for label, y in [('train',desarrollo[target]),('OOF',desarrollo[target]),('test',prueba[target])]:
            summary.update({k+'_'+label:v for k,v in metricas(y,predictions[method][label]).items()})
        summary.update({k:v for k,v in predictions[method].items() if k.startswith('segundos_')})
        h = history[history.optimizador == method]
        summary.update(evaluaciones=len(h), evaluaciones_unicas=int((~h.cache).sum()),
                       ajustes_internos=int((~h.cache).sum())*3, negativos_test=int((predictions[method]['test']<0).sum()))
        metrics.append(summary)
    result = {'resumen':pd.DataFrame(metrics), 'folds':folds, 'historial':history,
              'predicciones':predictions, 'mejor_metodo':best_method, 'ruta':str(path)}
    joblib.dump(result, salida/'corridas'/f'{target}_{familia.replace(" ", "_")}.joblib', compress=3)
    print(f'{target} / {familia}: {best_method}; {perf_counter()-started:.1f} s', flush=True)
    return result


def _worker(*args):
    with threadpool_limits(limits=1):
        return evaluar_familia(*args)


def diagnosticos(prueba, y, predicted):
    from statsmodels.stats.diagnostic import het_white, acorr_ljungbox
    from statsmodels.tsa.stattools import bds
    from statsmodels.tools.tools import add_constant
    residual = np.asarray(y)-np.asarray(predicted)
    white = het_white(residual, add_constant(StandardScaler().fit_transform(prueba[PREDICTORES])))
    # Use one complete, contiguous source block; never concatenate gaps for serial tests.
    sizes = prueba.groupby(['anio','bloque']).size()
    year, block = sizes[sizes == sizes.max()].index[0]
    mask = (prueba.anio == year) & (prueba.bloque == block)
    r = residual[mask.to_numpy()]
    bds_stat, bds_p = bds(r, max_dim=2, distance=1.5)
    lb = acorr_ljungbox(r, lags=[10], return_df=True)
    return {'p_White':float(white[1]), 'p_BDS':float(np.asarray(bds_p).item()),
            'p_Ljung_Box':float(lb.lb_pvalue.iloc[0]), 'p_normalidad':float(stats.normaltest(residual).pvalue),
            'anio_bloque_serial':int(year),'bloque_serial':int(block),'n_serial':len(r)}


def ejecutar(datos, salida, workers=4):
    salida = Path(salida)
    for name in ['modelos','corridas']:
        (salida/name).mkdir(parents=True,exist_ok=True)
    train_idx, test_idx, outer, assignment, counts = particiones(datos,168,24)
    desarrollo, prueba = datos.iloc[train_idx].copy(), datos.iloc[test_idx].copy()
    desarrollo['bloque'] = assignment.iloc[train_idx].bloque.to_numpy()
    prueba['bloque'] = assignment.iloc[test_idx].bloque.to_numpy()
    signature = hashlib.sha256(pd.util.hash_pandas_object(datos, index=False).values.tobytes()).hexdigest()
    config = {**CONFIG, 'datos_sha256':signature, 'codigo_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config_path = salida/'configuracion.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('La carpeta contiene corridas de otra configuracion; use otra carpeta de salida.')
    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    assignment.to_csv(salida/'asignacion_particiones.csv',index=False)
    counts.to_csv(salida/'conteos_externos.csv',index=False)
    jobs, completed = [], []
    for target, family in itertools.product(OBJETIVOS,FAMILIAS):
        path = salida/'corridas'/f'{target}_{family.replace(" ","_")}.joblib'
        if path.exists():
            completed.append(joblib.load(path))
        else:
            jobs.append((datos,desarrollo,prueba,outer,family,target,salida))
    completed.extend(joblib.Parallel(n_jobs=workers,backend='loky',verbose=10)(
        joblib.delayed(_worker)(*args) for args in jobs))
    summary = pd.concat([r['resumen'] for r in completed],ignore_index=True)
    folds = pd.concat([r['folds'] for r in completed],ignore_index=True)
    history = pd.concat([r['historial'] for r in completed],ignore_index=True)
    selected = summary[summary.mejor_en_familia].copy()
    selected['seleccionado_global'] = False
    for target in OBJETIVOS:
        idx = selected.loc[selected.objetivo == target].MSE_CV_externo.idxmin()
        selected.loc[idx,'seleccionado_global'] = True
    test_predictions = prueba[['anio','orden_registro','bloque']+OBJETIVOS].copy()
    oof_predictions = desarrollo[['anio','orden_registro','bloque']+OBJETIVOS].copy()
    diagnosis = []
    with threadpool_limits(limits=1):
        for result in completed:
            row = result['resumen'].loc[lambda d:d.mejor_en_familia].iloc[0]
            target, family = row.objetivo, row.modelo
            pred = result['predicciones'][result['mejor_metodo']]
            col = target+'_'+family.replace(' ','_')
            test_predictions[col] = pred['test']
            oof_predictions[col] = pred['OOF']
            diagnosis.append({'objetivo':target,'modelo':family,**diagnosticos(prueba,prueba[target],pred['test'])})
    selected = selected.merge(pd.DataFrame(diagnosis),on=['objetivo','modelo'],validate='one_to_one')
    for name, df in [('resultados_optimizadores',summary),('resultados_familias',selected),
                     ('validacion_externa',folds),('historial_busquedas',history),
                     ('predicciones_prueba',test_predictions),('predicciones_OOF',oof_predictions)]:
        df.to_csv(salida/(name+'.csv'),index=False)
    versions = {p:metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib',
                                               'statsmodels','joblib','xgboost','deap','threadpoolctl']}
    (salida/'versiones.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    registry = []
    for _, row in selected.iterrows():
        name = row.objetivo+'_'+row.modelo.replace(' ','_')+'.joblib'
        registry.append({'objetivo':row.objetivo,'familia':row.modelo,'optimizador':row.optimizador,
                         'ruta':'modelos/'+name,'seleccionado_global':bool(row.seleccionado_global),
                         'sha256':hashlib.sha256((salida/'modelos'/name).read_bytes()).hexdigest()})
    (salida/'registro_modelos.json').write_text(json.dumps({'modelos':registry,'versiones':versions,
            'predictores':PREDICTORES,'n_ajuste':len(desarrollo),'n_prueba':len(prueba),
            'criterio':'MSE medio de cinco folds externos anidados; prueba no usada para seleccion',
            'prueba_historica_examinada_previamente':True},indent=2),encoding='utf-8')
    return selected


def predecir(ruta, entradas):
    """Solo se deben cargar artefactos joblib procedentes de fuentes confiables."""
    artifact = joblib.load(ruta)
    x = entradas.loc[:,artifact['predictores']].apply(pd.to_numeric,errors='raise')
    if x.empty or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Se requieren sensores numericos finitos y al menos un registro.')
    return pd.Series(artifact['estimador'].predict(x),index=x.index,name=artifact['objetivo'])


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    root = HERE
    data = pd.concat([pd.read_csv(root/'data'/f'gt_{year}.csv').assign(anio=year)
                      for year in range(2011,2016)],ignore_index=True)
    data['orden_registro'] = np.arange(len(data))
    result = ejecutar(data,root/'informe_resultados'/'entregable2_anidada',workers=4)
    print(result[['objetivo','modelo','optimizador','RMSE_CV_media','R2_test','RMSE_test']].to_string(index=False))
