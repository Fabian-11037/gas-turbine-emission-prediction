"""Reentrenamiento reproducible del kernel, separado de la inferencia en Render."""
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from emission_models.kernel_operativo import OperationalStack, operational_features, weight_operational_features, PREDICTORS
from block_cv_comparison import particiones

WEIGHTS = [(.5, 1., 1.), (.5, .5, 2.), (.5, 2., .5), (2., 1., 1.)]


def purge(frame, candidates, reserved):
    keep = np.ones(len(candidates), dtype=bool)
    years, order = frame.anio.to_numpy(), frame.orden_registro.to_numpy()
    for year in np.unique(years):
        mask = years[candidates] == year
        points = np.sort(order[reserved[years[reserved] == year]])
        position = order[candidates[mask]]
        keep[mask] = (np.searchsorted(points, position-24, side='left') ==
                      np.searchsorted(points, position+24, side='right'))
    return np.asarray(candidates)[keep]


def inner_splits(frame):
    mapping, rng = {}, np.random.default_rng(42)
    for _, group in frame.groupby('anio', sort=True):
        blocks = np.unique(group.bloque)
        rng.shuffle(blocks)
        mapping.update({int(b): j % 3 for j, b in enumerate(blocks)})
    folds = frame.bloque.map(mapping).to_numpy()
    result = []
    for k in range(3):
        va = np.flatnonzero(folds == k)
        tr = purge(frame, np.flatnonzero(folds != k), va)
        assert not set(frame.iloc[tr].bloque) & set(frame.iloc[va].bloque)
        assert np.array_equal(purge(frame, tr, va), tr)
        result.append((tr, va))
    return result


def estimators():
    result = []
    for depth, leaf in [(20, 2), (None, 5)]:
        result.append(TransformedTargetRegressor(regressor=make_pipeline(
            FunctionTransformer(operational_features), StandardScaler(),
            ExtraTreesRegressor(n_estimators=200, max_depth=depth, min_samples_leaf=leaf,
                max_features=.75, random_state=42, n_jobs=1)), transformer=StandardScaler()))
    for environment, temperature, pressure in WEIGHTS:
        result.append(TransformedTargetRegressor(regressor=make_pipeline(
            FunctionTransformer(operational_features), StandardScaler(),
            FunctionTransformer(weight_operational_features, kw_args=dict(
                environmental=environment, temperature=temperature, pressure=pressure)),
            SVR(C=20., gamma=.2, epsilon=.1, cache_size=256)), transformer=StandardScaler()))
    return result


def candidates():
    return [dict(members=[tree, svr], **{'lambda': penalty})
        for tree in [0, 1] for svr in [2, 3, 4, 5] for penalty in [.001, .01, .1]]


class Suite:
    def fit(self, frame, target, factory=estimators):
        oof = np.full((len(frame), 6), np.nan)
        seen = np.zeros(len(frame), dtype=int)
        for tr, va in inner_splits(frame):
            for j, model in enumerate(factory()):
                model.fit(frame.iloc[tr][PREDICTORS], frame.iloc[tr][target])
                oof[va, j] = model.predict(frame.iloc[va][PREDICTORS])
            seen[va] += 1
        assert np.all(seen == 1) and np.isfinite(oof).all()
        self.models, self.meta = factory(), []
        for config in candidates():
            self.meta.append(TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),
                Ridge(alpha=config['lambda']*len(frame))), transformer=StandardScaler()).fit(
                    oof[:, config['members']], frame[target]))
        for model in self.models:
            model.fit(frame[PREDICTORS], frame[target])
        return self

    def predict_all(self, X):
        predictions = np.column_stack([m.predict(X[PREDICTORS]) for m in self.models])
        return [meta.predict(predictions[:, config['members']])
                for meta, config in zip(self.meta, candidates())]

    def predictor(self, candidate):
        return OperationalStack([self.models[j] for j in candidates()[candidate]['members']], self.meta[candidate])


def search(frame, target, output):
    history = []
    for fold, (tr, va) in enumerate(inner_splits(frame), 1):
        suite = Suite().fit(frame.iloc[tr], target)
        for candidate, prediction in enumerate(suite.predict_all(frame.iloc[va])):
            history.append(dict(fold=fold, candidato=candidate,
                MSE=float(mean_squared_error(frame.iloc[va][target], prediction))))
    table = pd.DataFrame(history)
    table.to_csv(output, index=False)
    selected = int(table.groupby('candidato').MSE.mean().idxmin())
    return Suite().fit(frame, target).predictor(selected), candidates()[selected]


def entrenar_kernel(datos, salida):
    """No sobrescribe los modelos publicados: exige una carpeta nueva de resultados."""
    output = Path(salida).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use una carpeta nueva para no mezclar resultados de entrenamiento.')
    output.mkdir(parents=True, exist_ok=True)
    tr, te, folds, assignment, _ = particiones(datos, 168, 24)
    frame = datos.assign(bloque=assignment.bloque.to_numpy())
    assert (len(tr), len(te)) == (27570, 7411)
    assignment.to_csv(output/'asignacion_particiones.csv', index=False)
    rows = []
    with threadpool_limits(limits=1):
        for target in ['CO', 'NOX']:
            for k, (a, b) in enumerate(folds, 1):
                model, config = search(frame.iloc[tr[a]], target, output/f'busqueda_{target}_{k}.csv')
                prediction = model.predict(frame.iloc[tr[b]][PREDICTORS])
                rows.append(dict(objetivo=target, fold=k, parametros=json.dumps(config),
                    MSE=mean_squared_error(frame.iloc[tr[b]][target], prediction),
                    R2=r2_score(frame.iloc[tr[b]][target], prediction)))
                frame.iloc[tr[b]][['anio', 'orden_registro', 'bloque', target]].assign(
                    prediccion=prediction).to_csv(output/f'OOF_{target}_{k}.csv.gz', index=False)
            model, config = search(frame.iloc[tr], target, output/f'busqueda_{target}_final.csv')
            joblib.dump(dict(formato='gas-turbine-entregable2-v1', estimador=model, objetivo=target,
                familia='Kernel operativo', predictores=PREDICTORS, n_ajuste=len(tr), parametros=config),
                output/f'modelo_{target}.joblib', compress=3)
            prediction = model.predict(frame.iloc[te][PREDICTORS])
            frame.iloc[te][['anio', 'orden_registro', 'bloque', target]].assign(prediccion=prediction).to_csv(
                output/f'prueba_{target}.csv.gz', index=False)
    pd.DataFrame(rows).to_csv(output/'validacion_externa.csv', index=False)
    return output
