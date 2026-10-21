"""Fixed-fold seed sensitivity and controlled kernel ablations, resumable offline."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits
from dashboard.models import registry, artifact_path
from emission_models.kernel_operativo import operational_features, PREDICTORS
from tools.export_explainability import protected_files
from tools.export_hierarchical_statistics import StationaryOriginalBlocks
from tools.export_kernel_significance import holm

OUT = ROOT/'data/stability_ablation'
CACHE = ROOT.parent/'tmp/stability_ablation_cache'
SEEDS = [42, 7, 21, 84, 123]
OFFICIAL = ROOT/'book/informe_resultados/entregable2_anidada'
KERNEL = ROOT/'book/informe_resultados/kernel_operativo'
KEYS = ['anio', 'orden_registro', 'bloque']
VARIANTS = ['completo', 'sin_derivadas', 'sin_pesos', 'sin_adaptaciones', 'solo_ET', 'solo_SVR']
VERSION = 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def frozen_hashes():
    return protected_files()


def purge(frame, candidates, reserved):
    keep = np.ones(len(candidates), dtype=bool)
    years, order = frame.anio.to_numpy(), frame.orden_registro.to_numpy()
    for year in np.unique(years):
        mask = years[candidates] == year
        points = np.sort(order[reserved[years[reserved] == year]])
        positions = order[candidates[mask]]
        keep[mask] = (np.searchsorted(points, positions-24, side='left') ==
                      np.searchsorted(points, positions+24, side='right'))
    return np.asarray(candidates)[keep]


def inner_splits(frame):
    mapping, rng = {}, np.random.default_rng(42)
    for _, group in frame.groupby('anio', sort=True):
        blocks = np.unique(group.bloque)
        rng.shuffle(blocks)
        mapping.update({int(b): j % 3 for j, b in enumerate(blocks)})
    folds = frame.bloque.map(mapping).to_numpy()
    splits = []
    for k in range(3):
        va = np.flatnonzero(folds == k)
        tr = purge(frame, np.flatnonzero(folds != k), va)
        assert not set(frame.iloc[tr].bloque) & set(frame.iloc[va].bloque)
        splits.append((tr, va))
    return splits


def load_frames():
    data = pd.read_csv(ROOT/'data/observaciones.csv.gz')
    assignment = pd.read_csv(OFFICIAL/'asignacion_particiones.csv.gz')
    pd.testing.assert_frame_equal(data[KEYS+['particion']], assignment[KEYS+['particion']])
    dev = data.loc[data.particion.eq('entrenamiento_CV')].reset_index(drop=True)
    dev['fold_validacion'] = assignment.loc[assignment.particion.eq('entrenamiento_CV'), 'fold_validacion'].to_numpy()
    test = data.loc[data.particion.eq('prueba_bloques')].reset_index(drop=True)
    assert (len(dev), len(test)) == (27570, 7411)
    splits = []
    counts = pd.read_csv(OFFICIAL/'conteos_externos.csv.gz')
    for k in range(1, 6):
        va = np.flatnonzero(dev.fold_validacion.eq(k))
        tr = purge(dev, np.flatnonzero(dev.fold_validacion.ne(k)), va)
        assert not set(dev.iloc[tr].bloque) & set(dev.iloc[va].bloque)
        expected = counts.loc[counts.fold.eq(k)].iloc[0]
        assert (len(tr), len(va)) == (expected.n_train, expected.n_val)
        splits.append((tr, va))
    return dev, test, splits


def weighted(X, environmental=1., temperature=1., pressure=1.):
    weights = np.array([environmental]*3+[pressure]*2+[temperature]*2+[1., pressure])
    if X.shape[1] == 11:
        weights = np.r_[weights, temperature, 1.]
    return np.asarray(X)*weights


def kernel_configuration(target, stage):
    stages = pd.read_csv(KERNEL/'etapas.csv')
    row = stages.loc[stages.objetivo.eq(target)&stages.etapa.eq(stage)].iloc[0]
    config = json.loads(row.parametros)
    weights = [(.5, 1., 1.), (.5, .5, 2.), (.5, 2., .5), (2., 1., 1.)]
    config['weights'] = weights[config['members'][1]-8]
    return config


def template(target, model_id):
    payload = joblib.load(artifact_path(registry()[model_id]['respuestas'][target]))
    estimator = payload['estimador']
    return estimator


def branch_template(original, kind, config, derived, use_weights, seed):
    base = clone(original)
    steps = []
    if derived:
        steps.append(FunctionTransformer(operational_features))
    steps.append(StandardScaler())
    if kind == 'svr':
        e, t, p = config['weights'] if use_weights else (1., 1., 1.)
        steps.append(FunctionTransformer(weighted, kw_args=dict(environmental=e, temperature=t, pressure=p)))
    estimator = clone(base.regressor.steps[-1][1])
    if kind == 'tree':
        depth, leaf = [(20, 2), (None, 5)][config['members'][0]-6]
        estimator.set_params(max_depth=depth, min_samples_leaf=leaf, random_state=seed, n_jobs=1)
    steps.append(estimator)
    base.set_params(regressor=make_pipeline(*steps))
    return base


def frame_hash(frame, target):
    return hashlib.sha256(pd.util.hash_pandas_object(frame[KEYS+PREDICTORS+[target]], index=False).values.tobytes()).hexdigest()


def branch_prediction(original, kind, config, derived, use_weights, seed, train, evaluation, target):
    # Cache keys include the exact train/evaluation rows, model parameters and code.
    estimator = branch_template(original, kind, config, derived, use_weights, seed)
    identity = dict(version=VERSION, code=sha(__file__), train=frame_hash(train, target),
                    evaluation=frame_hash(evaluation, target), target=target, kind=kind,
                    params=str(estimator.get_params(deep=True)))
    # Function reprs contain process addresses; use stable structured settings instead.
    identity['params'] = dict(derived=derived, weights=config['weights'] if kind=='svr' and use_weights else None,
        model=estimator.regressor.steps[-1][1].get_params())
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    path = CACHE/(key+'.joblib')
    if path.exists():
        return joblib.load(path)
    start = perf_counter()
    estimator.fit(train[PREDICTORS], train[target])
    assert estimator.transformer_.n_samples_seen_ == len(train)
    np.testing.assert_allclose(estimator.transformer_.mean_[0], train[target].mean(), rtol=1e-12)
    result = dict(prediction=estimator.predict(evaluation[PREDICTORS]), seconds=perf_counter()-start,
                  n_train=len(train), identity=identity)
    temporary = path.with_suffix('.partial')
    joblib.dump(result, temporary, compress=3)
    temporary.replace(path)
    del estimator
    gc.collect()
    return result


def kernel_stage(target, stage, train, evaluation, seed, variant, original):
    config = kernel_configuration(target, stage)
    derived = variant not in ['sin_derivadas', 'sin_adaptaciones']
    use_weights = variant not in ['sin_pesos', 'sin_adaptaciones']
    members = [0] if variant == 'solo_ET' else [1] if variant == 'solo_SVR' else [0, 1]
    start = perf_counter()
    columns, cross_columns = [], []
    for member in members:
        kind = ['tree', 'svr'][member]
        branch_seed = seed if member == 0 else 42
        full = branch_prediction(original.models[member], kind, config, derived, use_weights,
                                 branch_seed, train, evaluation, target)
        columns.append(full['prediction'])
        if len(members) > 1:
            oof = np.full(len(train), np.nan)
            for a, b in inner_splits(train):
                part = branch_prediction(original.models[member], kind, config, derived, use_weights,
                    branch_seed, train.iloc[a], train.iloc[b], target)
                oof[b] = part['prediction']
            assert np.isfinite(oof).all()
            cross_columns.append(oof)
    if len(members) == 1:
        prediction = columns[0]
    else:
        meta = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),
            Ridge(alpha=config['lambda']*len(train))), transformer=StandardScaler())
        meta.fit(np.column_stack(cross_columns), train[target])
        prediction = meta.predict(np.column_stack(columns))
    return prediction, perf_counter()-start


def deterministic(model):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.svm import LinearSVR
    from xgboost import XGBRegressor
    if isinstance(model, (KNeighborsRegressor, Ridge)):
        return True
    if isinstance(model, Lasso):
        return model.selection == 'cyclic'
    if isinstance(model, LinearSVR):
        return model.dual is False
    if isinstance(model, XGBRegressor):
        settings = model.get_params()
        return settings.get('booster', 'gbtree') in [None, 'gbtree'] and all(
            settings.get(key) in [None, 1, 1.] for key in
            ['subsample', 'colsample_bytree', 'colsample_bylevel', 'colsample_bynode'])
    return False


def base_stage(target, model_id, seed, stage, train, evaluation, original):
    spec = registry()[model_id]
    estimator = clone(original)
    if stage != 'final':
        selected = pd.read_csv(ROOT/'data/resultados.csv')
        method = selected.loc[selected.objetivo.eq(target)&selected.modelo.eq(spec['nombre']), 'optimizador'].iloc[0]
        external = pd.read_csv(OFFICIAL/'validacion_externa.csv.gz')
        row = external.loc[external.objetivo.eq(target)&external.modelo.eq(spec['nombre'])&
                           external.optimizador.eq(method)&external.fold.eq(int(stage.split('_')[-1]))].iloc[0]
        estimator.set_params(**{'regressor__modelo__'+key: value for key, value in json.loads(row.parametros).items()})
    random_keys = {key: seed for key in estimator.get_params() if key.endswith('random_state')}
    estimator.set_params(**random_keys)
    start = perf_counter()
    estimator.fit(train[PREDICTORS], train[target])
    assert estimator.transformer_.n_samples_seen_ == len(train)
    return estimator.predict(evaluation[PREDICTORS]), perf_counter()-start


def task(target, model_id, seed, variant, phase):
    with threadpool_limits(limits=1):
        dev, test, splits = load_frames()
        name = f'{target}_{model_id}_{seed}_{variant}_{phase}'
        marker = CACHE/(name+'.json')
        path = OUT/(name+'.csv.gz')
        signature = dict(code=sha(__file__), observations=sha(ROOT/'data/observaciones.csv.gz'),
                         assignment=sha(OFFICIAL/'asignacion_particiones.csv.gz'))
        if marker.exists() and path.exists() and json.loads(marker.read_text())['signature']==signature:
            return
        original = template(target, model_id)
        det = False if model_id=='kernel_operativo' else deterministic(original.regressor.steps[-1][1])
        canonical = pd.read_csv(OFFICIAL/'predicciones_OOF.csv.gz') if phase=='OOF' else pd.read_csv(OFFICIAL/'predicciones_prueba.csv.gz')
        reuse = model_id != 'kernel_operativo' and (seed==42 or det)
        frame = dev if phase=='OOF' else test
        prediction = np.full(len(frame), np.nan)
        stages = [(f'externo_{k}', dev.iloc[a], dev.iloc[b], b) for k,(a,b) in enumerate(splits,1)] if phase=='OOF' else [('final', dev, test, np.arange(len(test)))]
        details = []
        for stage, train, evaluation, indices in stages:
            start = perf_counter()
            if reuse:
                column = target+'_'+registry()[model_id]['nombre'].replace(' ', '_')
                prediction[indices] = canonical[column].to_numpy()[indices]
                seconds = 0.
            elif model_id=='kernel_operativo':
                prediction[indices], seconds = kernel_stage(target, stage, train, evaluation, seed, variant, original)
            else:
                prediction[indices], seconds = base_stage(target, model_id, seed, stage, train, evaluation, original)
            if model_id=='kernel_operativo' and seed==42 and variant=='completo':
                expected = pd.read_csv(KERNEL/f'predicciones_{"OOF" if phase=="OOF" else "prueba"}_{target}.csv.gz')
                np.testing.assert_allclose(prediction[indices], expected.prediccion.to_numpy()[indices], rtol=1e-8, atol=1e-7)
            details.append(dict(etapa=stage, R2=r2_score(evaluation[target], prediction[indices]),
                                segundos=seconds, n_train=len(train), n_val=len(evaluation)))
            print(f'{name} {stage} finalizado en {perf_counter()-start:.1f}s', flush=True)
        assert np.isfinite(prediction).all()
        frame[KEYS+[target]].assign(prediccion=prediction).to_csv(path, index=False)
        row = dict(objetivo=target, modelo_id=model_id, modelo=registry()[model_id]['nombre'],
            semilla=seed, variante=variant, fase=phase, determinista=det, reutilizado=reuse,
            R2=r2_score(frame[target], prediction), R2_folds_media=np.mean([s['R2'] for s in details]),
            MSE=mean_squared_error(frame[target], prediction), segundos=sum(s['segundos'] for s in details),
            etapas=details, archivo=path.name, sha256=sha(path), signature=signature)
        save_json(row, marker)
        return row


def aggregate():
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in CACHE.glob('*_OOF.json')]
    tests = [json.loads(p.read_text(encoding='utf-8')) for p in CACHE.glob('*_prueba.json')]
    data = pd.DataFrame([{k:v for k,v in r.items() if k not in ['etapas','signature']} for r in rows+tests])
    data.to_csv(OUT/'resultados.csv', index=False)
    seeds = data.loc[data.variante.eq('completo')&data.fase.eq('OOF')]
    summary = seeds.groupby(['objetivo','modelo_id','modelo','determinista']).R2.agg(['count','mean','std','min','max']).reset_index()
    summary['rango'] = summary['max']-summary['min']
    summary.to_csv(OUT/'sensibilidad_semillas.csv', index=False)
    dev, _, _ = load_frames()
    comparison = []
    for target in ['CO','NOX']:
        primary = next(r for r in rows if r['objetivo']==target and r['modelo_id']=='kernel_operativo' and r['semilla']==42 and r['variante']=='completo')
        base = pd.read_csv(OUT/primary['archivo']).prediccion.to_numpy()
        y = dev[target].to_numpy()
        for row in rows:
            if row['objetivo']!=target or row['modelo_id']!='kernel_operativo' or row['semilla']!=42 or row['variante']=='completo':
                continue
            other = pd.read_csv(OUT/row['archivo']).prediccion.to_numpy()
            delta = (y-other)**2-(y-base)**2
            for length in [1.,2.,4.]:
                bootstrap = StationaryOriginalBlocks(dev,length,seed=42)
                distribution = []
                for args,_ in bootstrap.bootstrap(5000):
                    index = args[0]
                    distribution.append(float(np.mean(delta[index])/np.var(y[index])))
                estimates = np.asarray(distribution)
                low,high = np.quantile(estimates,[.025,.975])
                p = min(1.,2*min((np.sum(estimates<=0)+1)/5001,(np.sum(estimates>=0)+1)/5001))
                comparison.append(dict(objetivo=target,variante=row['variante'],bloques_medios=length,
                    R2_completo=primary['R2'],R2_ablacion=row['R2'],delta_R2=primary['R2']-row['R2'],
                    IC95_inf=low,IC95_sup=high,p_bootstrap=p))
    comparisons = pd.DataFrame(comparison)
    comparisons['p_Holm'] = np.nan
    for _,group in comparisons.groupby('bloques_medios'):
        comparisons.loc[group.index,'p_Holm'] = holm(group.p_bootstrap.to_numpy())
    comparisons.to_csv(OUT/'ablacion_contrastes.csv', index=False)
    return data


def run_target(target, phase):
    for model_id in registry():
        for seed in SEEDS:
            task(target, model_id, seed, 'completo', phase)
    for variant in VARIANTS[1:]:
        task(target, 'kernel_operativo', 42, variant, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--phase',choices=['all','OOF','prueba','report'],default='all')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    manifest = OUT/'metodologia.json'
    hashes = frozen_hashes()
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding='utf-8'))['huellas_protegidas']==hashes
    else:
        save_json(dict(version=VERSION, semillas=SEEDS, bloque=168,purga=24,folds_externos=5,crossfit=3,
            validacion_modificada=False, modelos_publicados_sustituidos=False,
            configuraciones='Configuración original seleccionada en el entrenamiento de cada pliegue; final solo para prueba.',
            alcance='Estabilidad del ajuste condicionada a configuraciones y particiones fijas; no repite la búsqueda.',
            ablacion='Cambios controlados con semilla 42, sin reajustar hiperparámetros; pesos normalizados a uno al retirarlos.',
            pruebas='Bootstrap pareado, 5.000 réplicas; bloques originales con longitudes medias 1, 2, 4; Holm conjunto de 10 contrastes por longitud.',
            huellas_protegidas=hashes,estado='En ejecución'),manifest)
    from concurrent.futures import ProcessPoolExecutor
    if args.phase in ['all','OOF','prueba']:
        phases = ['OOF','prueba'] if args.phase=='all' else [args.phase]
        for phase in phases:
            if phase=='prueba':
                required = [(t,m,s,'completo') for t in ['CO','NOX'] for m in registry() for s in SEEDS]
                required += [(t,'kernel_operativo',42,v) for t in ['CO','NOX'] for v in VARIANTS[1:]]
                assert all((CACHE/f'{t}_{m}_{s}_{v}_OOF.json').exists() for t,m,s,v in required), 'OOF incomplete; test phase blocked'
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_target,target,phase) for target in ['CO','NOX']]
                for future in futures:
                    future.result()
    if args.phase in ['all','report']:
        aggregate()
        assert hashes==frozen_hashes(), 'Protected published evidence changed'
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        metadata.update(estado='Completado',codigo_sha256=sha(__file__))
        save_json(metadata,manifest)
        print('Validación finalizada; modelos y particiones publicados intactos.',flush=True)


if __name__=='__main__':
    main()
