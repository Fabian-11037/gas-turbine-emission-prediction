"""Curvas diagnosticas por bloques; no modifica modelos ni evalua la prueba."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from emission_models.kernel_operativo import OperationalStack, PREDICTORS
from kernel_operativo_training import estimators, inner_splits, purge

DATA = ROOT/'data'
EVIDENCE = ROOT/'book/informe_resultados'
FRACTIONS = [.2,.4,.6,.8,1.]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def sources():
    return {str(path.relative_to(ROOT)):sha(path) for path in [
        DATA/'observaciones.csv.gz',DATA/'modelos.json',
        EVIDENCE/'entregable2_anidada/asignacion_particiones.csv.gz',
        EVIDENCE/'entregable2_anidada/validacion_externa.csv.gz',
        EVIDENCE/'kernel_operativo/etapas.csv',Path(__file__),
        Path(__file__).with_name('kernel_operativo_training.py')]}


def development():
    observations = pd.read_csv(DATA/'observaciones.csv.gz')
    frame = observations.loc[observations.particion.eq('entrenamiento_CV')].copy()
    assignment = pd.read_csv(EVIDENCE/'entregable2_anidada/asignacion_particiones.csv.gz')
    frame = frame.merge(assignment[['anio','orden_registro','fold_validacion']],
                        on=['anio','orden_registro'],validate='one_to_one',sort=False).reset_index(drop=True)
    assert len(frame)==27570 and set(frame.fold_validacion)=={1,2,3,4,5}
    return frame


def subset_by_blocks(frame, candidates, fraction, fold):
    if not 0 < fraction <= 1:
        raise ValueError('La fraccion debe estar entre cero y uno.')
    pool = frame.iloc[candidates]
    rng = np.random.default_rng(42+fold)
    blocks = []
    for _, group in pool.groupby('anio',sort=True):
        ordered = np.sort(group.bloque.unique())
        rng.shuffle(ordered)
        blocks.extend(ordered[:max(1,int(np.ceil(len(ordered)*fraction)))])
    return np.asarray(candidates)[pool.bloque.isin(blocks).to_numpy()]


def separation(frame, train, valid):
    distances = []
    for year, group in frame.iloc[train].groupby('anio'):
        reserved = np.sort(frame.iloc[valid].loc[lambda d:d.anio.eq(year),'orden_registro'].to_numpy())
        positions = group.orden_registro.to_numpy()
        right = np.searchsorted(reserved,positions)
        distances.extend(np.abs(positions-reserved[np.clip(right,0,len(reserved)-1)]))
        distances.extend(np.abs(positions-reserved[np.clip(right-1,0,len(reserved)-1)]))
    return int(min(distances))


def baseline_template(model_id,target,params):
    import gc
    import joblib
    registry = json.loads((DATA/'modelos.json').read_text(encoding='utf-8'))
    spec = next(m for m in registry['modelos'] if m['id']==model_id)['respuestas'][target]
    payload = joblib.load(DATA/spec['ruta'])
    template = clone(payload['estimador'])
    del payload
    gc.collect()
    return template.set_params(**{f'regressor__modelo__{k}':v for k,v in params.items()})


def verify_scaling(model,n):
    assert model.transformer_.n_samples_seen_==n
    scalers = [step for _,step in model.regressor_.steps if isinstance(step,StandardScaler)]
    assert len(scalers)==1 and scalers[0].n_samples_seen_==n


def fit_kernel(frame,target,params):
    # El combinador solo ve predicciones de registros fuera del ajuste de cada rama.
    chosen = [estimators()[int(member)-6] for member in params['members']]
    oof = np.full((len(frame),2),np.nan)
    seen = np.zeros(len(frame),dtype=int)
    for train,valid in inner_splits(frame):
        assert separation(frame,train,valid)>24
        for j,template in enumerate(chosen):
            model = clone(template).fit(frame.iloc[train][PREDICTORS],frame.iloc[train][target])
            verify_scaling(model,len(train))
            oof[valid,j] = model.predict(frame.iloc[valid][PREDICTORS])
        seen[valid] += 1
    assert np.all(seen==1) and np.isfinite(oof).all()
    meta = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),
        Ridge(alpha=params['lambda']*len(frame))),transformer=StandardScaler()).fit(oof,frame[target])
    fitted = [clone(template).fit(frame[PREDICTORS],frame[target]) for template in chosen]
    for model in fitted:
        verify_scaling(model,len(frame))
    return OperationalStack(fitted,meta)


def fold_record(spec,target,fold):
    if spec['id']=='kernel_operativo':
        table = pd.read_csv(EVIDENCE/'kernel_operativo/etapas.csv')
        return table.loc[table.objetivo.eq(target)&table.etapa.eq(f'externo_{fold}')].iloc[0]
    table = pd.read_csv(EVIDENCE/'entregable2_anidada/validacion_externa.csv.gz')
    return table.loc[table.objetivo.eq(target)&table.modelo.eq(spec['nombre'])&
                     table.optimizador.eq(spec['respuestas'][target]['optimizador'])&table.fold.eq(fold)].iloc[0]


def run_job(job):
    model_id,target,fold,fraction,cache,fingerprint = job
    destination = Path(cache)/f'{model_id}_{target}_{fold}_{int(fraction*100)}.json'
    if destination.exists():
        row = json.loads(destination.read_text(encoding='utf-8'))
        if row['huella_fuentes']==fingerprint:
            return row
    tick = perf_counter()
    frame = development()
    valid = np.flatnonzero(frame.fold_validacion.eq(fold))
    candidates = np.flatnonzero(frame.fold_validacion.ne(fold))
    available = purge(frame,candidates,valid)
    train = subset_by_blocks(frame,available,fraction,fold)
    spec = next(m for m in json.loads((DATA/'modelos.json').read_text(encoding='utf-8'))['modelos'] if m['id']==model_id)
    reference = fold_record(spec,target,fold)
    expected = int(reference['n_ajuste'] if model_id=='kernel_operativo' else reference['n_train'])
    assert len(available)==expected
    assert not set(frame.iloc[train].bloque)&set(frame.iloc[valid].bloque)
    distance = separation(frame,train,valid)
    assert distance>24 and np.array_equal(purge(frame,train,valid),train)
    params = json.loads(reference.parametros)
    row = dict(modelo_id=model_id,modelo=spec['nombre'],objetivo=target,fold=fold,fraccion=fraction,
        n_train=len(train),n_train_disponible=len(available),n_validacion=len(valid),
        n_bloques_train=frame.iloc[train].bloque.nunique(),min_separacion=distance,
        huella_subconjunto=hashlib.sha256(frame.iloc[train][['anio','orden_registro']].to_csv(index=False).encode()).hexdigest(),
        parametros=json.dumps(params,sort_keys=True),huella_fuentes=fingerprint)
    with threadpool_limits(limits=1):
        if fraction==1.:
            row.update(R2_train=float(reference.R2_train),R2_validacion=float(reference.R2),
                RMSE_train=float(reference.RMSE_train),RMSE_validacion=float(reference.RMSE),
                origen='validacion_externa_guardada')
        else:
            subset = frame.iloc[train].reset_index(drop=True)
            model = fit_kernel(subset,target,params) if model_id=='kernel_operativo' else baseline_template(model_id,target,params).fit(subset[PREDICTORS],subset[target])
            if model_id!='kernel_operativo':
                verify_scaling(model,len(subset))
            p_train = model.predict(subset[PREDICTORS])
            p_valid = model.predict(frame.iloc[valid][PREDICTORS])
            row.update(R2_train=float(r2_score(subset[target],p_train)),
                R2_validacion=float(r2_score(frame.iloc[valid][target],p_valid)),
                RMSE_train=float(np.sqrt(mean_squared_error(subset[target],p_train))),
                RMSE_validacion=float(np.sqrt(mean_squared_error(frame.iloc[valid][target],p_valid))),
                origen='ajuste_auxiliar_bloques')
    row['segundos'] = perf_counter()-tick
    destination.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--cache',type=Path,default=ROOT.parent/'tmp/curvas_aprendizaje')
    args = parser.parse_args()
    args.cache.mkdir(parents=True,exist_ok=True)
    protected = {str(p.relative_to(ROOT)):sha(p) for p in (DATA/'models').glob('*.joblib')}
    provenance = sources()
    fingerprint = hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    registry = json.loads((DATA/'modelos.json').read_text(encoding='utf-8'))['modelos']
    registry.sort(key=lambda m:m['id']!='kernel_operativo')
    jobs = [(m['id'],target,fold,fraction,str(args.cache),fingerprint)
            for m in registry for target in ['CO','NOX'] for fraction in FRACTIONS for fold in range(1,6)]
    if args.pilot:
        row = run_job(jobs[0])
        print(json.dumps(row,ensure_ascii=False),flush=True)
        return
    rows, tick = [],perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job,job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            (args.cache/'avance.json').write_text(json.dumps(dict(completados=len(rows),total=len(jobs),
                segundos=perf_counter()-tick,ultimo={k:row[k] for k in ['modelo_id','objetivo','fold','fraccion']})),encoding='utf-8')
            if len(rows)%25==0:
                print(f'{len(rows)}/{len(jobs)} puntos terminados',flush=True)
    table = pd.DataFrame(rows).sort_values(['modelo_id','objetivo','fraccion','fold'])
    assert len(table)==400 and not table[['modelo_id','objetivo','fraccion','fold']].duplicated().any()
    assert protected=={name:sha(ROOT/name) for name in protected}, 'Se modificaron modelos publicados.'
    assert provenance==sources(), 'Las fuentes cambiaron durante el calculo.'
    table.to_csv(DATA/'curvas_aprendizaje.csv',index=False)
    metadata = dict(version=1,semilla=42,fracciones=FRACTIONS,pliegues=5,longitud_bloque=168,purga=24,
        seleccion='Bloques completos disponibles por anio, aleatorizados y anidados dentro de cada pliegue.',
        hiperparametros='Configuracion elegida por validacion interna del pliegue original; fija para los cinco tamanos.',
        apilamiento='Cross-fitting interno de tres grupos con purga para las predicciones de Ridge; escalado local.',
        extremo_100='Metricas guardadas del ajuste externo original; mismo entrenamiento y validacion.',
        dispersion='Media y desviacion estandar entre cinco pliegues; no intervalo de confianza.',
        alcance='Diagnostico de las configuraciones publicadas, no nueva seleccion de modelos.',
        usa_prueba=False,modelos_publicados_sin_cambios=True,filas=len(table),
        fuentes_sha256=provenance,modelos_sha256=protected,segundos=perf_counter()-tick)
    (DATA/'curvas_aprendizaje_metodologia.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Curvas exportadas; modelos y prueba sin cambios.',flush=True)


if __name__=='__main__':
    main()
