"""Measure frozen-model inference in isolated processes; never call fit."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dashboard.models import artifact_path,StreamingPipelinePredictor
from emission_models.kernel_operativo import PREDICTORS
from tools.export_explainability import protected_files
from tools.export_kernel import ImportancePredictor

OUT = ROOT/'data/efficiency'
REPEATS = 5


class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.observe,daemon=True)

    def observe(self):
        while not self.stop.wait(.01):
            self.peak = max(self.peak,self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.peak = max(self.peak,self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def measure(function,repeats=REPEATS):
    values = []
    prediction = None
    for _ in range(repeats):
        tick = perf_counter()
        prediction = np.asarray(function(),dtype=float)
        values.append(perf_counter()-tick)
    return prediction,values


def representation(estimator,model_id):
    if model_id=='kernel_operativo':
        tree,branch = estimator.models
        forest,svr = tree.regressor_[-1],branch.regressor_[-1]
        return dict(p_efectivas=11,n_arboles=len(forest.estimators_),
            profundidad=max(t.tree_.max_depth for t in forest.estimators_),n_soporte=len(svr.support_),
            cache_SVR_MB=svr.cache_size)
    fitted = estimator.regressor_[-1]
    result = dict(p_efectivas=int(fitted.n_features_in_))
    if hasattr(fitted,'estimators_'):
        result.update(n_arboles=len(fitted.estimators_),
                      profundidad=max(t.tree_.max_depth for t in fitted.estimators_))
    elif hasattr(fitted,'tree_'):
        result['profundidad'] = fitted.tree_.max_depth
    elif hasattr(fitted,'get_booster'):
        result.update(n_arboles=fitted.get_booster().num_boosted_rounds(),
                      profundidad=fitted.get_params()['max_depth'])
    elif hasattr(fitted,'_fit_method'):
        result.update(indice_vecinos=fitted._fit_method,k=fitted.n_neighbors)
    return result


def worker(model_id,target,mode):
    definition = next(m for m in json.loads((ROOT/'data/modelos.json').read_text(encoding='utf-8'))['modelos'] if m['id']==model_id)
    response = definition['respuestas'][target]
    frame = pd.read_csv(OUT/'lote_prueba.csv')
    x = frame[PREDICTORS]
    expected = frame[target+'_'+definition['nombre'].replace(' ','_')].to_numpy()
    base_memory = psutil.Process().memory_info().rss
    load_tick = perf_counter()
    description = {}
    with MemoryMonitor() as monitor,threadpool_limits(limits=1):
        if mode=='fragmentado':
            parts = response['fragmentos']
            for component in parts['arboles']+[parts['svr'],parts['meta']]:
                artifact_path(component)
            predict = lambda values:StreamingPipelinePredictor._predict_parts(parts,values)
        else:
            estimator = joblib.load(artifact_path(response))['estimador']
            description = representation(estimator,model_id)
            predictor = ImportancePredictor(estimator) if mode=='rbf_vectorizado' else estimator
            predict = predictor.predict
        loading = perf_counter()-load_tick
        predict(x.iloc[:1])
        rows = []
        for n in [1,100,1000]:
            predicted,times = measure(lambda:predict(x.iloc[:n]))
            np.testing.assert_allclose(predicted,expected[:n],rtol=1e-8,atol=1e-7)
            for trial,seconds in enumerate(times,1):
                rows.append(dict(modelo_id=model_id,modelo=definition['nombre'],objetivo=target,modo=mode,
                    n_lote=n,repeticion=trial,segundos=seconds,ms_por_registro=seconds/n*1000,
                    error_maximo=float(np.max(abs(predicted-expected[:n])))))
        loop = []
        if mode=='residente':
            predicted,times = measure(lambda:np.concatenate([np.asarray(predict(x.iloc[i:i+1])) for i in range(100)]),repeats=3)
            np.testing.assert_allclose(predicted,expected[:100],rtol=1e-8,atol=1e-7)
            loop = times
    result = dict(modelo_id=model_id,modelo=definition['nombre'],objetivo=target,modo=mode,
        segundos_carga=loading,RSS_base_MB=base_memory/2**20,RSS_pico_MB=monitor.peak/2**20,
        incremento_RSS_MB=(monitor.peak-base_memory)/2**20,
        artefacto_MB=artifact_path(response).stat().st_size/2**20 if mode!='fragmentado' else
            sum(artifact_path(c).stat().st_size for c in parts['arboles']+[parts['svr'],parts['meta']])/2**20,
        bucle_100_segundos=loop,estructura=description,mediciones=rows)
    (OUT/f'worker_{model_id}_{target}_{mode}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def run():
    OUT.mkdir(exist_ok=True)
    protected = protected_files()
    test = pd.read_csv(ROOT/'data/predicciones_prueba.csv.gz')
    data = pd.read_csv(ROOT/'data/observaciones.csv.gz')
    keys = ['anio','orden_registro','bloque']
    batch = test.merge(data[keys+PREDICTORS],on=keys,validate='one_to_one').sample(1000,random_state=42).reset_index(drop=True)
    batch.to_csv(OUT/'lote_prueba.csv',index=False)
    registry = json.loads((ROOT/'data/modelos.json').read_text(encoding='utf-8'))['modelos']
    completed = []
    for definition in registry:
        for target in ['CO','NOX']:
            modes = ['residente','fragmentado','rbf_vectorizado'] if definition['id']=='kernel_operativo' else ['residente']
            for mode in modes:
                path = OUT/f"worker_{definition['id']}_{target}_{mode}.json"
                command = [sys.executable,str(Path(__file__).resolve()),'--worker',definition['id'],target,mode]
                creation = subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
                subprocess.run(command,check=True,creationflags=creation,env={**os.environ,
                    'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
                completed.append(json.loads(path.read_text(encoding='utf-8')))
                print(f"{definition['id']} · {target} · {mode}: completado",flush=True)
    measurements = pd.DataFrame([row for item in completed for row in item['mediciones']])
    measurements.to_csv(OUT/'mediciones.csv',index=False)
    costs,comparison = [],[]
    historical = pd.read_csv(ROOT/'data/resultados.csv')
    for item in completed:
        rows = pd.DataFrame(item['mediciones'])
        medians = rows.groupby('n_lote').segundos.median()
        record = {key:value for key,value in item.items() if key not in ['mediciones','bucle_100_segundos','estructura']}
        record.update(item['estructura'])
        record.update({f'segundos_lote_{n}':medians.loc[n] for n in [1,100,1000]})
        h = historical.loc[historical.modelo.eq(item['modelo'])&historical.objetivo.eq(item['objetivo'])].iloc[0]
        record.update(segundos_ajuste_historico=h.segundos_ajuste,R2_prueba_original=h.R2_test,
            memoria_entrenamiento_MB=None,error_maximo=rows.error_maximo.max())
        if item['bucle_100_segundos']:
            record['segundos_bucle_100'] = float(np.median(item['bucle_100_segundos']))
            record['aceleracion_lote_100'] = record['segundos_bucle_100']/medians.loc[100]
        costs.append(record)
    cost_frame = pd.DataFrame(costs)
    cost_frame.to_csv(OUT/'resumen.csv',index=False)
    assert protected_files()==protected,'Models or validation changed'
    metadata = dict(version=1,semilla=42,modelos_reentrenados=0,validacion_modificada=False,
        repeticiones=REPEATS,repeticiones_bucle=3,n_lotes=[1,100,1000],hilos=1,
        memoria='Pico RSS observado en proceso aislado, muestreo cada 10 ms; incluye bibliotecas, carga y predicción.',
        tiempos='Tras una llamada de calentamiento; mediana de cinco repeticiones. La carga se registra aparte; el modo fragmentado vuelve a cargar componentes en cada llamada.',
        entrenamiento='Tiempos históricos del ajuste final cuando existen; no se vuelve a entrenar. La memoria de entrenamiento no fue registrada.',
        comparaciones='Mismos modelos y entradas: lote frente a llamadas individuales; kernel residente frente a fragmentado y RBF vectorizado solo para benchmark.',
        limites='Mediciones de una estación compartida; caché del sistema operativo no controlada. Se varía el tamaño del lote, no el tamaño de entrenamiento ni el número de sensores.',
        equipo=dict(sistema=platform.platform(),procesador=platform.processor(),nucleos_logicos=os.cpu_count(),RAM_GB=psutil.virtual_memory().total/2**30),
        huellas_protegidas=protected,codigo_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'metodologia.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',nargs=3,metavar=('MODELO','OBJETIVO','MODO'))
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
    else:
        run()
