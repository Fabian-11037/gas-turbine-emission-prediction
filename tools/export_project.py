"""Exportacion de artefactos y evidencias del JBook, sin entrenar modelos."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = [('knn','KNN'),('ridge','Ridge'),('lasso','Lasso'),
          ('decision_tree','Decision Tree'),('random_forest','Random Forest'),
          ('xgboost','XGBoost'),('svr_lineal','SVR lineal')]
DESCRIPTIONS = {
    'knn':'Regresión por vecinos próximos sobre sensores estandarizados.',
    'ridge':'Regresión lineal con regularización L2.',
    'lasso':'Regresión lineal con regularización L1.',
    'decision_tree':'Árbol de regresión con profundidad y tamaño de hoja ajustados.',
    'random_forest':'Ensamble de árboles de regresión con muestreo bootstrap.',
    'xgboost':'Árboles potenciados por gradiente; inferencia CPU.',
    'svr_lineal':'SVR lineal con pérdida ε-insensible cuadrática y penalización L2.'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exportar(book, repeats=10):
    import joblib
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits

    source = Path(book).resolve()/'informe_resultados'/'entregable2_anidada'
    data = ROOT/'data'
    destination = data/'models'
    destination.mkdir(exist_ok=True)
    read = lambda name: pd.read_csv(source/(name+'.csv.gz'))
    records = json.loads((source/'registro_modelos.json').read_text(encoding='utf-8'))
    summary = read('resultados_familias')
    observations = pd.read_csv(data/'observaciones.csv.gz')
    assignment = read('asignacion_particiones')
    for col in ['anio','orden_registro','bloque','particion']:
        np.testing.assert_array_equal(observations[col],assignment[col])
    predictors = records['predictores']
    holdout = observations.loc[observations.particion.eq('prueba_bloques')]
    expected = read('predicciones_prueba')
    np.testing.assert_array_equal(holdout.orden_registro,expected.orden_registro)
    definitions, importance, sources = [], [], []
    with threadpool_limits(limits=1):
        for mid, family in MODELS:
            responses, errors = {}, {}
            for target in ['CO','NOX']:
                item = next(r for r in records['modelos'] if r['familia']==family and r['objetivo']==target)
                path = source/item['ruta']
                assert digest(path)==item['sha256']
                payload = joblib.load(path)
                assert payload['predictores']==predictors and payload['n_ajuste']==27570
                estimator = payload['estimador']
                x = holdout[predictors]
                pred = estimator.predict(x)
                col = target+'_'+family.replace(' ','_')
                expected[col] = expected[col].astype(pred.dtype)
                np.testing.assert_allclose(pred,expected[col],rtol=1e-10,atol=1e-10)
                errors[target] = float(np.max(np.abs(pred-expected[col])))
                # Preserve native float32 predictions exactly in the exported CSV.
                expected[col] = expected[col].astype('float64')
                output = destination/path.name
                shutil.copy2(path,output)
                responses[target] = {'ruta':'models/'+path.name,'sha256':digest(output),
                    'optimizador':item['optimizador'],'parametros':payload['parametros'],
                    'seleccionado_por_cv':item['seleccionado_global']}
                sources.append(path)
                y = holdout[target].to_numpy()
                baseline = np.mean((pred-y)**2)
                rng = np.random.default_rng(42)
                for sensor in predictors:
                    changes = []
                    shuffled = x.copy()
                    for _ in range(repeats):
                        shuffled[sensor] = rng.permutation(x[sensor].to_numpy())
                        p = estimator.predict(shuffled)
                        changes.append(float(np.mean((p-y)**2)-baseline))
                    importance.append({'modelo':family,'objetivo':target,'sensor':sensor,
                        'aumento_MSE':float(np.mean(changes)),
                        'desviacion':float(np.std(changes,ddof=1)),
                        'MSE_base':float(baseline),'repeticiones':repeats,'semilla':42,'n':len(x)})
                print(f'{target} / {family}: artefacto verificado e importancia calculada.',flush=True)
            definitions.append({'id':mid,'nombre':family,'adaptador':'sklearn_pipeline_v1',
                'predictores':predictors,'respuestas':responses,'n_ajuste':27570,
                'descripcion':DESCRIPTIONS[mid],'error_max_exportacion':errors})
    summary = summary.rename(columns={'R2_CV_media':'R2_CV_promedio',
                                      'seleccionado_global':'seleccionado_por_CV'})
    summary.to_csv(data/'resultados.csv',index=False)
    read('intervalos_bootstrap').to_csv(data/'intervalos_bootstrap.csv',index=False)
    read('resultados_optimizadores').to_csv(data/'optimizacion.csv',index=False)
    pd.DataFrame(importance).to_csv(data/'importancia_permutacion.csv',index=False)
    expected.to_csv(data/'predicciones_prueba.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    (data/'modelos.json').write_text(json.dumps({'version':2,'modelos':definitions},ensure_ascii=False,indent=2),encoding='utf-8')
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    manifest['protocolo'].update({'n_folds':5,'n_folds_internos':3,
        'validacion':'CV anidada: cinco folds externos y tres internos por bloques, con purga de 24 registros originales.',
        'criterio_seleccion':'Menor MSE medio externo; búsqueda de hiperparámetros exclusivamente interna.',
        'metricas_CV':'Medias y desviaciones de los cinco folds externos anidados.',
        'familias_finales':{t:summary.loc[(summary.objetivo==t)&summary.seleccionado_por_CV,'modelo'].item() for t in ['CO','NOX']}})
    manifest['inferencia']='Catorce pipelines originales verificados, con transformaciones de sensores y respuesta; sin ajuste en el servidor.'
    manifest['versiones_modelos']=records['versiones']
    sources.extend([source/'registro_modelos.json',source/'resultados_familias.csv.gz',source/'configuracion.json'])
    manifest['fuentes_SHA256']={str(p.relative_to(Path(book).resolve())):digest(p) for p in sources}
    (data/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Exportación terminada: 7 familias, 14 modelos; no se ha entrenado ningún estimador.',flush=True)
    if (ROOT/'book'/'informe_resultados'/'kernel_operativo'/'registro.json').exists():
        from tools.export_kernel import exportar_kernel
        exportar_kernel()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--book',type=Path,default=ROOT/'book')
    args=parser.parse_args()
    exportar(args.book)
