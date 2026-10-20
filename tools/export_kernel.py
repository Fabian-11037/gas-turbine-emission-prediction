"""Publica evidencia ya entrenada del kernel; nunca ajusta modelos."""
import argparse
import ast
import gc
import copy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAMILY = 'Kernel operativo'
EVIDENCE = ROOT/'book'/'informe_resultados'/'kernel_operativo'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(source):
    from sklearn.preprocessing import FunctionTransformer
    from emission_models.kernel_operativo import OperationalStack, operational_features, weight_operational_features
    result = OperationalStack(copy.deepcopy(source.models), copy.deepcopy(source.meta))
    for model in result.models:
        for name in ['regressor', 'regressor_']:
            for _, step in getattr(model, name).steps:
                if isinstance(step, FunctionTransformer) and step.func is not None:
                    functions = {'engineering': operational_features, 'sensor_weights': weight_operational_features}
                    step.func = functions[step.func.__name__]
    return result


class ImportancePredictor:
    """Evalua el mismo SVR con bloques de kernel BLAS, solo para diagnosticos."""
    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, X):
        import numpy as np
        from sklearn.metrics.pairwise import rbf_kernel
        tree, branch = self.estimator.models
        pipeline = branch.regressor_
        transformed = pipeline[:-1].transform(X)
        svr = pipeline[-1]
        values = np.empty(len(transformed))
        for start in range(0, len(values), 256):
            kernel = rbf_kernel(transformed[start:start+256], svr.support_vectors_, gamma=svr._gamma)
            values[start:start+256] = kernel @ svr.dual_coef_[0] + svr.intercept_[0]
        prediction = branch.transformer_.inverse_transform(values.reshape(-1, 1)).ravel()
        return self.estimator.meta.predict(np.column_stack([tree.predict(X), prediction]))


def complete_summary():
    import pandas as pd
    summary = pd.read_csv(EVIDENCE/'resultados.csv')
    stages = pd.read_csv(EVIDENCE/'etapas.csv')
    for target in ['CO', 'NOX']:
        outer = stages.loc[(stages.objetivo == target) & stages.etapa.ne('final')]
        assert len(outer) == 5
        mask = summary.objetivo.eq(target)
        summary.loc[mask, 'RMSE_CV_std'] = outer.RMSE.std()
        summary.loc[mask, 'MAE_CV_media'] = outer.MAE.mean()
        summary.loc[mask, 'MAE_CV_std'] = outer.MAE.std()
    summary.to_csv(EVIDENCE/'resultados.csv', index=False)


def exportar_kernel(source=None):
    import joblib
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from emission_models.kernel_operativo import PREDICTORS

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    data = ROOT/'data'
    observations = pd.read_csv(data/'observaciones.csv.gz')
    dev = observations.loc[observations.particion == 'entrenamiento_CV']
    test = observations.loc[observations.particion == 'prueba_bloques']
    if source is not None:
        source = Path(source).resolve()
        sys.path.insert(0, str(source.parent))
        summary = pd.read_csv(source/'resultados_provisionales.csv')
        summary = summary.loc[summary.modelo == 'ET SVR kernel operativo'].copy()
        summary['modelo'], summary['optimizador'] = FAMILY, 'Grid kernel'
        summary['parametros_finales'] = summary.parametros
        summary['mejor_en_familia'] = True
        stages = pd.read_csv(source/'resultados_etapas.csv')
        stages.loc[stages.modelo == 'ET SVR kernel operativo'].to_csv(EVIDENCE/'etapas.csv', index=False)
        (EVIDENCE/'busqueda.json').write_bytes((source/'configuracion.json').read_bytes())
        for target in ['CO', 'NOX']:
            for cohort in ['OOF', 'prueba']:
                (EVIDENCE/f'predicciones_{cohort}_{target}.csv.gz').write_bytes(
                    (source/f'predicciones_{cohort}_{target}_ET SVR kernel operativo.csv.gz').read_bytes())
        sys.path.insert(0, str(ROOT/'book'/'training'))
        from scipy import stats
        from sklearn.preprocessing import StandardScaler
        # Reutiliza el diagnostico publicado sin importar optimizadores de entrenamiento.
        tree = ast.parse((ROOT/'book'/'training'/'regression_entregable2.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'diagnosticos')
        namespace = dict(np=np, stats=stats, StandardScaler=StandardScaler, PREDICTORES=PREDICTORS)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<diagnosticos publicados>', 'exec'), namespace)
        diagnosticos = namespace['diagnosticos']
        from block_model import bootstrap_prueba
        definitions, intervals, architecture, errors = {}, [], {}, {}
        with threadpool_limits(limits=1):
            for target in ['CO', 'NOX']:
                original = joblib.load(source/f'modelo_{target}_ET SVR kernel operativo.joblib')
                estimator = portable(original['estimador'])
                expected = pd.read_csv(EVIDENCE/f'predicciones_prueba_{target}.csv.gz')
                prediction = estimator.predict(test[PREDICTORS])
                np.testing.assert_array_equal(expected.orden_registro, test.orden_registro)
                np.testing.assert_allclose(prediction, expected.prediccion, rtol=1e-12, atol=1e-12)
                errors[target] = float(np.max(np.abs(prediction-expected.prediccion.to_numpy())))
                output = data/'models'/f'{target}_Kernel_operativo.joblib'
                payload = dict(formato='gas-turbine-entregable2-v1', estimador=estimator,
                    predictores=PREDICTORS, objetivo=target, familia=FAMILY, n_ajuste=27570,
                    parametros=original['parametros'], unidad_respuesta='mg/m3')
                joblib.dump(payload, output, compress=3)
                definitions[target] = dict(ruta='models/'+output.name, sha256=sha(output),
                    optimizador='Grid kernel', parametros=original['parametros'], seleccionado_por_cv=True)
                index = summary.index[summary.objetivo == target][0]
                for column, value in diagnosticos(test, test[target], prediction).items():
                    summary.loc[index, column] = value
                for cohort, frame, values in [('train', dev, estimator.predict(dev[PREDICTORS])),
                    ('OOF', dev, pd.read_csv(EVIDENCE/f'predicciones_OOF_{target}.csv.gz').prediccion.to_numpy()),
                    ('test', test, prediction)]:
                    residual = values-frame[target].to_numpy()
                    summary.loc[index, f'sesgo_{cohort}'] = float(residual.mean())
                    summary.loc[index, f'MAPE_pct_{cohort}'] = float(100*np.mean(abs(residual/frame[target].to_numpy())))
                summary.loc[index, 'negativos_test'] = int((prediction < 0).sum())
                for metric, draws in bootstrap_prueba(test, test[target], prediction).items():
                    low, high = np.quantile(draws, [.025, .975])
                    intervals.append(dict(objetivo=target, modelo=FAMILY, metrica=metric, IC95_inf=low, IC95_sup=high))
                meta = estimator.meta
                scaler, ridge = meta.regressor_[0], meta.regressor_[-1]
                weights = meta.transformer_.scale_[0]*ridge.coef_/scaler.scale_
                intercept = float(meta.transformer_.mean_[0]+meta.transformer_.scale_[0]*ridge.intercept_-weights@scaler.mean_)
                architecture[target] = dict(coeficientes=weights.tolist(), intercepto=intercept,
                    pesos_svr=estimator.models[1].regressor_[2].kw_args,
                    alpha_ridge=float(ridge.alpha), lambda_ridge=original['parametros']['lambda'],
                    n_ajuste=27570, n_variables_derivadas=2)
                del original, estimator, payload, meta
                gc.collect()
        summary.to_csv(EVIDENCE/'resultados.csv', index=False)
        pd.DataFrame(intervals).to_csv(EVIDENCE/'intervalos.csv', index=False)
        (EVIDENCE/'arquitectura.json').write_text(json.dumps(architecture, indent=2), encoding='utf-8')
        record = dict(id='kernel_operativo', nombre=FAMILY, adaptador='sklearn_operational_stream_v1',
            predictores=PREDICTORS, respuestas=definitions, n_ajuste=27570,
            descripcion='Stacking Extra Trees + SVR RBF ponderado por grupos, con variables operativas derivadas y metaaprendiz Ridge.',
            error_max_exportacion=errors)
        (EVIDENCE/'registro.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    record = json.loads((EVIDENCE/'registro.json').read_text(encoding='utf-8'))
    record['adaptador'] = 'sklearn_operational_stream_v1'
    (EVIDENCE/'registro.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    (data/'kernel_operativo.json').write_bytes((EVIDENCE/'arquitectura.json').read_bytes())
    complete_summary()
    importance_path = EVIDENCE/'importancia.csv'
    importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
    with threadpool_limits(limits=1):
        for target, response in record['respuestas'].items():
            estimator = joblib.load(data/response['ruta'])['estimador']
            assert sha(data/response['ruta']) == response['sha256']
            x, y = test[PREDICTORS], test[target].to_numpy()
            fast = ImportancePredictor(estimator)
            native = estimator.predict(x)
            expected = pd.read_csv(EVIDENCE/f'predicciones_prueba_{target}.csv.gz')
            np.testing.assert_array_equal(test.orden_registro, expected.orden_registro)
            np.testing.assert_allclose(native, expected.prediccion, rtol=1e-12, atol=1e-12)
            record['error_max_exportacion'][target] = float(np.max(np.abs(native-expected.prediccion.to_numpy())))
            tree = estimator.models[0].regressor_[-1]
            svr = estimator.models[1].regressor_[-1]
            response['parametros'] = dict(
                ExtraTrees={k: tree.get_params()[k] for k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features']},
                SVR={**{k: svr.get_params()[k] for k in ['C', 'gamma', 'epsilon']},
                     'pesos': estimator.models[1].regressor_[2].kw_args},
                **{'lambda_Ridge': record['respuestas'][target]['parametros'].get('lambda',
                    record['respuestas'][target]['parametros'].get('lambda_Ridge'))})
            np.testing.assert_allclose(fast.predict(x), native, rtol=1e-10, atol=1e-10)
            baseline = float(np.mean((native-y)**2))
            for sensor in PREDICTORS:
                if len(importance) and ((importance.objetivo == target) & (importance.sensor == sensor) &
                    (importance.modelo_sha256 == response['sha256'])).any():
                    continue
                rng, changes = np.random.default_rng(42), []
                for _ in range(10):
                    shuffled = x.copy()
                    shuffled[sensor] = rng.permutation(x[sensor].to_numpy())
                    changes.append(float(np.mean((fast.predict(shuffled)-y)**2)-baseline))
                row = dict(modelo=FAMILY, objetivo=target, sensor=sensor, aumento_MSE=np.mean(changes),
                    desviacion=np.std(changes, ddof=1), MSE_base=baseline, repeticiones=10,
                    semilla=42, n=len(x), modelo_sha256=response['sha256'])
                if len(importance):
                    importance = importance.loc[~((importance.objetivo == target) & (importance.sensor == sensor))]
                importance = pd.concat([importance, pd.DataFrame([row])], ignore_index=True)
                importance.to_csv(importance_path, index=False)
                print(f'Importancia kernel {target}/{sensor} lista.', flush=True)
            del estimator, fast
            gc.collect()
    (EVIDENCE/'registro.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    registry_path = data/'modelos.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    registry['modelos'] = [m for m in registry['modelos'] if m['id'] != record['id']]+[record]
    summary = pd.read_csv(data/'resultados.csv')
    summary = pd.concat([summary.loc[summary.modelo != FAMILY], pd.read_csv(EVIDENCE/'resultados.csv')], ignore_index=True)
    summary['seleccionado_por_CV'] = False
    winners = {target: group.MSE_CV_externo.idxmin() for target, group in summary.groupby('objetivo')}
    for index in winners.values():
        summary.loc[index, 'seleccionado_por_CV'] = True
    for model in registry['modelos']:
        for target, response in model['respuestas'].items():
            response['seleccionado_por_cv'] = bool(summary.loc[winners[target], 'modelo'] == model['nombre'])
    summary.to_csv(data/'resultados.csv', index=False)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding='utf-8')
    for filename, addition in [('intervalos_bootstrap.csv', pd.read_csv(EVIDENCE/'intervalos.csv')),
                               ('importancia_permutacion.csv', importance)]:
        original = pd.read_csv(data/filename)
        pd.concat([original.loc[original.modelo != FAMILY], addition], ignore_index=True).to_csv(data/filename, index=False)
    predictions = pd.read_csv(data/'predicciones_prueba.csv.gz')
    for target in ['CO', 'NOX']:
        extra = pd.read_csv(EVIDENCE/f'predicciones_prueba_{target}.csv.gz')
        keys = ['anio', 'orden_registro', 'bloque']
        values = predictions[keys].merge(extra[keys+['prediccion']], on=keys, validate='one_to_one')
        predictions[f'{target}_Kernel_operativo'] = values.prediccion.to_numpy()
    predictions.to_csv(data/'predicciones_prueba.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    manifest_path = data/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['protocolo']['familias_finales'] = {target: summary.loc[index, 'modelo'] for target, index in winners.items()}
    manifest['inferencia'] = 'Dieciséis artefactos verificados: catorce comparadores y dos modelos de kernel operativo, sin ajuste en servidor.'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Kernel publicado: artefactos portables, resultados e importancia; sin reentrenar.', flush=True)
    from tools.export_kernel_significance import export
    export()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    exportar_kernel(parser.parse_args().source)
