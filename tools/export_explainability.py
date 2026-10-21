"""Precompute SHAP/LIME using frozen artifacts; no predictive model is fitted."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.models import registry, artifact_path
from emission_models.kernel_operativo import PREDICTORS
from tools.export_kernel import ImportancePredictor

OUTPUT = ROOT/'data/explainability'
SEED = 42
GLOBAL_BUDGET = 38  # Two forward/reverse permutation cycles for nine sensors.
REFINED_BUDGET = 190
CHUNK = 100


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def protected_files():
    paths = [ROOT/'data/modelos.json', ROOT/'data/resultados.csv',
        ROOT/'data/observaciones.csv.gz', ROOT/'data/predicciones_prueba.csv.gz',
        ROOT/'book/informe_resultados/entregable2_anidada/asignacion_particiones.csv.gz',
        ROOT/'book/informe_resultados/entregable2_anidada/configuracion.json']
    paths += sorted((ROOT/'data/models').rglob('*.joblib'))
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def stratified_background(dev, per_year=2):
    return pd.concat([g.sample(per_year, random_state=SEED)
                      for _, g in dev.groupby('anio', sort=True)]).reset_index(drop=True)


def choose_cases(test, predictions, target):
    errors = np.column_stack([abs(predictions[(mid,target)]-test[target].to_numpy())
                              for mid in ['kernel_operativo','xgboost']])
    good = int(np.argmin(errors.max(axis=1)))
    bad = int(np.argmax(errors.min(axis=1)))
    return {'error_bajo': good, 'error_alto': bad}


class FrozenPredictor:
    def __init__(self, model_id, target):
        self.model_id = model_id
        response = registry()[model_id]['respuestas'][target]
        self.estimator = joblib.load(artifact_path(response))['estimador']
        self.accelerated = ImportancePredictor(self.estimator) if model_id == 'kernel_operativo' else self.estimator

    def __call__(self, values):
        frame = pd.DataFrame(np.asarray(values, dtype=float), columns=PREDICTORS)
        return np.asarray(self.accelerated.predict(frame), dtype=float)

    def verify(self, frame):
        expected = np.asarray(self.estimator.predict(frame[PREDICTORS]))
        actual = self(frame[PREDICTORS])
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9)
        return float(np.max(abs(actual-expected)))


def global_values(predict, test, background, stem, limit=None):
    import shap
    n = len(test) if limit is None else min(limit,len(test))
    paths = []
    masker = shap.maskers.Independent(background[PREDICTORS].to_numpy(), max_samples=len(background))
    for start in range(0,n,CHUNK):
        path = OUTPUT/f'checkpoint_{stem}_{start:05d}.csv.gz'
        paths.append(path)
        if path.exists():
            continue
        rows = test.iloc[start:min(start+CHUNK,n)]
        explainer = shap.PermutationExplainer(predict, masker, feature_names=PREDICTORS, seed=SEED+start)
        explanation = explainer(rows[PREDICTORS].to_numpy(), max_evals=GLOBAL_BUDGET,
                               batch_size=1024, silent=True)
        values = np.asarray(explanation.values)
        prediction = predict(rows[PREDICTORS])
        base = np.asarray(explanation.base_values).reshape(-1)
        np.testing.assert_allclose(base+values.sum(axis=1), prediction, rtol=1e-8, atol=1e-7)
        result = rows[['anio','orden_registro','CO','NOX']+PREDICTORS].reset_index(drop=True).copy()
        for j, feature in enumerate(PREDICTORS):
            result['shap_'+feature] = values[:,j]
        result['base'] = base
        result['prediccion'] = prediction
        result.to_csv(path, index=False, compression={'method':'gzip','mtime':0})
        print(f'{stem}: {min(start+CHUNK,n)}/{n}', flush=True)
    result = pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    result.to_csv(OUTPUT/f'global_{stem}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    return result


def local_values(predict, dev, test, background, cases, target):
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    rows, summaries = [], []
    for label,index in cases.items():
        x = test.iloc[index][PREDICTORS].to_numpy(dtype=float)
        np.random.seed(SEED)
        exact = shap.KernelExplainer(predict, background[PREDICTORS].to_numpy(), feature_names=PREDICTORS)
        phi = np.asarray(exact.shap_values(x[None,:], nsamples=510, l1_reg=0., silent=True)).reshape(-1)
        prediction = float(predict(x[None,:])[0])
        base = float(exact.expected_value)
        np.testing.assert_allclose(base+phi.sum(), prediction, rtol=1e-9, atol=1e-8)
        lime = LimeTabularExplainer(dev[PREDICTORS].to_numpy(), mode='regression',
            feature_names=PREDICTORS, discretize_continuous=True, discretizer='quartile',
            feature_selection='none', random_state=SEED)
        exp = lime.explain_instance(x, predict, num_features=9, num_samples=5000)
        weights = dict(exp.local_exp[1])
        conditions = dict((j,condition) for (j,_),(condition,_) in zip(exp.local_exp[1], exp.as_list()))
        local_prediction = float(np.asarray(exp.local_pred).reshape(-1)[0])
        np.testing.assert_allclose(float(exp.intercept[1])+sum(weights.values()), local_prediction,atol=1e-8)
        for j,feature in enumerate(PREDICTORS):
            rows.append({'caso':label,'sensor':feature,'valor':x[j], 'SHAP':phi[j],
                         'LIME':weights[j], 'condicion_LIME':conditions[j]})
        shap_top = set(np.argsort(abs(phi))[-3:].tolist())
        lime_values = np.array([weights[j] for j in range(9)])
        lime_top = set(np.argsort(abs(lime_values))[-3:].tolist())
        summaries.append({'caso':label,'anio':int(test.iloc[index].anio),
            'orden_registro':int(test.iloc[index].orden_registro), 'observado':float(test.iloc[index][target]),
            'prediccion':prediction,'error_absoluto':abs(prediction-float(test.iloc[index][target])),
            'base_SHAP':base, 'suma_SHAP':float(phi.sum()), 'aditividad_error':abs(base+phi.sum()-prediction),
            'intercepto_LIME':float(exp.intercept[1]), 'prediccion_LIME':local_prediction,
            'error_aproximacion_LIME':abs(prediction-local_prediction), 'R2_local_LIME':float(exp.score),
            'rho_magnitud':float(spearmanr(abs(phi),abs(lime_values)).statistic),
            'coincidencias_top3':len(shap_top & lime_top),
            'signos_coincidentes':int(np.sum(np.sign(phi)==np.sign(lime_values)))})
    return pd.DataFrame(rows),pd.DataFrame(summaries)


def precision_check(predict, test, background, global_frame):
    import shap
    # Fixed, outcome-independent subset; additional numerical budget, not model selection.
    indices = np.random.default_rng(SEED).choice(len(global_frame),min(64,len(global_frame)),replace=False)
    masker = shap.maskers.Independent(background[PREDICTORS].to_numpy(),max_samples=len(background))
    refined = shap.PermutationExplainer(predict,masker,seed=SEED,feature_names=PREDICTORS)(
        test.iloc[indices][PREDICTORS].to_numpy(), max_evals=REFINED_BUDGET,batch_size=1024,silent=True)
    initial = global_frame.iloc[indices][['shap_'+f for f in PREDICTORS]].to_numpy()
    a,b = np.mean(abs(initial),axis=0),np.mean(abs(refined.values),axis=0)
    return pd.DataFrame({'sensor':PREDICTORS,'media_abs_inicial':a,'media_abs_refinada':b}), {
        'n':len(indices), 'presupuesto_inicial':GLOBAL_BUDGET, 'presupuesto_refinado':REFINED_BUDGET,
        'rho_rangos':float(spearmanr(a,b).statistic),
        'coincidencias_top3':len(set(np.argsort(a)[-3:]) & set(np.argsort(b)[-3:]))}


def run(limit=None):
    import shap
    OUTPUT.mkdir(exist_ok=True)
    original = protected_files()
    observations = pd.read_csv(ROOT/'data/observaciones.csv.gz')
    dev = observations.loc[observations.particion.eq('entrenamiento_CV')].reset_index(drop=True)
    test = observations.loc[observations.particion.eq('prueba_bloques')].reset_index(drop=True)
    background = stratified_background(dev)
    background.to_csv(OUTPUT/'referencia_desarrollo.csv',index=False)
    predictions = {}
    all_local,all_cases,all_global,details = [],[],[],[]
    for mid in ['xgboost','kernel_operativo']:
        for target in ['CO','NOX']:
            frozen = FrozenPredictor(mid,target)
            predictions[(mid,target)] = frozen(test[PREDICTORS])
            del frozen
            gc.collect()
    cases = {t:choose_cases(test,predictions,t) for t in ['CO','NOX']}
    for mid in ['xgboost','kernel_operativo']:
        for target in ['CO','NOX']:
            tick = perf_counter()
            stem = f'{mid}_{target}'
            frozen = FrozenPredictor(mid,target)
            error_equivalence = frozen.verify(test.sample(64,random_state=SEED))
            global_frame = global_values(frozen,test,background,stem,limit)
            global_rows = pd.DataFrame({'sensor':PREDICTORS,
                'media_abs_SHAP':global_frame[['shap_'+f for f in PREDICTORS]].abs().mean().to_numpy()})
            local,case_summary = local_values(frozen,dev,test,background,cases[target],target)
            precision,diagnostic = precision_check(frozen,test,background,global_frame)
            precision.to_csv(OUTPUT/f'precision_{stem}.csv',index=False)
            for frame in [global_rows,local,case_summary]:
                frame['modelo_id'],frame['objetivo'] = mid,target
            all_local.append(local)
            all_cases.append(case_summary)
            all_global.append(global_rows)
            details.append({'modelo_id':mid,'objetivo':target,'n_global':len(global_frame),
                'error_equivalencia_max':error_equivalence,'precision':diagnostic,
                'segundos':perf_counter()-tick,
                'base_SHAP':float(global_frame.base.iloc[0]),
                'aditividad_global_max':float(abs(global_frame.base+global_frame[
                    ['shap_'+f for f in PREDICTORS]].sum(axis=1)-global_frame.prediccion).max())})
            print(f'Completado {stem}: {details[-1]}',flush=True)
            pd.concat(all_local).to_csv(OUTPUT/'local.csv',index=False)
            pd.concat(all_cases).to_csv(OUTPUT/'casos.csv',index=False)
            pd.concat(all_global).to_csv(OUTPUT/'resumen_global.csv',index=False)
            del frozen
            gc.collect()
    current = protected_files()
    if current != original:
        raise RuntimeError('A protected model, metric or validation artifact changed')
    metadata = {'version':1,'semilla':SEED,'modelos_reentrenados':0,'validacion_modificada':False,
        'n_desarrollo':len(dev),'n_prueba':len(test),'sensores':PREDICTORS,
        'referencia':{'n':len(background),'seleccion':'Dos registros aleatorios por año del desarrollo; semilla 42',
                      'identificadores':background[['anio','orden_registro']].to_dict('records')},
        'SHAP_global':{'metodo':'PermutationExplainer','max_evals':GLOBAL_BUDGET,
                       'ciclos_permutacion':2,'escala':'mg/m³','poblacion':'Prueba completa'},
        'SHAP_local':{'metodo':'KernelExplainer','nsamples':510,'l1_reg':0.,
                      'coaliciones':'Todas las 512 combinaciones, incluyendo vacío y conjunto completo'},
        'LIME':{'metodo':'LimeTabularExplainer','num_samples':5000,'discretizacion':'Cuartiles del desarrollo',
                'kernel_width':float(.75*np.sqrt(9)),'feature_selection':'none'},
        'seleccion_casos':'Comunes por emisión: mínimo del máximo error absoluto y máximo del mínimo error absoluto entre ambos modelos; solo interpretación.',
        'resultados':details,'huellas_protegidas':original,
        'versiones':{p:importlib.metadata.version(p) for p in ['shap','lime','numpy','pandas','scikit-learn','xgboost-cpu','scipy']},
        'codigo_sha256':sha(__file__),
        'fuentes':['https://shap.readthedocs.io/en/stable/generated/shap.PermutationExplainer.html',
                   'https://shap.readthedocs.io/en/stable/generated/shap.KernelExplainer.html',
                   'https://lime-ml.readthedocs.io/en/latest/lime.html']}
    (OUTPUT/'metodologia.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,help='Only for a separate smoke-test output directory')
    args = parser.parse_args()
    if args.limit:
        OUTPUT = ROOT.parent/'tmp/explainability_smoke'
    with threadpool_limits(limits=4):
        run(args.limit)
