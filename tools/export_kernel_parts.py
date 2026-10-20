"""Divide el bosque ya entrenado para inferencia con memoria acotada."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export():
    import gc
    import joblib
    from sklearn.base import clone

    registry_path = ROOT/'data'/'modelos.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    model = next(m for m in registry['modelos'] if m['id'] == 'kernel_operativo')
    output = ROOT/'data'/'models'/'kernel_parts'
    output.mkdir(exist_ok=True)

    def save(estimator, name, **metadata):
        path = output/name
        joblib.dump(estimator, path, compress=3)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return dict(ruta=str(path.relative_to(ROOT/'data')).replace('\\','/'), sha256=digest, **metadata)

    for target, response in model['respuestas'].items():
        with (ROOT/'data'/response['ruta']).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == response['sha256']
        original = joblib.load(ROOT/'data'/response['ruta'])['estimador']
        branch = original.models[0]
        forest = branch.regressor_[-1]
        groups = []
        for start in range(0, len(forest.estimators_), 50):
            part = copy.copy(branch)
            part.regressor = clone(branch.regressor)
            part.regressor_ = copy.copy(branch.regressor_)
            part.regressor_.steps = list(branch.regressor_.steps)
            subset = copy.copy(forest)
            subset.estimators_ = forest.estimators_[start:start+50]
            subset.n_estimators = len(subset.estimators_)
            part.regressor_.steps[-1] = (branch.regressor_.steps[-1][0], subset)
            groups.append(save(part, f'{target}_trees_{start//50}.joblib', n_arboles=len(subset.estimators_)))
            del part, subset
        response['fragmentos'] = dict(n_arboles=len(forest.estimators_), arboles=groups,
            svr=save(original.models[1], f'{target}_svr.joblib'),
            meta=save(original.meta, f'{target}_meta.joblib'))
        del original, branch, forest
        gc.collect()
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding='utf-8')
    record_path = ROOT/'book'/'informe_resultados'/'kernel_operativo'/'registro.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    for target in ['CO', 'NOX']:
        record['respuestas'][target]['fragmentos'] = model['respuestas'][target]['fragmentos']
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Dos bosques fragmentados; parametros y modelos originales sin cambios.')


if __name__ == '__main__':
    export()
