"""Registro de pipelines entrenados: inferencia local sin dependencias del JBook."""
from functools import lru_cache
import hashlib
import json
import gc
from threading import RLock
import pandas as pd
from .repository import DATA, importances, validate_inputs

_MODEL_LOCK = RLock()


def artifact_path(response):
    path=(DATA/response['ruta']).resolve()
    if not path.is_relative_to((DATA/'models').resolve()) or not path.is_file():
        raise ValueError('Ruta de modelo no válida o artefacto ausente.')
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest!=response['sha256']:
        raise ValueError('La huella del artefacto no coincide con el registro.')
    return path


class PipelinePredictor:
    def __init__(self,specification):
        import joblib
        self.predictors=specification['predictores']
        self.estimators={}
        for target,response in specification['respuestas'].items():
            payload=joblib.load(artifact_path(response))
            if (payload['formato']!='gas-turbine-entregable2-v1' or
                payload['predictores']!=self.predictors or payload['objetivo']!=target or
                payload['familia']!=specification['nombre'] or payload['n_ajuste']!=specification['n_ajuste']):
                raise ValueError('Metadatos del modelo incompatibles con el registro.')
            self.estimators[target]=payload['estimador']

    def predict(self,frame):
        x=validate_inputs(frame,self.predictors)
        return pd.DataFrame({t:m.predict(x) for t,m in self.estimators.items()},index=frame.index)


class StreamingPipelinePredictor:
    """Carga un contaminante a la vez para no retener dos bosques grandes."""
    def __init__(self, specification):
        self.specification = specification

    def predict(self, frame):
        import joblib
        spec = self.specification
        x = validate_inputs(frame, spec['predictores'])
        result = {}
        for target, response in spec['respuestas'].items():
            payload = joblib.load(artifact_path(response))
            if (payload['formato'] != 'gas-turbine-entregable2-v1' or
                payload['predictores'] != spec['predictores'] or payload['objetivo'] != target or
                payload['familia'] != spec['nombre'] or payload['n_ajuste'] != spec['n_ajuste']):
                raise ValueError('Metadatos del modelo incompatibles con el registro.')
            result[target] = payload['estimador'].predict(x)
            del payload
            gc.collect()
        return pd.DataFrame(result, index=frame.index)


ADAPTERS={'sklearn_pipeline_v1':PipelinePredictor,
          'sklearn_operational_stream_v1':StreamingPipelinePredictor}


@lru_cache(maxsize=1)
def registry():
    payload=json.loads((DATA/'modelos.json').read_text(encoding='utf-8'))
    if payload['version']!=2:
        raise ValueError('Versión del registro no soportada.')
    definitions=payload['modelos']
    if len({m['id'] for m in definitions})!=len(definitions):
        raise ValueError('Identificadores de modelos duplicados.')
    for m in definitions:
        if m['adaptador'] not in ADAPTERS or set(m['respuestas'])!={'CO','NOX'}:
            raise ValueError('Adaptador o respuestas no válidos.')
        if len(set(m['predictores']))!=9:
            raise ValueError('El modelo debe declarar los nueve sensores sin duplicados.')
        for response in m['respuestas'].values():
            artifact_path(response)
    return {m['id']:m for m in definitions}


@lru_cache(maxsize=1)
def _cached_predictor(model_id):
    definition=registry()[model_id]
    return ADAPTERS[definition['adaptador']](definition)


def predictor(model_id):
    # Avoid duplicate cold loads on concurrent requests.
    with _MODEL_LOCK:
        return _cached_predictor(model_id)


def predict(model_id,frame):
    with _MODEL_LOCK:
        return predictor(model_id).predict(frame)


def feature_importance(model_id,target):
    family=registry()[model_id]['nombre']
    frame=importances()
    result=frame.loc[(frame.modelo==family)&(frame.objetivo==target),
                     ['sensor','aumento_MSE','desviacion']]
    return result.sort_values('aumento_MSE',ascending=False).reset_index(drop=True).copy()
