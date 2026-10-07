"""Registro extensible: artefactos JSON y adaptadores independientes de la interfaz."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from .repository import DATA, validate_inputs


class LinearPredictor:
    def __init__(self, specification):
        self.specification = specification
        self.predictors = specification['predictores']
        if len(set(self.predictors)) != len(self.predictors):
            raise ValueError('Predictores duplicados en el registro.')
        for target, definition in specification['respuestas'].items():
            beta = np.asarray(definition['coeficientes'],dtype=float)
            if len(beta) != len(self.predictors) or not np.isfinite(beta).all():
                raise ValueError(f'Coeficientes inválidos para {target}.')
            if not np.isfinite(definition['intercepto']):
                raise ValueError(f'Intercepto inválido para {target}.')

    def predict(self, frame):
        x = validate_inputs(frame,self.predictors).to_numpy(dtype=float)
        return pd.DataFrame({t:x @ np.asarray(d['coeficientes']) + d['intercepto']
                             for t,d in self.specification['respuestas'].items()},index=frame.index)


ADAPTERS = {'linear_v1':LinearPredictor}


@lru_cache(maxsize=1)
def registry():
    payload = json.loads((DATA/'modelos.json').read_text(encoding='utf-8'))
    if payload['version'] != 1:
        raise ValueError('Versión del registro no soportada.')
    definitions = payload['modelos']
    if len({m['id'] for m in definitions}) != len(definitions):
        raise ValueError('Identificadores de modelos duplicados.')
    for m in definitions:
        if m['adaptador'] not in ADAPTERS:
            raise ValueError(f"Adaptador no implementado: {m['adaptador']}")
    return {m['id']:m for m in definitions}


@lru_cache(maxsize=8)
def predictor(model_id):
    definition = registry()[model_id]
    return ADAPTERS[definition['adaptador']](definition)


def predict(model_id, frame):
    return predictor(model_id).predict(frame)
