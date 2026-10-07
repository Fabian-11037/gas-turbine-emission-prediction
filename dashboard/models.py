"""Registro extensible: artefactos JSON y adaptadores independientes de la interfaz."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from .repository import DATA, observations, validate_inputs


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

    def permutation_importance(self, frame, target, repeats=10, seed=42):
        x = frame[self.predictors].to_numpy(dtype=float)
        definition = self.specification['respuestas'][target]
        beta = np.asarray(definition['coeficientes'],dtype=float)
        residual = x @ beta + definition['intercepto'] - frame[target].to_numpy(dtype=float)
        baseline = np.mean(residual**2)
        rng = np.random.default_rng(seed)
        rows = []
        for j, sensor in enumerate(self.predictors):
            # For a linear predictor this is exactly a full prediction after permutation.
            changes = [np.mean((residual + beta[j]*(rng.permutation(x[:,j])-x[:,j]))**2)-baseline
                       for _ in range(repeats)]
            rows.append({'sensor':sensor,'aumento_MSE':float(np.mean(changes)),
                         'desviacion':float(np.std(changes,ddof=1))})
        return pd.DataFrame(rows).sort_values('aumento_MSE',ascending=False).reset_index(drop=True)


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


@lru_cache(maxsize=32)
def feature_importance(model_id, target):
    adapter = predictor(model_id)
    if not hasattr(adapter,'permutation_importance'):
        return None
    holdout = observations().loc[observations().particion.eq('prueba_bloques')]
    return adapter.permutation_importance(holdout,target)
