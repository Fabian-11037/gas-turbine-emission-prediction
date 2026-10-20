"""Inferencia del stacking operativo; solo NumPy y estimadores sklearn guardados."""
import numpy as np

PREDICTORS = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']


def operational_features(X):
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[1] != 9:
        raise ValueError('Se requieren los nueve sensores en el orden declarado.')
    delta = values[:, 5]-values[:, 6]
    return np.column_stack([values, delta, values[:, 7]*delta])


def weight_operational_features(X, environmental=1., temperature=1., pressure=1.):
    weights = np.array([environmental, environmental, environmental, pressure,
                        pressure, temperature, temperature, 1., pressure, temperature, 1.])
    return np.asarray(X)*weights


class OperationalStack:
    def __init__(self, models, meta):
        self.models, self.meta = models, meta

    def predict(self, X):
        frame = X[PREDICTORS] if hasattr(X, 'columns') else X
        values = np.asarray(frame, dtype=float)
        if values.ndim != 2 or values.shape[1] != 9 or not np.isfinite(values).all():
            raise ValueError('Los nueve sensores deben ser numeros finitos.')
        predictions = np.column_stack([model.predict(frame) for model in self.models])
        return self.meta.predict(predictions)
