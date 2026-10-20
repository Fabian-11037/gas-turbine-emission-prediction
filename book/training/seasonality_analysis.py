"""Periodicidad exploratoria por posicion; nunca crea fechas ni ajusta predictores."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac

PERIODS = (24, 168)
BANDWIDTHS = (168, 336)


def harmonic_test(values, positions, period, bandwidth=336):
    y, t = np.asarray(values, dtype=float), np.asarray(positions)
    if y.ndim != 1 or t.shape != y.shape or len(y) < 4*period:
        raise ValueError('Se requieren cuatro ciclos y posiciones unidimensionales.')
    if not np.isfinite(y).all() or not np.isfinite(t).all():
        raise ValueError('Los valores y las posiciones deben ser finitos.')
    if np.any(t != t.astype(int)) or np.any(np.diff(t) <= 0):
        raise ValueError('Las posiciones deben ser enteras, unicas y crecientes.')
    t = t.astype(int)
    trend = (t-t.mean())/(t.max()-t.min())
    phase = 2*np.pi*t/period
    design = np.column_stack([np.ones(len(t)), trend, np.sin(phase), np.cos(phase)])
    fitted = OLS(y, design).fit()
    baseline = OLS(y, design[:, :2]).fit()
    if baseline.ssr <= np.finfo(float).eps:
        raise ValueError('La respuesta no tiene variacion residual suficiente.')
    # Ceros en las contribuciones HAC preservan huecos de desarrollo, sin imputar y.
    scores = np.zeros((t.max()-t.min()+1, design.shape[1]))
    scores[t-t.min()] = design*fitted.resid[:, None]
    covariance = cov_hac((scores, fitted.normalized_cov_params),
        nlags=min(bandwidth, len(scores)-1), use_correction=False)
    covariance *= len(y)/(len(y)-design.shape[1])
    restriction = np.eye(4)[2:]
    test = fitted.wald_test(restriction, cov_p=covariance, use_f=False, scalar=True)
    row = {'n': len(y), 'periodo': period, 'HAC_rezagos': bandwidth,
        'Wald_chi2': float(test.statistic), 'p': float(test.pvalue),
        'amplitud': float(np.hypot(*fitted.params[2:])),
        'R2_parcial': max(0., float(1-fitted.ssr/baseline.ssr)),
        'beta_seno': float(fitted.params[2]), 'beta_coseno': float(fitted.params[3])}
    return row, baseline.resid


def analyze(frame):
    required = ['anio', 'orden_registro', 'particion', 'CO', 'NOX']
    data = frame[required].sort_values(['anio', 'orden_registro']).copy()
    if data.duplicated(['anio', 'orden_registro']).any():
        raise ValueError('No puede haber posiciones duplicadas dentro de un ano.')
    rows, profiles = [], []
    for population in ['desarrollo', 'todos']:
        for year, annual in data.groupby('anio', sort=True):
            origin = annual.orden_registro.min()
            group = annual if population == 'todos' else annual.loc[
                annual.particion.eq('entrenamiento_CV')]
            positions = (group.orden_registro-origin).to_numpy()
            for target in ['CO', 'NOX']:
                for period in PERIODS:
                    for bandwidth in BANDWIDTHS:
                        row, residuals = harmonic_test(group[target], positions, period, bandwidth)
                        rows.append({'poblacion': population, 'anio': int(year),
                            'objetivo': target, **row})
                        if bandwidth == 336:
                            profile = pd.DataFrame({'fase': positions % period, 'residuo': residuals})
                            profile = profile.groupby('fase').residuo.agg(['mean', 'count']).reset_index()
                            profile['poblacion'], profile['anio'] = population, int(year)
                            profile['objetivo'], profile['periodo'] = target, period
                            profiles.append(profile)
    results = pd.DataFrame(rows)
    # Una familia fija de 40 contrastes por poblacion, no por filtro de pantalla.
    for population, group in results.groupby('poblacion'):
        results.loc[group.index, 'p_Holm'] = multipletests(group.p, method='holm')[1]
    results['evidencia_005'] = results.p_Holm.lt(.05)
    return results, pd.concat(profiles, ignore_index=True)


def export(data_path, output):
    data_path, output = Path(data_path), Path(output)
    results, profiles = analyze(pd.read_csv(data_path))
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output/'estacionalidad_pruebas.csv', index=False)
    profiles.to_csv(output/'estacionalidad_perfiles.csv.gz', index=False)
    metadata = {'periodos_registros': list(PERIODS), 'HAC_rezagos': list(BANDWIDTHS),
        'HAC_principal': 336, 'contrastes_Holm_por_poblacion': 40,
        'fecha_calendario_disponible': False, 'usa_modelos_predictivos': False,
        'datos_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'codigo_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'estacionalidad_metodologia.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    return results, profiles


def notebook_figure(results, profiles, population='todos'):
    import matplotlib.pyplot as plt
    colors = ['#157a7c', '#b65346', '#627f35', '#82648b', '#aa793b']
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout='constrained')
    for row, target in enumerate(['CO', 'NOX']):
        for col, period in enumerate(PERIODS):
            ax = axes[row, col]
            filtered = results.loc[(results.poblacion == population) &
                (results.objetivo == target) & (results.periodo == period) &
                (results.HAC_rezagos == 336)]
            phase = np.arange(period)
            for color, item in zip(colors, filtered.itertuples()):
                harmonic = item.beta_seno*np.sin(2*np.pi*phase/period)+item.beta_coseno*np.cos(2*np.pi*phase/period)
                ax.plot(phase, harmonic, label=str(item.anio), color=color)
            ax.axhline(0, color='#8b949c', lw=.7)
            ax.set(title=f'{target}: componente de {period} registros',
                xlabel='Posici\u00f3n dentro del ciclo (no calendario)',
                ylabel='Componente arm\u00f3nico (mg/m\u00b3)')
            ax.legend(ncol=5, fontsize=8)
    return fig


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    results, _ = export(root/'data/observaciones.csv.gz', root/'data')
    print(results.loc[results.HAC_rezagos == 336,
        ['poblacion', 'anio', 'objetivo', 'periodo', 'p_Holm', 'R2_parcial']].to_string(index=False))
