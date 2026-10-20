"""Contrastes pareados contra comparadores publicados, sin entrenar modelos."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'book'/'informe_resultados'/'kernel_operativo'
OFFICIAL = ROOT/'book'/'informe_resultados'/'entregable2_anidada'
KEYS = ['anio', 'orden_registro', 'bloque']


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values))
    result[order] = np.minimum(1., np.maximum.accumulate(values[order]*np.arange(len(values), 0, -1)))
    return result


def paired_bootstrap(frame, target, starts, length, replicates=20000):
    y = frame[target].to_numpy()
    old = frame.referencia.to_numpy()-y
    new = frame.nuevo.to_numpy()-y
    units = (frame.orden_registro-frame.anio.map(starts))//length
    moments = frame[['anio']].assign(unidad=units, n=1., sy=y, sy2=y*y,
                                   old=old*old, new=new*new)
    columns = ['n', 'sy', 'sy2', 'old', 'new']
    blocks = moments.groupby(['anio', 'unidad'], sort=True)[columns].sum()
    if length == 168:
        assert len(blocks) == frame.groupby(['anio', 'bloque']).ngroups
    point = moments[columns].sum().to_numpy()/len(frame)
    gain = point[3]-point[4]
    delta = gain/(point[2]-point[1]**2)
    rng = np.random.default_rng(42)
    draws = np.zeros((replicates, len(columns)))
    for _, group in blocks.groupby(level='anio', sort=True):
        values = group.to_numpy()
        assert len(values) >= 2
        counts = rng.multinomial(len(values), np.full(len(values), 1/len(values)), size=replicates)
        totals = counts@values
        # Preserva la mezcla anual y pondera por filas, no por medias de bloques.
        draws += values[:, 0].sum()/len(frame)*totals/totals[:, [0]]
    gains = draws[:, 3]-draws[:, 4]
    variance = draws[:, 2]-draws[:, 1]**2
    assert (variance > 0).all()
    low, high = np.quantile(gains/variance, [.025, .975])
    p = (1+np.count_nonzero(abs(gains-gain) >= abs(gain)))/(replicates+1)
    return dict(delta_R2=delta, IC95_inf=low, IC95_sup=high,
                p_prueba_nominal=p, longitud_bloque=length, n=len(frame), n_unidades=len(blocks))


def corrected_cv(frame, target, reference_rows, stages):
    details = []
    for row in reference_rows.sort_values('fold').itertuples():
        values = frame.loc[frame.fold_validacion.eq(row.fold)]
        stage = stages.loc[stages.etapa.eq(f'externo_{row.fold}')].iloc[0]
        assert len(values) == row.n_val and row.n_train == stage.n_ajuste
        old = float(np.mean((values.referencia-values[target])**2))
        new = float(np.mean((values.nuevo-values[target])**2))
        np.testing.assert_allclose(old, row.RMSE**2, rtol=1e-10)
        np.testing.assert_allclose(new, stage.MSE, rtol=1e-10)
        details.append(dict(objetivo=target, referencia=row.modelo, fold=row.fold,
            n_train=row.n_train, n_val=row.n_val, MSE_referencia=old, MSE_kernel=new,
            delta_MSE=old-new))
    assert len(details) == 5
    d = np.array([r['delta_MSE'] for r in details])
    ratio = np.mean([r['n_val']/r['n_train'] for r in details])
    se = np.sqrt((1/5+ratio)*d.var(ddof=1))
    t = d.mean()/se if se else (0. if d.mean() == 0 else np.sign(d.mean())*np.inf)
    return dict(p_CV_nominal=2*stats.t.sf(abs(t), 4), delta_MSE_CV=d.mean(),
                folds_favor_kernel=int((d > 0).sum()), error_estandar_corregido=se,
                ratio_n_val_n_train=ratio, grados_libertad=4), details


def align(new, reference, assignment, target, column, cohort):
    expected = 7411 if cohort == 'prueba' else 27570
    assert len(new) == len(reference) == expected
    assert not new.duplicated(KEYS).any() and not reference.duplicated(KEYS).any()
    frame = new[KEYS+[target, 'prediccion']].rename(columns={'prediccion':'nuevo'}).merge(
        reference[KEYS+[target, column]].rename(columns={target:'check_target', column:'referencia'}),
        on=KEYS, validate='one_to_one')
    np.testing.assert_allclose(frame[target], frame.check_target, rtol=1e-12, atol=1e-12)
    frame = frame.drop(columns='check_target').merge(assignment[KEYS+['particion', 'fold_validacion']],
                                                  on=KEYS, validate='one_to_one')
    assert len(frame) == expected
    partition = 'prueba_bloques' if cohort == 'prueba' else 'entrenamiento_CV'
    assert frame.particion.eq(partition).all()
    assert np.isfinite(frame[[target, 'nuevo', 'referencia']]).all().all()
    return frame.sort_values(['anio', 'orden_registro'])


def export():
    paths = []
    def read(path):
        paths.append(path)
        return pd.read_csv(path)

    results = read(ROOT/'data'/'resultados.csv')
    registry = json.loads((ROOT/'data'/'modelos.json').read_text(encoding='utf-8'))
    available = {m['nombre'] for m in registry['modelos']}
    data = read(ROOT/'data'/'observaciones.csv.gz')
    assignment = read(OFFICIAL/'asignacion_particiones.csv.gz')
    pd.testing.assert_frame_equal(data[KEYS+['particion']].reset_index(drop=True),
                                  assignment[KEYS+['particion']].reset_index(drop=True))
    starts = data.groupby('anio').orden_registro.min().to_dict()
    official_test = read(ROOT/'data'/'predicciones_prueba.csv.gz')
    official_oof = read(OFFICIAL/'predicciones_OOF.csv.gz')
    external = read(OFFICIAL/'validacion_externa.csv.gz')
    stages = read(EVIDENCE/'etapas.csv')
    all_rows, folds = [], []
    with threadpool_limits(limits=1):
        for target in ['CO', 'NOX']:
            candidates = results.loc[results.objetivo.eq(target) & results.modelo.ne('Kernel operativo')]
            ref = candidates.loc[candidates.MSE_CV_externo.idxmin()]
            assert ref.modelo in available
            column = target+'_'+ref.modelo.replace(' ', '_')
            oof = align(read(EVIDENCE/f'predicciones_OOF_{target}.csv.gz'), official_oof,
                        assignment, target, column, 'OOF')
            cv, details = corrected_cv(oof, target, external.loc[
                external.objetivo.eq(target) & external.modelo.eq(ref.modelo) &
                external.optimizador.eq(ref.optimizador)], stages.loc[stages.objetivo.eq(target)])
            folds.extend(details)
            frame = align(read(EVIDENCE/f'predicciones_prueba_{target}.csv.gz'), official_test,
                          assignment, target, column, 'prueba')
            y = frame[target].to_numpy()
            variance = np.var(y)
            r2_new = 1-np.mean((frame.nuevo-y)**2)/variance
            r2_old = 1-np.mean((frame.referencia-y)**2)/variance
            expected = results.loc[results.objetivo.eq(target) & results.modelo.eq('Kernel operativo')].iloc[0]
            np.testing.assert_allclose([r2_new, r2_old], [expected.R2_test, ref.R2_test], rtol=1e-10)
            for length in [168, 336, 672]:
                record = paired_bootstrap(frame, target, starts, length)
                np.testing.assert_allclose(record['delta_R2'], r2_new-r2_old, atol=1e-12)
                all_rows.append(dict(objetivo=target, referencia=ref.modelo,
                    optimizador_referencia=ref.optimizador, R2_kernel=r2_new, R2_referencia=r2_old,
                    **record, **cv))
    blocks = pd.DataFrame(all_rows).sort_values(['longitud_bloque', 'objetivo']).reset_index(drop=True)
    for _, group in blocks.groupby('longitud_bloque'):
        blocks.loc[group.index, 'p_prueba_Holm2'] = holm(group.p_prueba_nominal)
        blocks.loc[group.index, 'p_CV_Holm2'] = holm(group.p_CV_nominal)
    summary = blocks.loc[blocks.longitud_bloque.eq(168)].copy().reset_index(drop=True)
    assert len(summary) == 2 and summary.n_unidades.eq(45).all()
    summary.to_csv(ROOT/'data'/'kernel_significancia.csv', index=False)
    blocks.to_csv(ROOT/'data'/'kernel_significancia_bloques.csv', index=False)
    pd.DataFrame(folds).to_csv(EVIDENCE/'significancia_dashboard_folds.csv', index=False)
    metadata = dict(revision='2026-10-20', orientacion='delta R2 = kernel menos referencia; positivo favorece kernel',
        hipotesis='Igualdad bilateral de MSE', alfa=.05, replicas=20000, semilla=42, bloque_principal=168,
        bloques_sensibilidad=[336, 672], n_prueba=7411, purga_validacion=24,
        folds_externos=5, folds_internos=3, prueba_reutilizada=True, modelos_entrenados=0,
        referencias=summary[['objetivo', 'referencia', 'optimizador_referencia']].to_dict('records'),
        criterio_referencia='Menor MSE externo por emision entre los comparadores publicados, excluyendo kernel.',
        ajuste='Holm2 conjunto para CO y NOX; prueba y CV se ajustan por separado.',
        limitacion='Inferencia exploratoria; reserva previamente consultada y seleccion adaptativa.',
        metodo_bootstrap='Pareado por bloques originales dentro de cada anio, mezcla anual fija; p bilateral centrado bajo H0.',
        codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fuentes_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (ROOT/'data'/'kernel_significancia_metodologia.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary[['objetivo','referencia','delta_R2','IC95_inf','IC95_sup','p_prueba_nominal',
                   'p_prueba_Holm2','p_CV_nominal','p_CV_Holm2']].to_string(index=False))


if __name__ == '__main__':
    export()
