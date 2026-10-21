"""Archive recent completed summary metrics, keeping validation protocols distinct."""
from io import StringIO
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'Pruebas provisionales CO NOx'
COMMIT = 'aa8e9ae87324e969396bdb9ebe93b9093cbb08db'
GROUPS = {
    'bloques': 'Bloques 168; purga 24; CV externa por bloques; 27.570 desarrollo / 7.411 prueba',
    'aleatorio': '80/20 aleatorio sin estratificar; CV aleatoria 5 × 3; sin purga; 29.386 / 7.347',
    'estratificado': '80/20 estratificado por objetivo; CV estratificada 5 × 3; sin purga; 29.386 / 7.347',
    'comun_aleatoria': 'Mismos bloques externos y prueba del protocolo principal; selección y cross-fitting aleatorios'
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    sources, rows = [], []

    def load(relative):
        path = SOURCE/relative
        data = path.read_bytes()
        sources.append(dict(archivo=relative, sha256=sha(data)))
        return pd.read_csv(StringIO(data.decode('utf-8')))

    def add(frame, context, group, source, state='Provisional'):
        frame = frame.rename(columns={'R2_prueba':'R2_test', 'R2_CV':'R2_CV_promedio'}).copy()
        for record in frame.to_dict('records'):
            rows.append(dict(modelo=record['modelo'], objetivo=record['objetivo'],
                variante=context, grupo_validacion=group, protocolo=GROUPS[group],
                tipo_CV='Media de 5 pliegues externos' if state!='Pausado' else 'Pendiente',
                estado=state, politica='Sin limpieza', fuentes=source,
                **{m:record.get(m) for m in ['R2_train', 'R2_CV_promedio', 'R2_test', 'R2_OOF']}))

    for relative, context, group in [
        ('resultados_bases_random_20261021/resultados.csv', 'Comparadores · 80/20 aleatorio', 'aleatorio'),
        ('resultados_kernel_aleatorio_80_20_20261021/resultados_aleatorios.csv', 'Kernel · 80/20 aleatorio', 'aleatorio'),
        ('resultados_kernel_estratificado_80_20_20261021/resultados_estratificados.csv', 'Kernel · 80/20 estratificado por emisión', 'estratificado')]:
        add(load(relative), context, group, relative)

    relative = 'resultados_pesos_continuos_CO_20261020/resultados_comparacion.csv'
    frame = load(relative).loc[lambda d:d.modelo.eq('continuo')].copy()
    frame['modelo'] = 'Kernel operativo · pesos continuos del SVR'
    add(frame, 'Optimización continua de pesos del SVR', 'bloques', relative)

    relative = 'resultados_lmkl_pesos_stack_20261020/resultados_comparacion.csv'
    frame = load(relative).loc[lambda d:d.modelo.isin(['lmkl_et','et_svr_pesos'])].copy()
    frame['modelo'] = frame.modelo.map({'lmkl_et':'MKL localizado + Extra Trees → Ridge',
                                      'et_svr_pesos':'ET + SVR con pesos optimizados por stacking → Ridge'})
    add(frame, 'Selección de pesos por el error del stacking completo', 'bloques', relative)

    # These are the original controlled ablations, not new hyperparameter searches.
    path = ROOT/'data/stability_ablation/resultados.csv'
    data = path.read_bytes()
    source = 'dashboard:data/stability_ablation/resultados.csv'
    sources.append(dict(archivo=source, sha256=sha(data)))
    frame = pd.read_csv(StringIO(data.decode('utf-8')))
    frame = frame.loc[frame.modelo_id.eq('kernel_operativo') & frame.semilla.eq(42) & frame.variante.ne('completo')]
    names = {'sin_derivadas':'Kernel sin variables derivadas', 'sin_pesos':'Kernel sin pesos por sensores',
             'sin_adaptaciones':'Kernel sin derivadas ni pesos', 'solo_ET':'Extra Trees del kernel',
             'solo_SVR':'SVR ponderado del kernel'}
    ablations = []
    for (target, variant), group in frame.groupby(['objetivo','variante']):
        cv = group.loc[group.fase.eq('OOF')].iloc[0]
        test = group.loc[group.fase.eq('prueba')].iloc[0]
        ablations.append(dict(objetivo=target, modelo=names[variant], R2_CV_promedio=cv.R2_folds_media,
                              R2_OOF=cv.R2, R2_test=test.R2))
    add(pd.DataFrame(ablations), 'Ablación · configuración original fija por pliegue', 'bloques', source)

    relative = 'ablacion_TEY_20261021/resultados.csv'
    frame = load(relative).loc[lambda d:~d.modelo.eq('Kernel completo')].copy()
    frame['modelo'] = 'Kernel sin TEY ni su interacción'
    add(frame, 'Ablación TEY · configuración original fija por pliegue', 'bloques', relative)

    relative = 'comparacion_kernel_validaciones_20261021/resultados.csv'
    frame = load(relative)
    frame['modelo'] = 'Kernel operativo · selección interna aleatoria'
    add(frame, 'Comparación controlada sobre segmentos externos comunes', 'comun_aleatoria', relative)

    # Read immutable Git objects, without pulling code, cache, or model binaries.
    repo = ROOT.parent/'CO_NOx_Training'
    prefix = 'resultados/2026-10-21-stackings-linux/'
    payloads = {name:subprocess.check_output(['git','show',f'{COMMIT}:{prefix}{name}'], cwd=repo)
                for name in ['resultados.csv','etapas.csv','procedencia_publicacion.json']}
    provenance = json.loads(payloads['procedencia_publicacion.json'])
    assert provenance['completion']['estado'] == 'Completado'
    for name in ['resultados.csv','etapas.csv']:
        assert sha(payloads[name]) == provenance['files_sha256'][name]
    frame = pd.read_csv(StringIO(payloads['resultados.csv'].decode('utf-8')))
    stages = pd.read_csv(StringIO(payloads['etapas.csv'].decode('utf-8')))
    ids = {'Kernel operativo + KNN → Ridge':'kernel_knn',
           'Kernel operativo + CatBoost → Ridge':'kernel_catboost',
           'XGBoost + SVR operativo → Ridge':'xgb_svr'}
    for row in frame.itertuples():
        final = stages.loc[stages.objetivo.eq(row.objetivo)&stages.modelo.eq(ids[row.modelo])&stages.etapa.eq('final')]
        assert len(final) == 1 and final.iloc[0].n_train == 29386 and final.iloc[0].n_val == 7347
        frame.loc[frame.objetivo.eq(row.objetivo)&frame.modelo.eq(row.modelo), 'R2_train'] = final.iloc[0].R2_train
    private_source = f'entrenamiento@{COMMIT[:7]}/{prefix}resultados.csv'
    sources.append(dict(archivo=private_source, sha256=sha(payloads['resultados.csv']),
                        etapas_sha256=sha(payloads['etapas.csv']), commit=COMMIT))
    add(frame, 'Stackings adicionales · 80/20 aleatorio', 'aleatorio', private_source)

    pause = SOURCE/'kernel_catboost_bloques_NOX_20261021/configuracion.json'
    if pause.exists() and 'Pausado' in json.loads(pause.read_text(encoding='utf-8'))['estado']:
        add(pd.DataFrame([dict(modelo='Kernel operativo + CatBoost → Ridge', objetivo='NOX')]),
            'Prueba por bloques pausada; sin resultado final', 'bloques',
            'kernel_catboost_bloques_NOX_20261021/configuracion.json', 'Pausado')

    output = ROOT/'data/historial_adicional.csv'
    pd.DataFrame(rows).to_csv(output, index=False)
    metadata = dict(revision='2026-10-21', fuentes=sources, filas=len(rows),
                    archivo_sha256=sha(output.read_bytes()),
                    observacion='Solo resúmenes finales y estado explícito de la prueba pausada. '
                    'Las métricas ausentes no se estiman. CV siempre identifica la evaluación externa.')
    (ROOT/'data/historial_adicional.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'{len(rows)} resultados por emisión archivados, con protocolo y procedencia.')


if __name__ == '__main__':
    main()
