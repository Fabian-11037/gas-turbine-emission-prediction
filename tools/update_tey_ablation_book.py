"""Export completed TEY evidence and add saved tables without model execution."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from dashboard import tey_ablation as view, tey_content as text

SOURCE = ROOT.parent / 'Pruebas provisionales CO NOx/ablacion_TEY_20261021'
ARCHIVE = ROOT / 'book/informe_resultados/tey_ablation'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def score(frame, target):
    y = frame[target].to_numpy()
    return float(1-((y-frame.prediccion.to_numpy())**2).sum()/((y-y.mean())**2).sum())


def export_evidence():
    completed = json.loads((SOURCE/'completado.json').read_text(encoding='utf-8'))
    assert completed['estado'] == 'Completado' and completed['archivos_publicados_intactos']
    for group in ['codigo', 'evidencias']:
        for path, digest in completed['signature'][group].items():
            assert sha(path) == digest, f'Original evidence changed: {Path(path).name}'
    results = pd.read_csv(SOURCE/'resultados.csv')
    stages = []
    assignment = pd.read_csv(ROOT/'book/informe_resultados/entregable2_anidada/asignacion_particiones.csv.gz')
    dev = assignment.loc[assignment.particion.eq('entrenamiento_CV')].reset_index(drop=True)
    for target in ['CO', 'NOX']:
        for label in [f'fold{k}' for k in range(1, 6)]+['final']:
            marker = json.loads((SOURCE/f'completed_{target}_{label}.json').read_text(encoding='utf-8'))
            assert sha(SOURCE/marker['prediction']) == marker['prediction_sha']
            stages.append({name:marker[name] for name in
                ['target', 'stage', 'n_train', 'n_val', 'R2', 'MSE', 'seconds', 'config', 'prediction_sha']})
        group = results.loc[results.objetivo.eq(target)]
        for phase, filename, n in [('OOF', f'predicciones_OOF_{target}.csv.gz', 27570),
                                   ('prueba', f'predicciones_{target}_final.csv.gz', 7411)]:
            new = pd.read_csv(SOURCE/filename)
            old = pd.read_csv(ROOT/f'book/informe_resultados/kernel_operativo/predicciones_{phase}_{target}.csv.gz')
            pd.testing.assert_frame_equal(new[['anio', 'orden_registro', 'bloque', target]],
                                        old[['anio', 'orden_registro', 'bloque', target]])
            assert len(new) == n
            column = 'R2_OOF' if phase == 'OOF' else 'R2_prueba'
            np.testing.assert_allclose(score(old, target), group.loc[group.modelo.eq('Kernel completo'), column].iloc[0])
            np.testing.assert_allclose(score(new, target), group.loc[~group.modelo.eq('Kernel completo'), column].iloc[0])
            if phase == 'OOF':
                scores = [score(new.loc[dev.fold_validacion.eq(k)], target) for k in range(1, 6)]
                np.testing.assert_allclose(np.mean(scores), group.loc[~group.modelo.eq('Kernel completo'), 'R2_CV'].iloc[0])
    view.LOCATION.mkdir(exist_ok=True)
    ARCHIVE.mkdir(exist_ok=True)
    tables = ['resultados.csv', 'significancia_CV.csv', 'significancia_prueba.csv', 'diferencias_por_fold.csv']
    for name in tables:
        shutil.copy2(SOURCE/name, view.LOCATION/name)
    for target in ['CO', 'NOX']:
        for source, name in [(f'predicciones_OOF_{target}.csv.gz', f'sin_TEY_OOF_{target}.csv.gz'),
                             (f'predicciones_{target}_final.csv.gz', f'sin_TEY_prueba_{target}.csv.gz')]:
            shutil.copy2(SOURCE/source, ARCHIVE/name)
    pd.DataFrame(stages).to_csv(ARCHIVE/'etapas.csv', index=False)
    paths = [view.LOCATION/name for name in tables]+list(ARCHIVE.glob('*'))
    metadata = dict(estado='Completado', revision='2026-10-21', n_desarrollo=27570, n_prueba=7411,
        bloque=168, purga=24, folds_externos=5, crossfit=3, semilla=42,
        retiradas=['TEY', 'TEY*(TIT-TAT)'], conservadas=['TIT-TAT'],
        hiperparametros='Configuraciones originales por pliegue y final, sin nueva busqueda',
        replicas_bootstrap=20000, bloque_principal=168, bloques_sensibilidad=[336, 672],
        ajuste='Holm para CO y NOx; CV y prueba por separado, y por longitud de remuestreo',
        orientacion='sin TEY menos completo; negativo favorece al kernel completo',
        modelos_reentrenados_al_incorporar=0, prueba_usada_en_seleccion=False,
        modelos_publicados_sustituidos=False, fuente='Ablacion provisional completada el 21 de octubre de 2026',
        origen={group:{Path(p).name:digest for p,digest in completed['signature'][group].items()}
                for group in ['codigo', 'evidencias']},
        archivos={p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    (view.LOCATION/'metodologia.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    view.evidence.cache_clear()


CODE = '''from pathlib import Path
import sys
from IPython.display import display
_rutas_tey = [globals().get('REPO_KERNEL'), Path.cwd(), Path.cwd().parent]
REPO_TEY = next(p for p in _rutas_tey if p is not None and (p/'data/tey_ablation').exists())
sys.path.insert(0, str(REPO_TEY))
from dashboard.tey_ablation import comparison_table
display(comparison_table().round(5))
'''


def update_book():
    path = ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    prefix = 'tey-ablation-'
    old = copy.deepcopy([c for c in notebook['cells'] if not c.get('id', '').startswith(prefix)])
    notebook['cells'] = copy.deepcopy(old)
    outputs = []
    for frame in [view.comparison_table()]:
        frame = frame.round(5)
        outputs.append(dict(output_type='display_data', metadata={}, data={
            'text/html':frame.to_html(index=False), 'text/plain':frame.to_string(index=False)}))
    count = max(c.get('execution_count') or 0 for c in old if c['cell_type'] == 'code')+1
    cells = [dict(cell_type='markdown', id=prefix+'method', metadata={}, source=(
        '#### 4.14.3 '+text.TITLE+'\n\n'+text.AVAILABILITY+'\n\n'+text.DESIGN).splitlines(keepends=True)),
        dict(cell_type='code', id=prefix+'results', metadata={'saved_evidence_only':True, 'tags':['hide-input']},
             source=CODE.splitlines(keepends=True), execution_count=count, outputs=outputs),
        dict(cell_type='markdown', id=prefix+'conclusion', metadata={}, source=(
             text.READING+'\n\n'+text.CONCLUSIONS['CO']+'\n\n'+text.CONCLUSIONS['NOX']).splitlines(keepends=True))]
    index = next(i for i,c in enumerate(old) if c.get('id') == 'stability-results-NOX')+1
    notebook['cells'][index:index] = cells
    assert [c for c in notebook['cells'] if not c.get('id', '').startswith(prefix)] == old
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')


def main():
    prior_data = {p:sha(p) for p in (ROOT/'data').rglob('*') if p.is_file()
                  and 'tey_ablation' not in p.parts and '__pycache__' not in p.parts}
    export_evidence()
    update_book()
    assert all(sha(path) == digest for path,digest in prior_data.items())
    print('Ablacion de TEY incorporada sin entrenar ni alterar resultados o modelos previos.')


if __name__ == '__main__':
    main()
