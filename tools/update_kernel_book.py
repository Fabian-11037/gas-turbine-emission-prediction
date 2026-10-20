"""Amplia el notebook conservando resultados y graficos anteriores sin reejecutarlos."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.kernel_content import (ARCHITECTURE, MATHEMATICS, CONTRIBUTION,
    SELECTION, STATISTICAL_METHOD, STATISTICAL_INTERPRETATION)


SUMMARY = '''from pathlib import Path
import sys, json, joblib
import pandas as pd

_anterior = globals().get('RAIZ', Path.cwd())
_opciones = [_anterior.parent, _anterior.parent/'CO_NOx_Dashboard', Path.cwd(), Path.cwd().parent]
REPO_KERNEL = next(p.resolve() for p in _opciones if (p/'data'/'modelos.json').exists())
sys.path.insert(0, str(REPO_KERNEL))
EVIDENCIA_KERNEL = REPO_KERNEL/'book'/'informe_resultados'/'kernel_operativo'
resultados_publicados = pd.read_csv(REPO_KERNEL/'data'/'resultados.csv')
referencias_kernel = []
for objetivo in ['CO', 'NOX']:
    comparadores = resultados_publicados.loc[resultados_publicados.objetivo.eq(objetivo) &
        resultados_publicados.modelo.ne('Kernel operativo')]
    referencias_kernel.append(comparadores.loc[comparadores.MSE_CV_externo.idxmin()])
comparacion_kernel = pd.concat([resultados_publicados.loc[
    resultados_publicados.modelo.eq('Kernel operativo')], pd.DataFrame(referencias_kernel)], ignore_index=True)
display(comparacion_kernel[['objetivo','modelo','R2_train','R2_CV_promedio','R2_test',
    'RMSE_test','MAE_test','brecha_R2_train_test']].round(4))

arquitectura_kernel = json.loads((EVIDENCIA_KERNEL/'arquitectura.json').read_text(encoding='utf-8'))
configuracion_kernel = []
for objetivo, detalle in arquitectura_kernel.items():
    configuracion_kernel.append({'Emisión':objetivo, **detalle['pesos_svr'],
        'lambda_Ridge':detalle['lambda_ridge'],'alpha_Ridge':detalle['alpha_ridge'],
        'Intercepto':detalle['intercepto'],'Coef_ET':detalle['coeficientes'][0],
        'Coef_SVR':detalle['coeficientes'][1]})
display(pd.DataFrame(configuracion_kernel).round(4))
'''

INFERENCE = '''registro_kernel = json.loads((EVIDENCIA_KERNEL/'registro.json').read_text(encoding='utf-8'))
sensores_kernel = registro_kernel['predictores']
manifest_kernel = json.loads((REPO_KERNEL/'data'/'manifest.json').read_text(encoding='utf-8'))
ejemplo_kernel = pd.DataFrame([{s['id']:s['ejemplo'] for s in manifest_kernel['sensores']}])
prediccion_kernel = ejemplo_kernel.copy()
for objetivo, registro in registro_kernel['respuestas'].items():
    artefacto = joblib.load(REPO_KERNEL/'data'/registro['ruta'])
    prediccion_kernel[objetivo+'_estimado'] = artefacto['estimador'].predict(ejemplo_kernel[sensores_kernel])
display(prediccion_kernel.round(4))

# El reentrenamiento es explicito y no sustituye los artefactos publicados.
REENTRENAR_KERNEL = False
if REENTRENAR_KERNEL:
    sys.path.insert(0, str(REPO_KERNEL/'book'/'training'))
    from kernel_operativo_training import entrenar_kernel
    datos_kernel = pd.read_csv(REPO_KERNEL/'data'/'observaciones.csv.gz')
    entrenar_kernel(datos_kernel, REPO_KERNEL/'book'/'informe_resultados'/'kernel_nueva_ejecucion')
'''

STATISTICS = '''significancia_kernel = pd.read_csv(REPO_KERNEL/'data'/'kernel_significancia.csv')
display(significancia_kernel[['objetivo','referencia','delta_R2','IC95_inf','IC95_sup',
    'p_prueba_Holm2','p_CV_Holm2']].round(6))

display(significancia_kernel[['objetivo','referencia','p_prueba_nominal','p_prueba_Holm2',
    'p_CV_nominal','p_CV_Holm2']].round(6))

sensibilidad_kernel = pd.read_csv(REPO_KERNEL/'data'/'kernel_significancia_bloques.csv')
display(sensibilidad_kernel[['objetivo','referencia','longitud_bloque','n_unidades',
    'IC95_inf','IC95_sup','p_prueba_nominal','p_prueba_Holm2']].round(6))
'''


def cell(kind, source, identifier):
    result = dict(cell_type=kind, id=identifier, metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def save_notebook(path, notebook):
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    source_copy = ROOT.parent/'Gas Turbine Project'/path.name
    if source_copy.exists():
        source = copy.deepcopy(notebook)
        for current in source['cells']:
            if current.get('id') == 'kernel-intro':
                current['source'] = [line.replace('../assets/kernel_arquitectura.svg',
                    '../CO_NOx_Dashboard/assets/kernel_arquitectura.svg') for line in current['source']]
        source_copy.write_text(json.dumps(source, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')


def refresh_text(path, notebook):
    previous_code = [copy.deepcopy(c) for c in notebook['cells']
                     if c['cell_type'] == 'code' and c.get('id') not in {'kernel-statistics-code', 'kernel-results-code'}]
    texts = {
        'kernel-intro': '### 4.10 Propuesta: stacking con kernel operativo\n\n'+ARCHITECTURE+
            '\n\n![Arquitectura del kernel operativo](../assets/kernel_arquitectura.svg)\n',
        'kernel-math': '#### 4.10.1 Fundamentos matemáticos\n\n'+MATHEMATICS.replace('### ', '##### '),
        'kernel-results-text': '#### 4.10.2 Configuraciones y resultados\n\n'
            'Se mantienen las divisiones de datos anteriores. Las referencias son **Random Forest para CO '
            'y XGBoost para NOx**, los comparadores publicados con menor error de validación. '
            'Se muestran entrenamiento, validación y prueba para distinguir ajuste de capacidad de predicción.\n',
        'kernel-contribution': '#### 4.10.4 Diferenciación y evidencia\n\n'+CONTRIBUTION.replace('### ', '##### ')+
            '\n\n#### 4.10.5 Selección y significancia para ambas emisiones\n\n'+
            SELECTION.replace('### ', '##### ')+'\n\n'+STATISTICAL_METHOD.replace('### ', '##### ')+
            '\n\n**CO.** '+STATISTICAL_INTERPRETATION['CO']+'\n\n**NOx.** '+STATISTICAL_INTERPRETATION['NOX']+'\n'}
    for current in notebook['cells']:
        if current.get('id') in texts:
            current['source'] = texts[current['id']].splitlines(keepends=True)
        if current.get('id') in {'kernel-statistics-code', 'kernel-results-code'}:
            import pandas as pd
            outputs = []
            def display(value):
                outputs.append(dict(output_type='display_data', metadata={}, data={
                    'text/html': value.to_html(index=False), 'text/plain': value.to_string(index=False)}))
            code = STATISTICS if current['id'] == 'kernel-statistics-code' else SUMMARY
            exec(code, {'pd':pd, 'json':json, 'Path':Path, 'RAIZ':ROOT/'book',
                        'REPO_KERNEL':ROOT, 'display':display})
            current['source'] = code.splitlines(keepends=True)
            current['outputs'] = outputs
            current['metadata']['verified_artifact_outputs'] = True
    assert previous_code == [c for c in notebook['cells']
        if c['cell_type'] == 'code' and c.get('id') not in {'kernel-statistics-code', 'kernel-results-code'}]
    save_notebook(path, notebook)
    print('Seleccion y significancia sincronizadas; solo se actualizan tablas guardadas, sin entrenar.')


def update():
    path = ROOT/'book'/'gas_turbine_eda_hilo_conductor_redaccion.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    previous = copy.deepcopy(notebook)
    if any(c.get('id') == 'kernel-intro' for c in notebook['cells']):
        refresh_text(path, notebook)
        return
    new_cells = [
        cell('markdown', '### 4.10 Propuesta: stacking con kernel operativo\n\n'+ARCHITECTURE+
            '\n\n![Arquitectura del kernel operativo](../assets/kernel_arquitectura.svg)\n', 'kernel-intro'),
        cell('markdown', '#### 4.10.1 Fundamentos matemáticos\n\n'+MATHEMATICS.replace('### ', '##### '), 'kernel-math'),
        cell('markdown', '#### 4.10.2 Configuraciones y resultados\n\nLos comparadores anteriores se conservan. '
            'Esta ampliación mantiene las mismas particiones e incorpora una búsqueda de 24 configuraciones por emisión. '
            'Las referencias son los comparadores publicados con menor MSE externo por emisión. '
            'La comparación es exploratoria y los presupuestos no coinciden con todos los antecedentes.\n', 'kernel-results-text'),
        cell('code', SUMMARY, 'kernel-results-code'),
        cell('markdown', '#### 4.10.3 Artefactos y predicción\n\nSe conservan dos estimadores del kernel, '
            'además de los catorce comparadores. La inferencia utiliza únicamente el paquete local `emission_models` '
            'y las dependencias ya declaradas de scikit-learn; no requiere experimentos provisionales, R ni datos de entrenamiento. '
            'Los nueve sensores originales son las entradas y las transformaciones derivadas se calculan dentro del modelo. '
            'El módulo `training/kernel_operativo_training.py` reproduce la búsqueda y el cross-fitting sin depender de cachés históricas.\n', 'kernel-artifacts-text'),
        cell('code', INFERENCE, 'kernel-inference-code'),
        cell('markdown', '#### 4.10.4 Diferenciación y evidencia\n\n'+CONTRIBUTION.replace('### ', '##### '), 'kernel-contribution'),
        cell('code', STATISTICS, 'kernel-statistics-code')]
    namespace = {'__name__': '__kernel_evidence__', 'RAIZ': ROOT/'book'}
    execution = max(c.get('execution_count') or 0 for c in previous['cells'] if c['cell_type'] == 'code')
    for current in new_cells:
        if current['cell_type'] != 'code':
            continue
        outputs = []
        def display(value):
            if hasattr(value, 'to_html'):
                data = {'text/html': value.to_html(index=False), 'text/plain': value.to_string(index=False)}
            else:
                data = {'text/plain': str(value)}
            outputs.append(dict(output_type='display_data', metadata={}, data=data))
        namespace['display'] = display
        exec(''.join(current['source']), namespace)
        execution += 1
        current['execution_count'] = execution
        current['outputs'] = outputs
        current['metadata']['verified_artifact_outputs'] = True
    insertion = next(i for i, c in enumerate(notebook['cells']) if ''.join(c.get('source', [])).startswith('## 5.'))
    notebook['cells'][insertion:insertion] = new_cells
    for current in notebook['cells']:
        if current['cell_type'] != 'markdown' or current.get('id', '').startswith('kernel-'):
            continue
        text = ''.join(current['source'])
        text = text.replace('Los modelos de referencia no constituyen la contribución original; su formulación específica y validación de novedad siguen pendientes.',
            'Los comparadores no constituyen la contribución propia. La sección 4.10 especifica el kernel operativo y contrasta su integración con los antecedentes próximos; la revisión no es exhaustiva.')
        text = text.replace('Su evaluación proporciona una referencia para una propuesta posterior. El dashboard\ngestiona los artefactos vigentes mediante un registro independiente. La revisión\nbibliográfica no modifica los modelos ni incorpora experimentos provisionales.',
            'Su evaluación proporciona la referencia para el kernel operativo descrito en 4.10, incorporado con dos artefactos adicionales. El dashboard gestiona dieciséis artefactos mediante un registro independiente; otros ensayos provisionales no forman parte de esta publicación.')
        text = text.replace('La superioridad estadística, estabilidad entre semillas,\ninterpretabilidad SHAP/LIME y eficiencia algorítmica formal permanecen pendientes.',
            'La sección 4.10 presenta inferencia exploratoria para CO y NOx. La superioridad general frente a todos los comparadores publicados, estabilidad entre semillas, interpretabilidad SHAP/LIME y eficiencia algorítmica formal no quedan establecidas.')
        text = text.replace('| Comparación estadística jerárquica y tamaños de efecto | Pendiente |',
            '| Comparación estadística y tamaños de efecto | NOx: bootstrap pareado por bloques, sensibilidad y CV corregida; alcance exploratorio |')
        text = text.replace('| Propuesta metodológica original | Pendiente; los benchmarks no son novedosos |',
            '| Propuesta metodológica | Kernel operativo: representación, SVR ponderado y stacking ET–SVR–Ridge; revisión dirigida de antecedentes |')
        current['source'] = text.splitlines(keepends=True)
    old_code = [c for c in previous['cells'] if c['cell_type'] == 'code']
    kept_code = [c for c in notebook['cells'] if c['cell_type'] == 'code' and not c.get('id', '').startswith('kernel-')]
    assert old_code == kept_code, 'Las celdas de codigo/resultados anteriores deben permanecer intactas.'
    save_notebook(path, notebook)
    print('Notebook ampliado; codigo y outputs anteriores intactos. Nuevas celdas verificadas sin reentrenar.')


if __name__ == '__main__':
    update()
