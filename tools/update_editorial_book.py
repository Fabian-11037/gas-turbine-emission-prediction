"""Synchronize explanatory prose without executing any notebook code."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dashboard.kernel_content import ARCHITECTURE,TREE_MODELS,MATHEMATICS,CONTRIBUTION,SELECTION,STATISTICAL_METHOD,STATISTICAL_INTERPRETATION


def main():
    path = ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    code_before = copy.deepcopy([c for c in notebook['cells'] if c['cell_type']=='code'])
    texts = {
        '6c4797f3': '''### 4.9 Alcance de la evaluación

Se comparan siete familias mediante cuatro búsquedas y validación anidada. La sección 4.10 incorpora el kernel operativo; las siguientes presentan SHAP y LIME (4.11), comparación estadística (4.12), costo computacional (4.13), y estabilidad y ablación (4.14).

Referencias de implementación: [validación anidada](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [búsqueda con procesos gaussianos](https://scikit-learn.org/stable/modules/gaussian_process.html), [operadores genéticos DEAP](https://deap.readthedocs.io/en/master/api/tools.html) y [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
''',
        'kernel-intro':'### 4.10 Propuesta: stacking con kernel operativo\n\n'+ARCHITECTURE+
            '\n\n![Arquitectura del kernel operativo](../assets/kernel_arquitectura.svg)\n\n'+TREE_MODELS.replace('### ','#### '),
        'kernel-math':'#### 4.10.1 Fundamentos matemáticos\n\n'+MATHEMATICS.replace('### ','##### '),
        'kernel-contribution':'#### 4.10.4 Diferenciación y evidencia\n\n'+CONTRIBUTION.replace('### ','##### ')+
            '\n\n#### 4.10.5 Selección y significancia para ambas emisiones\n\n'+SELECTION.replace('### ','##### ')+
            '\n\n'+STATISTICAL_METHOD.replace('### ','##### ')+'\n\n**CO.** '+STATISTICAL_INTERPRETATION['CO']+
            '\n\n**NOx.** '+STATISTICAL_INTERPRETATION['NOX']+'\n',
        'fe9bdcbf':'''## 6. Discusión y conclusiones

**Modelo elegido: kernel operativo para CO y NOx**, por su menor MSE medio de validación externa entre las ocho familias publicadas. En prueba alcanza **R² de 0,7491 y 0,8242**, con **RMSE de 1,2838 y 5,3426 mg/m³**, respectivamente.

**NOx presenta la evidencia más consistente.** El kernel supera a XGBoost, la referencia elegida por validación. La comparación jerárquica conserva al kernel como único procedimiento en el grupo de mejores resultados con las tres longitudes de bloques. En CO hay evidencia favorable frente a Random Forest en la comparación de CV corregida, pero no en prueba; la comparación conjunta también pierde estabilidad al ampliar los bloques.

**El entrenamiento es estable entre las semillas evaluadas.** El R² OOF del kernel varía de 0,7073 a 0,7085 en CO y de 0,7859 a 0,7866 en NOx. Esta estabilidad no demuestra que el resultado se mantenga en otra planta o periodo.

**El aporte de las adaptaciones depende de la emisión.** El kernel completo obtiene mayor R² OOF que las ablaciones en ambas emisiones. En NOx también mantiene el mayor R² de prueba; en CO, algunas variantes simplificadas mejoran prueba, aunque empeoran OOF. No puede atribuirse una mejora uniforme a todos los componentes. La brecha entre entrenamiento y evaluación persiste.

SHAP describe qué sensores utiliza cada modelo y cómo se forma una predicción; no identifica causas físicas. LIME muestra poca fidelidad local en los casos analizados. La ejecución del kernel por fragmentos reduce memoria a costa de mayor tiempo, sin cambiar sus predicciones.

Los resultados siguientes conservan el detalle de la comparación inicial de siete familias; el kernel y los análisis complementarios se desarrollan en las secciones 4.10 a 4.14.
'''
    }
    for cell in notebook['cells']:
        if cell.get('id') in texts:
            cell['source'] = texts[cell['id']].splitlines(keepends=True)
        if cell.get('id')=='1b49e614':
            text = ''.join(cell['source']).replace(
                '**Siguiente paso.** Evaluar la estabilidad en nuevos periodos, verificar la disponibilidad de sensores y comparar cada componente de la propuesta por separado. Las cifras de otros artículos no son una competencia directa porque cambian los datos, las entradas y la evaluación.',
                '**Siguiente paso.** Evaluar datos nuevos y verificar sensores y unidades en la instalación de uso. Las cifras de otros artículos sirven como referencia, pero no son directamente comparables porque cambian los datos, las entradas y la evaluación.')
            cell['source'] = text.splitlines(keepends=True)
    assert code_before==[c for c in notebook['cells'] if c['cell_type']=='code'],'Notebook code or outputs changed'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Redacción sincronizada; código, métricas y salidas previas conservados.')


if __name__=='__main__':
    main()
