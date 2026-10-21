"""Shared, concise scientific text for statistical and computational assessment."""
STATISTICS = r'''### Cómo se comparan los modelos

Se comparan **29 procedimientos por emisión**: siete familias con cuatro métodos de búsqueda y el kernel operativo. El análisis usa las **27.570 predicciones OOF**: cada registro se estima con un modelo que no lo utilizó para entrenarse. El conjunto de prueba se analiza por separado.

1. **Comparación conjunta (MCS).** Conserva los procedimientos cuyo error no puede distinguirse del mejor. Si queda uno, hay evidencia a su favor; si quedan varios, no se identifica un ganador único. El umbral principal es 0,05 y se revisa también 0,10.
2. **Comparación con una referencia.** El kernel se compara con Random Forest para CO y XGBoost para NOx, elegidos previamente por validación. La prueba de Wald revisa la diferencia media de error y si se relaciona con la producción TEY.
3. **Comprobación adicional (DM-HLN).** Revisa la diferencia media de error considerando la variación entre bloques. Holm ajusta los valores p de ambas emisiones para reducir falsas alarmas por comparaciones múltiples.

La hipótesis nula plantea que los modelos tienen el mismo error medio. Para cada registro, $d_i=(y_i-\widehat y_{ref,i})^2-(y_i-\widehat y_{kernel,i})^2$: un valor positivo favorece al kernel. Wald utiliza $d_i$ y su producto por TEY estandarizada. DM divide la diferencia media por su error estándar y aplica la corrección $\sqrt{(B-1)/B}$, con B bloques.

Se realizan **5.000 remuestreos de bloques**, manteniendo el número de bloques de cada año. La longitud media es un bloque original de 168 posiciones; se revisan también dos y cuatro bloques para representar una dependencia más prolongada. Los recorridos consecutivos no cruzan huecos ni cambian de año o pliegue; los bloques parciales conservan sus registros.

**ΔR² positivo favorece al kernel.** Si su intervalo al 95% incluye cero, la comparación no permite establecer qué modelo es mejor. Un **valor p ajustado menor que 0,05** indica evidencia contra la igualdad de error. **Cohen dz** resume el tamaño de la diferencia respecto a la variación entre bloques; no es un porcentaje.

La conclusión se considera más consistente si se mantiene con bloques más largos, que permiten representar una dependencia más prolongada entre registros. Que una comparación no sea significativa no demuestra que los modelos sean iguales.

Wald y DM-HLN son adaptaciones al problema de estimar el registro actual, no pruebas aplicadas a un pronóstico futuro. Se comparan predicciones guardadas y no se repite todo el entrenamiento. Friedman–Nemenyi se incluye como comprobación auxiliar de rangos entre los cinco pliegues. Los procedimientos con errores numéricamente iguales comparten contraste. Clark–West no se aplica porque un modelo no es una versión restringida del otro.

Fuentes: [MCS, implementación arch](https://bashtage.github.io/arch/multiple-comparison/multiple-comparisons.html), [Giacomini y White](https://doi.org/10.1111/j.1468-0262.2006.00718.x) y [corrección HLN](https://doi.org/10.1016/S0169-2070(96)00719-4).
'''

COMPLEXITY = r'''### Costo esperado de los modelos

La **complejidad computacional** describe cuánto aumenta el trabajo al crecer los datos o el modelo. La notación **O(·)** expresa ese crecimiento, no segundos exactos. **n** son los registros; **p**, las entradas; **d**, la profundidad; **B**, los árboles; **I**, las iteraciones; **k**, los vecinos; **q**, los intervalos del histograma; **L**, las hojas; y **s**, los vectores soporte. La inferencia es el cálculo de una predicción, aquí expresado por registro.

| Modelo | Entrenamiento: costo orientativo | Inferencia por registro |
|---|---|---|
| KNN con KD-tree | $O(pn\log n)$, construcción del índice | Típico $O(p\log n+kp)$; peor caso $O(np)$ |
| Ridge con SVD | $O(np^2+p^3)$, para $n\geq p$ | $O(p)$ |
| Lasso | $O(Inp)$, descenso por coordenadas | $O(p)$ |
| Decision Tree | Cota $O(pnd\log n)$, ordenamiento y búsqueda de cortes | $O(d)$ |
| Random Forest | Cota $O(Bpnd\log n)$ | $O(Bd)$ |
| XGBoost con histogramas | Cota $O(np\log n+Bp(nd+qL))$ | $O(Bd)$ |
| SVR lineal | $O(Inp)$, según convergencia del solver | $O(p)$ |
| Kernel operativo | Extra Trees + SVR-RBF + Ridge, repetidos en el cross-fitting | $O(Bd+sp)$ |

En el kernel, **p = 11** después de calcular dos variables; en los comparadores, **p = 9**. Extra Trees evita ordenar candidatos de corte y tiene una cota $O(Bpnd)$. El SVR-RBF puede pasar de un costo cuadrático a cúbico en n, según los datos, la caché y la convergencia. Guardar sus soportes requiere $O(sp)$; la caché de entrenamiento tiene un límite configurado de 256 MB.

Con **r = 3** grupos de cross-fitting, una cota de un ajuste del stacking es $(r+1)[C_{ET}(n,p)+C_{SVR}(n,p)]+C_{Ridge}(n,2)$. La búsqueda y la validación anidada multiplican los ajustes; la reutilización de cálculos puede reducir el trabajo efectivo. La memoria del bosque crece con sus nodos y la del SVR con sus soportes. Cargar el bosque por fragmentos reduce los árboles residentes, pero añade lectura de archivos.

Fuentes de implementación: [árboles de scikit-learn](https://scikit-learn.org/stable/modules/tree.html#complexity), [SVM y SVR](https://scikit-learn.org/stable/modules/svm.html#complexity) y [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).
'''

EFFICIENCY = '''### Cómo se midió la eficiencia

Cada combinación de modelo y emisión se evalúa en un proceso aislado, con un hilo y el mismo lote de prueba. Se miden lotes de **1, 100 y 1.000 registros**, después de una llamada de calentamiento. Se presenta la mediana de cinco repeticiones; la lectura inicial del modelo se registra aparte.

Se compara el procesamiento por lote con 100 llamadas individuales. Para el kernel se añaden la carga por fragmentos y la evaluación vectorizada del mismo SVR. Son formas de ejecutar los estimadores guardados, no modelos nuevos. Todas las predicciones se contrastan con las originales.

**RSS** es la memoria física del proceso. Su pico se observa cada 10 ms e incluye bibliotecas, carga y predicción. Se expresa en **MiB**: 1.048.576 bytes. El tamaño del archivo del modelo no equivale a su memoria RAM.

Los tiempos de ajuste se recuperan del entrenamiento original. No se dispone de memoria de entrenamiento ni del tiempo aislado del ajuste final del kernel. Las mediciones nuevas varían el tamaño del lote, no n ni p; por tanto, no verifican experimentalmente el crecimiento del costo de entrenamiento. La estación es compartida y la caché del sistema operativo no está controlada.

El **frente de Pareto** identifica un equilibrio entre error y tiempo: un procedimiento queda fuera si otro logra igual o menor error en igual o menor tiempo, mejorando al menos uno. Se utilizan los tiempos históricos y se informa el tiempo medio de cálculos no repetidos. Como los presupuestos y la carga del equipo difieren, la comparación describe estas ejecuciones, no una ventaja universal de un optimizador.
'''


def statistical_interpretation(target,sizes):
    if target=='CO':
        return ('CO: el kernel mejora el rendimiento estimado, pero no hay evidencia suficientemente estable '
                'para afirmar que supera al conjunto de modelos evaluados. Algunas comparaciones son significativas '
                'con los bloques originales; al usar bloques más largos, aumenta la incertidumbre y el grupo de '
                f'procedimientos que no pueden distinguirse del mejor pasa de {sizes[0]} a {sizes[1]} y {sizes[2]}. '
                'Esto no demuestra que los modelos sean iguales.')
    return ('NOx: hay evidencia estadística consistente a favor del kernel. Supera a XGBoost, la referencia '
            'elegida por validación, y permanece como único procedimiento en el grupo de mejores resultados '
            f'con las tres longitudes de bloques ({sizes[0]}, {sizes[1]} y {sizes[2]} retenidos). '
            'La conclusión se mantiene al considerar una dependencia más prolongada entre registros.')


def computational_interpretation(frame):
    rows = frame.set_index('modo')
    native,stream,vector = [rows.loc[mode] for mode in ['residente','fragmentado','rbf_vectorizado']]
    reduction = 100*(1-stream.RSS_pico_MB/native.RSS_pico_MB)
    slowdown = stream.segundos_lote_1000/native.segundos_lote_1000
    speedup = native.segundos_lote_1000/vector.segundos_lote_1000
    f = lambda value:format(value,'.1f').replace('.',',')
    structure = f'El kernel conserva {int(native.n_arboles)} árboles y {int(native.n_soporte):,} vectores soporte. '.replace(',','.')
    return (structure+f'En este ensayo, fragmentar el kernel reduce el pico de memoria un {f(reduction)}%, '
        f'pero tarda {f(slowdown)} veces el tiempo de la carga residente para 1.000 registros. '
        f'La evaluación vectorizada del RBF es {f(speedup)} veces más rápida que la residente original, '
        'con memoria similar. Las tres formas conservan las mismas predicciones dentro de la tolerancia numérica.')
