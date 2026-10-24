"""Textos compartidos por el dashboard y el JBook."""

ARCHITECTURE = r"""El **kernel operativo** combina dos modelos: **Extra Trees**, que promedia las reglas de 200 árboles, y **SVR**, que compara las condiciones actuales con situaciones parecidas del entrenamiento. **Ridge** aprende a combinar sus estimaciones; esta estructura se llama *stacking*.

A los nueve sensores se añaden dos variables: la caída de temperatura **TIT − TAT** y su producto por **TEY**. SVR da distintos pesos a los grupos de ambiente, temperatura y presión para definir qué condiciones considera similares.

Se entrena un modelo independiente por emisión. La validación interna elige entre 24 configuraciones. Ridge aprende con predicciones obtenidas por turnos: cada registro se estima cuando su bloque queda fuera del ajuste de las bases. Este procedimiento, llamado *cross-fitting*, evita que el combinador aprenda de predicciones demasiado fáciles del entrenamiento. Se mantienen los bloques de 168 registros, el margen de 24 y la validación de cinco grupos externos y tres internos. El ajuste final utiliza los 27.570 registros de desarrollo.
"""

TREE_MODELS = r"""### Qué es Extra Trees y cómo se diferencia

Un **árbol de regresión** separa registros mediante reglas sobre los sensores, como «TIT por debajo de un límite». Cada recorrido termina en una **hoja**, cuya estimación se aprende de los registros que contiene.

Un **Extra Tree** propone límites aleatorios para las divisiones y elige el mejor entre esos candidatos. **Extra Trees** reúne muchos de estos árboles y promedia sus estimaciones; nuestra rama usa **200**.

| Modelo | Cómo construye y combina los árboles |
|---|---|
| Árbol de decisión | Un solo árbol; en cada división busca un límite que reduzca el error. |
| Random Forest | Construye árboles independientes con muestras con reemplazo y promedia sus resultados. Una muestra con reemplazo puede repetir registros y omitir otros. |
| Extra Trees | Introduce límites aleatorios candidatos y promedia los árboles. En nuestra configuración, cada árbol usa todo el entrenamiento disponible, sin remuestreo. |
| XGBoost | Añade árboles en secuencia: cada nuevo árbol corrige errores del conjunto anterior. |

La diversidad puede ayudar a compensar errores al promediar, pero **no garantiza superioridad ni elimina el sobreajuste**. En el kernel, Extra Trees aporta reglas que complementan la semejanza entre registros usada por SVR.

Referencias: [ensambles](https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees), [Extra Trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html) y [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), documentación de scikit-learn.
"""

MATHEMATICS = r"""### Variables operativas

$$
\Delta T=TIT-TAT,\qquad I=TEY\,\Delta T
$$

$\Delta T$ es la caída de temperatura entre entrada y salida; $I$ combina esa caída con la energía producida, **TEY**.

### Posible interpretación operativa

Una mayor $\Delta T$ representa una mayor caída térmica, **no necesariamente mayor eficiencia**. En una turbina ideal, el trabajo por unidad de masa se relaciona con esa caída mediante $w\approx c_p\Delta T$; $c_p$ representa cuánto calor necesita el gas para cambiar su temperatura ([NASA Glenn](https://www.grc.nasa.gov/WWW/K-12/airplane/powtrbth.html)).

La interacción $I$ distingue una misma caída térmica bajo distintos niveles de producción. Como TEY se expresa en MWh, $I$ tiene unidades **MWh·°C**, no de eficiencia ([diccionario UCI](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)). Ambas variables se calculan con los sensores existentes y su relación con las emisiones puede cambiar según las demás condiciones. Su aporte se evalúa retirándolas en el análisis de ablación.

### Poner los sensores en una escala comparable

$$
z_j=\frac{x_j-\mu_j}{s_j}
$$

A cada lectura $x_j$ se le resta su promedio de entrenamiento $\mu_j$ y se divide por su desviación estándar $s_j$, una medida de dispersión. Así, las unidades de presión o temperatura no dominan por su tamaño numérico.

### Medir la semejanza entre condiciones

$$
K(z,z_i)=\exp\!\left[-0{,}2\sum_{j=1}^{11}w_j^2(z_j-z_{ij})^2\right]
$$

Se comparan las once variables del registro actual $z$ con un registro de referencia $z_i$. El **kernel RBF** devuelve un valor cercano a 1 cuando se parecen y a 0 cuando difieren. Los pesos $w_j$ hacen que algunas diferencias cuenten más. Se eligen entre cuatro combinaciones mediante validación interna; no son efectos causales ni los coeficientes que aprende Ridge al combinar predicciones.

### Cómo aprende Extra Trees

Cada árbol divide los registros en grupos cada vez más pequeños. En cada nodo se toma una **selección aleatoria de variables** y se propone un límite aleatorio para cada una. Se conserva la división que más reduce la dispersión de las emisiones entre esos candidatos, no entre todos los límites posibles.

Para los $n_t$ registros de un nodo $t$, la dispersión se mide como:

$$
\bar y_t=\frac{1}{n_t}\sum_{i\in t}y_i,\qquad
H(t)=\frac{1}{n_t}\sum_{i\in t}(y_i-\bar y_t)^2
$$

$\bar y_t$ es el promedio de emisión del grupo y $H(t)$ su error cuadrático medio respecto a ese promedio. Una buena división produce grupos con emisiones más parecidas. Si $L$ y $R$ son los grupos resultantes, se elige el candidato con mayor reducción:

$$
\Delta H=H(t)-\frac{n_L}{n_t}H(L)-\frac{n_R}{n_t}H(R)
$$

Los cocientes ponderan cada grupo por su cantidad de registros. Cuando el árbol deja de dividir, la **hoja predice el promedio** de sus registros. El ensamble promedia las predicciones de todos los árboles:

$$
\widehat y_{ET}(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x),\qquad B=200
$$

$T_b(x)$ es la estimación del árbol $b$. Promediar árboles distintos ayuda a compensar errores individuales. Frente a Random Forest, Extra Trees añade aleatoriedad a los límites; en nuestra configuración no remuestrea registros y usa aproximadamente el 75 % de las variables como candidatas en cada división.

Se fijaron **200 árboles**. La validación eligió entre **profundidad máxima 20 y mínimo 2 registros por hoja**, o **sin límite de profundidad y mínimo 5 por hoja**. Para ambas emisiones se seleccionó la primera opción. El escalado de sensores y emisión se aprende únicamente con el entrenamiento; las predicciones se devuelven a la escala original.

Referencias: [Extra Trees](https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees), [criterio de regresión](https://scikit-learn.org/stable/modules/tree.html#regression-criteria) y [parámetros del estimador](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html), documentación de scikit-learn.

### Obtener la estimación de SVR

$$
\widehat y_{SVR}=b_S+\sum_i c_iK(z,z_i)
$$

El símbolo $\widehat y$ significa «valor estimado». SVR combina semejanzas con registros de referencia, llamados **vectores soporte**: $c_i$ es su influencia aprendida y $b_S$ un ajuste constante.

SVR usa $C=20$ para controlar la penalización de errores, $\gamma=0{,}2$ para definir qué tan local es la semejanza y $\epsilon=0{,}1$ como margen de tolerancia, sobre la emisión estandarizada.

### Combinar sin dar pesos excesivos

$$
\min_{b,\beta}\ \sum_i(\widetilde y_i-b-\beta^Tq_i)^2+\alpha\lVert\beta\rVert_2^2,
\qquad \alpha=\lambda n
$$

Ridge busca pesos $\beta$ y una constante $b$ que reduzcan el error entre la emisión estandarizada $\widetilde y_i$ y las dos predicciones estandarizadas $q_i$. El primer término mide errores al cuadrado; el segundo penaliza pesos excesivos. Esa **regularización** se controla con $\lambda$ y se escala por los $n$ registros de ajuste.

En unidades originales, el resultado es:

$$
\widehat y=b_0+b_{ET}\widehat y_{ET}+b_{SVR}\widehat y_{SVR}
$$

Los pesos se aprenden de los datos; no tienen que sumar uno.
"""

CONTRIBUTION = r"""### Aporte de la propuesta

La propuesta integra **dos variables operativas**, **semejanzas ponderadas por grupos de sensores** y **Extra Trees + SVR combinados con Ridge**. No se identificó esta combinación exacta para la tarea en las fuentes revisadas.

| Antecedente | Enfoque reportado | Diferencia de la propuesta |
|---|---|---|
| [Sinap (2025)](https://dergipark.org.tr/en/pub/jista/article/1566965) | Combina Gradient Boosting, LightGBM y CatBoost con XGBoost; evalúa filtrado de extremos | Combina Extra Trees y SVR ponderado con Ridge, conservando extremos |
| [Coelho et al. (2024)](https://doi.org/10.1016/j.fuel.2023.129366) | Resume sensores con PCA, t-SNE, UMAP y PHATE, según el resumen editorial | Usa dos variables de interpretación operativa en lugar de esas proyecciones |
| [Jhinaoui et al. (2024)](https://gpps.global/wp-content/uploads/2024/09/GPPS-TC-2024_paper_154.pdf) | CatBoost + Catch22 resume ventanas de 60 lecturas | Usa únicamente sensores del registro actual y sus transformaciones |

Ponderar las dimensiones de un kernel tiene antecedentes ([Rasmussen y Williams](https://gaussianprocess.org/gpml/chapters/RW5.pdf)). El aporte está en integrar y adaptar los componentes al problema. La ablación compara el modelo completo con versiones sin las variables operativas, sin los pesos y con cada modelo base por separado.
"""

SELECTION = r"""### Modelo elegido para CO y NOx

Se elige **Kernel operativo** para ambas emisiones porque obtiene el **menor error cuadrático medio (MSE) de validación externa** entre las ocho familias publicadas. Cada emisión tiene su propia configuración. La significancia de la mejora frente a los mejores comparadores se presenta a continuación.
"""

STATISTICAL_METHOD = r"""### Cómo se evaluó la mejora

Se parte de la **hipótesis nula** de que ambos modelos tienen el mismo error medio al cuadrado. Se comparan las mismas **7.411 observaciones de prueba** mediante **20.000 remuestreos de bloques** (*bootstrap pareado*), conservando los años y bloques de 168 registros. También se revisan bloques de 336 y 672.

**ΔR² = R² del kernel − R² del comparador.** Un valor positivo favorece al kernel. El **intervalo al 95%** resume la incertidumbre de esa diferencia; si incluye cero, admite ventajas para cualquiera de los dos modelos. Se comprueba además la diferencia de error en los cinco grupos de validación con un **test t corregido**, que considera que esos grupos comparten datos de entrenamiento.

Un **valor p** pequeño indica que los resultados son poco compatibles con la igualdad de error. Se usa el umbral **0,05** y el ajuste **Holm2** para reducir falsas alarmas al comparar las dos emisiones. Prueba y validación se ajustan por separado.

Los contrastes son exploratorios: la prueba se ha consultado durante el proyecto. Una diferencia no significativa no demuestra que los modelos sean equivalentes.
"""

STATISTICAL_INTERPRETATION = {
    'CO': 'Frente a Random Forest, la mejora de R² en prueba es +0,0231 '
        '(intervalo al 95%: −0,0537 a 0,0796; p ajustado = 0,5092). '
        'La validación favorece al kernel (p ajustado = 0,0345), '
        'pero la prueba no confirma una mejora significativa.',
    'NOX': 'Frente a XGBoost, la mejora de R² en prueba es +0,0272 '
        '(intervalo al 95%: 0,0144 a 0,0417; p ajustado = 0,0004). '
        'La validación también favorece al kernel (p ajustado = 0,0018). '
        'Ambas comprobaciones respaldan una mejora significativa.'}
