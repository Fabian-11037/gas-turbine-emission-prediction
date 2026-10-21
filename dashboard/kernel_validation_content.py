"""Interpretacion compartida de la comparacion controlada de validaciones."""

TITLE = 'Por qué mantenemos la validación por bloques'

INTRO = ('El mayor R² del 80/20 aleatorio previo corresponde a condiciones más cercanas '
    'al entrenamiento. Esta comparación evalúa ambas estrategias sobre los mismos segmentos externos.')

DESIGN = ('Se mantienen 27.570 registros de desarrollo, cinco pliegues externos por bloques '
    'y una prueba final de 7.411 registros. Cambian la selección interna de configuraciones '
    'y el cross-fitting: las predicciones por turnos con las que Ridge aprende a combinar '
    'Extra Trees y SVR. Los bloques tienen 168 registros y se apartan 24 posiciones alrededor '
    'de la evaluación. Así se compara la preparación del modelo, no la dificultad de dos '
    'conjuntos de prueba distintos.')

METHOD = ('En CV se contrastan las diferencias de error de los cinco pliegues mediante un '
    'test t corregido por compartir datos de entrenamiento. En prueba se realizan 20.000 '
    'remuestreos pareados de segmentos completos dentro de cada año. El análisis principal '
    'usa 168 registros y se comprueba con 336 y 672. Holm ajusta las dos emisiones, por '
    'separado para CV y prueba; el umbral es 0,05. Los intervalos del 95 % resumen la '
    'incertidumbre de cada diferencia. Los contrastes son exploratorios sobre datos ya examinados.')

READING = ('En prueba, ΔR² positivo favorece bloques y negativo, aleatoria. Si el intervalo '
    'cruza cero, no se distingue una ventaja. Los puntos muestran las diferencias observadas '
    'y las líneas, sus intervalos del 95 %.')

CONCLUSION = ('Ambas emisiones comparten el objetivo de estimar segmentos operativos separados. '
    'Por eso se conserva un protocolo por bloques: el EDA muestra que reduce la similitud con '
    'el entrenamiento. La comparación confirma una ventaja significativa en NOx, pero no '
    'demuestra superioridad de ninguna estrategia en CO. El 80/20 aleatorio se mantiene como '
    'complemento para condiciones cercanas a las observadas.')
