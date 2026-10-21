"""Shared editorial content for the EDA and the book."""

INTRO = ('La partición cambia la pregunta de evaluación. Se comparan dos escenarios con los mismos '
    'nueve sensores, sin utilizar CO ni NOx para medir la similitud.')
RANDOM_QUESTION = ('¿Qué tan bien estimamos las emisiones en registros con condiciones operativas '
    'cercanas a las observadas durante el entrenamiento?')
BLOCK_QUESTION = ('¿Qué tan bien estimamos las emisiones en segmentos del registro separados de '
    'los usados para entrenar?')
RANDOM_DESCRIPTION = 'Se reserva al azar el 20 % de los registros individuales.'
BLOCK_DESCRIPTION = ('Se reservan bloques de 168 registros y se apartan 24 posiciones alrededor '
    'de la evaluación. Los cinco años están representados.')
CONCLUSION = ('La partición por bloques reduce la similitud local entre entrenamiento y evaluación '
    'según los sensores. Esto respalda su uso para evaluar segmentos separados, mientras el 80/20 '
    'aleatorio describe un escenario de condiciones más cercanas. La separación no demuestra '
    'independencia ni implica necesariamente predecir períodos futuros.')
METHOD = ('Se comparan 1.540 registros que quedaron en prueba en ambas particiones guardadas '
    '(semilla 42). Para cada uno se busca el registro de entrenamiento más parecido mediante '
    'la distancia conjunta de los nueve sensores estandarizados: menor distancia significa '
    'mayor similitud. La escala se ajusta solo con registros compartidos por ambos entrenamientos.')
UNCERTAINTY = ('Los intervalos del 95 % se obtienen con 20.000 remuestreos pareados de segmentos '
    'originales dentro de cada año, conservando su proporción. Se usan 168 posiciones y se '
    'comprueban 336 y 672. Holm ajusta las dos comparaciones principales. Los intervalos '
    'describen la incertidumbre para los entrenamientos observados; no incluyen su reajuste.')
ROBUSTNESS = ('El aumento de distancia se mantiene con cinco semillas y con segmentos de '
    'remuestreo de 168, 336 y 672 posiciones: todos los intervalos de la diferencia quedan por '
    'encima de cero. Las semillas son comprobaciones de estabilidad, no muestras independientes.')
