"""Shared, accessible interpretation of the completed TEY ablation."""

TITLE = 'TEY: justificación de su uso'

AVAILABILITY = '''TEY representa la energía producida por la turbina. Aquí se utiliza para estimar emisiones del mismo período con las mediciones del registro, sin usar CO ni NOx como entradas ni anticipar el futuro. Por eso, incluir TEY no constituye fuga para este objetivo de estimación contemporánea.

Sí habría fuga si TEY se conociera después del momento de predicción o se calculara a partir de las emisiones objetivo ([criterio metodológico](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)). Los datos de UCI son agregados horarios: esta justificación no presupone disponibilidad instantánea ([diccionario UCI](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)).
'''

DESIGN = ('Para sustentar su permanencia en el modelo, se retiraron TEY y su interacción '
    'TEY × (TIT − TAT), manteniendo la misma validación por bloques y las configuraciones originales. '
    'La siguiente ablación cuantifica su aporte conjunto.')

READING = ('ΔR² = completo − sin TEY e interacción. Un valor positivo favorece conservarlas; '
    'p ajustado con Holm < 0,05 indica una diferencia significativa.')

CONCLUSIONS = {
    'CO': ('En CO, el kernel completo obtiene un R² ligeramente mayor, pero las diferencias '
           'no son significativas. La ablación no demuestra un aporte concluyente para esta emisión.'),
    'NOX': ('En NOx, la mejora del kernel completo es significativa en ambas evaluaciones. '
            'La ablación respalda conservar TEY y su interacción como entradas del modelo.')}
