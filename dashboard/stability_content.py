"""Shared interpretation of fixed-fold seed sensitivity and kernel ablation."""
NAMES = {'completo':'Kernel completo', 'sin_derivadas':'Sin variables derivadas',
         'sin_pesos':'Sin pesos por sensores', 'sin_adaptaciones':'Sin derivadas ni pesos',
         'solo_ET':'Solo Extra Trees', 'solo_SVR':'Solo SVR ponderado'}

SEED_HELP = ('La tabla resume cuánto cambia el R² OOF al repetir el entrenamiento con cinco semillas. '
    'El gráfico muestra cada resultado del modelo seleccionado: puntos cercanos indican poca variación '
    'por la aleatoriedad del ajuste, manteniendo las mismas particiones.')

ABLATION_HELP = ('Para evaluar qué aporta cada componente, se compara el kernel completo con versiones '
    'sin algunas de sus adaptaciones y con sus modelos por separado. La primera tabla muestra el R²; '
    'la segunda, las diferencias OOF y su incertidumbre. Un ΔR² positivo favorece al kernel completo; '
    'p ajustado menor que 0,05 indica una diferencia significativa.')

ABLATION_SENSITIVITY_HELP = ('Esta tabla comprueba si las conclusiones se mantienen al remuestrear '
    'grupos de uno, dos o cuatro bloques consecutivos. Agrupar más bloques permite representar una '
    'dependencia más prolongada entre registros, sin cambiar los modelos ni sus particiones.')

METHOD = '''### Estabilidad y aporte de los componentes

La **sensibilidad a semillas** comprueba cuánto cambia el resultado al variar la aleatoriedad del ajuste. Se utilizan cinco semillas (42, 7, 21, 84 y 123), con las mismas particiones, bloques de 168 registros, purga de 24 y configuraciones seleccionadas dentro de cada pliegue. El R² OOF reúne las predicciones de validación externa: cada registro se evalúa sin haber participado en su entrenamiento.

En las configuraciones utilizadas, KNN, Ridge, Lasso, SVR lineal y XGBoost no tienen una fuente activa de aleatoriedad. Se reutiliza su resultado original; no representan cinco entrenamientos independientes. En el árbol de decisión, Random Forest y el kernel sí se repite el ajuste. Se evalúa la estabilidad del entrenamiento con configuraciones fijas, no una nueva búsqueda de hiperparámetros.

Además de comprobar la estabilidad, la **ablación** evalúa qué aporta cada componente del kernel. Se quitan las dos variables operativas, los pesos por sensores o ambas adaptaciones; también se comparan Extra Trees y SVR por separado con su combinación mediante Ridge. Se mantienen la semilla 42 y los hiperparámetros. El escalado se aprende solo con entrenamiento y Ridge recibe predicciones obtenidas cuando cada registro queda fuera del ajuste de las bases, mediante cross-fitting.

**ΔR² = R² del kernel completo − R² de la variante.** Un valor positivo favorece al modelo completo. El intervalo al 95% se obtiene con 5.000 remuestreos de los mismos bloques para ambos modelos; si incluye cero, la comparación no permite establecer una ventaja clara. Se revisan longitudes medias de uno, dos y cuatro bloques. Holm ajusta las diez comparaciones de ambas emisiones por longitud. Se comparan ajustes guardados, sin repetir la búsqueda completa de configuraciones.

Los resultados de prueba se calculan después de completar la validación y no se utilizan para elegir semillas ni variantes.
'''


def seed_conclusion(summary, target):
    group = summary.loc[summary.objetivo.eq(target)&~summary.determinista]
    kernel = group.loc[group.modelo_id.eq('kernel_operativo')].iloc[0]
    f = lambda value: format(value,'.4f').replace('.',',')
    return (f'El R² OOF del kernel varía entre {f(kernel["min"])} y {f(kernel["max"])} '
            f'con las cinco semillas; su desviación estándar es {f(kernel["std"])}. '
            'El resultado cambia poco entre estas semillas. Esta estabilidad corresponde al entrenamiento '
            'con particiones fijas, no a cambios del conjunto de prueba.')


def ablation_conclusion(comparisons, target, results=None):
    group = comparisons.loc[comparisons.objetivo.eq(target)]
    robust = []
    for variant, rows in group.groupby('variante',sort=False):
        if len(rows)==3 and (rows.delta_R2>0).all() and (rows.IC95_inf>0).all() and (rows.p_Holm<.05).all():
            robust.append({'solo_ET':'Extra Trees por separado','solo_SVR':'SVR ponderado por separado'}.get(variant,NAMES[variant]))
    extra = ''
    if results is not None:
        final = results.loc[results.objetivo.eq(target)&results.modelo_id.eq('kernel_operativo')&
            results.semilla.eq(42)&results.fase.eq('prueba')].set_index('variante')
        simplified = final.loc[final.index.isin(['sin_derivadas','sin_pesos','sin_adaptaciones'])]
        if (simplified.R2>final.loc['completo','R2']).any():
            extra = (' En prueba, algunas variantes sin adaptaciones superan al kernel completo, '
                     'aunque obtienen menor R² OOF. El beneficio de esas adaptaciones no es uniforme '
                     'entre validación y prueba.')
    if robust:
        return ('El kernel completo mantiene una ventaja estadística frente a '+', '.join(robust)+
                ' con las tres longitudes de bloques. Las demás comparaciones no respaldan una ventaja consistente.'+extra)
    return ('Ninguna ablación permite afirmar una ventaja estadística del kernel completo que se mantenga '
            'con las tres longitudes de bloques. Esto no demuestra que los componentes sean equivalentes.'+extra)
