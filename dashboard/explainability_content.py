"""Spanish editorial content shared by the dashboard and scientific report."""
INTRODUCTION = ('SHAP muestra qué sensores elevan o reducen la estimación respecto a una referencia. '
    'LIME resume el comportamiento del modelo cerca de un caso mediante una relación lineal sencilla. '
    'Ambos explican al modelo, no las causas físicas de las emisiones.')

METHOD = r'''### Alcance y referencia

Se explican las predicciones de los modelos guardados, sin cambiar su entrenamiento ni la validación. El análisis global cubre los **7.411 registros de prueba**. SHAP utiliza diez registros de desarrollo como referencia: dos por año, elegidos con semilla 42. La referencia es el promedio de las predicciones sobre esos registros.

**SHAP global** aproxima las contribuciones mediante dos recorridos de permutación, cada uno en ambos sentidos. El orden de importancia se contrasta con diez recorridos en 64 registros. **SHAP local** evalúa las 512 combinaciones posibles de los nueve sensores respecto a la misma referencia, sin selección de variables.

SHAP reparte entre los sensores la diferencia entre la predicción del registro y una predicción de referencia:

$$\widehat y(x)=\underbrace{E[\widehat y(X)]}_{\text{referencia}}+\sum_{j=1}^{9}\underbrace{\phi_j(x)}_{\text{contribución del sensor}}.$$

Las contribuciones están en **mg/m³**. En el kernel se explican los nueve sensores originales y se recalculan sus variables derivadas; se incluye la combinación final de Extra Trees, SVR y Ridge.

**LIME** construye una aproximación lineal alrededor del caso usando 5.000 registros modificados, los cuartiles del desarrollo y semilla 42. Sus pesos describen intervalos del sensor y no equivalen a las contribuciones SHAP. El **R² local** mide cuánto se parece esa aproximación al modelo: si es bajo, LIME lo representa poco fielmente en esa zona. No mide la precisión frente a las emisiones observadas.

Los casos de error bajo y alto son comunes a ambos modelos por emisión. Se identifican con sus errores de prueba únicamente para interpretar las predicciones. Los sensores relacionados pueden compartir contribuciones; combinarlos artificialmente puede producir condiciones poco habituales. Las explicaciones dependen de la referencia y no identifican efectos causales.

Fuentes: [SHAP por permutaciones](https://shap.readthedocs.io/en/stable/generated/shap.PermutationExplainer.html), [Kernel SHAP](https://shap.readthedocs.io/en/stable/generated/shap.KernelExplainer.html) y [LIME](https://lime-ml.readthedocs.io/en/latest/lime.html).
'''
