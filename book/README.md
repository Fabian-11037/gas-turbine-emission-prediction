# Informe cientifico en GitHub Pages

Esta carpeta contiene la copia del notebook ejecutado, `_config.yml`, `_toc.yml`
y las dependencias de publicacion. La seccion de modelos incorpora siete familias
de regresion y cuatro optimizadores por respuesta con validacion anidada 5 x 3.
La seccion 4.10 incorpora el kernel operativo ET-SVR-Ridge como octava familia,
con arquitectura, formulas, comparacion bibliografica y evidencia exploratoria.
La seccion 4.10.5 y las conclusiones establecen Kernel operativo como modelo
elegido para CO y NOx por MSE externo entre las ocho familias publicadas.
Se incluyen pruebas pareadas de ambas emisiones, sensibilidad por bloques y
ajustes multiples; no se interpreta ausencia de significancia como equivalencia.
La copia del proyecto fuente y la del libro contienen el mismo informe actualizado.

La publicacion es estatica: `execute_notebooks: 'off'`. Construir el libro no
ejecuta codigo, no consulta datos externos y no reentrena modelos. No es necesario
instalar dependencias de entrenamiento para publicar. Se incluye el codigo en
`training/`, resultados comprimidos y catorce pipelines en
`informe_resultados/entregable2_anidada/modelos/` para su reutilizacion.
Ninguno se importa al arrancar el dashboard.
Los dos artefactos adicionales del kernel estan en `../data/models/`; sus
transformaciones portables residen en `../emission_models/`. Las evidencias de
validacion y comparacion estan en `informe_resultados/kernel_operativo/`.

## Activacion inicial

En el repositorio, abra Settings > Pages. En Build and deployment seleccione
Source: GitHub Actions. Luego ejecute el workflow `Publicar Jupyter Book` desde
Actions si el primer intento ocurrio antes de activar Pages.

Sitio previsto:
https://Fabian-11037.github.io/gas-turbine-emission-prediction/

El workflow `.github/workflows/publish-book.yml` se ejecuta al cambiar `book/`
o el propio workflow en `main`; tambien admite ejecucion manual. La construccion
y el despliegue estan separados, con permisos de Pages solo en el segundo job.
La configuracion no requiere guardar tokens personales en el repositorio.

## Actualizaciones

Reemplace `gas_turbine_eda_hilo_conductor_redaccion.ipynb` por una version
ejecutada y verificada. Confirme que las salidas estan guardadas y que no hay
errores de ejecucion. Haga commit y push: GitHub Actions reconstruye el sitio.
Las dependencias del libro no se mezclan con las del dashboard de Render.

Para construirlo localmente en un entorno independiente:

```text
python -m pip install -r book/requirements.txt
jupyter-book build book
```

Abra `book/_build/html/index.html`. Esa carpeta generada no se sube a Git:
el workflow construye y publica el HTML. No actualice automaticamente a
Jupyter Book 2: el informe usa la configuracion y el motor Sphinx de la version 1.

## Reproducir o usar los modelos

Descargue o clone el repositorio completo; el notebook aislado no incluye sus
modulos, datos ni modelos. En un entorno Python 3.12 independiente:

```text
python -m pip install -r book/requirements-training.txt
```

Ejecute el notebook desde `book/`. Lee los datos completos ya incluidos en
`data/observaciones.csv.gz` de la raiz del repositorio, sin descargar UCI ni
depender de la carpeta original Gas Turbine Project. Por defecto reutiliza los
resultados guardados. `REENTRENAR = True` ejecuta o reanuda la evaluacion anidada;
los puntos de control se conservan localmente en `corridas/` y no se publican.
Si cambia el espacio, presupuesto o codigo, utilice otra carpeta de salida.

El registro JSON documenta rutas, familias, objetivos, versiones y SHA256.
Para inferencia desde `book/`, con los nueve sensores disponibles:

```python
import sys
sys.path.insert(0, 'training')
from regression_entregable2 import predecir
prediccion = predecir(
    'informe_resultados/entregable2_anidada/modelos/CO_Ridge.joblib',
    sensores,
)
```

`sensores` es un DataFrame con AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY y CDP.
Los joblib se cargan solo desde fuentes confiables y con el entorno fijado.
Las ocho familias del dashboard se actualizan mediante `tools/export_project.py`,
que verifica y copia los artefactos a `data/models/`. Publicar el JBook no actualiza
por si solo ese registro ni ejecuta entrenamiento durante el despliegue de Render.

`REENTRENAR_KERNEL = False` conserva los resultados del kernel. Activarlo llama
a `training/kernel_operativo_training.py` con una carpeta nueva: reproduce las
24 configuraciones por respuesta, CV anidada y cross-fitting purgado, sin utilizar
prueba para elegir parametros ni depender de caches de ensayos provisionales.
El presupuesto de esta ampliacion difiere del benchmark original. El menor MSE
externo entre las familias publicadas corresponde ahora al kernel para CO y NOx;
las tablas originales del benchmark se mantienen identificadas como tales.

Referencias oficiales:
https://jupyterbook.org/v1/publish/gh-pages.html
https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site
