# Dashboard CO y NOx

Guia de presentacion y glosario: [Guia de sustentacion](docs/guia_sustentacion.md).

Aplicacion Dash Plotly con tres pestanas: contexto, EDA y modelos. Inferencia
con ocho familias y dieciseis estimadores ya entrenados; nunca entrena en Render.

Repositorio: https://github.com/Fabian-11037/gas-turbine-emission-prediction.
Esta carpeta es un repositorio independiente del proyecto que la contiene.
Los archivos del servicio estan en la raiz remota: Root Directory se deja vacio.

## Modelos y evidencias

- KNN, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, SVR lineal y Kernel operativo.
- Un regresor por familia y emision, con escalado de sensores y respuesta incluido.
- Validacion anidada: cinco pliegues externos y tres internos, bloques de 168
  registros y purga de 24 posiciones originales en ambos niveles.
- Ajuste final en 27.570 registros; prueba historica de 7.411, sin ajuste con prueba.
- Grid, Random, Bayesiano y Genetico: 28 procedimientos por emision.
- Benchmark inicial: Random Forest para CO y XGBoost para NOx, por MSE externo.
- Ampliacion: kernel operativo ET + SVR ponderado con Ridge; 24 configuraciones
  por emision y cross-fitting purgado adicional. Tiene el menor MSE externo entre
  las ocho familias publicadas para ambas respuestas. No se selecciona por prueba.

Cada familia del selector utiliza su instancia seleccionada por CV para cada
objetivo. La emision elegida filtra metricas, diagnosticos e importancia; la
prediccion individual y CSV siempre devuelven CO y NOx de esa misma familia.
La configuracion y el optimizador pueden diferir entre los dos regresores.

El contexto, la comparacion, las metricas y sus intervalos usan las evidencias
del JBook actualizado. Se conserva la advertencia de prueba previamente
examinada: no es validacion prospectiva ni evidencia de superioridad estadistica.

## Estructura

```text
app.py                         Entrada local y WSGI (app:server)
dashboard/                     Interfaz, datos y adaptadores de inferencia
assets/                        Estilos y ajuste de graficos
data/models/                   Dieciseis estimadores verificados mediante SHA256
emission_models/               Transformaciones e inferencia portable del kernel
data/modelos.json              Registro extensible y metadatos por objetivo
data/resultados.csv            Mejor estrategia de cada familia y emision
data/optimizacion.csv           Los 56 procedimientos de CV anidada
data/importancia_permutacion.csv Importancia precalculada, 144 filas
data/                          Observaciones, predicciones, intervalos y literatura
requirements.txt               Dependencias fijadas de inferencia
render.yaml                    Configuracion de despliegue
tools/export_project.py        Exportacion local sin entrenamiento
tests/                         Pruebas de datos, callbacks e inferencia
book/                          Informe y entrenamiento; no se importa al servir Dash
```

## Ejecucion local y Render

Use un entorno Python 3.12 independiente:

```text
python -m pip install -r requirements.txt
python app.py
```

Direccion local: http://127.0.0.1:8050. `PORT` permite elegir otro puerto.
Windows ejecuta Dash directamente; Gunicorn se usa en Linux/Render.

Conecte el repositorio en Render mediante Blueprint y `render.yaml`, o configure:

- Root Directory: vacio.
- Python: 3.12.10.
- Build Command: `pip install -r requirements.txt`.
- Start Command: `gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 90`.
- Health Check: `/healthz`.
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

No se requieren secretos, disco persistente ni descarga de datos al arrancar.
Los pipelines se cargan bajo demanda con una cache de una familia. El kernel
promedia cada bosque en cuatro grupos de 50 arboles y carga despues SVR y Ridge.
Los fragmentos conservan sus propios SHA256; no cambian los estimadores ni sus
predicciones. Los originales completos se conservan para reproducibilidad.
La inferencia se serializa para limitar picos de memoria con usuarios simultaneos.
Los optimizadores, DEAP, Matplotlib y Jupyter no son dependencias de produccion.
Se usa `xgboost-cpu`, la distribucion oficial de inferencia/entrenamiento CPU sin
las dependencias GPU de la distribucion completa:
https://xgboost.readthedocs.io/en/stable/install.html#minimal-installation-cpu-only.
La equivalencia de los artefactos se verifica con esa misma variante instalada.

El plan free propuesto por el Blueprint puede suspenderse por inactividad;
verifique las condiciones de su cuenta antes de confirmar el despliegue.
No se afirma que esta documentacion cree o compruebe un servicio en Render.

## Actualizar modelos y resultados

El entrenamiento se realiza fuera de Dash. Despues de actualizar los artefactos
y resultados de `book/informe_resultados/entregable2_anidada/`, ejecute desde
la raiz del repositorio, en el entorno de inferencia:

```text
python tools/export_project.py
python -m unittest discover -s tests -p test_*.py
```

La exportacion verifica la particion y SHA256, copia los pipelines a `data/models/`,
comprueba sus predicciones y exporta resultados, intervalos, optimizacion e
importancia por permutacion. Esta ultima usa los 7.411 registros originales de
prueba, diez repeticiones y semilla 42. Es un diagnostico posterior; no se utiliza
para seleccionar variables o modelos. Su desviacion no es un intervalo de confianza.
Los sensores correlacionados limitan su interpretacion; no implica causalidad.
La exportacion no llama a `fit` ni vuelve a optimizar, pero el calculo de importancia
puede tardar varios minutos. Nada de esto se ejecuta durante el despliegue.
`tools/export_kernel.py` reutiliza evidencia estable de `book/informe_resultados/`
y artefactos portables, sin necesitar la carpeta de ensayos provisionales. Para
acelerar solo la importancia evalua el mismo RBF por bloques y comprueba su
equivalencia numerica con el SVR nativo; no modifica los modelos guardados.

Una nueva familia de pipelines compatible puede incorporarse al registro y al
exportador; otro formato de modelo requiere un adaptador en `dashboard/models.py`.
Los selectores y las vistas se generan desde metadatos, no desde listas de botones.
Actualice tambien las leyendas cientificas de seleccion si cambia el protocolo.

Para desplegar solo Dash, bastan `app.py`, `dashboard/`, `emission_models/`, `assets/`, `data/`,
`requirements.txt` y `render.yaml`; no necesita `book/`, `tools/` ni los CSV
originales por ano. `Gas Turbine Project` no es una dependencia de ejecucion.

## Jupyter Book

El EDA incluye una vista de estacionalidad para CO y NOx, por ano y poblacion.
Consume `data/estacionalidad_pruebas.csv`, `estacionalidad_perfiles.csv.gz` y
`estacionalidad_metodologia.json`; no requiere Statsmodels en Render.
Para regenerar esas evidencias en el entorno de entrenamiento:

```text
python book/training/seasonality_analysis.py
```

Se contrastan armonicos de 24 y 168 posiciones originales, con tendencia lineal,
Wald-HAC (336 rezagos y sensibilidad a 168) y Holm sobre 40 contrastes por
poblacion. Sin fechas exactas no se confirman ciclos de calendario. El JBook
presenta la metodologia, tablas ejecutadas y figuras de la misma evidencia.

La explicacion del kernel se mantiene en `dashboard/kernel_content.py`.
`python tools/update_kernel_book.py` sincroniza esos textos con la seccion 4.10
sin reentrenar modelos ni reejecutar el notebook completo. Actualiza unicamente
las tablas estadisticas desde CSV estables cuando cambia la evidencia.

Kernel operativo es la seleccion vigente para ambas emisiones y el valor inicial
del selector. La pestana Evidencia compara CO frente a Random Forest y NOx frente
a XGBoost, los mejores comparadores publicados por MSE externo, excluyendo kernel.
`data/kernel_significancia*.csv` contiene dos contrastes y la sensibilidad
de bloques; distingue p nominales y Holm2 conjunto para CO y NOx.
`python tools/export_kernel_significance.py` regenera estas tablas a partir de
las predicciones y particiones guardadas, sin entrenar ni depender de la carpeta provisional.

`book/` conserva el informe y sus materiales reproducibles. La publicacion en
GitHub Pages usa `.github/workflows/publish-book.yml` y `book/requirements.txt`,
sin ejecutar ni reentrenar el notebook. Las dependencias para reentrenarlo estan
en `book/requirements-training.txt`, separadas del servicio. Consulte `book/README.md`.

## Validacion y limites

Las pruebas cubren las ocho familias, predicciones individuales y por lote,
equivalencia con los registros de prueba, SHA256, filtros EDA, tablas y callbacks.
Se comprueba que la inferencia no ajusta pipelines ni importa scripts de entrenamiento.
Los archivos joblib son propios y confiables; no se aceptan modelos subidos por usuarios.

La entrada exige sensores numericos finitos, humedad entre 0 y 100% y cantidades
o presiones positivas. Se advierte extrapolacion respecto al rango individual de
desarrollo. El analisis historico conserva la poblacion original, incluidas las
mediciones de humedad ligeramente superiores a 100% senaladas para revision.
Las predicciones negativas no se recortan y el bootstrap de metricas no se
presenta como intervalo para una prediccion individual.

Las dispersiones muestran hasta 2.200 puntos; las metricas usan toda la prueba.
Los lotes CSV admiten 2 MB y 2.000 filas, no se guardan en disco y sus resultados
permanecen en el navegador de cada usuario. No hay autenticacion ni certificacion
de cumplimiento ambiental. No incorpore informacion privada sin controles adicionales.
