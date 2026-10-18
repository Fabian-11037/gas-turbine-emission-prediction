# Dashboard CO y NOx

Aplicacion Dash Plotly con tres pestanas: contexto, EDA y modelos. Inferencia
con siete familias y catorce pipelines ya entrenados; nunca entrena en Render.

Repositorio: https://github.com/Fabian-11037/gas-turbine-emission-prediction.
Esta carpeta es un repositorio independiente del proyecto que la contiene.
Los archivos del servicio estan en la raiz remota: Root Directory se deja vacio.

## Modelos y evidencias

- KNN, Ridge, Lasso, Decision Tree, Random Forest, XGBoost y SVR lineal.
- Un regresor por familia y emision, con escalado de sensores y respuesta incluido.
- Validacion anidada: cinco pliegues externos y tres internos, bloques de 168
  registros y purga de 24 posiciones originales en ambos niveles.
- Ajuste final en 27.570 registros; prueba historica de 7.411, sin ajuste con prueba.
- Grid, Random, Bayesiano y Genetico: 28 procedimientos por emision.
- Seleccion por MSE externo: Random Forest para CO y XGBoost para NOx.

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
data/models/                   Catorce pipelines verificados mediante SHA256
data/modelos.json              Registro extensible y metadatos por objetivo
data/resultados.csv            Mejor estrategia de cada familia y emision
data/optimizacion.csv           Los 56 procedimientos de CV anidada
data/importancia_permutacion.csv Importancia precalculada, 126 filas
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
Los pipelines se cargan bajo demanda con una cache de hasta tres familias.
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
puede tardar unos minutos. Nada de esto se ejecuta durante el despliegue.

Una nueva familia de pipelines compatible puede incorporarse al registro y al
exportador; otro formato de modelo requiere un adaptador en `dashboard/models.py`.
Los selectores y las vistas se generan desde metadatos, no desde listas de botones.
Actualice tambien las leyendas cientificas de seleccion si cambia el protocolo.

Para desplegar solo Dash, bastan `app.py`, `dashboard/`, `assets/`, `data/`,
`requirements.txt` y `render.yaml`; no necesita `book/`, `tools/` ni los CSV
originales por ano. `Gas Turbine Project` no es una dependencia de ejecucion.

## Jupyter Book

`book/` conserva el informe y sus materiales reproducibles. La publicacion en
GitHub Pages usa `.github/workflows/publish-book.yml` y `book/requirements.txt`,
sin ejecutar ni reentrenar el notebook. Las dependencias para reentrenarlo estan
en `book/requirements-training.txt`, separadas del servicio. Consulte `book/README.md`.

## Validacion y limites

Las pruebas cubren las siete familias, predicciones individuales y por lote,
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
