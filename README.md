# Dashboard CO y NOx

Aplicacion Dash Plotly independiente del notebook. Tres pestanas: contexto,
EDA y modelos base. Usa datos y artefactos exportados; nunca entrena en Render.

Repositorio: https://github.com/Fabian-11037/gas-turbine-emission-prediction.
Esta carpeta tiene su propio repositorio Git, independiente del proyecto de
Machine Learning que la contiene. Los archivos de la aplicacion estan en la
raiz del repositorio remoto; no se configura Root Directory para este despliegue.

## Estructura

```text
app.py                 Entrada local y WSGI (app:server)
dashboard/             Componentes, pestanas, figuras, datos y adaptadores
assets/                CSS y ajuste de graficos al desplegar secciones
data/                  Observaciones, resultados, bibliografia y modelos JSON
requirements.txt       Cinco dependencias directas para produccion
render.yaml            Blueprint de despliegue
tools/                 Exportacion local opcional; no se ejecuta en produccion
tests/                 Pruebas; no se ejecutan al atender usuarios
```

## Ejecucion local

Desde esta carpeta, en un entorno Python 3.12:

```text
python -m pip install -r requirements.txt
python app.py
```

Direccion local: http://127.0.0.1:8050. La variable PORT permite elegir otro puerto.
En Windows el desarrollo no usa Gunicorn; Render lo usa en Linux.

## Render

Opcion recomendada: cree un repositorio que contenga el contenido de esta carpeta
en su raiz. Conectelo en Render y cree un Blueprint usando `render.yaml`.
No se requiere disco persistente, secretos ni descarga del dataset al arrancar.

Si comparte un repositorio con los otros proyectos, cree un Web Service y defina
Root Directory como `CO_NOx_Dashboard`:

- Runtime: Python; version 3.12.10.
- Build Command: `pip install -r requirements.txt`.
- Start Command: `gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 90`.
- Health Check Path: `/healthz`.
- Variables: `PYTHON_VERSION=3.12.10`, `OPENBLAS_NUM_THREADS=1`.

El Blueprint propone el plan free. Sus restricciones y disponibilidad dependen
de Render; la aplicacion puede tardar en responder tras inactividad. Verifique
el plan antes de confirmar el despliegue. El servicio todavia no se ha creado
en Render: conecte el repositorio indicado y confirme el Blueprint en su cuenta.

Documentacion: https://render.com/docs/deploy-flask,
https://render.com/docs/blueprint-spec y https://dash.plotly.com/installation.

## Actualizaciones sin dependencias del notebook

Cambios de texto y bibliografia: `dashboard/context.py` y `data/literatura.json`.
Aspecto: `assets/dashboard.css`. EDA: `dashboard/eda.py` y `dashboard/figures.py`.
Los callbacks de cada pestana se registran en su propio modulo.

Ridge, SVR lineal y Media se exportan como coeficientes e interceptos en unidades
originales. Se verifico la equivalencia con los modelos scikit-learn en las
36.733 observaciones (tolerancia absoluta 1e-10). Los JSON no ejecutan codigo
como un pickle y no necesitan scikit-learn, joblib ni bibliotecas del notebook.
Los modelos conservan el ajuste original solo en desarrollo.

Para actualizar los modelos y datos desde el proyecto original, use un entorno
local con sus dependencias de entrenamiento:

```text
python tools/export_project.py --project "../Gas Turbine Project"
```

Esta herramienta es opcional y no se importa desde `app.py`. No reentrena;
reexporta datos, metricas y artefactos. `data/manifest.json` documenta sensores,
rangos de desarrollo, ejemplo real, protocolo y SHA256 de las fuentes.

Para agregar una familia nueva:

1. Agregue una entrada con id unico en `data/modelos.json`.
2. Para modelos lineales, use `linear_v1` y coeficientes en unidades originales.
   Para modelos no lineales, implemente otro adaptador en `dashboard/models.py`
   con `predict(DataFrame)` y registrelo en `ADAPTERS`; no fuerce coeficientes
   lineales ni ejecute entrenamiento en callbacks.
3. Incluya las filas de metricas de la familia en `data/resultados.csv`, sus
   intervalos en `data/intervalos_bootstrap.csv` y predicciones de prueba en
   `data/predicciones_prueba.csv.gz`, con nombres `CO_Familia` y `NOX_Familia`
   (espacios sustituidos por guiones bajos). Use el mismo protocolo para que
   la comparacion sea valida.
4. Verifique sensores, unidades, equivalencia y pruebas; redepliegue la carpeta.

El selector de modelos se genera desde el registro. No modifica la seleccion
cientifica previa por MSE de CV. Actualice las leyendas de seleccion si cambia
el protocolo o la familia seleccionada.

## Validacion y limites

```text
python -m unittest discover -s tests -p test_*.py
```

La comprobacion visual opcional `tools/check_ui.py` requiere
`requirements-dev.txt` y Microsoft Edge instalado. No se ejecuta en Render.
Verifica las tres pestanas en escritorio, 390 y 360 pixeles, prediccion,
validacion de humedad, cambio de modelo, tablas plegables y descarga de CSV.
Las capturas quedan en `tests/screenshots/` y no se incluyen en el paquete.

Los graficos utilizan datos reales. Histogramas, correlaciones y tablas usan
todos los registros filtrados; dispersiones muestran hasta 2.200 puntos con
semilla 42. Las cajas anuales muestran estadisticos completos sin dibujar
puntos extremos. Desarrollo es el filtro EDA por defecto; la fuente completa
es exploracion descriptiva post hoc, no seleccion de modelos.

La prediccion valida tipos, finitud, humedad de 0 a 100% y cantidades/presiones
positivas. Advierte extrapolacion respecto al rango individual de desarrollo.
Permanecer dentro de esos rangos no garantiza que la combinacion sea observada.
Las predicciones negativas no se recortan. No se ofrece un intervalo individual
usando indebidamente el intervalo bootstrap de R2.

Los lotes CSV tienen limite de 2 MB y 2.000 filas. No se escriben uploads en
disco; los resultados de cada usuario permanecen en su navegador. No hay
autenticacion ni conformidad ambiental certificada. El dataset es publico;
el dashboard no debe incorporar datos privados sin controles adicionales.
