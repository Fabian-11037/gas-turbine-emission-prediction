# Informe cientifico en GitHub Pages

Esta carpeta contiene la copia del notebook ejecutado, `_config.yml`, `_toc.yml`
y las dependencias de publicacion. Conserva las 50 celdas, sus tablas y figuras.
El notebook del proyecto original no se modifica.

La publicacion es estatica: `execute_notebooks: 'off'`. Construir el libro no
ejecuta codigo, no consulta datos externos y no reentrena modelos. No es necesario
incluir datos crudos, scripts de entrenamiento, modelos joblib, entornos Python
ni los archivos auxiliares de edicion del notebook.

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

La descarga del notebook conserva codigo y salidas, pero no es un paquete
autonomo de entrenamiento. Para volver a ejecutarlo se requiere el material
reproducible del proyecto original, no incluido en esta publicacion estatica.

Referencias oficiales:
https://jupyterbook.org/v1/publish/gh-pages.html
https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site
