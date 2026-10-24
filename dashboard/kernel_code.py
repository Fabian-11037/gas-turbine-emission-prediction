"""Read-only presentation of the executable training pipeline shipped with the book."""
from functools import lru_cache

from .repository import ROOT

INTRODUCTION = ('El pipeline conserva los bloques de 168 registros, la purga de 24, los cinco '
    'pliegues externos y el cross-fitting interno. El escalado se ajusta dentro de cada entrenamiento. '
    'La búsqueda elige entre 24 configuraciones sin consultar la prueba; al final se guarda un modelo '
    'por emisión en una carpeta nueva, sin sustituir los modelos publicados.')

ROOT_SETUP = '''# Funciona al iniciar Jupyter desde la raíz del repositorio o desde book/.
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p/'data/observaciones.csv.gz').is_file()
            and (p/'book/training/kernel_operativo_training.py').is_file())
sys.path.insert(0, str(ROOT/'book/training'))
sys.path.insert(0, str(ROOT))'''

RUN_CODE = '''from datetime import datetime

# Cambiar a True únicamente para iniciar el entrenamiento completo.
EJECUTAR_ENTRENAMIENTO = False
datos_kernel = pd.read_csv(ROOT/'data/observaciones.csv.gz')
assert len(datos_kernel) == 36733

if EJECUTAR_ENTRENAMIENTO:
    salida_kernel = ROOT/'resultados_kernel'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    carpeta_resultados = entrenar_kernel(datos_kernel, salida_kernel)
    print(f"Modelos y evidencias guardados en: {carpeta_resultados}")
else:
    print("Pipeline listo. Active EJECUTAR_ENTRENAMIENTO para ajustar CO y NOx.")
'''


@lru_cache(maxsize=1)
def training_source():
    source = (ROOT/'book/training/kernel_operativo_training.py').read_text(encoding='utf-8')
    return source.replace('ROOT = Path(__file__).resolve().parents[2]\nsys.path.insert(0, str(ROOT))', ROOT_SETUP)
