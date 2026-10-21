"""Keep introductory methods before results without executing notebook cells."""
from copy import deepcopy
import json
from pathlib import Path

from dashboard import kernel_validation_content as validation
from dashboard.stability_content import METHOD as STABILITY_METHOD
from dashboard.underestimation import METHOD as ERROR_METHOD

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb'


def update_book():
    notebook = json.loads(BOOK.read_text(encoding='utf-8'))
    original = deepcopy(notebook)
    cells = {cell['id']:cell for cell in notebook['cells']}

    def replace(cell_id, text):
        assert cells[cell_id]['cell_type']=='markdown'
        cells[cell_id]['source'] = (text.strip()+'\n').splitlines(keepends=True)

    replace('stability-method', '### 4.14 Sensibilidad a semillas y ablación\n\n'+
            STABILITY_METHOD.replace('### ','#### '))
    replace('underestimation-method', '### 4.15 Subestimación en concentraciones altas\n\n'+ERROR_METHOD)
    replace('kernel-validation-method',
            '#### 4.10.6 Validación ante segmentos operativos separados\n\n'+
            validation.INTRO+'\n\n'+validation.DESIGN+'\n\n'+validation.METHOD)
    conclusion = ''.join(cells['kernel-validation-conclusion']['source'])
    replace('kernel-validation-conclusion', conclusion.replace('\n\n'+validation.METHOD,''))
    roadmap = ''.join(cells['6c4797f3']['source'])
    replace('6c4797f3', roadmap.replace(
        'costo computacional (4.13), y estabilidad y ablación (4.14).',
        'costo computacional (4.13), estabilidad y ablación (4.14) y subestimación en concentraciones altas (4.15).'))

    allowed = {'stability-method','underestimation-method','kernel-validation-method',
               'kernel-validation-conclusion','6c4797f3'}
    assert len(original['cells'])==len(notebook['cells'])
    for before,after in zip(original['cells'],notebook['cells']):
        if before['id'] not in allowed:
            assert before==after, before['id']
        elif before['cell_type']=='code':
            raise AssertionError('Only markdown may change')
    BOOK.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')


if __name__=='__main__':
    update_book()
