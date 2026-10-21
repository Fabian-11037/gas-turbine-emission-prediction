"""Insert executed explanation-only cells without rerunning previous analyses."""
import base64
import copy
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dashboard.explainability_content import INTRODUCTION, METHOD

THEORY = r'''#### Fundamento de las explicaciones

SHAP interpreta cada sensor como un participante que aporta información a una estimación. Compara su aporte al incorporarlo a distintas combinaciones de sensores. Este reparto sigue tres principios: **eficiencia**, las contribuciones suman la diferencia respecto a la referencia; **simetría**, entradas con el mismo aporte reciben el mismo valor; y **aditividad**, al sumar modelos también se suman sus contribuciones, si se mantiene la misma referencia.

LIME construye una aproximación lineal alrededor del registro. Da mayor peso a las perturbaciones más cercanas y resume cada sensor mediante el intervalo de desarrollo al que pertenece. Por eso sus pesos no tienen la misma referencia que SHAP y pueden diferir cuando hay interacciones o sensores relacionados.
'''

CODE = '''from pathlib import Path
import sys, json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

_candidatos_xai = [globals().get('REPO_KERNEL', Path.cwd()), Path.cwd(), Path.cwd().parent]
REPO_XAI = next(p for p in _candidatos_xai if (p/'data/explainability/metodologia.json').exists())
sys.path.insert(0, str(REPO_XAI/'book/training'))
from explanation_plots import global_figure, local_figure, subset

modelo_xai = {model!r}
objetivo_xai = {target!r}
resumen_xai = subset('resumen_global', modelo_xai, objetivo_xai).sort_values('media_abs_SHAP', ascending=False)
display(Markdown('Los sensores de mayor contribución absoluta media son **'+
    ', '.join(resumen_xai.sensor.head(3))+'**. El signo de cada contribución depende del registro y de los demás sensores.'))
fig_xai = global_figure(modelo_xai, objetivo_xai)
display(fig_xai)
plt.close(fig_xai)

casos_xai = subset('casos', modelo_xai, objetivo_xai)
display(casos_xai.assign(caso=casos_xai.caso.map({{'error_bajo':'Error bajo','error_alto':'Error alto'}}))[
    ['caso','anio','orden_registro','observado','prediccion','error_absoluto',
    'prediccion_LIME','R2_local_LIME','coincidencias_top3']].rename(columns={{
        'caso':'Caso','anio':'Año','orden_registro':'Registro','observado':'Observado',
        'prediccion':'Estimado','error_absoluto':'Error absoluto','prediccion_LIME':'Estimado LIME',
        'R2_local_LIME':'R² local LIME','coincidencias_top3':'Top 3 comunes'}}).round(4))
for caso_xai in ['error_bajo','error_alto']:
    resumen_caso = casos_xai.loc[casos_xai.caso.eq(caso_xai)].iloc[0]
    contribuciones = subset('local', modelo_xai, objetivo_xai)
    contribuciones = contribuciones.loc[contribuciones.caso.eq(caso_xai)].copy()
    fig_xai = local_figure(modelo_xai, objetivo_xai, caso_xai)
    display(fig_xai)
    plt.close(fig_xai)
    mayor = contribuciones.loc[contribuciones.SHAP.abs().idxmax()]
    direccion = 'aumento' if mayor.SHAP>0 else 'reducción'
    texto = ('**'+caso_xai.replace('_',' ').capitalize()+'.** '+mayor.sensor+
        ' presenta la contribución SHAP de mayor magnitud, en dirección de '+direccion+'. '
        'El error absoluto es '+format(resumen_caso.error_absoluto,'.3f').replace('.',',')+' mg/m³. '
        'Las contribuciones explican la estimación, no la causa de la diferencia frente al valor observado. '
        'SHAP y LIME coinciden en '+str(int(resumen_caso.coincidencias_top3))+
        ' de los tres sensores principales; el R² local de LIME es '+
        format(resumen_caso.R2_local_LIME,'.3f').replace('.',',')+'.')
    display(Markdown(texto))
'''


def cell(kind,source,identifier):
    result = {'cell_type':kind,'id':identifier,'metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code':
        result.update(execution_count=None,outputs=[])
    return result


def update():
    path = ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    previous = [copy.deepcopy(c) for c in notebook['cells']
                if c['cell_type']=='code' and not c.get('id','').startswith('xai-')]
    notebook['cells'] = [c for c in notebook['cells'] if not c.get('id','').startswith('xai-')]
    new = [cell('markdown','### 4.11 Interpretación global y local: SHAP y LIME\n\n'+
                INTRODUCTION+'\n\n'+THEORY+'\n\n'+METHOD.replace('### ','#### '),'xai-method')]
    execution = max(c.get('execution_count') or 0 for c in previous)
    namespace = {'__name__':'__explanation_report__','REPO_KERNEL':ROOT,
                 'Markdown':lambda text:SimpleNamespace(data=text)}
    for index,(mid,target,label) in enumerate([
        ('kernel_operativo','CO','Kernel operativo · CO'),('kernel_operativo','NOX','Kernel operativo · NOx'),
        ('xgboost','CO','XGBoost · CO'),('xgboost','NOX','XGBoost · NOx')],1):
        new.append(cell('markdown',f'#### 4.11.{index} {label}\n','xai-heading-'+mid+'-'+target))
        current = cell('code',CODE.format(model=mid,target=target),'xai-results-'+mid+'-'+target)
        outputs = []
        def display(value):
            if isinstance(value,Figure):
                buffer = BytesIO()
                value.savefig(buffer,format='png',dpi=150,bbox_inches='tight')
                data = {'image/png':base64.b64encode(buffer.getvalue()).decode('ascii'),
                        'text/plain':'<Figure SHAP/LIME>'}
            elif hasattr(value,'to_html'):
                data = {'text/html':value.to_html(index=False),'text/plain':value.to_string(index=False)}
            elif hasattr(value,'data'):
                data = {'text/markdown':value.data,'text/plain':value.data}
            else:
                data = {'text/plain':str(value)}
            outputs.append({'output_type':'display_data','metadata':{},'data':data})
        namespace['display'] = display
        # Capture only these new cells; the notebook retains its normal IPython imports.
        source = ''.join(current['source']).replace('from IPython.display import display, Markdown',
                                                     '')
        exec(source,namespace)
        execution += 1
        current.update(execution_count=execution,outputs=outputs)
        current['metadata']['verified_frozen_model_explanations'] = True
        new.append(current)
    new.append(cell('markdown','''#### 4.11.5 Lectura conjunta

Las explicaciones globales resumen qué sensores utiliza más cada modelo; las locales muestran cómo se forma una estimación concreta. Un registro con error elevado puede tener una descomposición SHAP correcta: explicar una predicción no demuestra que esta sea acertada.

En los casos examinados, LIME presenta un ajuste local bajo. Sus pesos no deben interpretarse como una reproducción fiel de todo el comportamiento del modelo. Las diferencias con SHAP son coherentes con las interacciones no lineales, el uso de intervalos y las distintas referencias. No permiten concluir que un sensor cause una mayor emisión ni que un explicador sea superior en todos los casos.
''','xai-conclusions'))
    metadata = json.loads((ROOT/'data/explainability/metodologia.json').read_text(encoding='utf-8'))
    checks = []
    for row in metadata['resultados']:
        checks.append({'Modelo':{'kernel_operativo':'Kernel operativo','xgboost':'XGBoost'}[row['modelo_id']],
            'Emisión':'NOx' if row['objetivo']=='NOX' else 'CO',
            'Registros':row['n_global'],'Correlación de rangos':row['precision']['rho_rangos'],
            'Top 3 comunes':row['precision']['coincidencias_top3']})
    new[-1]['source'] += ('\nLa precisión numérica del resumen global se revisa ampliando el presupuesto '
        'en 64 registros seleccionados sin utilizar sus errores, con la misma referencia. '
        'Los resultados locales son exhaustivos respecto a los diez registros de referencia.\n').splitlines(keepends=True)
    # A static table is sufficient here; calculation remains in the exported manifest.
    import pandas as pd
    new[-1]['source'] += ('\n'+pd.DataFrame(checks).round(4).to_html(index=False)+'\n').splitlines(keepends=True)
    insertion = next(i for i,c in enumerate(notebook['cells']) if ''.join(c['source']).startswith('## 5.'))
    notebook['cells'][insertion:insertion] = new
    for current in notebook['cells']:
        if current.get('id')=='1b49e614':
            text = ''.join(current['source']).replace(
                '| SHAP/LIME, semillas adicionales y eficiencia formal | Pendiente |',
                '| SHAP global/local y comparación con LIME | Kernel operativo y XGBoost, CO y NOx; sección 4.11 |\n'
                '| Semillas adicionales y eficiencia formal | Pendiente |')
            current['source'] = text.splitlines(keepends=True)
    kept = [c for c in notebook['cells'] if c['cell_type']=='code' and not c.get('id','').startswith('xai-')]
    assert kept==previous,'Previous code or results changed'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Sección 4.11: cuatro combinaciones, doce figuras y resultados locales; celdas anteriores intactas.')


if __name__=='__main__':
    update()
