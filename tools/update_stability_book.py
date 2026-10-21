"""Publish only completed saved seed/ablation analyses in notebook outputs."""
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
from matplotlib.figure import Figure
from dashboard.stability_content import METHOD

CODE = '''from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML
_rutas_estabilidad = [globals().get('REPO_KERNEL',Path.cwd()),Path.cwd(),Path.cwd().parent]
REPO_ESTABILIDAD = next(p for p in _rutas_estabilidad if (p/'data/stability_ablation/metodologia.json').exists())
sys.path.insert(0,str(REPO_ESTABILIDAD))
sys.path.insert(0,str(REPO_ESTABILIDAD/'book/training'))
from stability_plots import read as leer_estabilidad, seeds_figure, ablation_figure
from dashboard.stability_content import NAMES, seed_conclusion, ablation_conclusion
objetivo_estabilidad = {target!r}
resumen_estabilidad = leer_estabilidad('sensibilidad_semillas')
tabla_estabilidad = resumen_estabilidad.loc[resumen_estabilidad.objetivo.eq(objetivo_estabilidad)].copy()
tabla_estabilidad['Aleatoriedad'] = tabla_estabilidad.determinista.map({{True:'Sin aleatoriedad activa',False:'Cinco semillas de ajuste'}})
display(tabla_estabilidad[['modelo','mean','std','min','max','rango','Aleatoriedad']].rename(columns={{
    'modelo':'Modelo','mean':'R² OOF medio','std':'Desv. estándar','min':'Mínimo','max':'Máximo','rango':'Rango'}}).round(6))
display(Markdown(seed_conclusion(resumen_estabilidad,objetivo_estabilidad)))
fig_estabilidad = seeds_figure(objetivo_estabilidad)
display(fig_estabilidad)
plt.close(fig_estabilidad)
resultados_estabilidad = leer_estabilidad('resultados')
casos_estabilidad = resultados_estabilidad.loc[resultados_estabilidad.objetivo.eq(objetivo_estabilidad)&
    resultados_estabilidad.variante.eq('completo')]
display(HTML('<details><summary>Resultados por semilla: validación y prueba</summary>'+casos_estabilidad[
    ['modelo','semilla','fase','R2','determinista','reutilizado']].rename(columns={{'R2':'R²'}}).round(6).to_html(index=False)+'</details>'))
ablacion_estabilidad = resultados_estabilidad.loc[resultados_estabilidad.objetivo.eq(objetivo_estabilidad)&
    resultados_estabilidad.modelo_id.eq('kernel_operativo')&resultados_estabilidad.semilla.eq(42)]
tabla_ablacion = ablacion_estabilidad.pivot(index='variante',columns='fase',values='R2').reindex(NAMES)
tabla_ablacion.index = [NAMES[v] for v in tabla_ablacion.index]
display(tabla_ablacion.reset_index().rename(columns={{'index':'Variante','OOF':'R² OOF','prueba':'R² prueba'}}).round(5))
contrastes_estabilidad = leer_estabilidad('ablacion_contrastes')
tabla_contrastes = contrastes_estabilidad.loc[contrastes_estabilidad.objetivo.eq(objetivo_estabilidad)].copy()
tabla_contrastes['variante'] = tabla_contrastes.variante.map(NAMES)
display(tabla_contrastes[['variante','bloques_medios','delta_R2','IC95_inf','IC95_sup','p_Holm']].rename(columns={{
    'variante':'Variante','bloques_medios':'Bloques medios','delta_R2':'ΔR² a favor del kernel completo',
    'IC95_inf':'IC95% inferior','IC95_sup':'IC95% superior','p_Holm':'p ajustado · Holm'}}).round(5))
display(Markdown(ablation_conclusion(contrastes_estabilidad,objetivo_estabilidad,resultados_estabilidad)))
fig_estabilidad = ablation_figure(objetivo_estabilidad)
display(fig_estabilidad)
plt.close(fig_estabilidad)
'''


def main():
    metadata = json.loads((ROOT/'data/stability_ablation/metodologia.json').read_text(encoding='utf-8'))
    assert metadata['estado']=='Completado','Do not publish incomplete analyses'
    notebook_path = ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    old_code = [copy.deepcopy(c) for c in notebook['cells'] if c['cell_type']=='code' and not c.get('id','').startswith('stability-')]
    notebook['cells'] = [c for c in notebook['cells'] if not c.get('id','').startswith('stability-')]
    new = [dict(cell_type='markdown',id='stability-method',metadata={},source=(
        '### 4.14 Sensibilidad a semillas y ablación\n\n'+METHOD.replace('### ','#### ')).splitlines(keepends=True))]
    execution = max(c.get('execution_count') or 0 for c in old_code)
    namespace = dict(__name__='__stability_report__',REPO_KERNEL=ROOT,
        Markdown=lambda text:SimpleNamespace(data=text,mime='text/markdown'),
        HTML=lambda text:SimpleNamespace(data=text,mime='text/html'))
    for i,target in enumerate(['CO','NOX'],1):
        new.append(dict(cell_type='markdown',id='stability-heading-'+target,metadata={},
            source=[f'#### 4.14.{i} '+('NOx' if target=='NOX' else target)]))
        outputs = []
        def display(value):
            if isinstance(value,Figure):
                buffer = BytesIO()
                value.savefig(buffer,format='png',dpi=150,bbox_inches='tight')
                data = {'image/png':base64.b64encode(buffer.getvalue()).decode('ascii'),'text/plain':'<Figure stability>'}
            elif hasattr(value,'to_html'):
                data = {'text/html':value.to_html(index=False),'text/plain':value.to_string(index=False)}
            else:
                data = {value.mime:value.data,'text/plain':value.data}
            outputs.append(dict(output_type='display_data',metadata={},data=data))
        namespace['display'] = display
        source = CODE.format(target=target)
        exec(source.replace('from IPython.display import display, Markdown, HTML',''),namespace)
        execution += 1
        new.append(dict(cell_type='code',id='stability-results-'+target,
            metadata={'verified_saved_stability':True},source=source.splitlines(keepends=True),
            execution_count=execution,outputs=outputs))
    insertion = next(i for i,c in enumerate(notebook['cells']) if ''.join(c['source']).startswith('## 5.'))
    notebook['cells'][insertion:insertion] = new
    for current in notebook['cells']:
        if current.get('id')=='1b49e614':
            text = ''.join(current['source']).replace('| Semillas adicionales | Pendiente |',
                '| Sensibilidad a semillas | Cinco semillas con particiones y configuraciones fijas; sección 4.14 |')
            if '| Ablación del kernel |' not in text:
                text = text.replace('| Sensibilidad a semillas | Cinco semillas con particiones y configuraciones fijas; sección 4.14 |',
                    '| Sensibilidad a semillas | Cinco semillas con particiones y configuraciones fijas; sección 4.14 |\n'
                    '| Ablación del kernel | Variables operativas, pesos y combinación; contrastes por bloques; sección 4.14 |')
            current['source'] = text.splitlines(keepends=True)
    kept = [c for c in notebook['cells'] if c['cell_type']=='code' and not c.get('id','').startswith('stability-')]
    assert kept==old_code,'Prior code or outputs changed'
    notebook_path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Sección 4.14 guardada, sin ejecutar ni modificar celdas previas.')


if __name__=='__main__':
    main()
