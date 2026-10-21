"""Add only saved-data analyses; all prior notebook code/outputs remain intact."""
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
from dashboard.assessment_content import STATISTICS,COMPLEXITY,EFFICIENCY


def optimizer_efficiency():
    import hashlib
    import pandas as pd
    path = ROOT/'data/optimizacion.csv'
    history_path = ROOT/'book/informe_resultados/entregable2_anidada/historial_busquedas.csv.gz'
    rows = pd.read_csv(path)
    history = pd.read_csv(history_path)
    history = history.loc[~history.cache.astype(str).str.lower().eq('true')]
    times = history.groupby(['objetivo','modelo','optimizador']).segundos_evaluacion.mean().rename('segundos_evaluacion_media')
    rows = rows.merge(times,on=['objetivo','modelo','optimizador'],validate='one_to_one')
    rows['Pareto'] = True
    for _,group in rows.groupby(['objetivo','modelo']):
        for index,row in group.iterrows():
            dominated = ((group.MSE_CV_externo<=row.MSE_CV_externo)&(group.segundos_total<=row.segundos_total)&
                ((group.MSE_CV_externo<row.MSE_CV_externo)|(group.segundos_total<row.segundos_total))).any()
            rows.loc[index,'Pareto'] = not dominated
    rows.to_csv(ROOT/'data/efficiency/optimizacion_pareto.csv',index=False)
    provenance = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,history_path]}
    (ROOT/'data/efficiency/optimizacion_fuentes.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')

SETUP = '''from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML
_opciones_evaluacion = [globals().get('REPO_KERNEL', Path.cwd()), Path.cwd(), Path.cwd().parent]
REPO_EVALUACION = next(p for p in _opciones_evaluacion if (p/'data/statistical_hierarchy/metodologia.json').exists())
sys.path.insert(0, str(REPO_EVALUACION/'book/training'))
sys.path.insert(0, str(REPO_EVALUACION))
from assessment_plots import read as leer_evaluacion, metadata as metadatos_evaluacion, pvalues, mcs_table, ranks_figure, efficiency_figure
from dashboard.assessment_content import statistical_interpretation, computational_interpretation
objetivo_evaluacion = {target!r}
'''

STAT_CODE = '''
tabla_mcs = mcs_table(objetivo_evaluacion)
tabla_mcs['MCS'] = tabla_mcs.incluido_005.map({True:'Retenido',False:'Excluido'})
display(pvalues(tabla_mcs[['modelo','optimizador','MSE_OOF','R2_OOF','p_MCS','MCS']]).round(4))
tam_mcs = [int(mcs_table(objetivo_evaluacion, l, False).incluido_005.sum()) for l in [1,2,4]]
display(Markdown(statistical_interpretation(objetivo_evaluacion,tam_mcs)))
detalle_mcs = mcs_table(objetivo_evaluacion,1,False)
detalle_mcs['MCS'] = detalle_mcs.incluido_005.map({True:'Retenido',False:'Excluido'})
display(HTML('<details><summary>Los 29 procedimientos</summary>'+pvalues(detalle_mcs[
    ['procedimiento','MSE_OOF','p_MCS','MCS','representante']]).round(4).to_html(index=False)+'</details>'))
dirigidas_evaluacion = leer_evaluacion('statistical_hierarchy','dirigidas')
dirigidas_evaluacion = dirigidas_evaluacion.loc[dirigidas_evaluacion.objetivo.eq(objetivo_evaluacion)]
display(pvalues(dirigidas_evaluacion[['bloques_medios','referencia','delta_R2','IC95_inf','IC95_sup',
    'p_GW','p_GW_Holm2','p_DM','p_DM_Holm2','Cohen_dz_bloques']]).rename(columns={
        'bloques_medios':'Bloques medios','referencia':'Referencia','delta_R2':'ΔR² OOF',
        'p_GW':'p Wald condicional','p_GW_Holm2':'p Wald Holm','p_DM':'p DM-HLN',
        'p_DM_Holm2':'p DM-HLN Holm','Cohen_dz_bloques':'Cohen dz · bloques'}).round(5))
auxiliar_evaluacion = next(r for r in metadatos_evaluacion('statistical_hierarchy')['auxiliar'] if r['objetivo']==objetivo_evaluacion)
display(Markdown('**Comprobación auxiliar.** Friedman: p = '+format(auxiliar_evaluacion['p_Friedman'],'.2e')+
    '. Los intervalos del gráfico se superponen cuando la distancia de rangos no supera la diferencia crítica de Nemenyi. '
    'Los cinco pliegues comparten datos de entrenamiento.'))
fig_evaluacion = ranks_figure(objetivo_evaluacion)
display(fig_evaluacion)
plt.close(fig_evaluacion)
'''

EFF_CODE = '''
costos_evaluacion = leer_evaluacion('efficiency','resumen')
costos_evaluacion = costos_evaluacion.loc[costos_evaluacion.objetivo.eq(objetivo_evaluacion)]
residentes_evaluacion = costos_evaluacion.loc[costos_evaluacion.modo.eq('residente')]
display(residentes_evaluacion[['modelo','segundos_ajuste_historico','segundos_lote_1000','RSS_pico_MB',
    'aceleracion_lote_100','R2_prueba_original']].rename(columns={
    'segundos_ajuste_historico':'Ajuste histórico (s)','segundos_lote_1000':'ms/registro · lote 1.000',
    'RSS_pico_MB':'Pico RSS (MiB)','aceleracion_lote_100':'Aceleración por lote · 100',
    'R2_prueba_original':'R² prueba original'}).round(5).fillna('No registrado'))
fig_evaluacion = efficiency_figure(objetivo_evaluacion)
display(fig_evaluacion)
plt.close(fig_evaluacion)
kernel_costos = costos_evaluacion.loc[costos_evaluacion.modelo_id.eq('kernel_operativo')]
display(kernel_costos.assign(modo=kernel_costos.modo.map({'residente':'Residente',
    'fragmentado':'Fragmentado','rbf_vectorizado':'RBF vectorizado'}),
    error_maximo=kernel_costos.error_maximo.map(lambda valor:format(valor,'.2e')))[
    ['modo','segundos_carga','segundos_lote_1','segundos_lote_100',
    'segundos_lote_1000','RSS_pico_MB','error_maximo']].rename(columns={
        'modo':'Ejecución','segundos_carga':'Carga inicial (s)','segundos_lote_1':'Lote 1 (s)',
        'segundos_lote_100':'Lote 100 (s)','segundos_lote_1000':'Lote 1.000 (s)',
        'RSS_pico_MB':'Pico RSS (MiB)','error_maximo':'Máxima diferencia de predicción'}).round(8))
display(Markdown(computational_interpretation(kernel_costos)))
display(Markdown('Los tiempos de carga son separados. El modo fragmentado vuelve a leer componentes en cada llamada; '
    'la tabla general utiliza modelos residentes para mantener una referencia común. Los valores no disponibles de entrenamiento '
    'no se sustituyen por cero ni por el tamaño del archivo.'))
optimizadores_evaluacion = leer_evaluacion('efficiency','optimizacion_pareto')
optimizadores_evaluacion = optimizadores_evaluacion.loc[optimizadores_evaluacion.objetivo.eq(objetivo_evaluacion)]
display(HTML('<details><summary>Presupuesto y frente de Pareto de los 28 procedimientos originales</summary>'+optimizadores_evaluacion[
    ['modelo','optimizador','MSE_CV_externo','segundos_total','segundos_evaluacion_media','evaluaciones_unicas','Pareto']].round(4).to_html(index=False)+'</details>'))
'''


def cell(kind,source,identifier):
    row = dict(cell_type=kind,id=identifier,metadata={},source=source.splitlines(keepends=True))
    if kind=='code':
        row.update(outputs=[],execution_count=None)
    return row


def update():
    optimizer_efficiency()
    path = ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    previous = [copy.deepcopy(c) for c in notebook['cells'] if c['cell_type']=='code' and not c.get('id','').startswith('assessment-')]
    notebook['cells'] = [c for c in notebook['cells'] if not c.get('id','').startswith('assessment-')]
    new = [cell('markdown','### 4.12 Comparación estadística jerárquica\n\n'+STATISTICS.replace('### ','#### '),'assessment-stat-method')]
    for i,target in enumerate(['CO','NOX'],1):
        new += [cell('markdown',f'#### 4.12.{i} '+('NOx' if target=='NOX' else target),'assessment-stat-heading-'+target),
                cell('code',SETUP.format(target=target)+STAT_CODE,'assessment-stat-'+target)]
    new.append(cell('markdown','### 4.13 Complejidad y eficiencia computacional\n\n'+
                    COMPLEXITY.replace('### ','#### ')+'\n\n'+EFFICIENCY.replace('### ','#### '),'assessment-cost-method'))
    for i,target in enumerate(['CO','NOX'],1):
        new += [cell('markdown',f'#### 4.13.{i} '+('NOx' if target=='NOX' else target),'assessment-cost-heading-'+target),
                cell('code',SETUP.format(target=target)+EFF_CODE,'assessment-cost-'+target)]
    execution = max(c.get('execution_count') or 0 for c in previous)
    namespace = {'__name__':'__assessment_report__','REPO_KERNEL':ROOT,
        'Markdown':lambda text:SimpleNamespace(data=text,mime='text/markdown'),
        'HTML':lambda text:SimpleNamespace(data=text,mime='text/html')}
    for current in new:
        if current['cell_type']!='code':
            continue
        outputs = []
        def display(value):
            if isinstance(value,Figure):
                buffer = BytesIO()
                value.savefig(buffer,format='png',dpi=150,bbox_inches='tight')
                data = {'image/png':base64.b64encode(buffer.getvalue()).decode('ascii'),'text/plain':'<Figure assessment>'}
            elif hasattr(value,'to_html'):
                data = {'text/html':value.to_html(index=False),'text/plain':value.to_string(index=False)}
            else:
                data = {value.mime:value.data,'text/plain':value.data}
            outputs.append(dict(output_type='display_data',metadata={},data=data))
        namespace['display'] = display
        exec(''.join(current['source']).replace('from IPython.display import display, Markdown, HTML',''),namespace)
        execution += 1
        current.update(outputs=outputs,execution_count=execution)
        current['metadata']['verified_frozen_model_assessment'] = True
    insertion = next(i for i,c in enumerate(notebook['cells']) if ''.join(c['source']).startswith('## 5.'))
    notebook['cells'][insertion:insertion] = new
    for current in notebook['cells']:
        if current.get('id')=='1b49e614':
            text = ''.join(current['source']).replace(
                '| Comparación estadística y tamaños de efecto | CO y NOx: bootstrap pareado por bloques, CV corregida y sensibilidad de multiplicidad; alcance exploratorio |',
                '| Comparación estadística y tamaños de efecto | MCS de 29 procedimientos, contrastes dirigidos adaptados, sensibilidad y efecto; sección 4.12 |').replace(
                '| Semillas adicionales y eficiencia formal | Pendiente |',
                '| Complejidad y eficiencia | Cotas teóricas, tiempos históricos e inferencia/RSS medidos; memoria de entrenamiento no disponible; sección 4.13 |\n'
                '| Semillas adicionales | Pendiente |')
            current['source'] = text.splitlines(keepends=True)
    kept = [c for c in notebook['cells'] if c['cell_type']=='code' and not c.get('id','').startswith('assessment-')]
    assert previous==kept,'Previous code or outputs changed'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Secciones 4.12 y 4.13 guardadas; celdas anteriores y modelos intactos.')


if __name__=='__main__':
    update()
