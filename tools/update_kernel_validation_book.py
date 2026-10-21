"""Publica evidencia terminada y sus figuras, sin entrenar ni repetir contrastes."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from dashboard import kernel_validation as view, kernel_validation_content as text

SOURCE = ROOT.parent/'Pruebas provisionales CO NOx/comparacion_kernel_validaciones_20261021'
STAGES = ROOT.parent/'Pruebas provisionales CO NOx/resultados_comparacion_pendiente_20261021'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def r2(frame,target):
    y=frame[target].to_numpy()
    return float(1-np.sum((y-frame.prediccion.to_numpy())**2)/np.sum((y-y.mean())**2))


def export_evidence():
    completed=json.loads((SOURCE/'completado.json').read_text(encoding='utf-8'))
    assert completed['estado']=='Completado'
    location=view.LOCATION
    location.mkdir(exist_ok=True)
    archive=ROOT/'book/informe_resultados/kernel_validation'
    archive.mkdir(exist_ok=True)
    for src,dest in [('significancia_CV.csv','cv.csv'),('significancia_prueba.csv','prueba.csv'),
                     ('diferencias_por_fold.csv','diferencias_pliegues.csv'),('etapas.csv','etapas.csv')]:
        shutil.copy2(SOURCE/src,location/dest)
    cv=pd.read_csv(location/'cv.csv').set_index('objetivo')
    tests=pd.read_csv(location/'prueba.csv').loc[lambda d:d.longitud_bloque.eq(168)].set_index('objetivo')
    rows=[]
    for target in ['CO','NOX']:
        source_oof=SOURCE/f'OOF_{target}_kernel.csv.gz'
        source_test=STAGES/f'predicciones_comun_{target}_kernel_final.csv.gz'
        for path,name in [(source_oof,f'aleatoria_OOF_{target}.csv.gz'),
                          (source_test,f'aleatoria_prueba_{target}.csv.gz')]:
            shutil.copy2(path,archive/name)
        old_oof=pd.read_csv(ROOT/f'book/informe_resultados/kernel_operativo/predicciones_OOF_{target}.csv.gz')
        old_test=pd.read_csv(ROOT/f'book/informe_resultados/kernel_operativo/predicciones_prueba_{target}.csv.gz')
        new_oof,new_test=pd.read_csv(source_oof),pd.read_csv(source_test)
        keys=['anio','orden_registro','bloque',target]
        pd.testing.assert_frame_equal(old_oof[keys],new_oof[keys])
        pd.testing.assert_frame_equal(old_test[keys],new_test[keys])
        assert len(new_oof)==27570 and len(new_test)==7411
        for strategy,fold_score,oof,test_score in [
            ('bloques',cv.loc[target,'R2_CV_referencia'],old_oof,tests.loc[target,'R2_prueba_referencia']),
            ('aleatoria',cv.loc[target,'R2_CV_nuevo'],new_oof,tests.loc[target,'R2_prueba_nuevo'])]:
            rows.append(dict(objetivo=target,estrategia=strategy,R2_CV=float(fold_score),
                R2_OOF=r2(oof,target),R2_prueba=float(test_score)))
    pd.DataFrame(rows).to_csv(location/'resultados.csv',index=False)
    paths=list(location.glob('*.csv'))+list(archive.glob('*.csv.gz'))
    metadata=dict(revision='2026-10-21',n_desarrollo=27570,n_prueba=7411,
        pliegues_externos=5,pliegues_internos=3,cross_fitting=3,bloque=168,purga=24,
        replicas_bootstrap=20000,bloque_principal=168,bloques_sensibilidad=[168,336,672],
        orientacion='delta R2 = bloques menos aleatoria',
        alcance='Mismos datos de desarrollo, bloques externos y prueba; cambian seleccion y cross-fitting internos.',
        diferencia_frente_al_80_20_original='No se comparan modelos sobre filas que conocieron al entrenar.',
        ajuste='Holm para CO y NOx; separado en CV, prueba y cada longitud de remuestreo.',
        modelos_reentrenados_al_publicar=0,prueba_usada_en_seleccion=False,exploratorio=True,
        origen=completed['signature'],
        archivos={p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    (location/'metodologia.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    view.evidence.cache_clear()


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results,_,tests=view.evidence()
    tests=tests.loc[tests.longitud_bloque.eq(168)].set_index('objetivo')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(10,4.4))
    fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.73)
    for strategy,offset,color in [('bloques',-.18,view.f.TEAL),('aleatoria',.18,view.f.CORAL)]:
        values=[]
        for target in ['CO','NOX']:
            row=results.loc[results.objetivo.eq(target)&results.estrategia.eq(strategy)].iloc[0]
            values.extend([row.R2_CV,row.R2_prueba])
        bars=ax.bar(np.arange(4)+offset,values,width=.32,color=color,label=view.STRATEGIES[strategy],zorder=3)
        for bar,value in zip(bars,values):
            ax.annotate(view.number(value,4),(bar.get_x()+bar.get_width()/2,value),
                        xytext=(0,6),textcoords='offset points',ha='center',fontsize=11)
    ax.set(ylim=(0,1.02),xticks=range(4),xticklabels=['CO\nCV','CO\nPrueba','NOx\nCV','NOx\nPrueba'],
           ylabel='R² (más alto es mejor)')
    ax.grid(axis='y',alpha=.2,zorder=0)
    ax.axvline(1.5,color='#d7dce0',lw=1)
    fig.suptitle('Misma evaluación externa, dos estrategias de preparación',x=.09,y=.97,ha='left',fontsize=16)
    fig.text(.09,.86,'CV: promedio de cinco pliegues por bloques. Prueba: los mismos 7.411 registros.',fontsize=11)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.08,.85),frameon=False,ncol=2)
    paths=[ROOT/'assets/kernel_validation_r2.png',ROOT/'assets/kernel_validation_ci.png']
    fig.savefig(paths[0],dpi=160,facecolor='white')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.6))
    fig.subplots_adjust(left=.11,right=.97,bottom=.36,top=.69)
    for target,y,color in [('CO',.78,view.f.CORAL),('NOX',.22,view.f.TEAL)]:
        row=tests.loc[target]
        ax.errorbar(row.delta_R2,y,xerr=[[row.delta_R2-row.IC95_inf],[row.IC95_sup-row.delta_R2]],
            fmt='o',color=color,elinewidth=3,capsize=6,markersize=8)
        ax.text(.99,y+.105,'p ajustado = '+view.number(row.p_Holm,4),transform=ax.get_yaxis_transform(),
            ha='right',va='bottom',fontsize=12)
        ax.annotate(view.number(row.delta_R2,4),(row.delta_R2,y),xytext=(0,-19),
                    textcoords='offset points',ha='center',va='top',fontsize=12)
    ax.axvline(0,color='#36444b',linestyle='--',lw=1.2)
    ax.set(xlim=(-.075,.06),ylim=(-.05,1.07),yticks=[.78,.22],yticklabels=['CO','NOx'],
        xlabel='Diferencia de R² de prueba: bloques − aleatoria')
    ax.tick_params(axis='both',length=0,pad=9)
    ax.spines['left'].set_visible(False)
    fig.suptitle('Diferencia e incertidumbre en prueba',x=.11,y=.98,ha='left',fontsize=17)
    fig.text(.11,.85,'Punto: diferencia observada. Línea: intervalo de confianza del 95 %.',fontsize=12)
    fig.text(.11,.13,'Valores negativos: favorecen aleatoria',fontsize=11)
    fig.text(.97,.13,'Valores positivos: favorecen bloques',fontsize=11,ha='right')
    fig.text(.11,.035,'20.000 remuestreos pareados de segmentos de 168 registros. Ajuste Holm para CO y NOx.',fontsize=10)
    fig.savefig(paths[1],dpi=160,facecolor='white')
    plt.close(fig)
    return paths


CODE = '''from pathlib import Path
from IPython.display import Image, display
import sys
_rutas_validacion = [globals().get('REPO_KERNEL'), Path.cwd(), Path.cwd().parent]
REPO_VALIDACION = next(p for p in _rutas_validacion if p is not None and (p/'data/kernel_validation').exists())
sys.path.insert(0, str(REPO_VALIDACION))
from dashboard.kernel_validation import result_table, significance_table
display(Image(filename=str(REPO_VALIDACION/'assets/kernel_validation_r2.png')))
display(Image(filename=str(REPO_VALIDACION/'assets/kernel_validation_ci.png')))
display(result_table().round(4))
display(significance_table().round(4))
'''


def update_book(paths):
    path=ROOT/'book/gas_turbine_eda_hilo_conductor_redaccion.ipynb'
    notebook=json.loads(path.read_text(encoding='utf-8'))
    prefix='kernel-validation-'
    prior=copy.deepcopy([c for c in notebook['cells'] if not c.get('id','').startswith(prefix)])
    notebook['cells']=copy.deepcopy(prior)
    paragraphs=['#### 4.10.6 Validación ante segmentos operativos separados',text.INTRO,text.DESIGN]
    outputs=[dict(output_type='display_data',metadata={},data={
        'image/png':base64.b64encode(p.read_bytes()).decode('ascii'),
        'text/plain':'<Figure: '+p.stem+'>'}) for p in paths]
    for frame in [view.result_table(),view.significance_table()]:
        frame=frame.round(4)
        outputs.append(dict(output_type='display_data',metadata={},data={
            'text/html':frame.to_html(index=False),'text/plain':frame.to_string(index=False)}))
    count=max(c.get('execution_count') or 0 for c in prior if c['cell_type']=='code')+1
    cells=[dict(cell_type='markdown',id=prefix+'method',metadata={},source='\n\n'.join(paragraphs).splitlines(keepends=True)),
        dict(cell_type='code',id=prefix+'results',metadata={'saved_evidence_only':True,'tags':['hide-input']},
             source=CODE.splitlines(keepends=True),execution_count=count,outputs=outputs),
        dict(cell_type='markdown',id=prefix+'conclusion',metadata={},source=(text.READING+'\n\n'+
            view.interpretation('CO')+'\n\n'+view.interpretation('NOX')+'\n\n'+
            '**Decisión de validación.** '+text.CONCLUSION+'\n\n'+text.METHOD).splitlines(keepends=True))]
    index=next(i for i,c in enumerate(notebook['cells']) if c.get('id')=='kernel-statistics-code')+1
    notebook['cells'][index:index]=cells
    changed=[]
    for cell in notebook['cells']:
        if cell.get('id')=='fe9bdcbf':
            content=''.join(cell['source'])
            marker='**Validación para ambas emisiones.** '
            pieces=content.split('\n\n')
            pieces=[p for p in pieces if not p.startswith(marker)]
            pieces.insert(3,marker+text.CONCLUSION)
            cell['source']='\n\n'.join(pieces).replace('secciones 4.10 a 4.14','secciones 4.10 a 4.15').splitlines(keepends=True)
            changed.append(cell['id'])
        if cell.get('id')=='840fa979':
            content=''.join(cell['source'])
            old='La elección responde al objetivo de estimar emisiones del registro actual, no a anticipar el futuro.'
            new=(old+' La reducción de similitud entre sensores (sección 3.9) y la comparación controlada '
                 '(sección 4.10.6) respaldan mantener bloques para ambas emisiones: hay ventaja significativa '
                 'en NOx, mientras que en CO no se distingue una estrategia superior.')
            if new not in content:
                content=content.replace(old,new)
            for a,b in [('Validacion documentada','Validación documentada'),('primeros tres anos','primeros tres años'),
                        ('ultimos dos','últimos dos'),('dos anos separados','dos años separados'),
                        ('evaluacion','evaluación'),('validacion','validación'),('metodologia p.','metodología p.')]:
                content=content.replace(a,b)
            cell['source']=content.splitlines(keepends=True)
            changed.append(cell['id'])
    assert [c for c in notebook['cells'] if c.get('id') not in changed and not c.get('id','').startswith(prefix)]==[
        c for c in prior if c.get('id') not in changed]
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')


def main():
    export_evidence()
    update_book(figures())
    print('Comparación incorporada; resultados previos y artefactos de predicción conservados.')


if __name__=='__main__':
    main()
