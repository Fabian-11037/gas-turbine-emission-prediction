"""Scientific figures from saved SHAP/LIME values; no estimator is loaded."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dashboard.sensor_labels import sensor_label

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/explainability'
FEATURES = ['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP']
TEAL, CORAL, GRAY = '#157a7c','#b65346','#637274'
NAMES = {'kernel_operativo':'Kernel operativo','xgboost':'XGBoost'}


def read(name):
    return pd.read_csv(DATA/(name+'.csv'))


def subset(name,model_id,target):
    frame = read(name)
    return frame.loc[frame.modelo_id.eq(model_id) & frame.objetivo.eq(target)].copy()


def global_figure(model_id,target):
    frame = pd.read_csv(DATA/f'global_{model_id}_{target}.csv.gz')
    importance = subset('resumen_global',model_id,target).sort_values('media_abs_SHAP')
    fig,axes = plt.subplots(1,2,figsize=(14,6),layout='constrained')
    colors = LinearSegmentedColormap.from_list('sensor',[TEAL,'#d7dddd',CORAL])
    rng = np.random.default_rng(42)
    for index,sensor in enumerate(importance.sensor):
        points = axes[0].scatter(frame['shap_'+sensor],index+rng.uniform(-.27,.27,len(frame)),
            c=frame[sensor].rank(pct=True),cmap=colors,vmin=0,vmax=1,s=3,alpha=.5,rasterized=True)
    axes[0].set(yticks=range(9),yticklabels=[sensor_label(s, width=28) for s in importance.sensor],
        xlabel='Contribución a la estimación (mg/m³)',title='Distribución de contribuciones · SHAP')
    axes[0].axvline(0,c=GRAY,ls=':',lw=1)
    colorbar = fig.colorbar(points,ax=axes[0],fraction=.025,pad=.02)
    colorbar.set_ticks([0,1],labels=['Bajo','Alto'])
    colorbar.set_label('Valor del sensor')
    axes[1].barh(importance.sensor,importance.media_abs_SHAP,color=TEAL)
    axes[1].set_yticks(range(9), labels=[sensor_label(s, width=28) for s in importance.sensor])
    axes[1].set(xlabel='Contribución absoluta media (mg/m³)',title='Importancia global · SHAP')
    for ax in axes:
        ax.tick_params(axis='y', labelsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.12)
    fig.suptitle(NAMES[model_id]+' · '+('NOx' if target=='NOX' else target),fontsize=13)
    return fig


def local_figure(model_id,target,case):
    cases = subset('casos',model_id,target)
    summary = cases.loc[cases.caso.eq(case)].iloc[0]
    local = subset('local',model_id,target)
    local = local.loc[local.caso.eq(case)].copy()
    shap = local.assign(magnitude=local.SHAP.abs()).sort_values('magnitude',ascending=False)
    lime = local.assign(magnitude=local.LIME.abs()).sort_values('magnitude',ascending=False)
    fig,axes = plt.subplots(1,2,figsize=(14,7),layout='constrained')
    ax = axes[0]
    ax.barh(0,summary.base_SHAP,color=GRAY,height=.65)
    previous = summary.base_SHAP
    for index,row in enumerate(shap.itertuples(),1):
        end = previous+row.SHAP
        ax.barh(index,abs(row.SHAP),left=min(previous,end),color=CORAL if row.SHAP>0 else TEAL,height=.65)
        ax.plot([previous,previous],[index-1+.325,index-.325],color='#b7c0c0',lw=.8)
        ax.annotate(f'{row.SHAP:+.2f}',(end,index),xytext=(4 if row.SHAP>=0 else -4,0),
            textcoords='offset points',ha='left' if row.SHAP>=0 else 'right',va='center',fontsize=8)
        previous = end
    ax.barh(10,summary.prediccion,color=GRAY,height=.65)
    ax.set(yticks=range(11),yticklabels=['Referencia']+[sensor_label(s, width=28) for s in shap.sensor]+['Estimación'],
           title='Cómo se forma la estimación · SHAP',xlabel='Concentración y contribuciones (mg/m³)')
    ax.invert_yaxis()
    ax.margins(x=.2)
    ax = axes[1]
    ax.barh(lime.sensor,lime.LIME,color=[CORAL if v>0 else TEAL for v in lime.LIME])
    ax.set_yticks(range(9), labels=[sensor_label(s, width=28) for s in lime.sensor])
    ax.axvline(0,c=GRAY,lw=.8)
    ax.set(title=f'Aproximación LIME · R² local = {summary.R2_local_LIME:.3f}',
           xlabel='Peso del intervalo del sensor (mg/m³)')
    ax.invert_yaxis()
    for ax in axes:
        ax.tick_params(axis='y', labelsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.12)
    title = NAMES[model_id]+' · '+('NOx' if target=='NOX' else target)
    fig.suptitle(title+' · '+('error bajo' if case=='error_bajo' else 'error alto')+
        f' · año {int(summary.anio)}, registro {int(summary.orden_registro)}',fontsize=12)
    return fig
