"""Static figures from saved stability evidence; no fitting."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def read(name):
    return pd.read_csv(ROOT/'data/stability_ablation'/(name+'.csv'))


def seeds_figure(target):
    rows = read('sensibilidad_semillas').query('objetivo == @target').sort_values('mean')
    fig, ax = plt.subplots(figsize=(9,4.8),layout='constrained')
    for i,row in enumerate(rows.itertuples()):
        color = '#8b949c' if row.determinista else '#157a7c'
        ax.plot([row.min,row.max],[i,i],color=color,linewidth=3)
        ax.scatter(row.mean,i,color=color,s=35,zorder=3)
    ax.set_yticks(range(len(rows)),rows.modelo)
    ax.set(xlabel='R² OOF: media y rango entre semillas',title='Estabilidad con particiones fijas · '+target)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=.15)
    return fig


def ablation_figure(target):
    from dashboard.stability_content import NAMES
    data = read('resultados')
    data = data.loc[data.objetivo.eq(target)&data.modelo_id.eq('kernel_operativo')&data.semilla.eq(42)]
    rows = data.pivot(index='variante',columns='fase',values='R2').reindex(NAMES)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9,4.8),layout='constrained')
    ax.barh(y-.18,rows.OOF,height=.34,label='Validación OOF',color='#157a7c')
    ax.barh(y+.18,rows.prueba,height=.34,label='Prueba',color='#d96953')
    ax.set_yticks(y,[NAMES[v] for v in rows.index])
    ax.invert_yaxis()
    ax.set(xlabel='R²',title='Aporte de los componentes · '+target)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.10),ncol=2,frameon=False)
    ax.set_title('Aporte de los componentes · '+target,pad=40)
    ax.grid(axis='x',alpha=.15)
    return fig
