"""Figures and tabular evidence; no fitted models or statistical engines loaded."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data'


def read(folder,name):
    return pd.read_csv(DATA/folder/(name+'.csv'))


def metadata(folder):
    return json.loads((DATA/folder/'metodologia.json').read_text(encoding='utf-8'))


def pvalues(frame):
    frame = frame.copy()
    for column in frame:
        if column.startswith('p_'):
            frame[column] = ['<0,0002' if value==0 else f'{value:.2e}'.replace('.',',') for value in frame[column]]
    return frame


def mcs_table(target,length=1.,published_only=True):
    candidates = read('statistical_hierarchy','procedimientos')
    result = read('statistical_hierarchy','mcs')
    rows = candidates.loc[candidates.objetivo.eq(target)]
    if published_only:
        rows = rows.loc[rows.publicado]
    return rows.merge(result.loc[result.objetivo.eq(target)&result.bloques_medios.eq(length)],
        on=['objetivo','procedimiento'],validate='one_to_one').sort_values('MSE_OOF')


def ranks_figure(target):
    frame = read('statistical_hierarchy','rangos').query('objetivo == @target').sort_values('rango_medio')
    aux = next(row for row in metadata('statistical_hierarchy')['auxiliar'] if row['objetivo']==target)
    fig,ax = plt.subplots(figsize=(9,4.5),layout='constrained')
    y = np.arange(len(frame))
    ax.errorbar(frame.rango_medio,y,xerr=aux['CD']/2 if aux['posthoc_aplicado'] else None,
        fmt='o',color='#157a7c',ecolor='#9aafaf',capsize=4)
    ax.set(yticks=y,yticklabels=frame.procedimiento.str.split(' · ').str[0],
        xlabel='Rango medio: menor es mejor',title=f'Friedman–Nemenyi · {target} · CD = {aux["CD"]:.2f}')
    ax.invert_yaxis()
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=.15)
    return fig


def efficiency_figure(target):
    frame = read('efficiency','resumen')
    native = frame.loc[frame.objetivo.eq(target)&frame.modo.eq('residente')].sort_values('segundos_lote_1000')
    kernel = frame.loc[frame.objetivo.eq(target)&frame.modelo_id.eq('kernel_operativo')]
    fig,axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].barh(native.modelo,native.segundos_lote_1000,color='#157a7c')
    axes[0].set_xscale('log')
    axes[0].set(xlabel='ms/registro · lote de 1.000',title='Tiempo de predicción · carga residente')
    names = kernel.modo.map({'residente':'Residente','fragmentado':'Fragmentado','rbf_vectorizado':'RBF vectorizado'})
    axes[1].barh(names,kernel.RSS_pico_MB,color=['#8b949c','#b65346','#157a7c'])
    axes[1].set(xlabel='Pico RSS del proceso (MiB)',title='Kernel operativo · memoria observada')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('Eficiencia computacional · '+('NOx' if target=='NOX' else target))
    return fig
