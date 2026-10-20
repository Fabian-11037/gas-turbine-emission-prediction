"""Genera las miniaturas de la guia a partir de las curvas ya calculadas."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from dashboard.learning import summary
from dashboard.models import registry


def main():
    output = ROOT/'docs/figuras_guia'
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    for model_id,spec in registry().items():
        fig,axes = plt.subplots(1,2,figsize=(10,3.4),layout='constrained',sharey=True)
        for target,ax in zip(['CO','NOX'],axes):
            group = summary(model_id,target)
            assert len(group)==5
            for stage,label,color in [('train','Entrenamiento','#8b949c'),('validacion','Validación','#157a7c')]:
                mean = group[f'R2_{stage}_mean'].to_numpy()
                std = group[f'R2_{stage}_std'].to_numpy()
                x = group.n_train.to_numpy()
                ax.plot(x,mean,'o-',color=color,label=label,linewidth=2,markersize=4)
                ax.fill_between(x,mean-std,mean+std,color=color,alpha=.13,linewidth=0)
            ax.set(title=f"{spec['nombre']} · {target}",xlabel='Registros de entrenamiento\n(promedio entre pliegues)',ylim=(0,1.06))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value/1000:g} mil'))
            ax.set_xticks([5000,10000,15000,20000])
            ax.grid(alpha=.15)
            ax.legend(loc='lower right',fontsize=9,frameon=False)
        axes[0].set_ylabel('R²')
        fig.savefig(output/f'curva_{model_id}.png',dpi=180)
        plt.close(fig)
    print('Ocho figuras exportadas sin entrenar modelos.')


if __name__=='__main__':
    main()
