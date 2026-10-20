"""Vista de curvas precalculadas: no ajusta modelos durante las consultas."""
import plotly.graph_objects as go
from . import figures as f
from .components import number, table
from .repository import learning_curves


def summary(model_id,target):
    frame = learning_curves().loc[lambda d:d.modelo_id.eq(model_id)&d.objetivo.eq(target)]
    if frame.empty:
        return frame
    measures = {f'{metric}_{stage}_{stat}':(f'{metric}_{stage}',stat)
                for metric in ['R2','RMSE'] for stage in ['train','validacion'] for stat in ['mean','std']}
    return frame.groupby('fraccion',as_index=False).agg(
        modelo=('modelo','first'),objetivo=('objetivo','first'),
        n_train=('n_train','mean'),n_train_min=('n_train','min'),n_train_max=('n_train','max'),
        n_validacion_min=('n_validacion','min'),n_validacion_max=('n_validacion','max'),
        pliegues=('fold','nunique'),**measures).sort_values('fraccion')


def figure(group,metric='R2'):
    if metric not in ['R2','RMSE']:
        raise ValueError('Metrica de curva no reconocida.')
    if group.empty:
        return f.empty('Curva de aprendizaje no disponible para este modelo.')
    plot = go.Figure()
    x = group.n_train.to_numpy()
    for stage,label,color,shade in [('train','Entrenamiento',f.GRAY,'rgba(139,148,156,.13)'),
                                    ('validacion','Validación',f.TEAL,'rgba(21,122,124,.13)')]:
        mean = group[f'{metric}_{stage}_mean'].to_numpy()
        std = group[f'{metric}_{stage}_std'].to_numpy()
        plot.add_trace(go.Scatter(x=x,y=mean-std,mode='lines',line={'width':0},
                                 hoverinfo='skip',showlegend=False))
        plot.add_trace(go.Scatter(x=x,y=mean+std,mode='lines',line={'width':0},
                                 fill='tonexty',fillcolor=shade,hoverinfo='skip',showlegend=False))
        plot.add_trace(go.Scatter(x=x,y=mean,mode='lines+markers',name=label,
            line={'color':color,'width':2},marker={'size':7},customdata=group.fraccion.to_numpy()*100,
            hovertemplate='%{customdata:.0f}% de los bloques disponibles (aprox.)<br>Registros medios: %{x:,.0f}<br>'+label+': %{y:.3f}<extra></extra>'))
    label = 'R²' if metric=='R2' else 'RMSE (mg/m³)'
    title = f"{group.modelo.iloc[0]} · {'NOx' if group.objetivo.iloc[0]=='NOX' else 'CO'}"
    plot = f.style(plot,title,
                   'Registros de entrenamiento<br>(promedio entre pliegues)',label)
    plot.update_layout(margin={'t':85,'b':65},legend={'y':1.18,'x':0})
    return plot


def render(model_id,target,metric):
    group = summary(model_id,target)
    if group.empty:
        return figure(group,metric),'No hay una curva calculada para esta selección.',''
    last = group.iloc[-1]
    label = 'R²' if metric=='R2' else 'RMSE'
    note = (f"Cinco tamaños, de 20% a 100%, con los mismos cinco grupos de validación. "
            f"Al 100% se usan entre {number(int(last.n_train_min),0)} y {number(int(last.n_train_max),0)} registros por ajuste, "
            f"no los 27.570 del modelo final. Las franjas muestran ±1 desviación entre grupos, no un intervalo de confianza. "
            f"{label} al 100%: entrenamiento {number(last[f'{metric}_train_mean'])}; "
            f"validación {number(last[f'{metric}_validacion_mean'])}.")
    values = group[['fraccion','n_train','n_train_min','n_train_max',
                    f'{metric}_train_mean',f'{metric}_train_std',f'{metric}_validacion_mean',f'{metric}_validacion_std']].copy()
    values['fraccion'] = values.fraccion.map(lambda v:f'{v:.0%}')
    values['n_train'] = values.n_train.round().astype(int)
    labels = {'fraccion':'Bloques disponibles (aprox.)','n_train':'Registros medios',
              'n_train_min':'Mínimo por ajuste','n_train_max':'Máximo por ajuste',
              f'{metric}_train_mean':f'{label} entrenamiento',f'{metric}_train_std':'Desv. entrenamiento',
              f'{metric}_validacion_mean':f'{label} validación',f'{metric}_validacion_std':'Desv. validación'}
    return figure(group,metric),note,table(values,labels)
