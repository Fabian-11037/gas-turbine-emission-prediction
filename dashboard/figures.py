"""Figuras sobre observaciones reales; agregacion y muestreo explicitos."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

TEAL = '#157a7c'
CORAL = '#b65346'
GRAY = '#8b949c'
COLORS = [TEAL,CORAL,'#627f35','#82648b','#aa793b']
TARGET_COLOR = {'CO':TEAL,'NOX':CORAL}


def style(fig, title, xlabel=None, ylabel=None):
    fig.update_layout(template='plotly_white',title={'text':title,'font':{'size':16}},
                      font={'family':'Segoe UI, Arial, sans-serif','size':12,'color':'#36444b'},
                      paper_bgcolor='white',plot_bgcolor='white',height=350,
                      margin={'l':52,'r':20,'t':56,'b':54},
                      legend={'orientation':'h','y':1.12,'x':0},
                      hoverlabel={'font_size':12},uirevision=title)
    fig.update_xaxes(title=xlabel,gridcolor='#edf0f2',zeroline=False,automargin=True)
    fig.update_yaxes(title=ylabel,gridcolor='#edf0f2',zeroline=False,automargin=True)
    return fig


def empty(message='No hay observaciones con estos filtros.'):
    f = go.Figure()
    f.add_annotation(text=message,x=.5,y=.5,xref='paper',yref='paper',showarrow=False)
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False)
    return style(f,'')


def distribution(df, target):
    counts,edges = np.histogram(df[target],bins=55)
    fig = go.Figure(go.Bar(x=(edges[:-1]+edges[1:])/2,y=counts,width=np.diff(edges)*.97,
                          marker_color=TARGET_COLOR[target],customdata=np.column_stack([edges[:-1],edges[1:]]),
                          hovertemplate='Intervalo: %{customdata[0]:.2f}–%{customdata[1]:.2f}<br>Registros: %{y}<extra></extra>'))
    return style(fig,f'Distribución de {target}',f'{target} (mg/m³)','Registros')


def annual_boxes(df, target):
    f = go.Figure()
    for color,(year,g) in zip(COLORS,df.groupby('anio')):
        v = g[target]
        q1,med,q3 = v.quantile([.25,.5,.75])
        iqr = q3-q1
        lower = v[v >= q1-1.5*iqr].min()
        upper = v[v <= q3+1.5*iqr].max()
        f.add_trace(go.Box(name=str(year),x=[str(year)],q1=[q1],median=[med],q3=[q3],
                           lowerfence=[lower],upperfence=[upper],marker_color=color,
                           boxpoints=False,showlegend=False))
    return style(f,f'{target} por año · cajas sin puntos extremos','Año',f'{target} (mg/m³)')


def relationship(df, target, sensor):
    sample = df.sample(min(2200,len(df)),random_state=42).sort_values('anio')
    f = go.Figure()
    for color,(year,g) in zip(COLORS,sample.groupby('anio')):
        f.add_trace(go.Scattergl(x=g[sensor],y=g[target],mode='markers',name=str(year),
                                marker={'color':color,'size':4,'opacity':.5},
                                hovertemplate=f'{sensor}: %{{x:.2f}}<br>{target}: %{{y:.2f}}<extra>%{{fullData.name}}</extra>'))
    return style(f,f'{sensor} y {target} · hasta 2.200 registros',sensor,f'{target} (mg/m³)')


def associations(df, target, features):
    correlations = df[features+[target]].rank().corr()[target].drop(target).sort_values()
    f = go.Figure(go.Bar(x=correlations.values,y=correlations.index,orientation='h',
                        marker_color=[CORAL if x < 0 else TEAL for x in correlations]))
    f.update_xaxes(range=[-1,1])
    return style(f,f'Asociación con {target} · muestra completa','ρ de Spearman','Sensor')


def correlation(df, features):
    corr = df[features+['CO','NOX']].rank().corr()
    f = go.Figure(go.Heatmap(z=corr.to_numpy(),x=corr.columns,y=corr.index,
                            colorscale=[[0,CORAL],[.5,'#fafbfc'],[1,TEAL]],zmin=-1,zmax=1,
                            colorbar={'title':'ρ'},hovertemplate='%{x} / %{y}: %{z:.3f}<extra></extra>'))
    return style(f,'Correlación de Spearman')


def block_evolution(df, target):
    f = go.Figure()
    g = df.groupby(['anio','bloque']).agg(x=('orden_registro','mean'),y=(target,'mean'))
    for color,(year,group) in zip(COLORS,g.groupby(level=0)):
        f.add_trace(go.Scatter(x=group.x,y=group.y,name=str(year),mode='lines+markers',
                              line={'color':color,'width':1.8},marker={'size':3},
                              hovertemplate='Posición: %{x:.0f}<br>Media: %{y:.2f}<extra>%{fullData.name}</extra>'))
    return style(f,f'{target} · medias por bloque de origen','Posición del registro (no fecha)',f'{target} (mg/m³)')


def annual_summary(df, target):
    group = df.groupby('anio')[target].agg(['mean','median'])
    f = go.Figure()
    for col,label,color in [('mean','Media',TEAL),('median','Mediana',CORAL)]:
        f.add_trace(go.Scatter(x=group.index,y=group[col],mode='lines+markers',name=label,
                              line={'color':color,'width':2}))
    f.update_xaxes(dtick=1)
    return style(f,f'{target} · diferencias entre años','Año',f'{target} (mg/m³)')


def stage_scores(row):
    f = go.Figure(go.Bar(x=['Entrenamiento','CV (media)','Prueba'],
                        y=[row.R2_train,row.R2_CV_promedio,row.R2_test],
                        marker_color=[GRAY,TEAL,CORAL],
                        error_y={'type':'data','array':[0,row.R2_CV_std,0]},
                        hovertemplate='%{x}<br>R²: %{y:.3f}<extra></extra>'))
    f.add_hline(y=0,line_color='#dce3e7')
    return style(f,'R² por etapa · CV ± 1 desviación estándar',None,'R²')


def observed_predictions(frame, target, family):
    col = f'{target}_{family.replace(" ","_")}'
    sample = frame.sample(min(2200,len(frame)),random_state=42)
    f = go.Figure(go.Scattergl(x=sample[target],y=sample[col],mode='markers',name='Observaciones',
                              marker={'color':TARGET_COLOR[target],'size':4,'opacity':.45},
                              hovertemplate='Observado: %{x:.2f}<br>Estimado: %{y:.2f}<extra></extra>'))
    low = min(frame[target].min(),frame[col].min())
    high = max(frame[target].max(),frame[col].max())
    f.add_trace(go.Scatter(x=[low,high],y=[low,high],mode='lines',name='Predicción perfecta',
                          line={'color':GRAY,'dash':'dash'},hoverinfo='skip'))
    return style(f,'Observado frente a estimado · prueba',f'Observado (mg/m³)',f'Estimado (mg/m³)')


def residual_histogram(frame, target, family):
    residual = frame[target]-frame[f'{target}_{family.replace(" ","_")}']
    counts,edges = np.histogram(residual,bins=50)
    f = go.Figure(go.Bar(x=(edges[:-1]+edges[1:])/2,y=counts,marker_color=TARGET_COLOR[target]))
    f.add_vline(x=0,line_color=GRAY,line_dash='dash')
    return style(f,'Distribución de residuos · prueba','Observado − estimado (mg/m³)','Registros')


def partition_chart():
    f = go.Figure()
    for label,n,color in [('Desarrollo',27570,TEAL),('Prueba',7411,CORAL),('Purga',1752,GRAY)]:
        f.add_trace(go.Bar(x=[n],y=['Muestra'],orientation='h',name=label,marker_color=color,
                          text=f'{n:,}'.replace(',','.'),textposition='inside',textangle=0))
    style(f,'Población por partición','Registros')
    f.update_layout(barmode='stack',height=205,margin={'l':15,'r':15,'t':95,'b':30},
                    legend={'orientation':'h','y':1.55,'x':0},uniformtext_minsize=10,uniformtext_mode='hide')
    f.update_yaxes(visible=False)
    return f
