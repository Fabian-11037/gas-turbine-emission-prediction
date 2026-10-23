"""Interactive exploration of precomputed multivariate EDA, never fitting at runtime."""
from functools import lru_cache
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input,Output,dcc,html

from . import figures as f
from .components import disclosure,graph,metric,notice,number,table
from .repository import DATA,filtered,sensors

LOCATION = DATA/'multivariate'
INTRO = ('Se analizan los nueve sensores en conjunto para reconocer condiciones operativas similares '
    'y combinaciones inusuales. Estos análisis describen la operación, sin modificar los modelos de emisiones.')


@lru_cache(maxsize=1)
def evidence():
    return (pd.read_csv(LOCATION/'proyecciones.csv.gz'),pd.read_csv(LOCATION/'cargas.csv'),
            pd.read_csv(LOCATION/'varianza.csv'),json.loads((LOCATION/'metodologia.json').read_text(encoding='utf-8')))


def selected(years,population):
    points = evidence()[0]
    data = filtered(years,population)
    return data.merge(points.drop(columns=['anio','particion']),on='orden_registro',validate='one_to_one')


def loadings_figure():
    values = evidence()[1]
    fig = go.Figure(go.Heatmap(z=values[['PC1','PC2']].to_numpy(),x=['PC1','PC2'],y=values.sensor,
        colorscale=[[0,f.CORAL],[.5,'#fafbfc'],[1,f.TEAL]],zmin=-1,zmax=1,
        text=[[number(v,2) for v in row] for row in values[['PC1','PC2']].to_numpy()],texttemplate='%{text}',
        colorbar={'title':'Carga','thickness':10,'len':.8},
        hovertemplate='%{y} · %{x}<br>Correlación: %{z:.3f}<extra></extra>'))
    f.style(fig,'Cargas de sensores · PCA','Componente','Sensor')
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(height=380,margin={'l':53,'r':24,'t':70,'b':55})
    return fig


def map_figure(frame,color,target):
    if frame.empty:
        return f.empty()
    points = frame.sample(min(2200,len(frame)),random_state=42)
    fig = go.Figure()
    hover='PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Año: %{customdata[0]:.0f}'
    if color=='grupo':
        hover += '<br>Grupo: %{customdata[1]:.0f}'
    hover += '<extra>%{fullData.name}</extra>'
    if color=='emision':
        fig.add_trace(go.Scattergl(x=points.PC1,y=points.PC2,mode='markers',name='NOx' if target=='NOX' else target,
            marker={'color':points[target],'colorscale':[[0,'#e6f1f1'],[1,f.TEAL]],'size':5,'opacity':.65,
                    'colorbar':{'title':('NOx' if target=='NOX' else target)+'<br>mg/m³','thickness':10}},
            customdata=np.column_stack([points.anio,points.grupo,points[target]]),
            hovertemplate=hover.replace('<extra>','<br>Emisión: %{customdata[2]:.2f} mg/m³<extra>')))
    else:
        field = {'anio':'anio','grupo':'grupo','inusual':'inusual'}[color]
        groups = [False,True] if color=='inusual' else sorted(points[field].unique())
        for index,value in enumerate(groups):
            group = points.loc[points[field].eq(value)]
            label = ('Inusual' if value else 'Habitual') if color=='inusual' else 'Grupo '+str(value) if color=='grupo' else str(value)
            fig.add_trace(go.Scattergl(x=group.PC1,y=group.PC2,mode='markers',name=label,
                marker={'color':f.COLORS[index%len(f.COLORS)],'size':5,'opacity':.5},
                customdata=np.column_stack([group.anio,group.grupo]),hovertemplate=hover))
    variance = evidence()[2].varianza
    f.style(fig,'Mapa de operación · PCA',f'PC1 · {number(100*variance.iloc[0],1)} %',
            f'PC2 · {number(100*variance.iloc[1],1)} %')
    fig.update_layout(height=380,legend={'orientation':'h','y':-.25,'x':0,'font':{'size':10}},
        margin={'l':48,'r':20,'t':70,'b':90})
    return fig


def variance_figure():
    values = evidence()[2]
    fig = go.Figure(go.Bar(x=values.componente,y=100*values.varianza,name='Individual',marker_color=f.TEAL))
    fig.add_trace(go.Scatter(x=values.componente,y=100*values.acumulada,name='Acumulada',
        mode='lines+markers',line={'color':f.CORAL}))
    f.style(fig,'Variación explicada · PCA','Número de componente','Variación de sensores (%)')
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0,105])
    fig.update_layout(legend={'orientation':'h','y':-.25,'x':0},margin={'t':65,'b':90})
    return fig


def pca_panel(summary,figure):
    _,loadings,variance,_ = evidence()
    labels = {s['id']:s['nombre'] for s in sensors()}
    loads = loadings[['sensor','PC1','PC2']].copy()
    loads.insert(1,'nombre',loads.sensor.map(labels))
    explained = variance.assign(varianza=100*variance.varianza,acumulada=100*variance.acumulada)
    return html.Section([
        html.H4('PCA · resumen de los sensores'),
        html.P('PCA combina los nueve sensores estandarizados en componentes que conservan la mayor '
            'variación posible. El mapa utiliza las dos primeras para visualizar semejanzas entre registros y años.'),
        html.Div(summary,id='multivariate-summary',className='metric-strip compact'),
        html.Div([graph('multivariate-map',figure),graph('multivariate-loadings',loadings_figure())],className='chart-grid'),
        html.P('Cada punto representa un registro; se muestran hasta 2.200. Las cargas son correlaciones '
            'entre cada sensor y la componente: una magnitud cercana a 1 indica mayor asociación; '
            'el signo señala su dirección.',className='caption'),
        notice('PC1 resume principalmente producción y presiones, en contraste con la temperatura de salida. '
            'PC2 recoge sobre todo temperatura, humedad y presión ambiente. La variación explicada '
            'describe los sensores, no el R² de predicción de emisiones.'),
        disclosure('Varianza explicada y cargas del PCA',[
            graph('multivariate-variance',variance_figure()),table(explained,
                {'componente':'Componente','varianza':'Variación individual (%)','acumulada':'Acumulada (%)'},digits=2),
            table(loads,{'sensor':'Sensor','nombre':'Variable','PC1':'Carga PC1','PC2':'Carga PC2'},digits=3)]),
        disclosure('Método del PCA',[
            html.P('Los nueve sensores se estandarizan restando su media y dividiendo por su desviación. '
                'Así, las distintas unidades no dominan la comparación. Se utilizan los 27.570 registros de '
                'desarrollo; los restantes se proyectan sin volver a ajustar el escalado ni las componentes. '
                'CO y NOx no intervienen en el ajuste del PCA.')])
    ],id='multivariate-pca-panel')


def group_profiles(frame):
    if frame.empty:
        return f.empty()
    meta = evidence()[3]
    medians = frame.groupby('grupo')[meta['sensores']].median()
    values = ((medians-np.asarray(meta['escala']['media']))/np.asarray(meta['escala']['desviacion'])).T
    limit = max(1.,float(np.abs(values.to_numpy()).max()))
    fig = go.Figure(go.Heatmap(z=values.to_numpy(),x=['Grupo '+str(v) for v in values.columns],y=values.index,
        customdata=medians.T.to_numpy(),colorscale=[[0,f.CORAL],[.5,'#fafbfc'],[1,f.TEAL]],
        zmin=-limit,zmax=limit,colorbar={'title':'Perfil z','thickness':10},
        hovertemplate='%{y} · %{x}<br>Mediana: %{customdata:.3f}<br>Perfil z: %{z:.2f}<extra></extra>'))
    f.style(fig,'Perfil de los grupos','Grupo','Sensor')
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(height=380,margin={'l':53,'r':24,'t':70,'b':55})
    return fig


def group_interpretation(frame,target):
    if frame.empty:
        return 'Sin registros con estos filtros.'
    groups = frame.groupby('grupo')[['TEY',target]].median().sort_values('TEY')
    label = 'NOx' if target=='NOX' else target
    if len(groups)<2:
        return 'La selección contiene un solo grupo; no permite comparar perfiles entre grupos.'
    low,high = groups.iloc[0],groups.iloc[-1]
    return (f'En los registros seleccionados, el grupo {groups.index[0]} tiene la menor mediana de producción '
        f'({number(low.TEY,2)} MWh) y una mediana de {label} de {number(low[target],2)} mg/m³. '
        f'El grupo {groups.index[-1]} presenta la mayor producción mediana ({number(high.TEY,2)} MWh) '
        f'y {number(high[target],2)} mg/m³ de {label}. Son diferencias descriptivas, no efectos causales.')


def kmeans_panel(frame,target):
    meta = evidence()[3]['kmeans']
    return html.Section([
        html.H4('KMeans · grupos de condiciones operativas'),
        html.P('KMeans reúne registros con combinaciones de sensores parecidas. Cada grupo tiene un centro '
            'representativo y cada registro se asigna al centro más cercano. Se comparan los nueve sensores '
            'estandarizados, de modo que sus distintas unidades no determinen los grupos.'),
        html.P('Se fijaron tres grupos como exploración de la operación, sin usar CO ni NOx para formarlos. '
            'Después se comparan sus emisiones para describir qué condiciones aparecen juntas. Los grupos '
            'no son categorías de emisiones ni modos de operación confirmados por la planta.'),
        html.Div([metric('Registros',number(len(frame),0)),metric('Grupos de referencia','3'),
            metric('Silhouette',number(meta['silhouette'],3),'desarrollo · muestra de 2.000')],className='metric-strip compact'),
        html.Div([graph('multivariate-kmeans-map',map_figure(frame,'grupo',target)),
            graph('multivariate-kmeans-profiles',group_profiles(frame))],className='chart-grid'),
        html.P('El mapa muestra hasta 2.200 registros en los dos ejes del PCA; los grupos se calcularon '
            'con los nueve sensores, no solo con esos ejes. El perfil compara la mediana de cada grupo '
            'con la media de desarrollo: positivo significa un valor superior a esa referencia.',className='caption'),
        notice(group_interpretation(frame,target)),
        html.P('Silhouette compara la cercanía de cada registro a su grupo con su cercanía a los demás. '
            'Valores próximos a 1 indican grupos separados; cerca de 0, superposición; negativos sugieren '
            f'una asignación poco clara. El valor de {number(meta["silhouette"],3)} indica separación parcial: existen perfiles '
            'distintos, pero también condiciones compartidas. No es una medida de precisión predictiva.'),
        disclosure('Medianas y tamaños de los grupos',html.Div(group_table(frame),id='multivariate-groups')),
        disclosure('Método de KMeans',html.P('El escalado y los centros se estiman solo con desarrollo. '
            'Se prueban diez inicializaciones con semilla 42 y se conserva la de menor suma de distancias '
            'cuadradas a los centros. El número de grupos se fijó en tres; no se realizó una búsqueda del '
            'número óptimo. Los otros registros se asignan a los centros ya estimados.'))
    ],id='multivariate-kmeans-panel')


def isolation_distribution(frame):
    if frame.empty:
        return f.empty()
    values = frame.puntaje_anomalia.to_numpy()
    edges = np.unique(np.r_[np.histogram_bin_edges(values,bins=45),0.])
    counts,_ = np.histogram(values,bins=edges)
    centers = (edges[1:]+edges[:-1])/2
    fig = go.Figure(go.Bar(x=centers,y=counts,width=np.diff(edges)*.95,
        marker_color=[f.CORAL if value<0 else f.TEAL for value in centers],
        hovertemplate='Puntaje: %{x:.3f}<br>Registros: %{y}<extra></extra>'))
    fig.add_vline(x=0,line_dash='dash',line_color=f.GRAY)
    f.style(fig,'Rareza de las configuraciones','Puntaje de Isolation Forest','Registros')
    fig.update_layout(height=380,margin={'l':53,'r':24,'t':70,'b':65})
    return fig


def isolation_panel(frame,target):
    count = int(frame.inusual.sum())
    percentage = 100*count/len(frame) if len(frame) else None
    return html.Section([
        html.H4('Isolation Forest · configuraciones inusuales'),
        html.P('Isolation Forest busca registros fáciles de separar de los demás. Construye árboles '
            'que dividen aleatoriamente los sensores: una combinación poco habitual suele quedar aislada '
            'con menos divisiones. El análisis considera los nueve sensores en conjunto, no solo valores '
            'extremos de una variable.'),
        html.P('Se utilizan 100 árboles y un umbral que señala aproximadamente el 2 % del desarrollo. '
            'Esa proporción se fijó como criterio exploratorio; no representa una tasa real de errores. '
            'Al filtrar años o incluir otros registros, el porcentaje señalado puede ser diferente.'),
        html.Div([metric('Registros',number(len(frame),0)),metric('Señalados',number(count,0)),
            metric('Proporción',number(percentage,2)+' %' if percentage is not None else '—'),
            metric('Árboles','100')],className='metric-strip compact'),
        html.Div([graph('multivariate-isolation-map',map_figure(frame,'inusual',target)),
            graph('multivariate-isolation-scores',isolation_distribution(frame))],className='chart-grid'),
        html.P('El mapa muestra hasta 2.200 registros en los ejes del PCA, pero la detección utiliza los '
            'nueve sensores. En el histograma, un puntaje menor que cero identifica un registro señalado; '
            'cuanto más negativo, más inusual. El puntaje no es una probabilidad de falla ni una concentración de emisión.',className='caption'),
        notice('Una configuración inusual puede corresponder a una condición operativa poco frecuente '
            'o a un problema de medición; este análisis no distingue por sí solo entre ambas causas. '
            'Se requiere contrastarla con los registros de la planta antes de descartarla. Aquí se conservan '
            'todas las observaciones y no se modifican los modelos de emisiones.'),
        disclosure('Registros señalados por año',html.Div(unusual_table(frame),id='multivariate-unusual')),
        disclosure('Método de Isolation Forest',html.P('Se ajusta solo con los sensores de desarrollo, '
            'con semilla 42. CO y NOx no se utilizan para detectar registros inusuales. El escalado y el '
            'umbral quedan fijos al evaluar la fuente completa; no se vuelve a ajustar el detector con prueba.'))
    ],id='multivariate-isolation-panel')


def layout():
    return html.Section([
        html.H3('Estructura multivariada · condiciones operativas'),html.P(INTRO),
        html.P('El ajuste de referencia utiliza solo desarrollo. Los filtros cambian los registros '
            'visualizados, sin recalcular el análisis.',className='caption'),
        dcc.Tabs(id='multivariate-method',value='pca',children=[
            dcc.Tab(label='PCA',value='pca'),dcc.Tab(label='KMeans',value='kmeans'),
            dcc.Tab(label='Isolation Forest',value='isolation')],
            className='local-tabs multivariate-tabs',parent_className='local-tabs-parent',mobile_breakpoint=0),
        html.Div([html.Label('Color del mapa',htmlFor='multivariate-color'),
            dcc.Dropdown(id='multivariate-color',value='anio',clearable=False,options=[
                {'label':'Año','value':'anio'},{'label':'Emisión seleccionada','value':'emision'}])],
            id='multivariate-color-controls',className='multivariate-color'),
        dcc.Loading(html.Div(id='multivariate-content'),type='circle',color=f.TEAL)
    ])


def group_table(frame):
    if frame.empty:
        return html.P('Sin registros con estos filtros.')
    groups = frame.groupby('grupo').agg(Registros=('grupo','size'),TEY=('TEY','median'),TIT=('TIT','median'),
        CDP=('CDP','median'),CO=('CO','median'),NOX=('NOX','median')).reset_index()
    return [table(groups,{'grupo':'Grupo','TEY':'TEY · MWh','TIT':'TIT · °C','CDP':'CDP · mbar',
            'CO':'CO · mg/m³','NOX':'NOx · mg/m³'},digits=3),
        html.P('Las medianas y cantidades corresponden a los registros filtrados; los grupos se definen '
            'con desarrollo completo.',className='caption')]


def unusual_table(frame):
    if frame.empty:
        return html.P('Sin registros con estos filtros.')
    unusual = frame.groupby('anio').agg(Registros=('inusual','size'),Inusuales=('inusual','sum')).reset_index()
    unusual['Porcentaje'] = 100*unusual.Inusuales/unusual.Registros
    return [table(unusual,{'anio':'Año','Porcentaje':'Inusuales (%)'},digits=2),
        html.P(f'{number(int(frame.inusual.sum()),0)} de {number(len(frame),0)} registros filtrados '
            'se señalan como inusuales; esto no demuestra que sean fallas o errores de medición.',className='caption')]


def render(years,population,color,target):
    frame = selected(years,population)
    variance = evidence()[2].varianza
    summary = [metric('Registros',number(len(frame),0)),
        metric('PC1',number(100*variance.iloc[0],2)+' %','variación de referencia'),
        metric('PC2',number(100*variance.iloc[1],2)+' %','variación de referencia'),
        metric('PC1 + PC2',number(100*variance.iloc[:2].sum(),2)+' %','nueve sensores')]
    return summary,map_figure(frame,color,target),group_table(frame),unusual_table(frame)


def render_content(years,population,color,target,method):
    if method=='kmeans':
        return kmeans_panel(selected(years,population),target),{'display':'none'}
    if method=='isolation':
        return isolation_panel(selected(years,population),target),{'display':'none'}
    summary,figure,_,_ = render(years,population,color,target)
    return pca_panel(summary,figure),{}


def register(app):
    app.callback(Output('multivariate-content','children'),Output('multivariate-color-controls','style'),
        Input('eda-years','value'),Input('eda-population','value'),
        Input('multivariate-color','value'),Input('eda-target','value'),Input('multivariate-method','value'))(render_content)
