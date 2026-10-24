"""Diagnosticos temporales descriptivos sin crear fechas ni modificar modelos."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dashboard.sensor_labels import sensor_label
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def medias_bootstrap_bloques(values, largo=168, replicas=2000, semilla=42):
    """Moving-block bootstrap, incluyendo exactamente n registros por replica."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2 or not np.isfinite(values).all():
        raise ValueError('Se requieren al menos dos valores finitos y ordenados.')
    largo = min(largo, n)
    rng = np.random.default_rng(semilla)
    completos, resto = divmod(n, largo)
    acumulados = np.r_[0.0, np.cumsum(values)]
    inicio = rng.integers(0, n-largo+1, size=(replicas, completos))
    suma = (acumulados[inicio+largo]-acumulados[inicio]).sum(axis=1)
    if resto:
        inicio_resto = rng.integers(0, n-resto+1, size=replicas)
        suma += acumulados[inicio_resto+resto]-acumulados[inicio_resto]
    return suma/n


def diagnostico_distribuciones(frame, entrenamiento, variables, objetivos, salida, etiqueta):
    frame = frame.sort_values('orden_registro').copy()
    years = sorted(frame.anio.unique())
    medias_ref = entrenamiento[variables].mean()
    sd_ref = entrenamiento[variables].std(ddof=1)
    filas = []
    for year, group in frame.groupby('anio', sort=True):
        for var in variables:
            y = group[var]
            filas.append({'anio':year, 'variable':var, 'n':len(y), 'media':y.mean(),
                          'mediana':y.median(), 'std':y.std(ddof=1),
                          'p10':y.quantile(.1), 'p90':y.quantile(.9),
                          'desplazamiento_media_SD_train':(y.mean()-medias_ref[var])/sd_ref[var]})
    anual = pd.DataFrame(filas)
    cambios = []
    sensibilidad = []
    for antes, despues in zip(years[:-1], years[1:]):
        for j, var in enumerate(variables):
            y0 = frame.loc[frame.anio==antes, var].to_numpy()
            y1 = frame.loc[frame.anio==despues, var].to_numpy()
            delta = y1.mean()-y0.mean()
            fila = {'variable':var, 'desde':antes, 'hasta':despues,
                    'delta_media':delta, 'delta_mediana':np.median(y1)-np.median(y0),
                    'delta_media_SD_train':delta/sd_ref[var],
                    'Wasserstein_SD_train':stats.wasserstein_distance(y0,y1)/sd_ref[var],
                    'KS_D_descriptivo':stats.ks_2samp(y0,y1,method='asymp').statistic}
            # Dos bootstrap independientes por periodo, sin mezclar fronteras anuales.
            for largo in [48,168,336]:
                b0 = medias_bootstrap_bloques(y0,largo,semilla=42+j)
                b1 = medias_bootstrap_bloques(y1,largo,semilla=142+j)
                lo, hi = np.quantile(b1-b0,[.025,.975])
                sensibilidad.append({'variable':var, 'desde':antes, 'hasta':despues,
                                     'bloque_registros':largo,'delta_media':delta,
                                     'IC95_inf':lo,'IC95_sup':hi,
                                     'IC_excluye_cero':bool(lo>0 or hi<0)})
                if largo==168:
                    fila.update(IC95_inf=lo,IC95_sup=hi)
            cambios.append(fila)
    cambios = pd.DataFrame(cambios)
    sensibilidad = pd.DataFrame(sensibilidad)
    bloques, tendencias = [], []
    for year, group in frame.groupby('anio',sort=True):
        group = group.copy()
        group['bloque'] = np.arange(len(group))//168
        for target in objetivos:
            agregado = group.groupby('bloque').agg(orden=('orden_registro','mean'),
                                                   n=(target,'size'),media=(target,'mean'),
                                                   p10=(target,lambda s:s.quantile(.1)),
                                                   p90=(target,lambda s:s.quantile(.9)))
            agregado = agregado.assign(anio=year,objetivo=target)
            bloques.append(agregado.reset_index())
            trend = agregado.loc[agregado.n>=84]
            slope = stats.theilslopes(trend.media,trend.orden).slope
            tau = stats.kendalltau(trend.orden,trend.media).statistic
            tendencias.append({'anio':year,'objetivo':target,'bloques_utilizados':len(trend),
                               'pendiente_media_por_1000_registros':1000*slope,
                               'Kendall_tau_descriptivo':tau})
    bloques = pd.concat(bloques,ignore_index=True)
    tendencias = pd.DataFrame(tendencias)
    correlaciones = []
    for year, group in frame.groupby('anio',sort=True):
        for target in objetivos:
            for var in [v for v in variables if v not in objetivos]:
                correlaciones.append({'anio':year,'objetivo':target,'sensor':var,
                                      'Spearman':stats.spearmanr(group[var],group[target]).statistic})
    correlaciones = pd.DataFrame(correlaciones)
    for nombre,tabla in [('resumen_anual',anual),('cambios_anuales',cambios),
                         ('sensibilidad_bloques',sensibilidad),('bloques_168',bloques),
                         ('tendencias_intra_anio',tendencias),('asociaciones_anuales',correlaciones)]:
        tabla.to_csv(salida/f'temporal_{etiqueta}_{nombre}.csv',index=False)

    colores = {2011:'#297c79',2012:'#548d3b',2013:'#926534',2014:'#a44b6f',2015:'#4c64b5'}
    fig, axes = plt.subplots(len(objetivos),2,figsize=(14,7),layout='constrained')
    for i,target in enumerate(objetivos):
        for year in years:
            b = bloques[(bloques.objetivo==target)&(bloques.anio==year)]
            axes[i,0].plot(b.orden,b.media,color=colores[year],label=str(year))
            axes[i,0].fill_between(b.orden,b.p10,b.p90,color=colores[year],alpha=.12)
            orden = np.sort(frame.loc[frame.anio==year,target].to_numpy())
            axes[i,1].plot(orden,np.arange(1,len(orden)+1)/len(orden),color=colores[year],label=str(year))
        for year in years[1:]:
            axes[i,0].axvline(frame.loc[frame.anio==year,'orden_registro'].min(),color='grey',ls=':',lw=.8)
        axes[i,0].set(title=f'{target}: media y p10-p90 por bloques de 168',xlabel='Orden original (no fecha)',ylabel='mg/m3')
        axes[i,1].set(title=f'{target}: distribucion acumulada por ano',xlabel='mg/m3',ylabel='Fraccion acumulada')
        axes[i,0].legend(ncol=len(years),fontsize=8)
        axes[i,1].legend(fontsize=8)
    plt.show()
    tabla = anual.pivot(index='variable',columns='anio',values='desplazamiento_media_SD_train').reindex(variables)
    fig, ax = plt.subplots(figsize=(10,6),layout='constrained')
    im = ax.imshow(tabla,vmin=-2,vmax=2,cmap='coolwarm',aspect='auto')
    anios_ref = sorted(entrenamiento.anio.unique())
    ax.set(xticks=range(len(years)),xticklabels=years,yticks=range(len(variables)),yticklabels=[sensor_label(v, width=28) for v in variables],
           title=f'Cambio de medias respecto a {anios_ref[0]}-{anios_ref[-1]} (SD de desarrollo)')
    for i in range(len(variables)):
        for j in range(len(years)): ax.text(j,i,f'{tabla.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,label='Desviaciones estandar; color saturado fuera de [-2,2]')
    plt.show()
    return anual,cambios,sensibilidad,tendencias,correlaciones


def diagnostico_residuos(frame, modelos, predictores, objetivos, salida):
    rows, bloques = [], []
    for target in objetivos:
        for nombre in ['Ridge','SVR lineal']:
            pred = modelos[target,nombre].predict(frame[predictores])
            auxiliar = frame[['anio','orden_registro',target]].copy()
            auxiliar['sesgo'] = pred-frame[target].to_numpy()
            auxiliar['pred'] = pred
            for year, group in auxiliar.groupby('anio',sort=True):
                lo,hi = np.quantile(medias_bootstrap_bloques(group.sesgo),[.025,.975])
                rows.append({'objetivo':target,'modelo':nombre,'anio':year,
                             'particion':'entrenamiento final' if year<=2014 else 'prueba',
                             'R2':r2_score(group[target],group.pred),
                             'RMSE':np.sqrt(mean_squared_error(group[target],group.pred)),
                             'MAE':mean_absolute_error(group[target],group.pred),
                             'sesgo_predicho_menos_real':group.sesgo.mean(),'IC95_inf':lo,'IC95_sup':hi})
                if nombre=='SVR lineal':
                    group = group.copy()
                    group['bloque'] = np.arange(len(group))//168
                    b = group.groupby('bloque').agg(orden=('orden_registro','mean'),sesgo=('sesgo','mean'))
                    bloques.append(b.assign(anio=year,objetivo=target).reset_index())
    tabla = pd.DataFrame(rows)
    tabla.to_csv(salida/'temporal_residuos_modelos_congelados.csv',index=False)
    bloques = pd.concat(bloques,ignore_index=True)
    fig, axes = plt.subplots(2,1,figsize=(13,6),layout='constrained')
    for ax,target in zip(axes,objetivos):
        for year,g in bloques[bloques.objetivo==target].groupby('anio',sort=True):
            ax.plot(g.orden,g.sesgo,label=str(year))
        ax.axhline(0,color='black',ls='--',lw=.8)
        for year in [2012,2013,2014,2015]: ax.axvline(frame.loc[frame.anio==year,'orden_registro'].min(),color='grey',ls=':',lw=.8)
        ax.set(title=f'{target}: sesgo del SVR final por bloques de 168',xlabel='Orden original (no fecha)',ylabel='Predicho - real (mg/m3)')
        ax.legend(ncol=5,fontsize=8)
    plt.show()
    return tabla
