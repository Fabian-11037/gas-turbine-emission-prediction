"""Package notebook-equivalent multivariate EDA; prediction artifacts stay frozen."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dashboard.repository import observations,sensors
from tools.export_explainability import protected_files

OUT = ROOT/'data/multivariate'


def export():
    start = perf_counter()
    before = protected_files()
    data = observations().sort_values('orden_registro').reset_index(drop=True)
    names = [s['id'] for s in sensors()]
    train = data.particion.eq('entrenamiento_CV').to_numpy()
    assert train.sum()==27570 and len(data)==36733
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(data.loc[train,names])
        z = scaler.transform(data[names])
        pca = PCA().fit(z[train])
        projected = pca.transform(z)
        groups = KMeans(n_clusters=3,n_init=10,random_state=42).fit(z[train])
        isolation = IsolationForest(n_estimators=100,contamination=.02,random_state=42,n_jobs=1).fit(z[train])
        labels = groups.predict(z)
        unusual = isolation.predict(z)==-1
        silhouette = silhouette_score(z[train],labels[train],sample_size=2000,random_state=42)
    correlation = np.corrcoef(z[train].T,projected[train,:2].T)[:len(names),len(names):]
    loadings = pd.DataFrame({'sensor':names,'PC1':correlation[:,0],'PC2':correlation[:,1],
                            'coef_PC1':pca.components_[0],'coef_PC2':pca.components_[1]})
    variance = pd.DataFrame({'componente':np.arange(1,len(names)+1),
        'varianza':pca.explained_variance_ratio_, 'acumulada':np.cumsum(pca.explained_variance_ratio_)})
    scores = data[['orden_registro','anio','particion']].copy()
    scores['PC1'],scores['PC2'] = projected[:,0],projected[:,1]
    scores['grupo'] = labels+1
    scores['inusual'] = unusual
    scores['puntaje_anomalia'] = isolation.decision_function(z)
    OUT.mkdir(exist_ok=True)
    scores.to_csv(OUT/'proyecciones.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    loadings.to_csv(OUT/'cargas.csv',index=False)
    variance.to_csv(OUT/'varianza.csv',index=False)
    metadata = dict(n_desarrollo=int(train.sum()),n_fuente=len(data),sensores=names,
        ajuste_solo_desarrollo=True,objetivos_usados_en_ajuste=False,semilla=42,
        kmeans=dict(n_clusters=3,n_init=10,silhouette=float(silhouette),muestra_silhouette=2000),
        isolation_forest=dict(n_estimators=100,contamination=.02,n_inusuales_desarrollo=int(unusual[train].sum())),
        referencia='Sección 3.5 del notebook: StandardScaler, PCA, KMeans e Isolation Forest sobre desarrollo.',
        proyeccion_fuente='Los otros registros se transforman con el ajuste de desarrollo; no intervienen en su estimación.',
        escala=dict(media=scaler.mean_.tolist(),desviacion=scaler.scale_.tolist()),
        versiones={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn']},
        segundos=perf_counter()-start,huellas_protegidas=before,
        archivos={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                  for name in ['proyecciones.csv.gz','cargas.csv','varianza.csv']})
    assert before==protected_files(),'Changed prediction data, artifacts or partitions.'
    (OUT/'metodologia.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(n_desarrollo=int(train.sum()),varianza_PC1_PC2=float(variance.varianza.iloc[:2].sum()),
                         silhouette=float(silhouette),inusuales=int(unusual[train].sum())),ensure_ascii=False))


if __name__=='__main__':
    export()
