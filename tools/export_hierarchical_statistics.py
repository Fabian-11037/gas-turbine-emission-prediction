"""Hierarchical comparison from existing OOF predictions; no estimator fitting."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.export_explainability import protected_files
from tools.export_kernel_significance import holm

OUT = ROOT/'data/statistical_hierarchy'
OFFICIAL = ROOT/'book/informe_resultados/entregable2_anidada'
KERNEL = ROOT/'book/informe_resultados/kernel_operativo'
KEYS = ['anio','orden_registro','bloque']
SEED, REPS = 42, 5000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DiscardIndices(list):
    """arch stores bootstrap indices only for its tests; avoid a >1 GB audit list."""
    def append(self,value):
        pass


class StationaryOriginalBlocks:
    """Stationary bootstrap over original blocks, stratified by year.

    Circular continuation is restricted to uninterrupted runs of the same outer
    fold. A restart can select any block of that year. Partial blocks retain
    their rows, so replicate row counts can differ; the number of blocks is fixed.
    """
    def __init__(self,frame,mean_blocks=1.,seed=SEED):
        self.frame = frame.reset_index(drop=True)
        self.mean_blocks,self.seed = float(mean_blocks),seed
        self.blocks = []
        self.by_year = {}
        self.successor = []
        for year,group in self.frame.groupby('anio',sort=True):
            indices = []
            runs = []
            previous = None
            run = []
            for _,block in group.groupby('bloque',sort=False):
                positions = block.index.to_numpy()
                if block.fold_validacion.nunique()!=1:
                    raise ValueError('An original block belongs to more than one fold')
                if not np.all(np.diff(block.orden_registro)==1):
                    raise ValueError('A retained block contains an internal gap')
                first,last = block.orden_registro.iloc[[0,-1]]
                fold = int(block.fold_validacion.iloc[0])
                if previous is not None and (first!=previous[0]+1 or fold!=previous[1]):
                    runs.append(run)
                    run = []
                j = len(self.blocks)
                self.blocks.append(positions)
                self.successor.append(j)
                indices.append(j)
                run.append(j)
                previous = (last,fold)
            if run:
                runs.append(run)
            for chain in runs:
                for a,b in zip(chain,chain[1:]+chain[:1]):
                    self.successor[a] = b
            self.by_year[int(year)] = np.asarray(indices)

    def bootstrap(self,reps):
        rng = np.random.default_rng(self.seed)
        probability = 1/self.mean_blocks
        for _ in range(reps):
            chosen = []
            for indices in self.by_year.values():
                starts = rng.choice(indices,size=len(indices))
                restart = rng.random(len(indices))<probability
                current = int(starts[0])
                for j in range(len(indices)):
                    current = int(starts[j]) if j==0 or restart[j] else self.successor[current]
                    chosen.append(self.blocks[current])
            yield ((np.concatenate(chosen),),{})


def load_losses(cache_root):
    data = pd.read_csv(ROOT/'data/observaciones.csv.gz')
    assignment = pd.read_csv(OFFICIAL/'asignacion_particiones.csv.gz')
    dev = data.loc[data.particion.eq('entrenamiento_CV')].merge(
        assignment[KEYS+['fold_validacion']],on=KEYS,validate='one_to_one').reset_index(drop=True)
    canonical = pd.read_csv(OFFICIAL/'predicciones_OOF.csv.gz')
    pd.testing.assert_frame_equal(dev[KEYS],canonical[KEYS])
    published = pd.read_csv(ROOT/'data/resultados.csv')
    configuration = json.loads((OFFICIAL/'configuracion.json').read_text())
    source_configuration = cache_root.parent/'configuracion.json'
    assert json.loads(source_configuration.read_text())==configuration
    sources = {str(source_configuration.name):sha(source_configuration)}
    families = ['KNN','Ridge','Lasso','Decision Tree','Random Forest','XGBoost','SVR lineal']
    metadata,frames = [],{}
    for target in ['CO','NOX']:
        frame = dev[KEYS+['fold_validacion','TEY',target]].copy()
        y = dev[target].to_numpy()
        for family in families:
            path = cache_root/f'{target}_{family.replace(" ","_")}.joblib'
            cached = joblib.load(path)
            sources[path.name] = sha(path)
            for method,prediction in cached['predicciones'].items():
                label = family+' · '+method
                p = np.asarray(prediction['OOF'],dtype=float)
                assert len(p)==len(dev) and np.isfinite(p).all()
                loss = (y-p)**2
                summary = cached['resumen'].loc[cached['resumen'].optimizador.eq(method)].iloc[0]
                np.testing.assert_allclose(1-loss.mean()/np.var(y),summary.R2_OOF,rtol=1e-9)
                for row in cached['folds'].loc[cached['folds'].optimizador.eq(method)].itertuples():
                    actual = loss[dev.fold_validacion.eq(row.fold)].mean()
                    np.testing.assert_allclose(actual,row.RMSE**2,rtol=1e-9)
                selected = published.loc[published.objetivo.eq(target)&published.modelo.eq(family)].iloc[0]
                current = method==selected.optimizador
                if current:
                    np.testing.assert_allclose(p,canonical[target+'_'+family.replace(' ','_')],rtol=1e-9,atol=1e-9)
                frame[label] = loss
                metadata.append(dict(objetivo=target,procedimiento=label,modelo=family,
                    optimizador=method,publicado=current,MSE_OOF=loss.mean(),R2_OOF=1-loss.mean()/np.var(y)))
            del cached
        saved = pd.read_csv(KERNEL/f'predicciones_OOF_{target}.csv.gz')
        aligned = dev[KEYS+[target]].merge(saved[KEYS+['prediccion']],on=KEYS,validate='one_to_one')
        assert len(aligned)==len(dev)
        loss = (aligned[target].to_numpy()-aligned.prediccion.to_numpy())**2
        label = 'Kernel operativo · Grid kernel'
        frame[label] = loss
        row = published.loc[published.objetivo.eq(target)&published.modelo.eq('Kernel operativo')].iloc[0]
        np.testing.assert_allclose(loss.mean(),row.MSE_OOF,rtol=1e-9)
        metadata.append(dict(objetivo=target,procedimiento=label,modelo='Kernel operativo',
            optimizador='Grid kernel',publicado=True,MSE_OOF=loss.mean(),R2_OOF=1-loss.mean()/np.var(y)))
        frames[target] = frame
    return frames,pd.DataFrame(metadata),sources


def unique_losses(frame,labels):
    groups = []
    for label in labels:
        for representative,aliases in groups:
            if np.allclose(frame[label],frame[representative],rtol=0,atol=1e-12):
                aliases.append(label)
                break
        else:
            groups.append((label,[label]))
    return groups


def confidence_set(frame,labels,mean_blocks,reps=REPS):
    from arch.bootstrap import MCS
    groups = unique_losses(frame,labels)
    names = [name for name,_ in groups]
    if len(names)==1:
        return pd.DataFrame([dict(procedimiento=alias,representante=names[0],p_MCS=1.,
            incluido_005=True,incluido_010=True,bloques_medios=mean_blocks) for alias in groups[0][1]])
    mcs = MCS(frame[names],size=.05,reps=reps,method='R',block_size=168,seed=SEED)
    mcs.bootstrap = StationaryOriginalBlocks(frame,mean_blocks)
    # This private diagnostic list is not used by the arch R statistic.
    mcs._bootstrap_indices = DiscardIndices()
    mcs.compute()
    rows = []
    for representative,aliases in groups:
        p = float(mcs.pvalues.loc[representative].iloc[0])
        for alias in aliases:
            rows.append(dict(procedimiento=alias,representante=representative,p_MCS=p,
                             incluido_005=p>.05,incluido_010=p>.10,bloques_medios=mean_blocks))
    return pd.DataFrame(rows)


def directed(frame,reference,kernel,mean_blocks=1.,reps=REPS):
    d = frame[reference].to_numpy()-frame[kernel].to_numpy()
    z = (frame.TEY.to_numpy()-frame.TEY.mean())/frame.TEY.std(ddof=0)
    moments = np.column_stack([d,d*z])
    y = frame[['CO' if 'CO' in frame else 'NOX']].to_numpy().ravel()
    boot = np.empty((reps,4))
    resampler = StationaryOriginalBlocks(frame,mean_blocks)
    for j,(args,_) in enumerate(resampler.bootstrap(reps)):
        idx = args[0]
        boot[j,:2] = moments[idx].mean(axis=0)
        boot[j,2:] = np.mean(y[idx]),np.mean(y[idx]**2)
    g = moments.mean(axis=0)
    covariance = np.cov(boot[:,:2],rowvar=False)
    rank = np.linalg.matrix_rank(covariance)
    wald = float(g@np.linalg.pinv(covariance)@g)
    se = float(boot[:,0].std(ddof=1))
    n_blocks = len(resampler.blocks)
    # h=1 is an index-level convention for block scores, not a forecast horizon.
    hln = np.sqrt((n_blocks-1)/n_blocks)
    dm = g[0]/se*hln if se else (0. if g[0]==0 else np.sign(g[0])*np.inf)
    delta_r2 = d.mean()/np.var(y)
    intervals = boot[:,0]/(boot[:,3]-boot[:,2]**2)
    lower,upper = np.quantile(intervals,[.025,.975])
    block_d = np.array([d[idx].mean() for idx in resampler.blocks])
    deviation = block_d.std(ddof=1)
    dz = block_d.mean()/deviation if deviation else (0. if block_d.mean()==0 else np.sign(block_d.mean())*np.inf)
    return dict(referencia=reference,n=len(frame),n_bloques=n_blocks,
        delta_MSE=g[0],delta_R2=delta_r2,IC95_inf=lower,IC95_sup=upper,
        Wald_condicional=wald,gl_condicional=rank,p_GW=stats.chi2.sf(wald,rank) if rank else (1. if np.allclose(g,0) else 0.),
        momento_carga=g[1],DM_HLN=dm,p_DM=2*stats.t.sf(abs(dm),n_blocks-1),
        Cohen_dz_bloques=dz,bloques_medios=mean_blocks,
        RMSE_referencia=np.sqrt(frame[reference].mean()),RMSE_kernel=np.sqrt(frame[kernel].mean()))


def auxiliary_ranks(frame,labels):
    means = frame.groupby('fold_validacion')[labels].mean()
    statistic,p = stats.friedmanchisquare(*[means[col] for col in labels])
    ranks = means.rank(axis=1,method='average').mean()
    k,n = len(labels),len(means)
    cd = stats.studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n))
    return pd.DataFrame({'procedimiento':labels,'rango_medio':ranks[labels].to_numpy()}),dict(
        Friedman=float(statistic),p_Friedman=float(p),n_folds=n,n_modelos=k,CD=float(cd),
        posthoc_aplicado=bool(p<.05),alcance='Auxiliar: cinco pliegues con entrenamiento compartido')


def run(cache_root,reps=REPS):
    OUT.mkdir(exist_ok=True)
    original = protected_files()
    saved = [OUT/f'perdidas_OOF_{target}.csv.gz' for target in ['CO','NOX']]
    if cache_root is None and all(path.exists() for path in saved):
        previous = json.loads((OUT/'metodologia.json').read_text(encoding='utf-8'))
        if previous['huellas_protegidas']!=original:
            raise ValueError('Frozen models or validation differ; export fresh OOF evidence with --cache-root')
        for path in saved:
            if previous.get('perdidas_sha256') and sha(path)!=previous['perdidas_sha256'][path.name]:
                raise ValueError('A saved OOF loss matrix changed')
        frames = {target:pd.read_csv(path) for target,path in zip(['CO','NOX'],saved)}
        candidates = pd.read_csv(OUT/'procedimientos.csv')
        sources = previous['fuentes_caches_sha256']
    else:
        cache_root = cache_root or ROOT.parent/'Gas Turbine Project/informe_resultados/entregable2_anidada/corridas'
        frames,candidates,sources = load_losses(cache_root)
    candidates.to_csv(OUT/'procedimientos.csv',index=False)
    mcs_rows,directed_rows,ranks,aux = [],[],[],[]
    for target,frame in frames.items():
        frame.to_csv(OUT/f'perdidas_OOF_{target}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        group = candidates.loc[candidates.objetivo.eq(target)]
        labels = group.procedimiento.tolist()
        kernel = 'Kernel operativo · Grid kernel'
        # Preserve the already published CV comparator, not a new selection by OOF or test.
        published = pd.read_csv(ROOT/'data/resultados.csv')
        ref = published.loc[published.objetivo.eq(target)&published.modelo.ne('Kernel operativo')].sort_values('MSE_CV_externo').iloc[0]
        reference = ref.modelo+' · '+ref.optimizador
        for length in [1.,2.,4.]:
            mcs = confidence_set(frame,labels,length,reps)
            mcs['objetivo'] = target
            mcs_rows.append(mcs)
            record = directed(frame,reference,kernel,length,reps)
            record['objetivo'] = target
            directed_rows.append(record)
            print(f'{target}: MCS y contrastes, longitud {length:g}; {int(mcs.incluido_005.sum())}/29 retenidos',flush=True)
        ranking,info = auxiliary_ranks(frame,group.loc[group.publicado].procedimiento.tolist())
        ranking['objetivo'] = target
        ranks.append(ranking)
        aux.append({'objetivo':target,**info})
        gc.collect()
    pd.concat(mcs_rows).to_csv(OUT/'mcs.csv',index=False)
    directed_frame = pd.DataFrame(directed_rows)
    for _,group in directed_frame.groupby('bloques_medios'):
        for name in ['GW','DM']:
            directed_frame.loc[group.index,'p_'+name+'_Holm2'] = holm(group['p_'+name])
    directed_frame.to_csv(OUT/'dirigidas.csv',index=False)
    pd.concat(ranks).to_csv(OUT/'rangos.csv',index=False)
    assert protected_files()==original,'A protected artifact changed'
    metadata = dict(version=1,semilla=SEED,replicas=reps,modelos_reentrenados=0,
        validacion_modificada=False,n_OOF=27570,procedimientos_por_emision=29,
        alfa=.05,sensibilidad_alfa=.1,longitudes_bloques=[1,2,4],
        MCS='arch 8.0.0, método R; bootstrap estacionario a nivel de bloques originales',
        remuestreo='Número de bloques por año fijo; continuidad circular solo en tramos sin huecos del mismo pliegue. Los bloques parciales conservan su tamaño.',
        duplicados='Vectores de pérdidas iguales con tolerancia absoluta 1e-12 comparten contraste; se conservan todos los nombres.',
        condicional='Wald de momentos [d, d·carga estandarizada], inspirado en Giacomini–White; covarianza por remuestreo de bloques.',
        DM='Adaptación de DM con corrección HLN h=1 y grados de libertad por número de bloques; no es una prueba de pronóstico temporal.',
        Clark_West='No aplicable: el kernel y RF/XGBoost no son modelos paramétricos anidados.',
        multiplicidad='Holm conjunto para CO y NOx, por tipo de contraste y longitud.',
        alcance='Inferencia exploratoria condicional a predicciones OOF guardadas; no elimina selección adaptativa ni entrenamiento compartido.',
        fuentes_caches_sha256=sources,huellas_protegidas=original,
        perdidas_sha256={path.name:sha(path) for path in saved},
        codigo_sha256=sha(__file__),auxiliar=aux,
        fuentes=['https://bashtage.github.io/arch/multiple-comparison/multiple-comparisons.html',
            'https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0262.2006.00718.x',
            'https://doi.org/10.1016/S0169-2070(96)00719-4'])
    (OUT/'metodologia.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,help='Only required for the initial extraction; later runs use portable saved losses')
    parser.add_argument('--replicas',type=int,default=REPS)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.cache_root,args.replicas)
