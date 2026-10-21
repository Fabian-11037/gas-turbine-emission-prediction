"""Export final experiment summaries; never load or fit estimators."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    'resultados_20261017': 'Búsqueda inicial',
    'resultados_adaboost_20261018': 'AdaBoost',
    'resultados_adaptativos_20261018': 'Adaptación y expertos',
    'resultados_centros_lssvr_20261019': 'Selección de centros LS-SVR',
    'resultados_CO_cadena_tweedie_20261020': 'Cadena y función Tweedie',
    'resultados_CO_expertos_20261020': 'Expertos para CO',
    'resultados_cuatro_mejoras_20261019': 'Ponderación y multiescala',
    'resultados_et_svr_20261018': 'Ajuste ET–SVR',
    'resultados_kernel_regimen_20261019': 'Kernel y régimen operativo',
    'resultados_lgbm_ebm_mars_20261019': 'LightGBM, EBM y MARS',
    'resultados_limpieza_dashboard_20261019': 'Auditoría de atípicos',
    'resultados_mkl_arbol_20261018': 'MKL y árbol de modelos lineales',
    'resultados_restringidos_20261017': 'Complejidad restringida',
    'resultados_robust_bart_20261019': 'Escalado robusto',
    'resultados_sparse_lssvr_20261018': 'Sparse LS-SVR inicial',
    'resultados_stacking_20261018': 'Stacking inicial',
    'resultados_stacking_grupos_20261019': 'Stacking por grupos',
    'resultados_stacking_lgbm_20261019': 'Stacking con LightGBM',
    'resultados_xgboost_intermedio_20261017': 'Complejidad intermedia',
}
ALIASES = {
    'ET SVR kernel operativo': 'Kernel operativo',
    'Stack ET XGB SVR referencia': 'Stack ET XGB SVR',
}
METRICS = ['R2_train', 'R2_CV_promedio', 'R2_test']


def consolidate(source_root):
    sources = [(ROOT/'data/resultados.csv', 'Modelos del dashboard', True)]
    sources += [(source_root/name/'resultados_provisionales.csv', label, False)
                for name, label in EXPERIMENTS.items()]
    records, evidence, protocols = {}, [], []
    for path, context, published in sources:
        frame = pd.read_csv(path)
        source = 'data/resultados.csv' if published else path.relative_to(source_root).as_posix()
        evidence.append({'archivo': source, 'filas': len(frame),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        config_path = (ROOT/'book/informe_resultados/entregable2_anidada/configuracion.json'
                       if published else path.parent/'configuracion.json')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        protocol = (f"Bloques {config.get('bloque', '?')}; purga {config.get('purga', '?')}; "
                    f"CV {config.get('folds_externos', '?')} externa × {config.get('folds_internos', '?')} interna")
        protocols.append(dict(archivo=source, protocolo=protocol,
            configuracion_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            tipo_CV='Media del R² de cinco pliegues externos; semilla de partición 42',
            configuracion=config))
        frame['modelo'] = frame.modelo.replace(ALIASES)
        frame['politica'] = 'Sin limpieza'
        if 'limpieza' in frame:
            frame.loc[frame.limpieza.astype(str).str.lower().eq('true'), 'politica'] = 'Limpieza de entrenamiento'
        for (model, policy), group in frame.groupby(['modelo', 'politica'], sort=False):
            if group.objetivo.duplicated().any():
                raise ValueError(f'Duplicate target in {source}: {model}, {policy}')
            values = {}
            for target in ['CO', 'NOX']:
                rows = group.loc[group.objetivo.eq(target)]
                for metric in METRICS:
                    values[f'{metric}_{target}'] = float(rows.iloc[0][metric]) if len(rows) else None
            # Merge repeated controls only when all final, CV and test scores agree.
            key = (model, policy, 'bloques', *(None if v is None or pd.isna(v) else round(v, 12)
                                    for v in values.values()))
            if key in records:
                records[key]['fuentes'].append(source)
                continue
            variant = context if policy == 'Sin limpieza' else context+' · '+policy.lower()
            records[key] = {'modelo': model, 'variante': variant, **values,
                'estado': 'Dashboard' if published else 'Provisional',
                'politica': policy, 'fuentes': [source], 'protocolo': protocol,
                'tipo_CV': 'Media de 5 pliegues externos', 'grupo_validacion': 'bloques'}
            for target in ['CO', 'NOX']:
                rows = group.loc[group.objetivo.eq(target)]
                records[key][f'R2_OOF_{target}'] = (rows.iloc[0].get('R2_OOF', None) if len(rows) else None)
    recent_path = ROOT/'data/historial_adicional.csv'
    recent_metadata = {}
    if recent_path.exists():
        recent = pd.read_csv(recent_path)
        recent_metadata = json.loads((ROOT/'data/historial_adicional.json').read_text(encoding='utf-8'))
        digest = hashlib.sha256(recent_path.read_bytes()).hexdigest()
        assert digest == recent_metadata['archivo_sha256']
        evidence.append(dict(archivo='data/historial_adicional.csv', filas=len(recent), sha256=digest))
        for _, record in recent[['grupo_validacion','protocolo','tipo_CV']].drop_duplicates().iterrows():
            protocols.append(dict(archivo='data/historial_adicional.csv', **record.to_dict()))
        for (model, variant, protocol_group, status), group in recent.groupby(
                ['modelo','variante','grupo_validacion','estado'], sort=False):
            assert not group.objetivo.duplicated().any()
            values = {}
            for target in ['CO','NOX']:
                selected = group.loc[group.objetivo.eq(target)]
                for metric in METRICS:
                    values[f'{metric}_{target}'] = float(selected.iloc[0][metric]) if len(selected) else None
            key = (model, 'Sin limpieza', protocol_group, *(None if v is None or pd.isna(v) else round(v,12)
                                                          for v in values.values()))
            sources_recent = group.fuentes.drop_duplicates().tolist()
            if key in records:
                records[key]['fuentes'].extend(sources_recent)
                continue
            records[key] = dict(modelo=model, variante=variant, **values, estado=status,
                politica='Sin limpieza', fuentes=sources_recent, protocolo=group.protocolo.iloc[0],
                tipo_CV=group.tipo_CV.iloc[0], grupo_validacion=protocol_group)
            for target in ['CO','NOX']:
                selected = group.loc[group.objetivo.eq(target)]
                records[key][f'R2_OOF_{target}'] = selected.iloc[0].R2_OOF if len(selected) else None
    rows = list(records.values())
    for row in rows:
        row['fuentes'] = ' | '.join(row['fuentes'])
    rows.append({'modelo': 'BART', 'variante': 'Muestreo bayesiano interrumpido',
                 'estado': 'Interrumpido', 'politica': 'Sin limpieza',
                 'grupo_validacion':'bloques', 'protocolo':'Bloques 168; purga 24; sin evaluación final',
                 'tipo_CV':'Pendiente',
                 'fuentes': 'resultados_robust_bart_20261019 (sin resumen final BART)'})
    output = pd.DataFrame(rows).sort_values(['modelo', 'variante'], kind='stable')
    metadata = {'fuentes': evidence, 'protocolos': protocols, 'n_variantes': len(output),
        'evidencia_reciente':recent_metadata,
        'criterio': 'Resultados finales por variante; controles idénticos consolidados. '
                    'R2_train se evalúa sobre el desarrollo original, incluso en la auditoría de limpieza. '
                    'Los protocolos distintos no se consolidan entre sí. '
                    'Los casos pausados o interrumpidos no reciben métricas parciales como resultado final.'}
    return output, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    output, metadata = consolidate(args.source_root)
    output.to_csv(ROOT/'data/historial_modelos.csv', index=False)
    (ROOT/'data/historial_modelos.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'{len(output)} variantes; {len(metadata["fuentes"])} archivos de evidencia.')
