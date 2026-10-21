"""Package completed sensor-distance diagnostics; never train or change models."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from tools.export_explainability import protected_files

SOURCE = ROOT.parent/'Pruebas provisionales CO NOx/similitud_particiones_20261021'
OUT = ROOT/'data/partition_similarity'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = protected_files()
    meta = json.loads((SOURCE/'metodologia.json').read_text(encoding='utf-8'))
    assert meta['estado'] == 'Completado y verificado' and not meta['objetivos_usados']
    results = pd.read_csv(SOURCE/'comparacion_distancias.csv')
    queries = pd.read_csv(SOURCE/'distancias_42_original.csv.gz')
    assert len(results) == 30 and len(queries) == 1540
    assert results.loc[results.semilla.eq(42)&results.longitud_segmento.eq(168)].shape[0] == 2
    OUT.mkdir(exist_ok=True)
    results.to_csv(OUT/'resumen.csv', index=False)
    queries[['anio', 'orden_registro', 'd_aleatoria', 'd_bloques']].to_csv(OUT/'distancias.csv.gz', index=False)
    fields = ['sensores', 'objetivos_usados', 'modelos_entrenados', 'semillas', 'escala', 'distancia',
        'evaluacion', 'control', 'diseño', 'bootstrap', 'longitud_principal', 'sensibilidad',
        'semilla_bootstrap', 'multiplicidad', 'alcance', 'codigo']
    evidence = {key:meta[key] for key in fields}
    evidence['fecha'] = '2026-10-21'
    evidence['procedencia'] = {name:sha(SOURCE/name) for name in
        ['comparacion_distancias.csv', 'distancias_42_original.csv.gz', 'metodologia.json']}
    evidence['particiones_originales'] = {key:meta['protegidos'][key] for key in ['data', 'random', 'blocked']}
    evidence['archivos'] = {name:sha(OUT/name) for name in ['resumen.csv', 'distancias.csv.gz']}
    (OUT/'metodologia.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    assert before == protected_files(), 'Modelos, resultados o particiones cambiaron.'
    print('Evidencia compacta exportada; modelos, resultados y particiones intactos.')


if __name__ == '__main__':
    main()
