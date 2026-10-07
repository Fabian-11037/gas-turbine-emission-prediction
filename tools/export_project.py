"""Exportacion fuera de linea del proyecto cientifico; no se usa en Render."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def exportar(origen):
    origen = Path(origen).resolve()
    sys.path.insert(0, str(origen/'notebook_dependencies'))
    import joblib
    import numpy as np
    import pandas as pd
    salida = ROOT/'data'
    salida.mkdir(parents=True, exist_ok=True)
    resultados = origen/'informe_resultados'/'validacion_bloques_168_purga_24'
    artefacto = joblib.load(origen/'modelos_finales'/'modelo_final_co_nox_bloques.joblib')
    predictores = artefacto['predictores']
    datos = pd.concat([pd.read_csv(origen/'data'/f'gt_{a}.csv').assign(anio=a)
                       for a in range(2011,2016)],ignore_index=True)
    datos['orden_registro'] = np.arange(len(datos))
    asignacion = pd.read_csv(resultados/'asignacion_particiones.csv')
    assert len(datos) == len(asignacion) == 36733
    for c in ['anio','orden_registro']:
        np.testing.assert_array_equal(datos[c],asignacion[c])
    datos['particion'] = asignacion.particion
    datos['bloque'] = asignacion.bloque
    datos.to_csv(salida/'observaciones.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    desarrollo = datos.loc[datos.particion.eq('entrenamiento_CV')]
    escalado = (desarrollo[predictores]-desarrollo[predictores].median())/desarrollo[predictores].std()
    ejemplo = desarrollo.loc[escalado.pow(2).sum(axis=1).idxmin(),predictores]
    units = {'AT':'°C','AP':'mbar','AH':'%','AFDP':'mbar','GTEP':'mbar',
             'TIT':'°C','TAT':'°C','TEY':'MWh','CDP':'mbar'}
    nombres = {'AT':'Temperatura ambiente','AP':'Presión ambiente','AH':'Humedad relativa',
               'AFDP':'Presión diferencial del filtro','GTEP':'Presión de escape',
               'TIT':'Temperatura de entrada','TAT':'Temperatura de salida',
               'TEY':'Rendimiento energético','CDP':'Presión del compresor'}
    sensores = [{'id':c,'nombre':nombres[c],'unidad':units[c],
                 'min':float(desarrollo[c].min()),'max':float(desarrollo[c].max()),
                 'ejemplo':float(ejemplo[c])} for c in predictores]
    protocolo = json.loads((resultados/'protocolo.json').read_text(encoding='utf-8'))
    fuentes = list((origen/'data').glob('gt_*.csv')) + [resultados/'resultados.csv',
              resultados/'asignacion_particiones.csv']
    modelos = []
    for identificador,nombre in [('ridge','Ridge'),('svr_lineal','SVR lineal'),('media','Media')]:
        respuestas = {}
        errores = {}
        for objetivo in ['CO','NOX']:
            ruta = resultados/f'modelo_{objetivo}_{nombre.replace(" ","_")}.joblib'
            m = joblib.load(ruta)
            if nombre == 'Media':
                coef = np.zeros(len(predictores))
                intercepto = float(m.constant_.ravel()[0])
            else:
                xescala,base = m.regressor_.steps[0][1],m.regressor_.steps[-1][1]
                yescala = m.transformer_
                coef = base.coef_.ravel()*yescala.scale_[0]/xescala.scale_
                intercepto = float(yescala.mean_[0]+yescala.scale_[0]*np.asarray(base.intercept_).ravel()[0]
                                   -np.dot(coef,xescala.mean_))
            pred = datos[predictores].to_numpy() @ coef + intercepto
            np.testing.assert_allclose(pred,m.predict(datos[predictores]),rtol=1e-11,atol=1e-10)
            errores[objetivo] = float(np.max(np.abs(pred-m.predict(datos[predictores]))))
            respuestas[objetivo] = {'coeficientes':coef.tolist(),'intercepto':intercepto}
            fuentes.append(ruta)
        modelos.append({'id':identificador,'nombre':nombre,'adaptador':'linear_v1',
                        'predictores':predictores,'respuestas':respuestas,
                        'seleccionado_por_cv':nombre == 'Ridge',
                        'descripcion':{'ridge':'Regresión lineal con regularización L2.',
                                       'svr_lineal':'Regresión de soporte vectorial con kernel lineal.',
                                       'media':'Referencia constante: media del desarrollo.'}[identificador],
                        'error_max_exportacion':errores})
    registro = {'version':1,'modelos':modelos}
    (salida/'modelos.json').write_text(json.dumps(registro,ensure_ascii=False,indent=2),encoding='utf-8')
    for nombre in ['resultados.csv','intervalos_bootstrap.csv','auditoria_predictiva_univariada.csv']:
        pd.read_csv(resultados/nombre).to_csv(salida/nombre,index=False)
    pred = pd.read_csv(resultados/'predicciones_prueba.csv')
    pred.to_csv(salida/'predicciones_prueba.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.read_csv(origen/'informe_resultados'/'catalogo_savedrecs.csv').to_csv(
        salida/'bibliografia.csv',index=False)
    manifest = {'version':1,'n_original':len(datos),'sensores':sensores,'protocolo':protocolo,
                'diccionario_objetivos':{'CO':'Monóxido de carbono','NOX':'Óxidos de nitrógeno'},
                'unidades_objetivos':'mg/m³','ejemplo_indice':int(ejemplo.name),
                'fuentes_SHA256':{str(p.relative_to(origen)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in fuentes},
                'inferencia':'Coeficientes en unidades originales; equivalencia comprobada en las 36.733 filas.',
                'entrenamiento':'No se realiza ajuste ni selección de modelos en el dashboard.'}
    (salida/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Exportacion completa: {salida}. Predicciones equivalentes en toda la fuente.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    exportar(parser.parse_args().project)
