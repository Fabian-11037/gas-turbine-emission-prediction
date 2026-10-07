"""Comprobacion local con Playwright; no forma parte de las dependencias Render."""
from pathlib import Path
import json
import sys
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dashboard.repository import sample_frame


def main():
    captures = ROOT/'tests'/'screenshots'
    captures.mkdir(parents=True,exist_ok=True)
    failures = []
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge',headless=True)
        page = browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
        page.on('pageerror',lambda error:failures.append(str(error)))
        for width,height,device in [(1440,1000,'desktop'),(390,844,'mobile'),(360,800,'narrow')]:
            page.set_viewport_size({'width':width,'height':height})
            page.goto('http://127.0.0.1:8050/',wait_until='networkidle')
            page.get_by_text('El problema y su alcance',exact=True).wait_for()
            for tab,label in [('contexto','Contexto'),('eda','EDA'),('modelos','Modelos base')]:
                page.locator('#main-tabs').get_by_text(label,exact=True).click()
                page.wait_for_timeout(1000)
                if tab == 'eda':
                    page.locator('#eda-chart-one .js-plotly-plot').wait_for()
                    for view in ['Relaciones','Cambios anuales','Correlaciones','Distribuciones']:
                        page.locator('#eda-view').get_by_text(view,exact=True).click()
                        page.wait_for_timeout(600)
                    page.locator('#eda-target').get_by_text('NOx',exact=True).click()
                    page.wait_for_timeout(600)
                if tab == 'modelos':
                    page.locator('#model-stage-chart .js-plotly-plot').wait_for()
                    page.locator('#predict-button').click()
                    page.wait_for_timeout(600)
                    assert '—' not in page.locator('#prediction-result').inner_text()
                    page.locator('#sensor-AH').fill('101')
                    page.locator('#predict-button').click()
                    page.wait_for_timeout(600)
                    page.wait_for_function('document.querySelector("#prediction-feedback").innerText.toLowerCase().includes("humedad")')
                    page.locator('#reset-inputs').click()
                    page.wait_for_timeout(400)
                    page.locator('#predict-button').click()
                    page.wait_for_timeout(400)
                    page.locator('#model-target').get_by_text('NOx',exact=True).click()
                    page.wait_for_timeout(500)
                page.screenshot(path=str(captures/f'{device}_{tab}.png'),full_page=True)
                sizes = page.evaluate('({width:innerWidth,document:document.documentElement.scrollWidth})')
                assert sizes['document'] <= sizes['width']+1,(device,tab,sizes)
                checks.append({'device':device,'tab':tab,'overflow':False})
            if device == 'desktop':
                page.locator('#model-select').click()
                page.locator('#model-select').get_by_text('SVR lineal',exact=True).click()
                page.wait_for_timeout(600)
                assert 'SVR lineal' in page.locator('#model-description').inner_text()
                page.locator('#predict-button').click()
                page.wait_for_timeout(500)
                assert '—' not in page.locator('#prediction-result').inner_text()
                page.get_by_text('Predicciones por lote · CSV',exact=True).click()
                page.locator('#batch-upload input[type=file]').set_input_files({
                    'name':'sensores.csv','mimeType':'text/csv','buffer':sample_frame().to_csv(index=False).encode()})
                page.wait_for_timeout(700)
                assert not page.locator('#batch-download-button').is_disabled()
                with page.expect_download() as info:
                    page.locator('#batch-download-button').click()
                assert info.value.suggested_filename == 'estimaciones_CO_NOx.csv'
                page.get_by_text('Comparación entre modelos · entrenamiento, validación y prueba',exact=True).click()
                assert page.locator('#model-comparison table').is_visible()
                page.get_by_text('Diagnóstico de errores',exact=True).click()
                page.wait_for_timeout(500)
                page.screenshot(path=str(captures/'desktop_modelos_expanded.png'),full_page=True)
                checks.append({'model_switch':True,'batch_csv':True,'download':True,'disclosure':True})
        browser.close()
    assert not failures,failures
    (captures/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps({'checks':checks,'browser_errors':failures},indent=2))


if __name__ == '__main__':
    main()
