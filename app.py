"""Punto de entrada local y WSGI para Render: app:server."""
import os
from pathlib import Path
from dash import Dash,Input,Output,dcc,html
from dashboard import context,eda,model_view
from dashboard.models import registry
from dashboard.repository import manifest

ROOT = Path(__file__).resolve().parent
TITLE = 'Emisiones de Turbinas de Gas: Predicción de CO y NOx a partir de Datos Operativos'
app = Dash(__name__,assets_folder=str(ROOT/'assets'),title=TITLE,
           update_title='Actualizando…',suppress_callback_exceptions=True)
server = app.server
server.config['MAX_CONTENT_LENGTH'] = 4*1024*1024


@server.get('/healthz')
def health():
    return {'status':'ok','dataset':manifest()['n_original'],'modelos':list(registry())}


app.layout = html.Div([
    html.Header([html.Div([html.P('PROYECTO DE MACHINE LEARNING',className='eyebrow'),
                          html.H1(TITLE)]),
                 html.Div([html.Span(className='live-dot'),html.Span('Datos UCI · 2011–2015')],className='header-meta')],className='app-header'),
    html.Figure([
        html.Img(src='/assets/turbina_portada.jpg',
                 alt='Inspección de los componentes de una turbina de gas FR6 en la sala de máquinas de DOGA Enerji, Estambul.',
                 width=1920, height=1440),
        html.Figcaption([
            html.Span('Turbina de gas FR6 · inspección en sala de máquinas'),
            html.Span(['Foto: ', html.A('Pro-Per Energy Services',
                href='https://commons.wikimedia.org/wiki/File:Frame_Turbine.JPG', target='_blank', rel='noopener noreferrer'),
                ' · ', html.A('CC BY-SA 4.0', href='https://creativecommons.org/licenses/by-sa/4.0/',
                              target='_blank', rel='noopener noreferrer')])])], className='project-cover'),
    dcc.Tabs(id='main-tabs',value='contexto',children=[
        dcc.Tab(label='Contexto',value='contexto'),dcc.Tab(label='EDA',value='eda'),
        dcc.Tab(label='Modelos',value='modelos')],className='main-tabs',parent_className='main-tabs-parent',mobile_breakpoint=0),
    html.Main(id='page-content',children=context.layout(),className='main-content'),
    html.Footer([html.Span('Estimación contemporánea · Investigación académica'),
                 html.Span('Bloques 168 · Purga 24 · CV anidada 5 × 3')],className='app-footer')
],className='app-shell')


@app.callback(Output('page-content','children'),Input('main-tabs','value'))
def page(tab):
    return {'contexto':context.layout,'eda':eda.layout,'modelos':model_view.layout}.get(tab,context.layout)()


eda.register(app)
model_view.register(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT','8050')),debug=False)
