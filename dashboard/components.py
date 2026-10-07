"""Componentes comunes de presentacion accesible."""
import math
from dash import dcc, html

GRAPH_CONFIG = {'displaylogo':False,'responsive':True,
                'modeBarButtonsToRemove':['lasso2d','select2d'],
                'toImageButtonOptions':{'format':'png','scale':2}}


def graph(identifier, figure=None):
    height = figure.layout.height if figure is not None and figure.layout.height else 350
    return dcc.Graph(id=identifier,figure=figure,config=GRAPH_CONFIG,
                     className='chart',responsive=True,style={'height':f'{height}px','width':'100%'})


def number(value, digits=3):
    if value is None or (isinstance(value,float) and not math.isfinite(value)):
        return '—'
    return f'{value:,.{digits}f}'.replace(',','X').replace('.',',').replace('X','.')


def table(frame, labels=None, digits=3):
    labels = labels or {}
    rows = []
    for row in frame.itertuples(index=False,name=None):
        cells = [number(v,digits) if isinstance(v,float) else str(v) for v in row]
        rows.append(html.Tr([html.Td(v) for v in cells]))
    return html.Div(html.Table([html.Thead(html.Tr([html.Th(labels.get(c,c)) for c in frame.columns])),
                               html.Tbody(rows)]),className='table-scroll')


def disclosure(title, children, identifier=None, open=False):
    kwargs = {'id':identifier} if identifier else {}
    return html.Details([html.Summary(title),html.Div(children,className='disclosure-body')],
                        open=open,className='disclosure',**kwargs)


def notice(text, kind='info', identifier=None):
    return html.Div(text,className=f'notice {kind}',**({'id':identifier} if identifier else {}))


def metric(label, value, note=None):
    return html.Div([html.Span(label,className='metric-label'),html.Strong(value),
                     html.Small(note) if note else None],className='metric')
