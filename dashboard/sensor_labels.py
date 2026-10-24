"""Sensor labels shared by interactive and notebook figures."""
from textwrap import fill

from .repository import sensors


def sensor_label(code, unit=False, width=None, html=False):
    names = {item['id']: item for item in sensors()}
    if code not in names:
        return {'CO': 'CO: monóxido de carbono', 'NOX': 'NOx: óxidos de nitrógeno'}.get(code, code)
    item = names[code]
    label = code+': '+item['nombre'].lower()
    if unit:
        label += ' ('+item['unidad']+')'
    if width:
        label = fill(label, width=width, break_long_words=False, break_on_hyphens=False)
    return label.replace('\n', '<br>') if html else label


def predictor_axes(fig, codes, positions=None):
    """Keep underlying sensor IDs while showing their meaning on the vertical axis."""
    codes = list(codes)
    fig.update_yaxes(tickmode='array', tickvals=codes if positions is None else list(positions),
        ticktext=[sensor_label(c, width=24, html=True) for c in codes],
        tickfont={'size': 10}, title=None, automargin=True)
    fig.update_layout(height=max(fig.layout.height or 350, 440), margin={'l': 155})
    return fig


def label_figure(fig):
    """Enrich categorical predictor axes and hover labels, preserving all plot values."""
    ids = {item['id'] for item in sensors()}
    for trace in fig.data:
        values = list(trace.y) if trace.y is not None else []
        if not any(isinstance(v, str) and v in ids for v in values):
            continue
        predictor_axes(fig, values)
        if trace.type == 'heatmap':
            trace.hovertext = [[sensor_label(y)+' / '+sensor_label(x) for x in trace.x] for y in values]
            trace.hovertemplate = trace.hovertemplate.replace('%{y} · %{x}', '%{hovertext}').replace('%{x} / %{y}', '%{hovertext}')
        else:
            trace.hovertext = [sensor_label(y) for y in values]
            if trace.hovertemplate:
                trace.hovertemplate = trace.hovertemplate.replace('%{y}', '%{hovertext}')
    return fig
