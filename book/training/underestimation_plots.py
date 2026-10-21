"""Reproducible scientific plot from saved OOF/test predictions, without fitting."""
from pathlib import Path
import sys
import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dashboard.underestimation import read, trend


def observed_figure(target):
    frames = [read(target, phase) for phase in ['OOF', 'prueba']]
    low = min(0., *[min(frame[target].min(), frame.prediccion.min()) for frame in frames])
    high = max(max(frame[target].max(), frame.prediccion.max()) for frame in frames)
    pad = (high-low)*.03
    bounds = [low-pad, high+pad]
    name = 'NOx' if target == 'NOX' else target
    figure = Figure(figsize=(11.5, 5.4), facecolor='white')
    axes = figure.subplots(1, 2)
    for ax, phase, frame, color in zip(axes, ['OOF', 'prueba'], frames, ['#157a7c', '#627f35']):
        ax.scatter(frame[target], frame.prediccion, s=7, alpha=.16, color=color, edgecolors='none', rasterized=True)
        ax.plot(bounds, bounds, color='#68747c', linestyle='--', linewidth=1.1, label='Estimación perfecta')
        averages = trend(target, phase)
        ax.plot(averages.observado, averages.estimado, color='#b65346', marker='o', linewidth=1.6,
                markersize=4, label='Promedios por intervalo')
        ax.set(xlim=bounds, ylim=bounds, xlabel=f'{name} observado (mg/m³)', ylabel=f'{name} estimado (mg/m³)')
        title = 'Desarrollo · OOF' if phase == 'OOF' else 'Prueba reservada'
        ax.set_title(f'{title} · n = {len(frame):,}'.replace(',', '.'), fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.2)
        ax.spines[['right', 'top']].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .055), ncol=2, frameon=False)
    figure.suptitle(f'{name}: concentraciones observadas y estimadas', fontsize=15, fontweight='bold', y=.99)
    figure.text(.5, .015, 'Por debajo de la diagonal: subestimación. Se conservan todos los registros.', ha='center', fontsize=9)
    figure.tight_layout(rect=[0, .13, 1, .94], w_pad=2.)
    return figure
