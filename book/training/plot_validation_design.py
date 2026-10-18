"""Figuras de las particiones reales y de los margenes de exclusion."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


COLORES = ['#24857e', '#694393', '#b9bec5', '#e9bb38', '#d96143']
ETIQUETAS = ['Entrenamiento / desarrollo', 'Prueba reservada',
             'Exclusion junto a prueba', 'Validacion del pliegue',
             'Exclusion junto a validacion']


def _pintar(ax, estados, inicio=0, altura=.74):
    cortes = np.r_[0, np.flatnonzero(np.diff(estados)) + 1, len(estados)]
    for a, b in zip(cortes[:-1], cortes[1:]):
        ax.broken_barh([(inicio + a, b-a)], (-altura/2, altura),
                       facecolors=COLORES[int(estados[a])], linewidth=0)


def mostrar_particiones(datos, train_idx, test_idx, splits, asignacion, conteos):
    """Usa los indices del protocolo existente; no ajusta modelos."""
    n = len(datos)
    estado = np.full(n, 2, dtype=int)
    estado[train_idx] = 0
    estado[test_idx] = 1
    anios = sorted(datos.anio.unique())
    fig, axes = plt.subplots(len(anios), 1, figsize=(12, 5.8), sharex=True)
    max_n = max((datos.anio == a).sum() for a in anios)
    for ax, anio in zip(axes, anios):
        ids = np.flatnonzero(datos.anio.to_numpy() == anio)
        _pintar(ax, estado[ids])
        cuentas = np.bincount(estado[ids], minlength=3)
        ax.set(ylim=(-.55, .55), yticks=[], ylabel=str(anio))
        ax.text(1.01, .5, f'D: {cuentas[0]:,}\nP: {cuentas[1]:,}\nE: {cuentas[2]:,}',
                transform=ax.transAxes, va='center', fontsize=9)
        ax.spines[['left', 'right', 'top']].set_visible(False)
    axes[-1].set(xlim=(0, max_n), xlabel='Posicion del registro dentro del archivo anual (base cero)')
    fig.suptitle('A. Desarrollo, prueba y exclusion: asignacion real en 2011-2015', fontsize=13)
    fig.legend(handles=[Patch(color=COLORES[i], label=ETIQUETAS[i]) for i in range(3)],
               loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(top=.89, bottom=.15, right=.88, hspace=.45, left=.08)
    plt.show()

    fig, axes = plt.subplots(5, 1, figsize=(12, 6.7), sharex=True)
    inicio_anios = [np.flatnonzero(datos.anio.to_numpy() == a)[0] for a in anios]
    fin_anios = inicio_anios[1:] + [n]
    estados_folds = []
    for i, (ax, (tr, va)) in enumerate(zip(axes, splits)):
        estados = estado.copy()
        estados[train_idx] = 4
        estados[train_idx[tr]] = 0
        estados[train_idx[va]] = 3
        estados_folds.append(estados)
        _pintar(ax, estados)
        ax.set(ylim=(-.55, .55), yticks=[], ylabel=f'Pliegue {i+1}')
        for lim in inicio_anios[1:]:
            ax.axvline(lim, color='white', linewidth=1.2)
        ax.spines[['left', 'right', 'top']].set_visible(False)
        if i == 0:
            for a, b, anio in zip(inicio_anios, fin_anios, anios):
                ax.text((a+b)/2, .69, str(anio), ha='center', fontsize=10)
        fila = conteos.iloc[i]
        ax.text(1.01, .5, f'A: {int(fila.n_train):,}\nV: {int(fila.n_val):,}\nE CV: {int(fila.n_purgados_CV):,}',
                transform=ax.transAxes, va='center', fontsize=9)
    axes[-1].set(xlim=(0, n), xlabel='Posicion global del registro; archivos anuales concatenados en orden')
    fig.suptitle('B. Cinco pliegues de validacion: mismos bloques de prueba fuera del ajuste', fontsize=13)
    fig.legend(handles=[Patch(color=COLORES[i], label=ETIQUETAS[i]) for i in range(5)],
               loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(top=.86, bottom=.17, right=.86, hspace=.5, left=.09)
    plt.show()

    # El acercamiento requiere ambos margenes completos y entrenamiento a sus lados.
    ids_anio = np.flatnonzero(datos.anio.to_numpy() == 2013)
    detalles = []
    for nombre, estados, reservado, purga in [
            ('Prueba reservada', estado, 1, 2),
            ('Validacion del pliegue 1', estados_folds[0], 3, 4)]:
        elegido = None
        for _, grupo in asignacion.iloc[ids_anio].groupby('bloque', sort=True):
            ids = grupo.index.to_numpy()
            a, b = int(ids.min()), int(ids.max()) + 1
            if len(ids) != 168 or a-60 < ids_anio[0] or b+60 > ids_anio[-1]+1:
                continue
            if (np.all(estados[a:b] == reservado) and
                    np.all(estados[a-24:a] == purga) and
                    np.all(estados[b:b+24] == purga) and
                    np.all(estados[a-60:a-24] == 0) and
                    np.all(estados[b+24:b+60] == 0)):
                elegido = (a, b, estados)
                break
        if elegido is None:
            raise AssertionError('No se encontro un bloque real con ambos margenes completos.')
        detalles.append((nombre, *elegido))
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.4))
    for ax, (nombre, a, b, estados) in zip(axes, detalles):
        _pintar(ax, estados[a-60:b+60], inicio=-60)
        ax.set(xlim=(-60, 228), ylim=(-.7, 1.3), yticks=[],
               xlabel='Posicion respecto al inicio del bloque reservado', title=nombre)
        ax.spines[['left', 'right', 'top']].set_visible(False)
        for x, etiqueta in [(-12, '24 excluidos'), (84, '168 reservados'), (180, '24 excluidos')]:
            ax.text(x, .65, etiqueta, ha='center', fontsize=9)
        ax.set_xticks([-60, -24, 0, 168, 192, 228])
        ax.text(.99, .95, f'2013: posiciones anuales {a-ids_anio[0]} a {b-1-ids_anio[0]}',
                transform=ax.transAxes, ha='right', fontsize=8)
    fig.suptitle('C. Detalle de dos bloques reales y sus margenes de 24 registros', fontsize=13)
    fig.subplots_adjust(top=.82, bottom=.14, hspace=.8, left=.08, right=.97)
    plt.show()
