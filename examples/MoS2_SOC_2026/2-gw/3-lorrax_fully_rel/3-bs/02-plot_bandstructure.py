"""Plot the eight-column BerkeleyGW-layout LORRAX band path against the plane-wave code.

Dots mark the 3 x 3 x 1 coarse k-points that lie on the path (Gamma, the grid point
two thirds of the way to M, and K): there the QP energies are computed directly, and
between them they are interpolated.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8})

# Path nodes in ht.in: Gamma (0), M (40), K (55), Gamma (105).  The 3 x 3 x 1 grid
# points on the path are Gamma, (0, 1/3) on Gamma-M (2/3 of the way to M at (0, 1/2)),
# and K = (1/3, 1/3).
NODES = (0, 40, 55, 105)


def bands(path):
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 8 or not np.isfinite(data).all():
        raise ValueError(f'{path}: expected finite eight-column bandstructure data')
    for band in np.unique(data[:, 1].astype(int)):
        yield data[data[:, 1].astype(int) == band]


def coarse_positions(distance):
    d0, dm, dk, dg = (distance[i] for i in NODES)
    return np.array([d0, d0 + 2.0 / 3.0 * (dm - d0), dk, dg])


fig, ax = plt.subplots(figsize=(3, 2.9))
for source, column, color, label, dots in (
    (Path('../../2-fully_rel/3-bs/bandstructure.dat'), 5, '0.8', 'DFT', False),
    (Path('bandstructure.dat'), 6, 'tab:blue', 'LORRAX GW', True),
    (Path('../../2-fully_rel/3-bs/bandstructure.dat'), 6, 'tab:orange', 'Plane-wave GW', True),
):
    if not source.exists():
        continue
    source_data = np.loadtxt(source)
    occupied = source_data[:, 1] <= 26
    zero = np.max(source_data[occupied, column])
    first = True
    for values in bands(source):
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(values[:, 2:5], axis=0), axis=1))]
        energy = values[:, column] - zero
        ax.plot(distance, energy, color=color, lw=.7,
                label=label if first else None)
        if dots:
            x = coarse_positions(distance)
            ax.plot(x, np.interp(x, distance, energy), 'o', color=color, ms=2.2,
                    mec='none', zorder=3)
        first = False
        ticks = distance[list(NODES)]
ax.set(xticks=ticks, xticklabels=[r'$\Gamma$', 'M', 'K', r'$\Gamma$'],
       yticks=[-4, 0, 4, 8], ylabel='Energy relative to VBM (eV)',
       ylim=(-5, 8), xlim=(ticks[0], ticks[-1]))
ax.minorticks_off()
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=3,
          frameon=False, columnspacing=.6, handlelength=1.3,
          handletextpad=.4, borderaxespad=0)
fig.tight_layout(pad=.4)
fig.savefig('MoS2_bandstructure_BGW_LORRAX.pdf')
fig.savefig('MoS2_bandstructure_BGW_LORRAX.png', dpi=300)
