"""Plot the eight-column BerkeleyGW-layout LORRAX band path."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def bands(path):
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 8 or not np.isfinite(data).all():
        raise ValueError(f'{path}: expected finite eight-column bandstructure data')
    for band in np.unique(data[:, 1].astype(int)):
        yield data[data[:, 1].astype(int) == band]


fig, ax = plt.subplots(figsize=(5, 5))
for source, column, color, label in (
    (Path('../../2-fully_rel/3-bs/bandstructure.dat'), 5, '0.8', 'DFT'),
    (Path('bandstructure.dat'), 6, 'tab:blue', 'LORRAX GW'),
    (Path('../../2-fully_rel/3-bs/bandstructure.dat'), 6, 'tab:orange', 'BerkeleyGW GW'),
):
    if not source.exists():
        continue
    source_data = np.loadtxt(source)
    occupied = source_data[:, 1] <= 26
    zero = np.max(source_data[occupied, column])
    first = True
    for values in bands(source):
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(values[:, 2:5], axis=0), axis=1))]
        ax.plot(distance, values[:, column] - zero, color=color, lw=.7,
                label=label if first else None)
        first = False
        ticks = distance[[0, 40, 55, 105]]
ax.set(xticks=ticks, xticklabels=[r'$\Gamma$', 'M', 'K', r'$\Gamma$'],
       ylabel='Energy relative to each curve’s VBM (eV)', ylim=(-5, 8), xlim=(ticks[0], ticks[-1]))
ax.legend()
fig.tight_layout()
fig.savefig('MoS2_bandstructure_BGW_LORRAX.pdf')
fig.savefig('MoS2_bandstructure_BGW_LORRAX.png', dpi=160)
