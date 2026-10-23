"""Plot the two fully relativistic implementations on the workshop axes."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8})

reference = Path('../../2-fully_rel/2-absorption')
fig, ax = plt.subplots(figsize=(3, 2.9))
for directory, label, color in ((Path('.'), 'LORRAX', 'tab:blue'),
                                (reference, 'BGW', 'tab:orange')):
    for name, style, kind in (('absorption_eh.dat', '-', 'BSE'),
                              ('absorption_noeh.dat', '--', 'IP')):
        if not (directory / name).exists():
            continue
        data = np.loadtxt(directory / name)
        ax.plot(data[:, 0], data[:, 1], style, color=color, lw=.9,
                label=f'{label} {kind}')
ax.set(xlabel='Energy (eV)', ylabel=r'$\epsilon_2$', xlim=(0, 10),
       xticks=[0, 5, 10], ylim=(0, None))
ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
ax.minorticks_off()
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=2,
          frameon=False, columnspacing=.8, handlelength=1.8,
          handletextpad=.4, labelspacing=.2, borderaxespad=0)
fig.tight_layout(pad=.4)
fig.savefig('MoS2_absorption_BGW_LORRAX.pdf')
fig.savefig('MoS2_absorption_BGW_LORRAX.png', dpi=300)
