"""Plot the two fully relativistic implementations on the workshop axes."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

reference = Path('../../2-fully_rel/2-absorption')
for name, style in (('absorption_eh.dat', '-'), ('absorption_noeh.dat', '--')):
    for directory, label in ((Path('.'), 'LORRAX SOC'), (reference, 'BerkeleyGW SOC')):
        if not (directory / name).exists():
            continue
        data = np.loadtxt(directory / name)
        plt.plot(data[:, 0], data[:, 1], style,
                 label=label + (' BSE' if name == 'absorption_eh.dat' else ' independent particle'))
plt.xlabel('Energy (eV)')
plt.ylabel(r'$\epsilon_2$')
plt.xlim(0, 10)
plt.legend()
plt.tight_layout()
plt.savefig('MoS2_absorption_BGW_LORRAX.pdf')

plt.savefig('MoS2_absorption_BGW_LORRAX.png', dpi=160)
