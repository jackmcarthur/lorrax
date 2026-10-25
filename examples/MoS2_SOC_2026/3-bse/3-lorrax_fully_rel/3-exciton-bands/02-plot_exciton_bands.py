"""Plot 128 computed exciton levels on Gamma--K, with a near-Gamma detail."""
import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def read_bands(prefix):
    """Read path energies and an optional matched continuum from one prefix."""
    prefix = Path(prefix)
    source = prefix.with_suffix('.dat')
    rows = np.atleast_2d(np.loadtxt(source, dtype=str))
    rows = rows[rows[:, 5] == 'interp']
    if not len(rows):
        raise ValueError(f'{source} has no path rows')
    coordinates = rows[:, :5].astype(float)
    energies = rows[:, 6:].astype(float)
    if not np.isfinite(coordinates).all() or not np.isfinite(energies).all():
        raise ValueError(f'{source} contains nonfinite path data')
    nodes = []
    for line in source.read_text().splitlines():
        if line.startswith('# nodes:'):
            nodes = [(int(i), label) for i, label in
                     re.findall(r'(\d+):([^\s]*)', line)]
    nodes = [(i, 'Γ' if label in ('G', 'Gamma', 'Γ') else label)
             for i, label in nodes if i < len(rows) and label not in ('None', '')]

    threshold_file = prefix.parent / (prefix.name + '_continuum.dat')
    threshold = None
    if threshold_file.exists():
        continuum = np.atleast_2d(np.loadtxt(threshold_file))
        # iQ, path distance, Qx, Qy, Qz, min(Ec(k+Q)-Ev(k)) in eV.
        if (continuum.shape != (len(rows), 6)
                or not np.allclose(continuum[:, :5], coordinates, atol=1e-6, rtol=0)
                or not np.isfinite(continuum[:, 5]).all()):
            raise ValueError(f'{threshold_file} does not match the exciton path')
        threshold = continuum[:, 5]
    return coordinates, energies, nodes, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='exciton_bands')
    args = parser.parse_args()
    prefix = Path(args.prefix)
    coordinates, energies, _, threshold = read_bands(prefix)
    q = coordinates[:, 2:]
    # The tutorial follows Gamma--K in this cell's 60-degree reciprocal basis.
    # Refuse an old full-path table rather than relabelling its endpoints.
    if (not np.allclose(q[0], 0, atol=1e-6, rtol=0)
            or not np.allclose(q[-1], [1/3, 1/3, 0], atol=1e-6, rtol=0)
            or not np.allclose(q[:, 0], q[:, 1], atol=1e-6, rtol=0)
            or not np.allclose(q[:, 2], 0, atol=1e-6, rtol=0)
            or not np.all(np.diff(q[:, 0]) > 0)):
        raise ValueError('Expected the computed Gamma--K path, with K=(1/3,1/3,0)')
    if energies.shape[1] < 128:
        raise ValueError('The tutorial plot requires at least 128 computed levels')
    energies = energies[:, :128]
    # The driver's path distance is bohr^-1. Convert it to angstrom^-1.
    distance = coordinates[:, 1] / 0.529177210903
    if abs(distance[0]) > 1e-6 or not np.all(np.diff(distance) > 0):
        raise ValueError('Expected increasing path distance starting at Gamma')
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 8})
    fig, (ax, zoom) = plt.subplots(2, 1, figsize=(3, 4.5),
                                  gridspec_kw={'height_ratios': [1.55, 1]})
    # Straight segments join actual Q solves. No spline or spectral smoothing.
    ax.plot(distance, energies, color='tab:blue', lw=.45, alpha=.75)
    if threshold is not None:
        ax.plot(distance, threshold, '--', color='.35', lw=.9,
                label='QP continuum')
    ax.set_title('LORRAX: lowest 128 levels', fontsize=9)
    ax.set_xlim(0, distance[-1])
    ax.set_ylim(energies.min() - .025, 3.0)
    ax.set_xticks([0, distance[-1]], ['Γ', 'K'])
    ax.set_xlabel('Exciton momentum')
    if threshold is not None and np.any(threshold <= 3.0):
        ax.fill_between(distance, threshold, 3.0, where=threshold <= 3.0,
                        color='.93', zorder=0, interpolate=True)
        ax.legend(loc='lower left', frameon=False, handlelength=1.5)
    # Choose an actual endpoint for the detail, never extrapolate a short scan.
    near = distance <= min(.12, distance[-1])
    near_distance = distance[near]
    shifts = 1000 * (energies[near, 2:4] - energies[0, 2:4])
    for j, color in enumerate(('tab:blue', 'tab:orange')):
        zoom.plot(near_distance, shifts[:, j], color=color, lw=.8,
                  marker='.', markersize=2.5, label=f'Level {j + 3}')
    zoom.set_xlim(0, near_distance[-1])
    low, high = shifts.min(), shifts.max()
    margin = max(.5, .08*(high-low))
    zoom.set_ylim(low-margin, high+margin)
    zoom.set_xlabel(r'$|Q|$ along Γ–K (Å$^{-1}$)')
    zoom.set_title('Near Γ: energy shifts', fontsize=9)
    zoom.legend(loc='upper left', frameon=False, handlelength=1.5)
    zoom.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    for panel in (ax, zoom):
        panel.spines[['top', 'right']].set_visible(False)
        panel.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        panel.minorticks_off()
    ax.set_ylabel('Exciton energy (eV)')
    zoom.set_ylabel('Shift from Γ (meV)')
    fig.tight_layout(pad=.4, h_pad=.9)
    fig.savefig(prefix.parent / 'MoS2_exciton_bands.pdf')
    fig.savefig(prefix.parent / 'MoS2_exciton_bands.png', dpi=300)
    if threshold is None:
        print(f'No {prefix.name}_continuum.dat: continuum threshold was not plotted.')


if __name__ == '__main__':
    main()
