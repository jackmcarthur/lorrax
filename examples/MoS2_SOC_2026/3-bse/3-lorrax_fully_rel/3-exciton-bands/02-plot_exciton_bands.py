"""Plot the finite-Q levels and their matched independent-particle threshold."""
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
    parser.add_argument('--near-gamma-prefix',
                        help='Optional separate dense near-Gamma scan prefix, without .dat')
    args = parser.parse_args()
    prefix = Path(args.prefix)
    coordinates, energies, nodes, threshold = read_bands(prefix)
    near_coordinates, near_energies, _, near_threshold = (
        read_bands(args.near_gamma_prefix) if args.near_gamma_prefix else
        (coordinates, energies, nodes, threshold))
    if args.near_gamma_prefix and (
            abs(near_coordinates[0, 1]) > 1e-6
            or np.linalg.norm(near_coordinates[0, 2:]) > 1e-6):
        raise ValueError('The near-Gamma scan must start at Gamma with zero path distance')
    # The driver's path distance is bohr^-1. Convert it to angstrom^-1.
    distance = coordinates[:, 1] / 0.529177210903
    near_distance = near_coordinates[:, 1] / 0.529177210903
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 8})
    fig, (ax, zoom) = plt.subplots(2, 1, figsize=(3, 4.5),
                                  gridspec_kw={'height_ratios': [1.45, 1]})
    for panel, x, e, edge in ((ax, distance, energies, threshold),
                              (zoom, near_distance, near_energies, near_threshold)):
        panel.plot(x, e, color='tab:blue', lw=.65, alpha=.8)
        if edge is not None:
            panel.plot(x, edge, '--', color='.35', lw=.9,
                       label='Independent-particle threshold')
        panel.spines[['top', 'right']].set_visible(False)
        panel.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        panel.set_ylabel('Exciton energy (eV)')
        panel.minorticks_off()
    ax.set_title(f'LORRAX: lowest {energies.shape[1]} levels', fontsize=9)
    ax.set_xlim(distance[0], distance[-1])
    highest = max(energies.max(), threshold.max()) if threshold is not None else energies.max()
    ax.set_ylim(energies.min() - .03, highest + .05)
    if nodes:
        ax.set_xticks([distance[i] for i, _ in nodes], [label for _, label in nodes])
        for i, _ in nodes[1:-1]:
            ax.axvline(distance[i], color='.85', lw=.5, zorder=0)
    ax.set_xlabel('Exciton momentum')
    if threshold is not None:
        ax.fill_between(distance, threshold, ax.get_ylim()[1],
                        color='.92', zorder=0)
        ax.legend(loc='best', frameon=False, handlelength=1.5)
    zoom_limit = min(.12, near_distance[-1])
    zoom.set_xlim(0, zoom_limit)
    if args.near_gamma_prefix:
        visible = near_energies[near_distance <= zoom_limit]
        low, high = visible.min(), visible.max()
        margin = max(.003, .08*(high-low))
        zoom.set_ylim(low-margin, high+margin)
    else:
        zoom.set_ylim(energies[0].min() - .005, energies[0].min() + .25)
    zoom.set_xlabel(r'$|Q|$ near Γ (Å$^{-1}$)')
    zoom.set_title(f'Near Γ: lowest {near_energies.shape[1]} levels'
                   if args.near_gamma_prefix else 'Near Γ', fontsize=9)
    zoom.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    fig.tight_layout(pad=.4, h_pad=.9)
    fig.savefig(prefix.parent / 'MoS2_exciton_bands.pdf')
    fig.savefig(prefix.parent / 'MoS2_exciton_bands.png', dpi=300)
    if threshold is None:
        print(f'No {prefix.name}_continuum.dat: continuum threshold was not plotted.')


if __name__ == '__main__':
    main()
