#!/usr/bin/env python
"""
get_windows.py

Compute optimal energy windows for the conduction and valence bands (minimizing total quadrature points)
for the O(N^3) polarizability and self energy calculations given in Kim, Martyna, and Ismail-Beigi, PRB 101, 035139 (2020).

These energy windows also define the discrete imaginary-time grids used by the
CTSP method.  Once we move beyond static COHSEX these same windows will control
the frequency resolution of the full GW calculations.
"""

import sys
import numpy as np
import symmetry_maps               # imported as requested
from wfnreader import WFNReader
from scipy.special import roots_laguerre
from gpu_utils import cp, xp, GPU_AVAILABLE

class GL_window:
    def __init__(self, start_ind, end_ind, wfn):
        self.start_ind = start_ind
        self.end_ind = end_ind
        self.start_energy = wfn.energies[start_ind]
        self.end_energy = wfn.energies[end_ind]

class WindowInfo:
    def __init__(self, start_energy, end_energy):
        self.start_energy = start_energy
        self.end_energy = end_energy

class WindowPair:
    def __init__(self, val_window, cond_window, epsq):
        self.val_window = val_window
        self.cond_window = cond_window
        self.ntau = round(N_tau_window(cond_window, val_window, epsq))
        self.tau_i, self.w_i = roots_laguerre(self.ntau)
        self.z_lm = np.sqrt(1.0/((cond_window.end_energy - val_window.start_energy)*(cond_window.start_energy - val_window.end_energy))) # very important bug fixed here...

def compute_dos(wfn_file, n_points=2000):
    """
    Load all band energies from WFN file, flatten them,
    and compute a Gaussian‑broadened DOS on a linear grid.

    Args:
        wfn_file (str): path to WFN .h5 file
        n_points (int): number of energy grid points

    Returns:
        energies (np.ndarray): 1D array of length n_points
        dos      (np.ndarray): 1D DOS array of same length
    """
    # load wavefunction reader
    wfn = WFNReader(wfn_file)

    # flatten all eigen‑energies (shape nkpts × nbands) and move to xp
    all_e = wfn.energies.flatten()
    all_e = xp.asarray(all_e, dtype=float)

    # set up energy grid on xp
    e_min, e_max = float(all_e.min()), float(all_e.max())
    energy_grid = xp.linspace(e_min, e_max, n_points)
    dx = 1.1*(energy_grid[1] - energy_grid[0])   # broadening = grid spacing

    # Gaussian broadening: sum_i exp[−(E – Ei)^2/(2*dx^2)]/(dx*√(2π))
    eg = energy_grid[None, :]             # shape (1, n_points)
    es = all_e[:, None]                   # shape (n_states, 1)
    kernel = xp.exp(-(eg - es)**2 / (2*dx*dx))
    dos = xp.sum(kernel, axis=0) / (dx * xp.sqrt(2*xp.pi))

    # convert back to pure NumPy if we used CuPy
    if GPU_AVAILABLE:
        energy_grid = cp.asnumpy(energy_grid)
        dos = cp.asnumpy(dos)

    efermi = np.amax(wfn.energies[:,:,int(np.sum(wfn.occs[0,0])-1)])
    return energy_grid, dos, efermi

def N_tau_window(window_c, window_v, epsq):
    E_bw = window_c.end_energy - window_v.start_energy
    E_gap = window_c.start_energy - window_v.end_energy
    alpha = np.sqrt(E_bw/E_gap)
    return alpha*(0.4 - 0.3 * np.log(epsq))

def find_optimal_partitions(energies, n_windows):
    """
    Find the optimal partition points for a given number of windows.

    Args:
        energies (np.ndarray): Sorted array of energies.
        n_windows (int): Number of windows to partition into.

    Returns:
        partitions (list): List of partition indices.
    """
    n_points = len(energies)
    # Initialize partitions to divide the energy range into equal segments
    energy_min, energy_max = energies[0], energies[-1]
    energy_step = (energy_max - energy_min) / n_windows
    partitions = [0]
    
    for i in range(1, n_windows):
        # Find the index where the energy exceeds the current partition point
        partition_energy = energy_min + i * energy_step
        partition_index = np.searchsorted(energies, partition_energy)
        partitions.append(partition_index)
    
    partitions.append(n_points)
    min_cost = float('inf')
    best_partitions = partitions

    max_iterations = n_points * 2
    since_accepted = 0
    # Local search over partition points
    for _ in range(max_iterations):  # Number of iterations can be adjusted
        for i in range(1, n_windows):
            # Try moving the partition up or down by one state
            for delta in [-1, 1]:
                new_partitions = partitions.copy()
                new_partitions[i] = max(1, min(n_points-1, new_partitions[i] + delta))
                cost = 0
                for j in range(n_windows):
                    start, end = new_partitions[j], new_partitions[j+1]
                    cost += (energies[end-1] - energies[start])  # Example cost function
                if cost < min_cost:
                    min_cost = cost
                    best_partitions = new_partitions
                    since_accepted = 0
                else:
                    since_accepted += 1
                    if since_accepted > n_windows*2:
                        break

    return best_partitions

def minimize_cost_fn(wfn, epsq, max_val_windows=8, max_cond_windows=8):
    """
    Minimize the cost function by finding optimal partition points for valence and conduction windows.

    Args:
        wfn (WFNReader): Wavefunction reader object.
        epsq (float): Epsilon squared value.
        max_val_windows (int): Maximum number of valence windows.
        max_cond_windows (int): Maximum number of conduction windows.

    Returns:
        cost_matrix (np.ndarray): Cost matrix for each combination of valence and conduction windows.
        window_bounds (dict): Dictionary of window boundaries for each combination.
    """
    # flatten & sort all band energies
    all_e = wfn.energies.flatten()
    sorted_e = np.sort(all_e)
    # get Fermi level
    efermi = np.amax(wfn.energies[:, :, int(np.sum(wfn.occs[0,0]) - 1)])

    # split into valence (<= efermi) and conduction (> efermi)
    val_e = sorted_e[sorted_e <= efermi]
    cond_e = sorted_e[sorted_e >  efermi]

    cost_matrix = np.zeros((max_val_windows, max_cond_windows))
    window_bounds = {}

    for nval in range(1, max_val_windows+1):
        v_partitions = find_optimal_partitions(val_e, nval)
        for ncond in range(1, max_cond_windows+1):
            c_partitions = find_optimal_partitions(cond_e, ncond)
            total_cost = 0.0

            # accumulate cost over every (valence, conduction) window pair
            for iv in range(nval):
                v0, v1 = val_e[v_partitions[iv]], val_e[v_partitions[iv+1]-1]
                nv = v_partitions[iv+1] - v_partitions[iv]
                for jc in range(ncond):
                    c0, c1 = cond_e[c_partitions[jc]], cond_e[c_partitions[jc+1]-1]
                    nc = c_partitions[jc+1] - c_partitions[jc]
                    # make minimal window‐like objects for N_tau_window
                    class _W: pass
                    wv = _W(); wv.start_energy, wv.end_energy = v0, v1
                    wc = _W(); wc.start_energy, wc.end_energy = c0, c1
                    Nt = N_tau_window(wc, wv, epsq)
                    total_cost += Nt * (nv + nc)

            cost_matrix[nval-1, ncond-1] = total_cost
            window_bounds[(nval, ncond)] = (v_partitions, c_partitions)

    # Find the optimal configuration
    min_cost_idx = np.unravel_index(np.argmin(cost_matrix, axis=None), cost_matrix.shape)
    optimal_val_windows, optimal_cond_windows = min_cost_idx[0] + 1, min_cost_idx[1] + 1
    optimal_v_partitions = window_bounds[(optimal_val_windows, optimal_cond_windows)][0]
    optimal_c_partitions = window_bounds[(optimal_val_windows, optimal_cond_windows)][1]

    # Print the energy ranges of the most optimal windows
    print(f"Optimal valence windows ({optimal_val_windows}):")
    for iv in range(optimal_val_windows):
        v0, v1 = val_e[optimal_v_partitions[iv]], val_e[optimal_v_partitions[iv+1]-1]
        print(f"  Window {iv+1}: {v0:.4f} to {v1:.4f}")

    print(f"Optimal conduction windows ({optimal_cond_windows}):")
    for jc in range(optimal_cond_windows):
        c0, c1 = cond_e[optimal_c_partitions[jc]], cond_e[optimal_c_partitions[jc+1]-1]
        print(f"  Window {jc+1}: {c0:.4f} to {c1:.4f}")

    # Print the table of N_tau_window values
    print("\nN_tau_window Table (rounded):")
    header = "Val\\Cond" + "".join([f"\t{j+1}" for j in range(optimal_cond_windows)])
    print("-" * (8 + 8 * optimal_cond_windows))
    print(header)
    print("-" * (8 + 8 * optimal_cond_windows))

    for iv in range(optimal_val_windows):
        row = f"{iv+1}\t"
        v0, v1 = val_e[optimal_v_partitions[iv]], val_e[optimal_v_partitions[iv+1]-1]
        for jc in range(optimal_cond_windows):
            c0, c1 = cond_e[optimal_c_partitions[jc]], cond_e[optimal_c_partitions[jc+1]-1]
            # Create window-like objects
            class _W: pass
            wv = _W(); wv.start_energy, wv.end_energy = v0, v1
            wc = _W(); wc.start_energy, wc.end_energy = c0, c1
            Nt = N_tau_window(wc, wv, epsq)
            row += f"\t{round(Nt):<7}"
        print(row)
    print("-" * (8 + 8 * optimal_cond_windows))

    return cost_matrix, window_bounds

def get_window_info(epsq, wfn):
    """
    Calculate window information for valence and conduction bands.

    Args:
        epsq (float): Epsilon squared value.
        wfn (WFNReader): Wavefunction reader object.

    Returns:
        list: A list of WindowPair objects containing window information.
    """
    # Use minimize_cost_fn to find the optimal number of windows
    cost_matrix, window_bounds = minimize_cost_fn(wfn, epsq, max_val_windows=8, max_cond_windows=8)

    # Find the optimal configuration
    min_cost_idx = np.unravel_index(np.argmin(cost_matrix, axis=None), cost_matrix.shape)
    optimal_val_windows, optimal_cond_windows = min_cost_idx[0] + 1, min_cost_idx[1] + 1
    optimal_v_partitions = window_bounds[(optimal_val_windows, optimal_cond_windows)][0]
    optimal_c_partitions = window_bounds[(optimal_val_windows, optimal_cond_windows)][1]

    # flatten & sort all band energies
    all_e = wfn.energies.flatten()
    sorted_e = np.sort(all_e)
    # get Fermi level
    efermi = np.amax(wfn.energies[:, :, int(np.sum(wfn.occs[0,0]) - 1)])

    # split into valence (<= efermi) and conduction (> efermi)
    val_e = sorted_e[sorted_e <= efermi]
    cond_e = sorted_e[sorted_e >  efermi]

    # Create a list to hold window pairs
    window_pairs = []

    # Fill valence and conduction window information and create pairs
    for iv in range(optimal_val_windows):
        v0, v1 = val_e[optimal_v_partitions[iv]], val_e[optimal_v_partitions[iv+1]-1]
        val_window = WindowInfo(v0, v1)
        for jc in range(optimal_cond_windows):
            c0, c1 = cond_e[optimal_c_partitions[jc]], cond_e[optimal_c_partitions[jc+1]-1]
            cond_window = WindowInfo(c0, c1)
            window_pairs.append(WindowPair(val_window, cond_window, epsq))

    return window_pairs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # allow passing filename, else default
    wfn_file = sys.argv[1] if len(sys.argv) > 1 else "WFN.h5"
    energies, dos, efermi = compute_dos(wfn_file, n_points=200)
    print(f"Fermi level: {efermi:.4f}")

    # Example ε(q) value; adjust as needed
    epsq = 0.01

    # Re-read WFN to pass into cost routine
    wfn = WFNReader(wfn_file)

    # Get window information
    window_pairs = get_window_info(epsq, wfn)

    # Print table of window information
    print("\nWindow Information:")
    print(f"{'Pair':<10}{'Valence Emin':<15}{'Valence Emax':<15}{'Cond Emin':<15}{'Cond Emax':<15}{'z_lm':<10}{'tau_i':<20}{'w_i':<20}")
    print("-" * 120)

    for i, pair in enumerate(window_pairs, start=1):
        val_window = pair.val_window
        cond_window = pair.cond_window
        print(f"{i:<10}{val_window.start_energy:<15.4f}{val_window.end_energy:<15.4f}{cond_window.start_energy:<15.4f}{cond_window.end_energy:<15.4f}{pair.z_lm:<10.4f}{pair.tau_i[:5]}...{pair.tau_i[-5:]}{pair.w_i[:5]}...{pair.w_i[-5:]}")

    # Continue with the rest of the main code... 