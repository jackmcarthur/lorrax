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
from common import symmetry_maps  # imported as requested
from isdf_io import WFNReader
from scipy.special import roots_laguerre
from common.gpu_utils import cp, xp, GPU_AVAILABLE
import builtins
import jax

# Gate prints in this module to rank 0 only
_def_print = builtins.print

def print(*args, **kwargs):
	if jax.process_index() == 0:
		_def_print(*args, **kwargs)

class GL_window:
    def __init__(self, start_ind, end_ind, wfn):
        self.start_ind = start_ind
        self.end_ind = end_ind
        self.start_energy = wfn.energies[start_ind]
        self.end_energy = wfn.energies[end_ind]

class WindowInfo:
    def __init__(self, start_energy, end_energy, index=None, count=None):
        self.start_energy = start_energy
        self.end_energy = end_energy
        self.index = index
        self.count = count

    @property
    def upper_inclusive(self) -> bool:
        if self.index is None or self.count is None:
            return True
        return self.index >= self.count - 1

class WindowPair:
    def __init__(self, val_window, cond_window, epsq):
        self.val_window = val_window
        self.cond_window = cond_window
        self.epsq = epsq  # store for lazy HGL initialization
        self.ntau = round(N_tau_window(cond_window, val_window, epsq))
        self.tau_i, self.w_i = roots_laguerre(self.ntau)
        self.z_lm = np.sqrt(1.0/((cond_window.end_energy - val_window.start_energy)*(cond_window.start_energy - val_window.end_energy))) # very important bug fixed here...
        # Populated on-demand when the JAX kernels need per-k band ranges.
        # All fields are typed for safe use in JAX kernels without casting.
        self.val_band_start: np.ndarray | None = None  # (nk,) int32
        self.val_band_len: np.ndarray | None = None    # (nk,) int32
        self.val_band_offset: np.ndarray | None = None
        self.cond_band_start: np.ndarray | None = None # (nk,) int32
        self.cond_band_len: np.ndarray | None = None   # (nk,) int32
        self.cond_band_offset: np.ndarray | None = None
        self.max_val_len: int = 0   # int, ready for static_argnames
        self.max_cond_len: int = 0  # int, ready for static_argnames
        self._has_band_ranges: bool = False
        # HGL quadrature fields (None until needed for crossing frequencies)
        # gamma = z_lm (same scaling parameter)
        self.tau_hgl: np.ndarray | None = None
        self.w_hgl: np.ndarray | None = None
        self.ntau_hgl: int = 0

    def init_hgl_quadrature(self, epsq: float | None = None):
        """Initialize HGL nodes/weights. Call once per window that needs HGL.
        
        HGL quadrature is used when a frequency ω causes an energy crossing,
        i.e., when E_gap < ω < E_bw for this window pair. The caller is
        responsible for determining which windows need HGL and calling this
        exactly once for each.
        
        Args:
            epsq: Target fractional error (defaults to self.epsq from __init__)
        """
        from .hgl_quadrature import hgl_nodes_weights, n_tau_hgl
        if epsq is None:
            epsq = self.epsq
        E_bw = self.cond_window.end_energy - self.val_window.start_energy
        self.ntau_hgl = n_tau_hgl(self.z_lm, E_bw, epsq)
        self.tau_hgl, self.w_hgl = hgl_nodes_weights(self.ntau_hgl)

    def classify_frequencies(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Classify which ω values use GL vs HGL quadrature for this window.
        
        GL (Gauss-Laguerre): |ω| < E_gap or |ω| > E_bw (denominator doesn't change sign)
        HGL (Hermite-Gauss-Laguerre): E_gap < |ω| < E_bw (energy crossing region)
        
        Args:
            omega: Array of frequencies to classify (can be complex for contour integration)
        
        Returns:
            gl_indices: Indices into omega array for GL treatment (non-crossing)
            hgl_indices: Indices into omega array for HGL treatment (crossing)
        """
        E_gap = self.cond_window.start_energy - self.val_window.end_energy
        E_bw = self.cond_window.end_energy - self.val_window.start_energy
        omega_abs = np.abs(np.real(omega))  # Use real part for classification
        
        # GL for non-crossing: outside the [E_gap, E_bw] range
        gl_mask = (omega_abs < E_gap) | (omega_abs > E_bw)
        gl_indices = np.where(gl_mask)[0]
        hgl_indices = np.where(~gl_mask)[0]
        return gl_indices, hgl_indices


def classify_frequencies(omega: np.ndarray, win: 'WindowPair') -> tuple[np.ndarray, np.ndarray]:
    """
    Classify which ω values use GL vs HGL quadrature for a window pair.
    
    This is a standalone function that can be used without a WindowPair object
    (useful for testing or when window bounds are known but not wrapped in a class).
    
    Args:
        omega: Array of frequencies to classify (can be complex)
        win: WindowPair object with val_window and cond_window
    
    Returns:
        gl_indices: Indices into omega array for GL treatment (non-crossing)
        hgl_indices: Indices into omega array for HGL treatment (crossing)
    """
    return win.classify_frequencies(omega)


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

def minimize_cost_fn(wfn, epsq, max_val_windows=8, max_cond_windows=8, cond_bounds=None, nband_max=None):
    """
    Minimize the cost function by finding optimal partition points for valence and conduction windows.

    Args:
        wfn (WFNReader): Wavefunction reader object.
        epsq (float): Epsilon squared value.
        max_val_windows (int): Maximum number of valence windows.
        max_cond_windows (int): Maximum number of conduction windows.
        cond_bounds (tuple[float,float]|None): Optional (emin, emax) to restrict conduction energies.
        nband_max (int|None): If provided, restrict energies to the first nband_max bands.

    Returns:
        cost_matrix (np.ndarray): Cost matrix for each combination of valence and conduction windows.
        window_bounds (dict): Dictionary of window boundaries for each combination.
    """
    # select energy source, optionally restrict to first nband_max bands
    if nband_max is not None:
        energies_src = wfn.energies[:, :, :nband_max]
    else:
        energies_src = wfn.energies

    # flatten & sort all band energies
    all_e = energies_src.flatten()
    sorted_e = np.sort(all_e)
    # get Fermi level (unchanged)
    efermi = np.amax(wfn.energies[:, :, int(np.sum(wfn.occs[0,0]) - 1)])

    # split into valence (<= efermi) and conduction (> efermi)
    val_e = sorted_e[sorted_e <= efermi]
    cond_e = sorted_e[sorted_e >  efermi]
    # Restrict conduction energies to active range if requested
    if cond_bounds is not None:
        cmin, cmax = cond_bounds
        cond_e = cond_e[(cond_e >= cmin) & (cond_e <= cmax)]

    cost_matrix = np.zeros((max_val_windows, max_cond_windows))
    window_bounds = {}

    for nval in range(1, max_val_windows+1):
        v_partitions = find_optimal_partitions(val_e, nval)
        for ncond in range(1, max_cond_windows+1):
            # Guard against empty or too-small cond_e
            if len(cond_e) < ncond or len(val_e) < nval:
                cost_matrix[nval-1, ncond-1] = np.inf
                continue
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
    optimal_v_partitions = window_bounds.get((optimal_val_windows, optimal_cond_windows), (None, None))[0]
    optimal_c_partitions = window_bounds.get((optimal_val_windows, optimal_cond_windows), (None, None))[1]

    # Conversion: Ry to eV
    RY_TO_EV = 13.605693122994
    
    # Compute Fermi level and gap
    # Fermi level is max of occupied bands
    efermi = np.max(val_e)
    efermi_ev = efermi * RY_TO_EV
    
    # Gap = minimum Ec - maximum Ev
    if optimal_v_partitions is not None and optimal_c_partitions is not None:
        v_max = val_e[optimal_v_partitions[-1] - 1]
        c_min = cond_e[optimal_c_partitions[0]]
        gap_ry = c_min - v_max
        gap_ev = gap_ry * RY_TO_EV
    else:
        gap_ry = gap_ev = 0.0
    
    # Section header
    print("")
    print("=" * 72)
    print("  FREQUENCY INTEGRATION WINDOWS")
    print("=" * 72)
    print(f"  Fermi level: {efermi_ev:8.3f} eV       Band gap: {gap_ev:8.3f} eV")
    print("")
    
    # Print valence windows
    print(f"  Valence windows ({optimal_val_windows}):")
    print(f"  {'Win':>4}  {'E_min (eV)':>12}  {'E_max (eV)':>12}  {'Width (eV)':>11}")
    print("  " + "-" * 48)
    if optimal_v_partitions is not None:
        for iv in range(optimal_val_windows):
            v0, v1 = val_e[optimal_v_partitions[iv]], val_e[optimal_v_partitions[iv+1]-1]
            v0_ev, v1_ev = v0 * RY_TO_EV, v1 * RY_TO_EV
            width_ev = (v1 - v0) * RY_TO_EV
            print(f"  {iv+1:>4}  {v0_ev:>12.3f}  {v1_ev:>12.3f}  {width_ev:>11.3f}")
    print("")
    
    # Print conduction windows
    print(f"  Conduction windows ({optimal_cond_windows}):")
    print(f"  {'Win':>4}  {'E_min (eV)':>12}  {'E_max (eV)':>12}  {'Width (eV)':>11}")
    print("  " + "-" * 48)
    if optimal_c_partitions is not None:
        for jc in range(optimal_cond_windows):
            c0, c1 = cond_e[optimal_c_partitions[jc]], cond_e[optimal_c_partitions[jc+1]-1]
            c0_ev, c1_ev = c0 * RY_TO_EV, c1 * RY_TO_EV
            width_ev = (c1 - c0) * RY_TO_EV
            print(f"  {jc+1:>4}  {c0_ev:>12.3f}  {c1_ev:>12.3f}  {width_ev:>11.3f}")
    print("")
    
    # Build window pair info for tables
    window_data = []
    if optimal_v_partitions is not None and optimal_c_partitions is not None:
        for iv in range(optimal_val_windows):
            v0, v1 = val_e[optimal_v_partitions[iv]], val_e[optimal_v_partitions[iv+1]-1]
            for jc in range(optimal_cond_windows):
                c0, c1 = cond_e[optimal_c_partitions[jc]], cond_e[optimal_c_partitions[jc+1]-1]
                class _W: pass
                wv = _W(); wv.start_energy, wv.end_energy = v0, v1
                wc = _W(); wc.start_energy, wc.end_energy = c0, c1
                n_gl = round(N_tau_window(wc, wv, epsq))
                # Compute HGL nodes (using the same z_lm formula)
                E_bw = c1 - v0
                E_gap_pair = c0 - v1
                z_lm = np.sqrt(1.0 / (E_bw * E_gap_pair)) if E_bw > 0 and E_gap_pair > 0 else 1.0
                from .hgl_quadrature import n_tau_hgl
                n_hgl = n_tau_hgl(z_lm, E_bw, epsq)
                window_data.append((iv, jc, n_gl, n_hgl))
    
    # Print quadrature node tables
    col_width = 6
    n_cond = optimal_cond_windows
    n_val = optimal_val_windows
    header_nums = "".join([f"{j+1:>{col_width}}" for j in range(n_cond)])
    
    print("  Quadrature nodes per window pair (val × cond):")
    print("")
    
    # GL table
    print(f"  Gauss-Laguerre (static/non-crossing):")
    print(f"  {'V\\C':<4}{header_nums}")
    print("  " + "-" * (4 + col_width * n_cond))
    for iv in range(n_val):
        row = f"  {iv+1:<4}"
        for jc in range(n_cond):
            for (v, c, n_gl, n_hgl) in window_data:
                if v == iv and c == jc:
                    row += f"{n_gl:>{col_width}}"
                    break
        print(row)
    print("")
    
    # HGL table
    print(f"  Hermite-Gauss-Laguerre (dynamic/crossing):")
    print(f"  {'V\\C':<4}{header_nums}")
    print("  " + "-" * (4 + col_width * n_cond))
    for iv in range(n_val):
        row = f"  {iv+1:<4}"
        for jc in range(n_cond):
            for (v, c, n_gl, n_hgl) in window_data:
                if v == iv and c == jc:
                    row += f"{n_hgl:>{col_width}}"
                    break
        print(row)
    
    print("=" * 72)
    print("")

    return cost_matrix, window_bounds

def get_window_info(epsq, wfn, cond_bounds=None, nband_max=None):
    """
    Calculate window information for valence and conduction bands.

    Args:
        epsq (float): Epsilon squared value.
        wfn (WFNReader): Wavefunction reader object.
        cond_bounds (tuple[float,float]|None): Optional (emin, emax) for conduction energies to restrict to active band range.
        nband_max (int|None): If provided, restrict energies to the first nband_max bands.

    Returns:
        list: A list of WindowPair objects containing window information.
    """
    # Use minimize_cost_fn to find the optimal number of windows
    cost_matrix, window_bounds = minimize_cost_fn(wfn, epsq, max_val_windows=8, max_cond_windows=8, cond_bounds=cond_bounds, nband_max=nband_max)

    # Find the optimal configuration
    min_cost_idx = np.unravel_index(np.argmin(cost_matrix, axis=None), cost_matrix.shape)
    optimal_val_windows, optimal_cond_windows = min_cost_idx[0] + 1, min_cost_idx[1] + 1
    optimal_v_partitions = window_bounds[(optimal_val_windows, optimal_cond_windows)][0]
    optimal_c_partitions = window_bounds[(optimal_val_windows, optimal_cond_windows)][1]

    # select energy source, optionally restrict to first nband_max bands
    if nband_max is not None:
        energies_src = wfn.energies[:, :, :nband_max]
    else:
        energies_src = wfn.energies

    # flatten & sort all band energies
    all_e = energies_src.flatten()
    sorted_e = np.sort(all_e)
    # get Fermi level
    efermi = np.amax(wfn.energies[:, :, int(np.sum(wfn.occs[0,0]) - 1)])

    # split into valence (<= efermi) and conduction (> efermi)
    val_e = sorted_e[sorted_e <= efermi]
    cond_e = sorted_e[sorted_e >  efermi]
    if cond_bounds is not None:
        cmin, cmax = cond_bounds
        cond_e = cond_e[(cond_e >= cmin) & (cond_e <= cmax)]

    # Create a list to hold window pairs
    window_pairs = []

    # Fill valence and conduction window information and create pairs
    for iv in range(optimal_val_windows):
        v0, v1 = val_e[optimal_v_partitions[iv]], val_e[optimal_v_partitions[iv+1]-1]
        val_window = WindowInfo(v0, v1, index=iv, count=optimal_val_windows)
        for jc in range(optimal_cond_windows):
            c0, c1 = cond_e[optimal_c_partitions[jc]], cond_e[optimal_c_partitions[jc+1]-1]
            cond_window = WindowInfo(c0, c1, index=jc, count=optimal_cond_windows)
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
    # right bandrange indices (HOMO-nv .. nband)
    r_start, r_end = brange_r  # r_end == nband

    # min energy at lower conduction index across all k-points
    cmin_bound = float(np.min(wfn.energies[0, :, r_start]))
    # max energy at the highest included band (nband-1) across all k-points
    cmax_bound = float(np.max(wfn.energies[0, :, r_end - 1]))

    window_pairs = get_window_info(0.01, wfn, cond_bounds=(cmin_bound, cmax_bound))

    # Print table of window information
    print("\nWindow Information:")
    print(f"{'Pair':<10}{'Valence Emin':<15}{'Valence Emax':<15}{'Cond Emin':<15}{'Cond Emax':<15}{'z_lm':<10}{'tau_i':<20}{'w_i':<20}")
    print("-" * 120)

    for i, pair in enumerate(window_pairs, start=1):
        val_window = pair.val_window
        cond_window = pair.cond_window
        print(f"{i:<10}{val_window.start_energy:<15.4f}{val_window.end_energy:<15.4f}{cond_window.start_energy:<15.4f}{cond_window.end_energy:<15.4f}{pair.z_lm:<10.4f}{pair.tau_i[:5]}...{pair.tau_i[-5:]}{pair.w_i[:5]}...{pair.w_i[-5:]}")

    # Continue with the rest of the main code... 
