"""Parser for BerkeleyGW's vcoul file (produced by write_vcoul in sigma.inp or epsilon.inp).

File format: one line per (q, G) entry with
    qx qy qz Gx Gy Gz vcoul
where qx,qy,qz are fractional k-point coords and Gx,Gy,Gz are integer Miller
indices. BGW's vcoul value is ``<8π/|q+G+δq|²>_miniBZ`` (MC-averaged) in
Rydberg units (before multiplication by ``fact = 1/(Nk·Ω)``).

This is useful to bypass LORRAX's approximate G=0-only mini-BZ average in the
3D semiconductor case, when BGW's all-G MC averaging matters (small k-grids).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BGWVcoulTable:
    """BGW vcoul values grouped by q-point.

    Attributes
    ----------
    q_fracs : (nq, 3) fractional coordinates of each q-point in the file order
    G_miller_per_q : list of (nG_q, 3) int arrays — Miller indices per q-point
    vcoul_per_q : list of (nG_q,) float arrays — v(q+G) per q-point
    """
    q_fracs: np.ndarray
    G_miller_per_q: list
    vcoul_per_q: list

    def find_q_index(self, q_frac, tol: float = 1e-4, sym_mats_k=None) -> tuple[int, np.ndarray, np.ndarray]:
        """Find stored q_table symmetry-equivalent to q_frac.

        Uses BGW/LORRAX's `k_full = S · k_bar + kg0` convention (see
        symmetry_maps.SymMaps._get_umklapp_vector).  Here the role of
        "full" is played by q_frac (what LORRAX asks for) and "bar" by
        q_table (what BGW stored).

        Parameters
        ----------
        q_frac : (3,) fractional coords of the requested q-point
        tol : match tolerance
        sym_mats_k : optional (nsym, 3, 3) reciprocal-space symmetry
            matrices (LORRAX's `sym_mats_k`).  If None, only direct
            matches are tried.

        Returns
        -------
        (iq, S_k, kg0) : index into the table, the reciprocal-space
            symmetry matrix, and the integer umklapp vector such that
                q_frac = S_k · q_table[iq] + kg0.
            For a direct match, S_k is the identity and kg0 = q_frac - q_table.
        """
        q = np.asarray(q_frac, dtype=np.float64)
        eye = np.eye(3, dtype=np.int32)

        def _umklapp(q_target, q_source):
            """BGW kg0 convention: q_target = q_source + kg0."""
            diff = q_target - q_source
            diff_int = np.rint(diff)
            if np.all(np.abs(diff - diff_int) < tol):
                return diff_int.astype(np.int32)
            return None

        # 1. Direct match: q_in = q_tbl + kg0
        for i, qf in enumerate(self.q_fracs):
            kg0 = _umklapp(q, qf)
            if kg0 is not None:
                return i, eye, kg0

        # 2. Crystal symmetries: q_in = S_k · q_tbl + kg0
        if sym_mats_k is not None:
            for S_k in np.asarray(sym_mats_k):
                for i, qf in enumerate(self.q_fracs):
                    kg0 = _umklapp(q, S_k @ qf)
                    if kg0 is not None:
                        return i, np.rint(S_k).astype(np.int32), kg0

        raise ValueError(f"No BGW q-point matches {q_frac} (after symmetry search)")


def read_bgw_vcoul(path: str) -> BGWVcoulTable:
    """Parse a BGW vcoul text file.

    Returns a BGWVcoulTable with one entry per *unique* q-point (BGW
    re-emits each q once per (q, k) combination; repeated contiguous
    blocks for the same q are deduplicated).
    """
    arr = np.loadtxt(path)
    q_all = arr[:, 0:3]
    G_all = arr[:, 3:6].astype(np.int32)
    v_all = arr[:, 6]

    # Round q's to 8 decimals so floating-point noise groups together
    q_key_all = np.round(np.mod(q_all, 1.0) * 1e8).astype(np.int64)

    q_fracs, G_miller_per_q, vcoul_per_q = [], [], []
    seen_keys = {}
    i = 0
    while i < arr.shape[0]:
        key = tuple(q_key_all[i])
        block_end = i
        while block_end < arr.shape[0] and tuple(q_key_all[block_end]) == key:
            block_end += 1
        if key not in seen_keys:
            seen_keys[key] = len(q_fracs)
            q_fracs.append(np.asarray(key, dtype=np.float64) / 1e8)
            G_miller_per_q.append(np.asarray(G_all[i:block_end], dtype=np.int32))
            vcoul_per_q.append(np.asarray(v_all[i:block_end], dtype=np.float64))
        i = block_end

    return BGWVcoulTable(
        q_fracs=np.asarray(q_fracs, dtype=np.float64),
        G_miller_per_q=G_miller_per_q,
        vcoul_per_q=vcoul_per_q,
    )


def fill_v_grid_for_q(
    table: BGWVcoulTable,
    q_frac,
    fft_grid,
    cell_volume: float,
    tol: float = 1e-4,
    sym_mats_k=None,
) -> np.ndarray:
    """Build a single q-point's v(q+G) array on the LORRAX FFT grid.

    BGW's stored vcoul is `8π × <1/|q+G+δq|²>_miniBZ` in Ry units (no
    cell-volume or Nk factor). LORRAX's internal convention is
    `v_scaled = 8π/|q+G|² / cell_volume` — i.e. BGW value divided by
    cell_volume.

    Parameters
    ----------
    table : BGWVcoulTable from read_bgw_vcoul
    q_frac : (3,) fractional crystal coordinates of the q-point (may be
        a q0-shift for head).  Matched to the BGW table via find_q_index.
    fft_grid : (fft_nx, fft_ny, fft_nz) LORRAX FFT grid dimensions
    cell_volume : unit cell volume in Bohr^3 (for the fact=1/Ω scaling)
    tol : tolerance for matching q-points

    Returns
    -------
    v_scaled : (fft_nx, fft_ny, fft_nz) array, in LORRAX's v_scaled units.
        Entries with no BGW value are zero.  G=(0,0,0) is forced to zero
        (the q=0, G=0 head is handled separately via head correction).
    """
    fft_nx, fft_ny, fft_nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    v_scaled = np.zeros((fft_nx, fft_ny, fft_nz), dtype=np.float64)

    iq, S_k, kg0 = table.find_q_index(q_frac, tol=tol, sym_mats_k=sym_mats_k)
    G_miller = table.G_miller_per_q[iq]
    vcoul = table.vcoul_per_q[iq]

    # Map G-vectors using the same integer formula as
    # common/symmetry_maps.py:get_gvecs_kfull (lines 689-690):
    #     G_full = sym_krep @ G_irr - kg0
    # Here q_frac plays the role of k_full, q_table plays k_bar, and kg0
    # satisfies q_frac = S_k @ q_table + kg0.  Both S_k and kg0 are
    # integer-valued so the transform preserves the FFT grid.
    G_input = np.einsum('ij,gj->gi', S_k.astype(np.int32), G_miller) - kg0[None, :]

    # Wrap Miller indices into FFT grid
    gx = np.mod(G_input[:, 0], fft_nx)
    gy = np.mod(G_input[:, 1], fft_ny)
    gz = np.mod(G_input[:, 2], fft_nz)

    v_scaled[gx, gy, gz] = vcoul / float(cell_volume)

    # Zero G=(0,0,0) *only at q=0* — that's the true head, handled
    # separately via the rank-1 head correction.  For q≠0 the G=0 entry
    # is just v(q, G=0)=8π/|q|², a finite body contribution that must be
    # preserved.
    q_arr = np.asarray(q_frac, dtype=np.float64)
    q_wrapped_bz = np.mod(q_arr + 0.5, 1.0) - 0.5
    if np.all(np.abs(q_wrapped_bz) < tol):
        v_scaled[0, 0, 0] = 0.0

    return v_scaled
