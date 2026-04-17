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

    def find_q_index(self, q_frac, tol: float = 1e-4, sym_matrices=None) -> tuple[int, np.ndarray | None]:
        """Return the index of the q-point matching q_frac.

        BGW deduplicates q-points by symmetry, so LORRAX q-points that are
        symmetry-equivalent to a stored q-point need to be matched via the
        crystal symmetry.

        Parameters
        ----------
        q_frac : (3,) fractional coords
        tol : match tolerance
        sym_matrices : optional (nsym, 3, 3) array of crystal symmetry
            matrices acting on fractional q.  If None, only direct and
            BZ-shifted matches are tried.

        Returns
        -------
        (iq, Sym) : index into table, and the symmetry matrix such that
            S @ q_frac ≡ q_table[iq] (mod BZ).  Sym is None if no symmetry
            was needed (direct match).
        """
        q = np.asarray(q_frac, dtype=np.float64)

        def _match_mod_bz(q_try):
            qw = np.mod(q_try + 0.5, 1.0) - 0.5
            for i, qf in enumerate(self.q_fracs):
                qfw = np.mod(qf + 0.5, 1.0) - 0.5
                if np.all(np.abs(qw - qfw) < tol):
                    return i
            return -1

        # 1. direct match
        idx = _match_mod_bz(q)
        if idx >= 0:
            return idx, None

        # 2. time reversal (q → -q)
        idx = _match_mod_bz(-q)
        if idx >= 0:
            return idx, -np.eye(3)

        # 3. crystal symmetries.
        # BGW's mtrx acts on real-space fractional vectors (preserves adot).
        # For reciprocal-space q the corresponding operator is S.T
        # (preserves bdot in LORRAX convention).
        if sym_matrices is not None:
            for S in np.asarray(sym_matrices):
                S_recip = S.T
                q_sym = S_recip @ q
                idx = _match_mod_bz(q_sym)
                if idx >= 0:
                    return idx, S_recip

        raise ValueError(f"No BGW q-point matches {q_frac} (after TR + symmetry search)")


def read_bgw_vcoul(path: str) -> BGWVcoulTable:
    """Parse a BGW vcoul text file.

    Returns a BGWVcoulTable with one entry per *unique* q-point (BGW's
    Sigma writes each q multiple times when sigma is evaluated at
    multiple k-points).  q-points that appear more than once must carry
    identical (G, vcoul) pairs; repeated blocks are deduplicated.
    """
    arr = np.loadtxt(path)
    q_all = arr[:, 0:3]
    G_all = arr[:, 3:6].astype(np.int32)
    v_all = arr[:, 6]

    # Group by unique q-points (wrap to [0,1) for comparison)
    q_wrapped = np.mod(q_all, 1.0)
    # Round to 8 decimals so "identical" q's with float noise group together
    q_key = np.round(q_wrapped * 1e8).astype(np.int64)

    q_fracs = []
    G_miller_per_q = []
    vcoul_per_q = []
    seen_keys = {}  # q_key_tuple -> index in output lists

    for i in range(arr.shape[0]):
        key = tuple(q_key[i])
        if key not in seen_keys:
            seen_keys[key] = len(q_fracs)
            q_fracs.append(q_wrapped[i].copy())
            G_miller_per_q.append([G_all[i]])
            vcoul_per_q.append([v_all[i]])
        else:
            # Duplicate q-block; skip (BGW re-emits the same q, G, vcoul)
            pass

    # After the first unique pass we only captured one entry per q — but
    # the full block follows contiguously. Walk again and append.
    q_fracs = []
    G_miller_per_q = []
    vcoul_per_q = []
    seen_keys = {}
    i = 0
    while i < arr.shape[0]:
        key = tuple(q_key[i])
        if key not in seen_keys:
            G_buf = []
            v_buf = []
            while i < arr.shape[0] and tuple(q_key[i]) == key:
                G_buf.append(G_all[i])
                v_buf.append(v_all[i])
                i += 1
            seen_keys[key] = len(q_fracs)
            q_fracs.append(np.asarray(key, dtype=np.float64) / 1e8)
            G_miller_per_q.append(np.asarray(G_buf, dtype=np.int32))
            vcoul_per_q.append(np.asarray(v_buf, dtype=np.float64))
        else:
            # Duplicate q block (same key already captured); skip it
            while i < arr.shape[0] and tuple(q_key[i]) == key:
                i += 1

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
    sym_matrices=None,
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

    iq, S = table.find_q_index(q_frac, tol=tol, sym_matrices=sym_matrices)
    G_miller = table.G_miller_per_q[iq]
    vcoul = table.vcoul_per_q[iq]

    # If we matched via a symmetry S (so S·q_input ≡ q_table[iq]), we need
    # to map G-vectors in the LORRAX q_input frame back via S^{-1}:
    # v(q+G) at input frame = v(S·q + S·G) at table frame, so for each
    # G_miller stored under the table q we look up, the corresponding
    # input-frame G is S^{-1} · G_table.  Since sym ops on q are integer
    # reciprocal-lattice matrices, S and S^{-1} preserve the integer grid.
    if S is not None:
        S_inv = np.linalg.inv(np.asarray(S, dtype=np.float64))
        G_input = np.rint(G_miller.astype(float) @ S_inv.T).astype(np.int32)
    else:
        G_input = G_miller

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
