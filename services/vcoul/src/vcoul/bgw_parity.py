"""Parser for BerkeleyGW's vcoul file (produced by write_vcoul in sigma.inp or epsilon.inp).

File format: one line per (q, G) entry with
    qx qy qz Gx Gy Gz vcoul
where qx,qy,qz are fractional k-point coords and Gx,Gy,Gz are integer Miller
indices. BGW's vcoul value is ``<8π/|q+G+δq|²>_miniBZ`` (MC-averaged) in
Rydberg units (before multiplication by ``fact = 1/(Nk·Ω)``).

Sigma writes the q walk once per requested outer k-point.  The first walk is
the one used for the first (normally Gamma) Sigma k-point and is the table
consumed here; later walks can use a different little group and repeat or add
q blocks.  The parser therefore stops at the first repeated q coordinate.

NUMPY ONLY.  Path resolution and WFN/grid validation stay in ``gw``; this
service owns only the text grammar and integer q/G mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os

import numpy as np

__all__ = [
    "BGWVcoulTable",
    "fill_v_grid_for_q",
    "fill_v_sphere_for_q",
    "read_bgw_vcoul",
]


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
    source_path: str | None = None
    sha256: str | None = None

    @property
    def n_G(self) -> int:
        """Total number of (q,G) rows in the retained q walk."""
        return sum(int(np.asarray(g).shape[0]) for g in self.G_miller_per_q)

    def q0_vcoul_raw(self) -> float:
        """Return BGW's raw-Ry q=0,G=0 value, before ``1/(Nq*Omega)``."""
        iq, _, _ = self.find_q_index((0.0, 0.0, 0.0), sym_mats_k=None)
        G = np.asarray(self.G_miller_per_q[iq], dtype=np.int32)
        rows = np.nonzero(np.all(G == 0, axis=1))[0]
        if rows.size != 1:
            raise ValueError(
                "BGW vcoul q=0 block must contain exactly one G=(0,0,0) "
                f"row; found {int(rows.size)} in {self.source_path or '<table>'}.")
        return float(np.asarray(self.vcoul_per_q[iq])[int(rows[0])])

    def find_q_index(self, q_frac, tol: float = 1e-4, sym_mats_k=None) -> tuple[int, np.ndarray, np.ndarray]:
        """Find stored q_table symmetry-equivalent to q_frac.

        Uses BGW/LORRAX's `k_full = S · k_bar + kg0` convention (see
        symmetry_maps.SymMaps.get_umklapp_vector).  Here the role of
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


def _q_key(q) -> tuple[int, int, int]:
    return tuple(np.rint(np.mod(np.asarray(q, dtype=np.float64), 1.0) * 1e8)
                 .astype(np.int64).tolist())


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_bgw_vcoul(path: str, *, compute_sha256: bool = True) -> BGWVcoulTable:
    """Parse a BGW vcoul text file.

    Returns a BGWVcoulTable with one entry per IBZ q-point — the q-blocks
    written during sigma's *first* outer-k iteration.  Subsequent outer-k
    iterations re-emit blocks for the same q-list (and may add new
    non-IBZ q's), each computed from a fresh mini-BZ Monte-Carlo draw, so
    redundant blocks for the same q are NOT bit-identical at the few-meV
    level.  Trusting the first block per q is what matches BGW's internal
    behaviour at outer-k = ``rk(1)``.  Later occurrences are repeats of
    the same q emitted on subsequent passes of the writer's outer-k loop;
    only the first block of each q is consumed, and the remaining full-BZ
    q are resolved through the symmetry map rather than from the file.

    We detect the boundary as "first repeated q-coord": once an
    already-seen q reappears, sigma has wrapped to its second outer-k
    iteration and we stop reading.  Non-IBZ q's needed downstream are
    obtained via the BGW symmetry path in
    :meth:`BGWVcoulTable.find_q_index`, not from later blocks.
    """
    source_path = os.path.realpath(os.fspath(path))
    q_fracs, G_miller_per_q, vcoul_per_q = [], [], []
    seen_keys: set[tuple[int, int, int]] = set()
    current_key = None
    current_G: list[tuple[int, int, int]] = []
    current_v: list[float] = []

    def finish_block():
        if current_key is None:
            return
        if not current_G:
            raise ValueError(
                f"BGW vcoul file {source_path} has an empty q block "
                f"for {current_key}.")
        q_fracs.append(np.asarray(current_key, dtype=np.float64) / 1e8)
        G_miller_per_q.append(np.asarray(current_G, dtype=np.int32))
        vcoul_per_q.append(np.asarray(current_v, dtype=np.float64))

    with open(source_path, "rt", encoding="ascii") as handle:
        for line_no, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 7:
                raise ValueError(
                    f"BGW vcoul file {source_path}, line {line_no}: expected "
                    f"7 fields (qx qy qz Gx Gy Gz vcoul), found {len(fields)}.")
            try:
                q = tuple(float(fields[i]) for i in range(3))
                G = tuple(int(fields[i]) for i in range(3, 6))
                value = float(fields[6])
            except ValueError as exc:
                raise ValueError(
                    f"BGW vcoul file {source_path}, line {line_no}: invalid "
                    "q/G/value field.") from exc
            if not np.all(np.isfinite(q)) or not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"BGW vcoul file {source_path}, line {line_no}: q and "
                    "vcoul must be finite and vcoul must be nonnegative.")
            key = _q_key(q)
            if current_key is None:
                current_key = key
                seen_keys.add(key)
            elif key != current_key:
                finish_block()
                if key in seen_keys:
                    break
                seen_keys.add(key)
                current_key = key
                current_G = []
                current_v = []
            current_G.append(G)
            current_v.append(value)
        else:
            finish_block()

    if not q_fracs:
        raise ValueError(f"BGW vcoul file {source_path} contains no complete q block.")

    return BGWVcoulTable(
        q_fracs=np.asarray(q_fracs, dtype=np.float64),
        G_miller_per_q=G_miller_per_q,
        vcoul_per_q=vcoul_per_q,
        source_path=source_path,
        sha256=_sha256_file(source_path) if compute_sha256 else None,
    )


def fill_v_sphere_for_q(
    table: BGWVcoulTable,
    q_frac,
    g_miller,
    cell_volume: float,
    *,
    tol: float = 1e-4,
    sym_mats_k=None,
    require_exact_sphere: bool = True,
) -> np.ndarray:
    """Map one BGW q block onto an explicit LORRAX Miller-G list.

    Unlike :func:`fill_v_grid_for_q`, this strict production form has no
    zero-valued "missing" sentinel.  Every requested G must occur exactly
    once and, by default, the transformed BGW sphere must contain no extra G.
    Returned values are divided by ``cell_volume`` but not by ``Nq``; the
    latter is applied by LORRAX's q summation, matching BGW's later ``fact``.
    """
    requested = np.asarray(g_miller, dtype=np.int32)
    if requested.ndim != 2 or requested.shape[-1] != 3:
        raise ValueError(
            f"g_miller must have shape (n_G,3); got {requested.shape}.")
    iq, S_k, kg0 = table.find_q_index(
        q_frac, tol=tol, sym_mats_k=sym_mats_k)
    stored = np.asarray(table.G_miller_per_q[iq], dtype=np.int32)
    transformed = np.einsum(
        "ij,gj->gi", S_k.astype(np.int32), stored) - kg0[None, :]
    values = np.asarray(table.vcoul_per_q[iq], dtype=np.float64)

    lookup: dict[tuple[int, int, int], float] = {}
    for G, value in zip(transformed, values):
        key = tuple(int(x) for x in G)
        if key in lookup:
            raise ValueError(
                f"BGW vcoul file {table.source_path or '<table>'} has duplicate "
                f"G={key} after q/umklapp mapping for q={tuple(q_frac)}.")
        lookup[key] = float(value)

    requested_keys = [tuple(int(x) for x in G) for G in requested]
    requested_set = set(requested_keys)
    missing = sorted(requested_set.difference(lookup))
    extra = sorted(set(lookup).difference(requested_set))
    if missing or (require_exact_sphere and extra):
        raise ValueError(
            "bgw_metal_vcoul_file G-sphere/cutoff mismatch at "
            f"q={tuple(float(x) for x in q_frac)}: missing {len(missing)} "
            f"G and extra {len(extra)} G; first missing={missing[:3]}, "
            f"first extra={extra[:3]}. Check bare_coulomb_cutoff and the "
            "WFN used by BerkeleyGW.")
    return np.asarray([lookup[key] for key in requested_keys], dtype=np.float64) \
        / float(cell_volume)


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
