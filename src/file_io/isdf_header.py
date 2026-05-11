"""``isdf_header`` — ζ-specific metadata group attached to ``zeta_q.h5``.

The header is intentionally small: only the irreducible content goes on
disk.  Everything that can be derived from the file's ``mf_header``
(crystal, k-grid, symmetry, FFT grid, ρ-cutoff) is **not** stored — the
reader rebuilds those tables on the fly via
:class:`common.symmetry_maps.SymMaps` and
:mod:`centroid.orbit_syms`.

What lives in ``isdf_header``
-----------------------------
- ``density`` (scalar str): the ISDF centroid weight that built this
  ζ — ``'scalar'`` (charge), ``'current'`` (Pauli current), or
  ``'unknown'`` for legacy files.  Matches the tag from
  :class:`centroid.centroid_io.CentroidFile`.
- ``vertex_mu_L`` (scalar int): the Lorentz vertex this ζ is for —
  ``0`` (charge γ̃⁰ = I) or ``1, 2, 3`` (transverse γ̃ⁱ = αⁱ).
- ``centroids/r_mu_fft_idx``  (n_rmu, 3) int32: FFT-grid indices of
  the centroid positions.  Primary representation — closure under
  the WFN symmetry group is checked against this table.
- ``centroids/r_mu_crystal``  (n_rmu, 3) float64: fractional coords
  (= ``r_mu_fft_idx / FFTgrid``).  Carried for human-readable
  inspection and for downstream callers that work in fractional
  coords; redundant with ``r_mu_fft_idx``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py as h5
import numpy as np


_GROUP = 'isdf_header'


@dataclass(frozen=True)
class IsdfHeader:
    density: str                 # 'scalar' | 'current' | 'mu_L=<int>' | 'unknown'
    vertex_mu_L: int             # 0 (charge) | 1, 2, 3 (transverse)
    r_mu_fft_idx: np.ndarray     # (n_rmu, 3) int32
    r_mu_crystal: np.ndarray     # (n_rmu, 3) float64

    @property
    def n_rmu(self) -> int:
        return int(self.r_mu_fft_idx.shape[0])

    @classmethod
    def build(
        cls,
        *,
        r_mu_fft_idx: np.ndarray,
        fft_grid: np.ndarray | tuple[int, int, int],
        density: str,
        vertex_mu_L: int,
    ) -> 'IsdfHeader':
        """Build a header from centroid FFT-grid indices.

        ``r_mu_crystal`` is derived as ``r_mu_fft_idx / fft_grid``.
        """
        idx = np.asarray(r_mu_fft_idx, dtype=np.int32)
        if idx.ndim != 2 or idx.shape[1] != 3:
            raise ValueError(
                f"r_mu_fft_idx must be (n_rmu, 3); got {idx.shape}")
        fg = np.asarray(fft_grid, dtype=np.float64).reshape(3)
        crystal = idx.astype(np.float64) / fg[None, :]
        return cls(
            density=str(density),
            vertex_mu_L=int(vertex_mu_L),
            r_mu_fft_idx=idx,
            r_mu_crystal=crystal,
        )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _read_group(f: h5.File) -> IsdfHeader:
    g = f[_GROUP]
    return IsdfHeader(
        density=_decode_str(g['density'][()]),
        vertex_mu_L=int(g['vertex_mu_L'][()]),
        r_mu_fft_idx=np.asarray(g['centroids/r_mu_fft_idx'][:], dtype=np.int32),
        r_mu_crystal=np.asarray(g['centroids/r_mu_crystal'][:], dtype=np.float64),
    )


def _decode_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode('utf-8')
    return str(v)


def read_isdf_header(path: str | Path) -> IsdfHeader:
    """Open ``path`` and return its ``isdf_header`` group."""
    with h5.File(str(path), 'r') as f:
        return _read_group(f)


def read_isdf_header_from_file(f: h5.File) -> IsdfHeader:
    """Same as :func:`read_isdf_header` but operates on an open handle."""
    return _read_group(f)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_isdf_header(
    path: str | Path,
    header: IsdfHeader,
    *,
    mode: str = 'a',
) -> None:
    """Write ``isdf_header`` into ``path``.

    Refuses to overwrite an existing ``isdf_header`` group — the
    caller should delete it explicitly first if a rewrite is needed.
    """
    with h5.File(str(path), mode) as f:
        if _GROUP in f:
            raise ValueError(
                f"write_isdf_header: {path} already has an "
                f"'{_GROUP}' group; refusing to overwrite.")
        g = f.create_group(_GROUP)
        g.create_dataset('density', data=np.bytes_(header.density))
        g.create_dataset('vertex_mu_L', data=np.int32(header.vertex_mu_L))
        c = g.create_group('centroids')
        c.create_dataset('r_mu_fft_idx', data=header.r_mu_fft_idx)
        c.create_dataset('r_mu_crystal', data=header.r_mu_crystal)


__all__ = [
    'IsdfHeader',
    'read_isdf_header',
    'read_isdf_header_from_file',
    'write_isdf_header',
]
