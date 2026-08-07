"""``isdf_header`` — ζ-specific metadata group attached to ``zeta_q.h5``.

The header is intentionally small: only the irreducible content goes on
disk.  Everything that can be derived from the file's ``mf_header``
(crystal, k-grid, symmetry, FFT grid, ρ-cutoff) is **not** stored — the
reader rebuilds those tables on the fly via
:class:`symmetry_maps.SymMaps` and the
:mod:`symmetry_maps` orbit helpers.

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
- ``zeta_is_done`` (scalar bool, dataset shape ()):  ``True`` once the
  whole ``zeta_q`` dataset has been written.  The writer initially
  writes ``False`` alongside ``mf_header`` / centroid tables, and
  flips it to ``True`` via :func:`mark_zeta_done` after the last
  chunk's H5Dwrite has drained.  Restart paths check this flag to
  decide whether to reuse the on-disk ζ or refit.  Legacy files that
  lack the field are treated as ``True`` (they pre-date the flag and
  were always written atomically at end-of-fit).
- ``zeta_layout`` (scalar str): ``'r_space'`` (legacy) or ``'G_flat'``.
  In G-flat mode the writer produces an extra metadata block —
  ``ngkmax``, ``ngk`` (per-q logical sphere size), ``gvec_components``
  (per-q Miller indices padded with the FFT-box pad sentinel —
  :func:`common.gvec_fft_box.fft_box_pad_sentinel`, the Miller index
  sitting in the Nyquist-corner cell ``(nx//2, ny//2, nz//2)``)
  and ``zeta_cutoff_ry``.  These mirror the WFN.h5 ``wfns``
  group layout with the ragged G-axis replaced by a fixed
  ``ngkmax``-padded axis.
- ``fit_provenance`` (scalar str, optional): JSON description of the
  INPUTS the fit consumed — band windows, cutoffs, solver knobs, source
  WFN identity.  Written by :func:`stamp_fit_provenance` AFTER
  :func:`mark_zeta_done`.  ``zeta_is_done`` says "this file is
  complete"; ``fit_provenance`` says "…and here is what it is complete
  FOR".  Both are required for ``gw.gw_init`` to reuse a ζ instead of
  refitting; a legacy file missing either is refit.
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
    zeta_is_done: bool = True    # ``False`` between writer header-write and the
                                 # final ``mark_zeta_done`` call.  Defaulted
                                 # ``True`` for in-memory construction so
                                 # synthetic test headers don't need to set it.
    zeta_layout: str = 'r_space'  # 'r_space' | 'G_flat'.
                                 # 'r_space' (legacy default): on-disk dataset
                                 # is ``zeta_q`` shape (n_q_disk, n_rtot, n_rmu),
                                 # consumer applies forward FFT + sphere gather
                                 # via ``ZetaLoader.load(layout='G_flat',
                                 # qvec_frac=, sphere_idx=)``.
                                 # 'G_flat' (Phase C of zeta migration): on-disk
                                 # dataset is ``zeta_q_G`` shape (n_q_disk,
                                 # n_rmu, ngkmax), already FFT'd + sphere-
                                 # gathered by the writer with the per-q
                                 # ``|q+G|² ≤ zeta_cutoff_ry`` sphere
                                 # padded to a uniform ``ngkmax`` (WFN.h5
                                 # ``wfns`` layout, non-ragged).  Pad slots
                                 # carry zero coeffs and sentinel components.
                                 # Legacy files that lack this field default
                                 # to 'r_space'.

    # G-flat metadata (None when ``zeta_layout == 'r_space'``).
    gvec_components: np.ndarray | None = None
                                 # (n_q_disk, 3, ngkmax) int32 — per-q Miller
                                 # indices in the ``wfns/gvecs`` style.  Pad
                                 # slots carry the FFT-box pad sentinel,
                                 # ``common.gvec_fft_box.fft_box_pad_sentinel``
                                 # (``(-nx/2, -ny/2, -nz/2)`` on EVEN extents;
                                 # ``+(n-1)/2`` on odd ones — MEASURED on the
                                 # production 36x36x135 MoS2 ζ, whose sentinel
                                 # is ``(-18, -18, +67)``.  Read it off
                                 # ``fftfreq``; do not write ``-n//2``).
    ngk_per_q: np.ndarray | None = None
                                 # (n_q_disk,) int32 — per-q logical sphere
                                 # size.  ``zeta_q_G[q, :, ngk[q]:]`` is zero
                                 # by construction.
    zeta_cutoff_ry: float | None = None
                                 # Bare Coulomb cutoff used to build the per-q
                                 # sphere.  Stashed so a consumer that wants
                                 # to verify or rebuild the sphere can do so
                                 # without trusting the components table.

    fit_provenance: str | None = None
                                 # JSON blob describing the INPUTS this ζ was
                                 # fit from (band windows, cutoffs, solver
                                 # knobs, source WFN identity, ...).  Written
                                 # by :func:`stamp_fit_provenance` after
                                 # ``mark_zeta_done``; consumed by
                                 # ``gw.gw_init``'s ζ-reuse check, which
                                 # refits unless this matches the current run
                                 # EXACTLY.  ``None`` on legacy files => no
                                 # reuse (refit), which is the safe direction.

    @property
    def n_rmu(self) -> int:
        return int(self.r_mu_fft_idx.shape[0])

    @property
    def ngkmax(self) -> int | None:
        if self.gvec_components is None:
            return None
        return int(self.gvec_components.shape[-1])

    @classmethod
    def build(
        cls,
        *,
        r_mu_fft_idx: np.ndarray,
        fft_grid: np.ndarray | tuple[int, int, int],
        density: str,
        vertex_mu_L: int,
        zeta_is_done: bool = False,
        zeta_layout: str = 'r_space',
        gvec_components: np.ndarray | None = None,
        ngk_per_q: np.ndarray | None = None,
        zeta_cutoff_ry: float | None = None,
        fit_provenance: str | None = None,
    ) -> 'IsdfHeader':
        """Build a header from centroid FFT-grid indices.

        ``r_mu_crystal`` is derived as ``r_mu_fft_idx / fft_grid``.
        ``zeta_is_done`` defaults ``False`` for the writer path — the
        writer flips it to ``True`` via :func:`mark_zeta_done` after
        the final chunk is on disk.  ``zeta_layout`` defaults to
        ``'r_space'`` (the legacy on-disk format); the
        ``accumulate_rchunk_to_gflat`` writer path sets it to
        ``'G_flat'``.  In G-flat mode, ``gvec_components`` /
        ``ngk_per_q`` / ``zeta_cutoff_ry`` are required.
        """
        if zeta_layout not in ('r_space', 'G_flat'):
            raise ValueError(
                f"zeta_layout must be 'r_space' or 'G_flat'; got "
                f"{zeta_layout!r}")
        idx = np.asarray(r_mu_fft_idx, dtype=np.int32)
        if idx.ndim != 2 or idx.shape[1] != 3:
            raise ValueError(
                f"r_mu_fft_idx must be (n_rmu, 3); got {idx.shape}")
        fg = np.asarray(fft_grid, dtype=np.float64).reshape(3)
        crystal = idx.astype(np.float64) / fg[None, :]

        # G-flat metadata coercion / validation.
        gv = None
        nk = None
        cutoff = None
        if zeta_layout == 'G_flat':
            if gvec_components is None or ngk_per_q is None \
                    or zeta_cutoff_ry is None:
                raise ValueError(
                    "zeta_layout='G_flat' requires gvec_components, "
                    "ngk_per_q, and zeta_cutoff_ry.")
            gv = np.asarray(gvec_components, dtype=np.int32)
            nk = np.asarray(ngk_per_q, dtype=np.int32)
            cutoff = float(zeta_cutoff_ry)
            if gv.ndim != 3 or gv.shape[1] != 3:
                raise ValueError(
                    f"gvec_components must be (n_q, 3, ngkmax); got "
                    f"{gv.shape}")
            if nk.shape != (gv.shape[0],):
                raise ValueError(
                    f"ngk_per_q must be (n_q,)={gv.shape[0]}; got "
                    f"{nk.shape}")
            if int(nk.max()) > int(gv.shape[-1]):
                raise ValueError(
                    f"max(ngk_per_q)={int(nk.max())} > ngkmax="
                    f"{int(gv.shape[-1])}.")

        return cls(
            density=str(density),
            vertex_mu_L=int(vertex_mu_L),
            r_mu_fft_idx=idx,
            r_mu_crystal=crystal,
            zeta_is_done=bool(zeta_is_done),
            zeta_layout=str(zeta_layout),
            gvec_components=gv,
            ngk_per_q=nk,
            zeta_cutoff_ry=cutoff,
            fit_provenance=(None if fit_provenance is None
                            else str(fit_provenance)),
        )


# ---------------------------------------------------------------------------
# Attribute binder (drop-in isdf_header surface for reader objects)
# ---------------------------------------------------------------------------

def bind_isdf_attrs(obj: object, isdf: IsdfHeader) -> None:
    """Mirror the :class:`IsdfHeader` surface onto ``obj`` as attributes.

    Both zeta readers expose the same isdf attribute surface so callers
    can treat them interchangeably.  This is not a straight field copy:
    it applies the readers' scalar coercions (``int``/``bool``/``str``)
    and renames the ``ngkmax`` property to ``ngkmax_zeta`` (to avoid
    colliding with the WFN's own ``ngkmax``).  ``n_rmu`` and
    ``ngkmax_zeta`` come from int-returning properties, so the coercions
    are idempotent — this binder is value-identical to both readers'
    former inline blocks.
    """
    obj.density = isdf.density
    obj.vertex_mu_L = isdf.vertex_mu_L
    obj.r_mu_fft_idx = isdf.r_mu_fft_idx
    obj.r_mu_crystal = isdf.r_mu_crystal
    obj.n_rmu = int(isdf.n_rmu)
    obj.zeta_is_done = bool(isdf.zeta_is_done)
    obj.zeta_layout = str(isdf.zeta_layout)   # 'r_space' | 'G_flat'
    # G-flat metadata surface (None for r-space files).
    obj.gvec_components = isdf.gvec_components
    obj.ngk_per_q = isdf.ngk_per_q
    obj.zeta_cutoff_ry = isdf.zeta_cutoff_ry
    obj.fit_provenance = isdf.fit_provenance
    obj.ngkmax_zeta = isdf.ngkmax   # WFN.h5-style padded G-axis size


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _read_group(f: h5.File) -> IsdfHeader:
    g = f[_GROUP]
    # Legacy files predate the ``zeta_is_done`` field; treat as ``True``
    # (they were always written atomically at end-of-fit).  Legacy files
    # also predate ``zeta_layout``; treat as ``'r_space'``.  New files
    # carry both fields explicitly.
    zeta_done = (bool(g['zeta_is_done'][()]) if 'zeta_is_done' in g
                 else True)
    zeta_layout = (_decode_str(g['zeta_layout'][()]) if 'zeta_layout' in g
                   else 'r_space')
    # G-flat metadata (only present when zeta_layout == 'G_flat').
    gv = (np.asarray(g['gvec_components'][:], dtype=np.int32)
          if 'gvec_components' in g else None)
    nk = (np.asarray(g['ngk'][:], dtype=np.int32)
          if 'ngk' in g else None)
    cutoff = (float(g['zeta_cutoff_ry'][()])
              if 'zeta_cutoff_ry' in g else None)
    prov = (_decode_str(g['fit_provenance'][()])
            if 'fit_provenance' in g else None)
    return IsdfHeader(
        density=_decode_str(g['density'][()]),
        vertex_mu_L=int(g['vertex_mu_L'][()]),
        r_mu_fft_idx=np.asarray(g['centroids/r_mu_fft_idx'][:], dtype=np.int32),
        r_mu_crystal=np.asarray(g['centroids/r_mu_crystal'][:], dtype=np.float64),
        zeta_is_done=zeta_done,
        zeta_layout=zeta_layout,
        gvec_components=gv,
        ngk_per_q=nk,
        zeta_cutoff_ry=cutoff,
        fit_provenance=prov,
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
        g.create_dataset('zeta_is_done', data=np.bool_(header.zeta_is_done))
        g.create_dataset('zeta_layout', data=np.bytes_(header.zeta_layout))
        c = g.create_group('centroids')
        c.create_dataset('r_mu_fft_idx', data=header.r_mu_fft_idx)
        c.create_dataset('r_mu_crystal', data=header.r_mu_crystal)
        # G-flat metadata — only written when present.
        if header.gvec_components is not None:
            g.create_dataset('gvec_components', data=header.gvec_components)
        if header.ngk_per_q is not None:
            g.create_dataset('ngk', data=header.ngk_per_q)
        if header.zeta_cutoff_ry is not None:
            g.create_dataset(
                'zeta_cutoff_ry',
                data=np.float64(header.zeta_cutoff_ry))
        if header.fit_provenance is not None:
            g.create_dataset('fit_provenance',
                             data=np.bytes_(header.fit_provenance))


def mark_zeta_done(path: str | Path) -> None:
    """Flip ``isdf_header/zeta_is_done`` to ``True`` (idempotent).

    Called by the writer after the final ζ chunk has drained to disk
    so future readers / restart paths can trust the file is complete.
    Idempotent: a missing dataset is created; an existing one is
    overwritten in place.
    """
    with h5.File(str(path), 'a') as f:
        if _GROUP not in f:
            raise ValueError(
                f"mark_zeta_done: {path} has no '{_GROUP}' group")
        g = f[_GROUP]
        if 'zeta_is_done' in g:
            g['zeta_is_done'][...] = np.bool_(True)
        else:
            g.create_dataset('zeta_is_done', data=np.bool_(True))


def stamp_fit_provenance(path: str | Path, provenance: str) -> None:
    """Write ``isdf_header/fit_provenance`` (idempotent, overwrite in place).

    Called by the ζ driver AFTER :func:`mark_zeta_done`, so a run killed
    between the two leaves a complete-but-unstamped file — which the reuse
    check treats as "cannot verify" and refits.  That ordering is
    deliberate: every failure mode falls back to recomputing, never to
    reusing an unverified ζ.

    ``provenance`` is a JSON string; see ``gw.gw_init._zeta_fit_provenance``
    for the schema and ``_zeta_reuse_ok`` for the comparison.
    """
    with h5.File(str(path), 'a') as f:
        if _GROUP not in f:
            raise ValueError(
                f"stamp_fit_provenance: {path} has no '{_GROUP}' group")
        g = f[_GROUP]
        if 'fit_provenance' in g:
            del g['fit_provenance']
        g.create_dataset('fit_provenance', data=np.bytes_(str(provenance)))


__all__ = [
    'IsdfHeader',
    'read_isdf_header',
    'read_isdf_header_from_file',
    'bind_isdf_attrs',
    'write_isdf_header',
    'mark_zeta_done',
    'stamp_fit_provenance',
]
