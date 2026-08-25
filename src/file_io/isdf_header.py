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
import hashlib
from pathlib import Path

import h5py as h5
import numpy as np


_GROUP = 'isdf_header'

#: Existing restart/ISDF centroid-table content identity, now named once.
#: The bytes are deliberately unchanged from the historical
#: ``gw.gw_init._centroid_table_md5`` contract: int64, C order, bare MD5
#: hexadecimal digest.  The scheme name lets compound provenance receipts
#: state which established artifact identity they embedded without inventing
#: another hash of the same table.
CENTROID_TABLE_FINGERPRINT_SCHEME = 'int64-c-order-md5-v1'


def centroid_table_md5(centroid_fft_idx) -> str:
    """Return the canonical restart/ISDF centroid-table content digest.

    This is the one spelling of the ``centroids_{charge,transverse}_md5``
    root attributes on restart tensor files.  Hash FFT-grid INDICES, not a
    fractional-coordinate text file: two text representations that snap to
    the same grid points describe the same basis.
    """
    # ``centroid_fft_idx`` is a small replicated table on production paths,
    # but can already be a JAX array.  Preserve the historical explicit
    # device_get without importing JAX when this format helper is merely read.
    try:
        import jax
        values = jax.device_get(centroid_fft_idx)
    except ImportError:  # pragma: no cover - h5py-only inspection installs
        values = centroid_fft_idx
    table = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    if table.ndim != 2 or table.shape[1] != 3:
        raise ValueError(
            "centroid_table_md5 requires FFT-grid indices with shape "
            f"(n_rmu, 3); got {table.shape}")
    return hashlib.md5(table.tobytes()).hexdigest()


@dataclass(frozen=True)
class WavefunctionBasisReceipt:
    """Immutable identity of ψ sampled at one ordered centroid table.

    This is host/static provenance, never a JAX array.  In particular,
    ``layout`` is absent: legacy four-copy and low-memory two-face carriers
    built from the same sampled ψ compare equal.  ``source_identity`` is the
    canonical full-Bloch transform convention; the WFN content fingerprint
    alone cannot distinguish two real-space transform gauges.

    ``n_rmu_padded`` describes the live in-memory carrier and may therefore
    change when a restart is read on another processor count.  The remaining
    fields describe the physical source/basis and are process-layout
    independent.
    """

    role: str
    wfn_fingerprint_scheme: str
    wfn_fingerprint: str
    band_interval: tuple[int, int]
    fft_grid: tuple[int, int, int]
    centroid_fingerprint_scheme: str
    centroid_table_md5: str
    n_rmu_logical: int
    n_rmu_padded: int
    source_identity: str

    def __post_init__(self) -> None:
        from common.parallel_transport import WFN_FINGERPRINT_SCHEME
        from common.wfn_transforms import FULL_BLOCH_TRANSFORM_SCHEME

        if str(self.role) not in ('charge', 'transverse'):
            raise ValueError(
                "WavefunctionBasisReceipt.role must be 'charge' or "
                f"'transverse'; got {self.role!r}")
        if self.wfn_fingerprint_scheme != WFN_FINGERPRINT_SCHEME:
            raise ValueError(
                "WavefunctionBasisReceipt requires the canonical WFN "
                f"fingerprint scheme {WFN_FINGERPRINT_SCHEME!r}; got "
                f"{self.wfn_fingerprint_scheme!r}")
        fingerprint = str(self.wfn_fingerprint)
        if (len(fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in fingerprint)):
            raise ValueError(
                "WavefunctionBasisReceipt.wfn_fingerprint must be a "
                "64-digit lowercase hexadecimal SHA-256")
        start, stop = (int(v) for v in self.band_interval)
        if start < 0 or stop <= start:
            raise ValueError(
                "WavefunctionBasisReceipt.band_interval must satisfy "
                f"0 <= start < stop; got [{start},{stop})")
        grid = tuple(int(v) for v in self.fft_grid)
        if len(grid) != 3 or any(v <= 0 for v in grid):
            raise ValueError(
                "WavefunctionBasisReceipt.fft_grid must contain three "
                f"positive integers; got {self.fft_grid!r}")
        if (str(self.centroid_fingerprint_scheme)
                != CENTROID_TABLE_FINGERPRINT_SCHEME):
            raise ValueError(
                "WavefunctionBasisReceipt requires the canonical centroid "
                f"fingerprint scheme {CENTROID_TABLE_FINGERPRINT_SCHEME!r}; "
                f"got {self.centroid_fingerprint_scheme!r}")
        centroid_md5 = str(self.centroid_table_md5)
        if (len(centroid_md5) != 32
                or any(c not in "0123456789abcdef" for c in centroid_md5)):
            raise ValueError(
                "WavefunctionBasisReceipt.centroid_table_md5 must be the "
                "canonical 32-digit lowercase MD5 stamp")
        logical, padded = int(self.n_rmu_logical), int(self.n_rmu_padded)
        if logical <= 0 or padded < logical:
            raise ValueError(
                "WavefunctionBasisReceipt centroid extents must satisfy "
                f"0 < logical <= padded; got {logical}/{padded}")
        if self.source_identity != FULL_BLOCH_TRANSFORM_SCHEME:
            raise ValueError(
                "WavefunctionBasisReceipt requires the canonical full-Bloch "
                f"source identity {FULL_BLOCH_TRANSFORM_SCHEME!r}; got "
                f"{self.source_identity!r}")
        # Frozen is meaningful only when every nested value is immutable.
        # Canonicalize numpy scalars/list-like intervals at the boundary so a
        # caller cannot mutate a list behind an otherwise frozen receipt.
        object.__setattr__(self, 'role', str(self.role))
        object.__setattr__(
            self, 'wfn_fingerprint_scheme',
            str(self.wfn_fingerprint_scheme))
        object.__setattr__(self, 'wfn_fingerprint', fingerprint)
        object.__setattr__(self, 'band_interval', (start, stop))
        object.__setattr__(self, 'fft_grid', grid)
        object.__setattr__(
            self, 'centroid_fingerprint_scheme',
            str(self.centroid_fingerprint_scheme))
        object.__setattr__(self, 'centroid_table_md5', centroid_md5)
        object.__setattr__(self, 'n_rmu_logical', logical)
        object.__setattr__(self, 'n_rmu_padded', padded)
        object.__setattr__(self, 'source_identity', str(self.source_identity))

    @classmethod
    def from_source(
        cls,
        *,
        wfn,
        role: str,
        band_interval,
        fft_grid,
        centroid_fft_idx,
        n_rmu_logical: int,
        n_rmu_padded: int,
    ) -> 'WavefunctionBasisReceipt':
        """Build the one receipt from canonical WFN/transform/hash owners."""
        from common.parallel_transport import (
            WFN_FINGERPRINT_SCHEME, wfn_fingerprint)
        from common.wfn_transforms import FULL_BLOCH_TRANSFORM_SCHEME

        start, stop = (int(v) for v in band_interval)
        nbands = getattr(wfn, 'nbands', None)
        if nbands is not None and stop > int(nbands):
            raise ValueError(
                "WavefunctionBasisReceipt band interval exceeds the source "
                f"WFN: stop={stop}, WFN.nbands={int(nbands)}")
        grid = tuple(int(v) for v in np.asarray(fft_grid).reshape(3))
        import jax
        centroids = np.ascontiguousarray(np.asarray(
            jax.device_get(centroid_fft_idx), dtype=np.int64))
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError(
                "WavefunctionBasisReceipt centroid table must have shape "
                f"(n_rmu,3); got {centroids.shape}")
        logical = int(n_rmu_logical)
        if int(centroids.shape[0]) != logical:
            raise ValueError(
                "WavefunctionBasisReceipt logical centroid extent differs "
                f"from the exact table: {logical} vs {centroids.shape[0]}")
        grid_array = np.asarray(grid, dtype=np.int64)
        if (np.any(centroids < 0)
                or np.any(centroids >= grid_array[None, :])):
            raise ValueError(
                "WavefunctionBasisReceipt centroid indices must lie inside "
                f"fft_grid={grid}")
        return cls(
            role=str(role),
            wfn_fingerprint_scheme=WFN_FINGERPRINT_SCHEME,
            wfn_fingerprint=wfn_fingerprint(wfn),
            band_interval=(start, stop),
            fft_grid=grid,
            centroid_fingerprint_scheme=(
                CENTROID_TABLE_FINGERPRINT_SCHEME),
            centroid_table_md5=centroid_table_md5(centroids),
            n_rmu_logical=logical,
            n_rmu_padded=int(n_rmu_padded),
            source_identity=FULL_BLOCH_TRANSFORM_SCHEME,
        )

    def assert_matches_source(
        self,
        *,
        wfn,
        role: str,
        band_interval,
        fft_grid,
        centroid_fft_idx,
        n_rmu_logical: int,
        n_rmu_padded: int,
        where: str,
    ) -> None:
        """Refuse unless current source inputs reproduce this exact receipt."""
        expected = type(self).from_source(
            wfn=wfn, role=role, band_interval=band_interval,
            fft_grid=fft_grid,
            centroid_fft_idx=centroid_fft_idx,
            n_rmu_logical=n_rmu_logical, n_rmu_padded=n_rmu_padded)
        if self == expected:
            return
        names = tuple(self.__dataclass_fields__)
        differing = [name for name in names
                     if getattr(self, name) != getattr(expected, name)]
        raise ValueError(
            f"{where}: supplied WavefunctionBasisReceipt disagrees with "
            f"the canonical source in fields {differing}")

    def assert_same_source(
        self, other: 'WavefunctionBasisReceipt', *, where: str,
    ) -> None:
        """Refuse unless two receipts name one physical sampled basis.

        The processor-dependent padded extent is deliberately excluded from
        this cross-runtime comparison.  :meth:`assert_same_carrier` adds it
        back for arrays that coexist in one runtime.
        """
        if not isinstance(other, WavefunctionBasisReceipt):
            raise TypeError(
                f"{where}: expected WavefunctionBasisReceipt, got "
                f"{type(other).__name__}")
        fields = (
            'role', 'wfn_fingerprint_scheme', 'wfn_fingerprint',
            'band_interval', 'fft_grid', 'centroid_fingerprint_scheme',
            'centroid_table_md5',
            'n_rmu_logical', 'source_identity')
        differing = [name for name in fields
                     if getattr(self, name) != getattr(other, name)]
        if differing:
            raise ValueError(
                f"{where}: wavefunction-at-centroids receipts differ in "
                f"physical source fields {differing}")

    def assert_same_carrier(
        self, other: 'WavefunctionBasisReceipt', *, where: str,
    ) -> None:
        """Refuse unless physical source and current padded extent agree."""
        self.assert_same_source(other, where=where)
        if int(self.n_rmu_padded) != int(other.n_rmu_padded):
            raise ValueError(
                f"{where}: receipt padded centroid extents differ: "
                f"{int(self.n_rmu_padded)} vs {int(other.n_rmu_padded)}")


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
                                 # is ``zeta_q`` shape (n_q_disk, n_rtot, n_rmu).
                                 # NO READER SINCE 2026-08-07 — the consumer's
                                 # forward FFT + sphere gather was deleted with
                                 # the zeta_loader extraction (no writer here
                                 # emits it).  ``ZetaLoader`` still reads the
                                 # HEADER of such a file; its ζ block refuses.
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
    'CENTROID_TABLE_FINGERPRINT_SCHEME',
    'centroid_table_md5',
    'IsdfHeader',
    'WavefunctionBasisReceipt',
    'read_isdf_header',
    'read_isdf_header_from_file',
    'bind_isdf_attrs',
    'write_isdf_header',
    'mark_zeta_done',
    'stamp_fit_provenance',
]
