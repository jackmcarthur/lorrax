"""Shared helper to build bare-Coulomb G-sphere flat-FFT indices.

One source of truth for the radius condition
``|q + G|² ≤ bare_coulomb_cutoff``:

* :func:`compute_per_q_bare_coulomb_components` — a per-q sphere
  ``{G : |q + G|² ≤ cutoff}`` for every IBZ q, padded uniformly to
  ``ngkmax = max_q ngk[q]`` with the sentinel Miller index
  ``(nx/2, ny/2, nz/2)``.  Returns both the flat-FFT indices (for
  gather/scatter on the FFT box) AND the Miller components (for on-disk
  storage in the WFN.h5 ``gspace/components`` convention).  Used by the
  G-flat ζ writer (``gw.isdf_fitting``) so the on-disk dataset
  ``zeta_q_G(n_q, n_rmu, ngkmax)`` matches the WFN.h5 layout — fixed-size
  G-axis, per-q components written alongside.

The shared sphere is a strict superset of every per-q sphere — for any
G outside the shared sphere, ``|G| > √cutoff + |q_max|_cart`` so
``|q + G| ≥ |G| - |q_max|_cart > √cutoff`` at every q.  This means the
consumer's shared-sphere gather is loss-less for the per-q content the
writer produced; at load time the reader scatters the per-q on-disk
coeffs into the consumer's shared sphere via the components table.

``sys_dim`` matters only for the 0-D box case (no sphere reduction
there); 2-D slab and 3-D bulk share the same construction.
"""
from __future__ import annotations

import itertools

import numpy as np


def compute_per_q_bare_coulomb_components(
    fft_grid,
    bvec: np.ndarray,
    q_irr_frac: np.ndarray,
    vcoul_cutoff_ry: float,
    *,
    sys_dim: int = 3,
) -> dict:
    """Return per-q spheres in WFN.h5-style padded layout.

    For each IBZ q (``q_irr_frac`` row), build the sphere
    ``{G : |q + G|² ≤ vcoul_cutoff_ry}`` over the full FFT grid, then
    pad each q's index list to a uniform ``ngkmax = max_q ngk[q]`` with
    the sentinel Miller index ``(nx/2, ny/2, nz/2)``.  Sentinel slots
    have a valid (non-OOB) flat-FFT index — the corresponding ζ
    coefficient is zeroed by the writer so downstream consumers can
    skip the slot via ``ngk[q]`` or by tagging the components row.

    Parameters
    ----------
    fft_grid
        ``(nx, ny, nz)`` int.
    bvec
        ``(3, 3)`` reciprocal lattice rows (Bohr⁻¹).
    q_irr_frac
        ``(n_q_ibz, 3)`` fractional q-vectors (BGW wrap convention —
        the writer divides BGW-wrapped integer q's by kgrid).
    vcoul_cutoff_ry
        Bare-Coulomb cutoff in Ry.  Caller is expected to set this to
        match the V_q consumer's ``vcoul_cutoff_ry``.
    sys_dim
        0 / 2 / 3.  ``sys_dim == 0`` is not narrowed (no analytic
        sphere reduction for box truncation) — caller falls back to the
        full FFT axis.

    Returns
    -------
    dict with keys:

    ``sphere_idx_padded`` : ``(n_q_ibz, ngkmax)`` int32
        Flat-FFT indices into ``[0, nx·ny·nz)`` (fftfreq order).
        Padded with ``sentinel_flat_idx``.  Used by the writer to
        gather coeffs from ``FFT(ζ_q)`` after the chunk FFT.
    ``gvec_components_padded`` : ``(n_q_ibz, 3, ngkmax)`` int32
        Miller indices for each q's G-list.  Layout mirrors WFN.h5's
        ``mf_header/gspace/components`` (3, ng) with an added leading
        q axis.  Padded with ``(nx/2, ny/2, nz/2)``.
    ``ngk_per_q`` : ``(n_q_ibz,)`` int32
        Per-q logical sphere size — pad slots start at index ``ngk[q]``
        along the trailing axis of both arrays above.
    ``ngkmax`` : int
        ``max_q ngk[q]``.
    ``vcoul_cutoff_ry`` : float
        Echoed cutoff (for the writer to stash in the on-disk header).
    """
    nx, ny, nz = (int(s) for s in fft_grid)
    n_rtot = nx * ny * nz
    bvec_f = np.asarray(bvec, dtype=np.float64)
    q_irr_frac = np.asarray(q_irr_frac, dtype=np.float64).reshape(-1, 3)
    n_q_ibz = int(q_irr_frac.shape[0])

    # Miller indices on the FFT grid (fftfreq order), in floats for the
    # |q+G|² calculation, then int32 for the components table.
    gx_f = (np.fft.fftfreq(nx) * nx).astype(np.float64)
    gy_f = (np.fft.fftfreq(ny) * ny).astype(np.float64)
    gz_f = (np.fft.fftfreq(nz) * nz).astype(np.float64)
    G_frac = np.stack(np.meshgrid(gx_f, gy_f, gz_f, indexing="ij"),
                       axis=-1).reshape(-1, 3)               # (n_rtot, 3)
    G_int = np.rint(G_frac).astype(np.int32)                   # Miller ints

    # Per-q |q + G|² in Cartesian.  Broadcast over the n_rtot G's.
    # qG_cart[q, r, :] = (q + G[r]) @ bvec_f
    qG_frac = q_irr_frac[:, None, :] + G_frac[None, :, :]      # (n_q, n_rtot, 3)
    qG_cart = qG_frac @ bvec_f                                  # (n_q, n_rtot, 3)
    qG_norm2 = np.sum(qG_cart * qG_cart, axis=-1)               # (n_q, n_rtot)
    mask = qG_norm2 <= float(vcoul_cutoff_ry)                   # (n_q, n_rtot)

    # G=(0,0,0) is flat-index 0 and is always inside: |q+0|² = |q|² ≤ |q_max|²
    # which is ≤ vcoul_cutoff_ry for any sane cutoff (≥ ecutwfc ≫ |q_max|²).
    if not bool(np.all(mask[:, 0])):
        bad = np.nonzero(~mask[:, 0])[0]
        raise RuntimeError(
            f"compute_per_q_bare_coulomb_components: G=(0,0,0) not in "
            f"the per-q sphere at q-rows {bad.tolist()}.  Check "
            f"vcoul_cutoff_ry ({vcoul_cutoff_ry} Ry) vs |q_max|².")

    # Per-q sphere sizes; ngkmax = max for the padded array dim.
    ngk_per_q = mask.sum(axis=1).astype(np.int32)               # (n_q,)
    ngkmax = int(ngk_per_q.max())

    # Sentinel flat-FFT index for pad slots: Miller (nx/2, ny/2, nz/2)
    # is a valid in-bounds flat index in fftfreq order — every axis's
    # mid-point lives at flat position ``N//2`` (for even N; for odd N
    # it's ``(N-1)//2``, still in-bounds).
    sentinel_mx, sentinel_my, sentinel_mz = (nx // 2, ny // 2, nz // 2)
    sentinel_flat = sentinel_mx * ny * nz + sentinel_my * nz + sentinel_mz
    # Miller components of the sentinel (taken from the fftfreq tables
    # so a downstream reader that recomputes ``G_frac[sentinel_flat]``
    # gets the same triple).
    sentinel_components = np.array(
        [G_int[sentinel_flat, 0], G_int[sentinel_flat, 1],
         G_int[sentinel_flat, 2]], dtype=np.int32)

    sphere_idx_padded = np.full((n_q_ibz, ngkmax), sentinel_flat,
                                  dtype=np.int32)
    gvec_components_padded = np.broadcast_to(
        sentinel_components[None, :, None],
        (n_q_ibz, 3, ngkmax)).copy()                            # int32 fill

    for q in range(n_q_ibz):
        idx_q = np.nonzero(mask[q])[0].astype(np.int32)         # (ngk[q],)
        # Sort ascending so the (q=0, identity) row matches the
        # natural FFT ordering — downstream consumers don't depend on
        # this beyond "sphere_idx_padded[q, 0] == 0", which holds
        # because G=(0,0,0) is always the smallest index and is inside.
        idx_q.sort()
        nk = int(ngk_per_q[q])
        sphere_idx_padded[q, :nk] = idx_q
        # Components: scatter Miller triples into (3, ngkmax) for this q.
        gvec_components_padded[q, :, :nk] = G_int[idx_q].T       # (3, ngk[q])

    return {
        "sphere_idx_padded": sphere_idx_padded,
        "gvec_components_padded": gvec_components_padded,
        "ngk_per_q": ngk_per_q,
        "ngkmax": ngkmax,
        "vcoul_cutoff_ry": float(vcoul_cutoff_ry),
    }


__all__ = [
    "compute_per_q_bare_coulomb_components",
]
