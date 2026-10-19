"""Bare-Coulomb G-sphere in the WFN.h5 PADDED layout.

The radius condition ``|q + G|² ≤ bare_coulomb_cutoff`` moved to the
``vcoul`` service on 2026-08-07 (:func:`vcoul.bare_coulomb_sphere_mask`).
What stays here is the LAYOUT, and the split is the point:

* the PREDICATE — which G are in q's sphere — is Coulomb arithmetic
  through the same ``bvec`` every kernel uses.  It belongs to vcoul, and
  three other callers there want it without any padding at all.
* the PADDED SENTINEL LAYOUT — ``ngkmax``, the Miller-``(nx/2, ny/2,
  nz/2)`` pad row, the flat-FFT index arithmetic — is a property of the
  FFT box, shared with ψ, and lives in :mod:`common.gvec_fft_box`.  A
  service that knew about it would be a service that knew about our
  on-disk format.

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

import numpy as np

from common.gvec_fft_box import pad_gvecs_to_sentinel
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from vcoul import bare_coulomb_sphere_mask                  # noqa: E402


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

        NEVER READ, and kept anyway: this parameter has no effect on the
        returned tables and never did (the docstring above describes a
        0-D contract the body does not implement).  The service-side
        predicate :func:`vcoul.bare_coulomb_sphere_mask` does NOT have
        it — a parameter that changes nothing forces every caller to
        decide what to pass.  It survives HERE because two in-tree call
        sites pass it by keyword and this is a compatibility surface,
        not a new one; deleting it is the replumb's business.

    Returns
    -------
    dict with keys:

    ``sphere_idx_padded`` : ``(n_q_ibz, ngkmax)`` int32
        Flat-FFT indices into ``[0, nx·ny·nz)`` (fftfreq order).
        Padded with the pad sentinel's flat slot.  Used by the writer to
        gather coeffs from ``FFT(ζ_q)`` after the chunk FFT.  Derived
        FROM ``gvec_components_padded`` (one wrap-and-flatten), so the
        two tables cannot describe different G.
    ``gvec_components_padded`` : ``(n_q_ibz, 3, ngkmax)`` int32
        Miller indices for each q's G-list.  Layout mirrors WFN.h5's
        ``mf_header/gspace/components`` (3, ng) with an added leading
        q axis.  Padded with the shared FFT-box pad sentinel
        (:func:`common.gvec_fft_box.fft_box_pad_sentinel`).
    ``ngk_per_q`` : ``(n_q_ibz,)`` int32
        Per-q logical sphere size — pad slots start at index ``ngk[q]``
        along the trailing axis of both arrays above.
    ``ngkmax`` : int
        ``max_q ngk[q]``.
    ``vcoul_cutoff_ry`` : float
        Echoed cutoff (for the writer to stash in the on-disk header).
    """
    del sys_dim                       # never read; see the parameter note
    nx, ny, nz = (int(s) for s in fft_grid)

    # THE PREDICATE, from the service.  ``mask[q, r]`` is
    # ``|q + G_r|² ≤ vcoul_cutoff_ry``; ``G_int[r]`` is the Miller index at
    # flat-FFT index ``r`` (fftfreq order, C-order meshgrid).
    mask, G_int = bare_coulomb_sphere_mask(
        (nx, ny, nz), bvec, q_irr_frac, vcoul_cutoff_ry)
    n_q_ibz = int(mask.shape[0])

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

    # Per-q G-lists, ascending in flat-FFT index.  Ascending order is why
    # ``sphere_idx_padded[q, 0] == 0`` (G=(0,0,0) is flat index 0 and is
    # always inside, checked above) — the only ordering property any
    # downstream consumer relies on.  ``np.nonzero`` already returns
    # ascending indices, so no sort is needed.
    idx_per_q = [np.nonzero(mask[q])[0].astype(np.int32)
                 for q in range(n_q_ibz)]

    # THE shared padded-layout step — identical call to the one
    # ``WfnLoader.gvecs`` makes on the ragged on-disk ψ G-lists.  It fills
    # pad slots with the FFT-box pad sentinel and REFUSES if any sphere
    # reaches the box's Nyquist corner (which would make "sentinel row"
    # stop meaning "pad row" for every consumer of this table).
    gvecs_padded, ngk_per_q = pad_gvecs_to_sentinel(
        [G_int[i] for i in idx_per_q], (nx, ny, nz), ngkmax=ngkmax)

    # On-disk components layout is (n_q, 3, ngkmax) — WFN.h5's
    # ``(3, ng)`` with a leading q axis.
    gvec_components_padded = np.ascontiguousarray(
        gvecs_padded.transpose(0, 2, 1))

    # Flat-FFT indices DERIVED from the components table rather than
    # accumulated alongside it.  ``G_int[r]`` wraps back to flat index
    # ``r`` by construction (fftfreq order + C-order meshgrid), and the
    # sentinel row wraps to the sentinel slot — so this reproduces the
    # per-q sphere indices exactly while making it impossible for the
    # two on-disk tables to describe different G.
    _wrapped = gvecs_padded.astype(np.int64) % np.asarray(
        (nx, ny, nz), dtype=np.int64)[None, None, :]
    sphere_idx_padded = (
        _wrapped[..., 0] * (ny * nz) + _wrapped[..., 1] * nz
        + _wrapped[..., 2]).astype(np.int32)                    # (n_q, ngkmax)

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
