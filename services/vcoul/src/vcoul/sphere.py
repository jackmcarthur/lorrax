"""WHICH G are inside ``|q + G|² ≤ cutoff`` — the predicate, and nothing else.

One source of truth for the bare-Coulomb radius condition.  This module
answers "which G of the FFT box are in q's sphere"; it does NOT lay them
out.  That split is deliberate and it is where the service boundary falls:

* the RADIUS CONDITION is Coulomb physics — ``|q+G|²`` against a cutoff in
  Ry, computed through the same ``bvec`` every other kernel here uses.
  It belongs to vcoul.
* the PADDED SENTINEL LAYOUT (``ngkmax``, the Miller-``(nx/2, ny/2,
  nz/2)`` pad row, the flat-FFT index arithmetic) is a property of the
  FFT box, shared with ψ, and belongs to the ψ/ζ file-format contract —
  ``common.gvec_fft_box`` in lorrax.  A caller that wants the padded
  table calls this function and then pads.

So this module imports nothing but numpy, and the ``sys_dim`` parameter
the old ``compute_per_q_bare_coulomb_components`` carried simply does not
exist here: it was never read in that function's body (its docstring
described a 0-D contract the code did not implement), and a parameter
that changes nothing is worse than no parameter — every caller has to
decide what to pass.

The shared sphere is a strict superset of every per-q sphere — for any
G outside the shared sphere, ``|G| > √cutoff + |q_max|_cart`` so
``|q + G| ≥ |G| - |q_max|_cart > √cutoff`` at every q.
"""
from __future__ import annotations

import numpy as np

__all__ = ["fft_box_miller", "bare_coulomb_sphere_mask",
           "bare_coulomb_sphere_indices"]


def fft_box_miller(fft_grid):
    """Miller indices of every point of the FFT box, in fftfreq order.

    Returns ``(G_frac, G_int)`` — ``(n_rtot, 3)`` float64 and int32, where
    ``n_rtot = nx*ny*nz`` and row ``r`` is the G at flat index ``r``
    (C-order meshgrid over fftfreq axes).  The float copy exists because
    the ``|q+G|²`` arithmetic below needs it and rounding twice would be a
    second convention.
    """
    nx, ny, nz = (int(s) for s in fft_grid)
    gx_f = (np.fft.fftfreq(nx) * nx).astype(np.float64)
    gy_f = (np.fft.fftfreq(ny) * ny).astype(np.float64)
    gz_f = (np.fft.fftfreq(nz) * nz).astype(np.float64)
    G_frac = np.stack(np.meshgrid(gx_f, gy_f, gz_f, indexing="ij"),
                      axis=-1).reshape(-1, 3)                 # (n_rtot, 3)
    G_int = np.rint(G_frac).astype(np.int32)                  # Miller ints
    return G_frac, G_int


def bare_coulomb_sphere_mask(fft_grid, bvec, q_irr_frac, vcoul_cutoff_ry):
    """``(mask, G_int)`` — the predicate, evaluated for every (q, G).

    Parameters
    ----------
    fft_grid
        ``(nx, ny, nz)`` int.
    bvec
        ``(3, 3)`` Cartesian reciprocal lattice ROWS (bohr⁻¹, blat folded
        in).  A :class:`~vcoul.geometry.CoulombGeometry` caller passes
        ``geom.bvec``.
    q_irr_frac
        ``(n_q, 3)`` fractional q-vectors (BGW wrap convention — the
        writer divides BGW-wrapped integer q's by kgrid).
    vcoul_cutoff_ry
        Bare-Coulomb cutoff in Ry.

    Returns
    -------
    mask : ``(n_q, n_rtot)`` bool
        ``|q + G|² ≤ vcoul_cutoff_ry``.
    G_int : ``(n_rtot, 3)`` int32
        Miller indices, flat-FFT order, shared by every q.
    """
    bvec_f = np.asarray(bvec, dtype=np.float64)
    q_irr_frac = np.asarray(q_irr_frac, dtype=np.float64).reshape(-1, 3)
    G_frac, G_int = fft_box_miller(fft_grid)

    # Per-q |q + G|² in Cartesian.  Broadcast over the n_rtot G's.
    # qG_cart[q, r, :] = (q + G[r]) @ bvec_f
    qG_frac = q_irr_frac[:, None, :] + G_frac[None, :, :]      # (n_q, n_rtot, 3)
    qG_cart = qG_frac @ bvec_f                                  # (n_q, n_rtot, 3)
    qG_norm2 = np.sum(qG_cart * qG_cart, axis=-1)               # (n_q, n_rtot)
    mask = qG_norm2 <= float(vcoul_cutoff_ry)                   # (n_q, n_rtot)
    return mask, G_int


def bare_coulomb_sphere_indices(fft_grid, bvec, q_irr_frac, vcoul_cutoff_ry):
    """Per-q sphere ENUMERATION: ascending flat-FFT indices, plus sizes.

    Returns a dict with

    ``G_int`` : ``(n_rtot, 3)`` int32
        Miller indices in flat-FFT order (index into this with the rows of
        ``idx_per_q`` to get a q's G-list).
    ``idx_per_q`` : list of ``(ngk_q,)`` int32
        Flat-FFT indices into ``[0, nx·ny·nz)``, ASCENDING.  Ascending is
        the one ordering property downstream relies on, and it is why
        ``idx_per_q[q][0] == 0`` (G=(0,0,0) is flat index 0 and is always
        inside — refused below if it is not).  ``np.nonzero`` already
        returns ascending indices, so no sort is needed.
    ``ngk_per_q`` : ``(n_q,)`` int32
    ``ngkmax`` : int
        ``max_q ngk[q]`` — what a padded layout would pad to.  Computed
        here because it is a property of the spheres; APPLIED by the
        caller, which owns the layout.

    REFUSAL.  G=(0,0,0) is flat index 0 and must be inside every q's
    sphere: ``|q+0|² = |q|² ≤ |q_max|²``, which is ≤ any sane cutoff
    (≥ ecutwfc ≫ |q_max|²).  A q where it is not means the cutoff is
    wrong, and every consumer downstream assumes ``idx[0] == 0``.
    """
    mask, G_int = bare_coulomb_sphere_mask(
        fft_grid, bvec, q_irr_frac, vcoul_cutoff_ry)
    n_q_ibz = int(mask.shape[0])

    if not bool(np.all(mask[:, 0])):
        bad = np.nonzero(~mask[:, 0])[0]
        raise RuntimeError(
            f"bare_coulomb_sphere_indices: G=(0,0,0) not in "
            f"the per-q sphere at q-rows {bad.tolist()}.  Check "
            f"vcoul_cutoff_ry ({vcoul_cutoff_ry} Ry) vs |q_max|².")

    ngk_per_q = mask.sum(axis=1).astype(np.int32)               # (n_q,)
    ngkmax = int(ngk_per_q.max())
    idx_per_q = [np.nonzero(mask[q])[0].astype(np.int32)
                 for q in range(n_q_ibz)]
    return {
        "G_int": G_int,
        "idx_per_q": idx_per_q,
        "ngk_per_q": ngk_per_q,
        "ngkmax": ngkmax,
    }
