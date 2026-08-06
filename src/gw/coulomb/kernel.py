"""The ONE bare-Coulomb formula ``v(q+G)`` — pure arithmetic, no I/O, no MC.

Before 2026-08-05 this formula was written out ~10 times: twice in
``gw/compute_vcoul.py``, twice per dimension in ``gw/coulomb/{bulk_3d,
slab_2d}.py``, twice in ``bse/vq_interp.py`` (host + jitted device body),
and once in ``gw/coulomb/base.py``'s MC kernel.  The copies had drifted:
different guard tolerances, different volume conventions, different
channel coverage.  :func:`v_qG` is the single source they now share.

    v(K) = 8pi / |K|^2  *  T_dim(K)  *  C_channel(K)  *  U_units

    T_bulk = 1                                              (sys_dim 3)
    T_slab = 1 - exp(-zc |K_par|) cos(K_z zc),  zc = pi/b_zz (sys_dim 2)

    C_full = 1
    C_lr   = exp(-|K|^2 / 4 alpha^2)          long-range  (b26p split)
    C_sr   = -expm1(-|K|^2 / 4 alpha^2)       short-range (stable form)

    U_bare = 1        U_per_volume = 1/Omega_cell

``C_sr`` is spelled with ``-expm1`` rather than ``1 - exp`` because the
small-|K| limit is where the SR channel matters and ``1 - exp(-x)`` loses
every significant digit there.  ``C_lr + C_sr == 1`` identically.

WHY 8pi AND NOT 4pi
-------------------
Rydberg atomic units (e^2 = 2), the BerkeleyGW convention: ``v(K) =
8pi/|K|^2`` in Ry-bohr.  ``docs/theory/physics.md`` said 4pi until
2026-08-05; the code has always been 8pi and BGW agreement is anchored to
it (tests/regression/si_cohsex_debug).

THE TWO GUARD TOLERANCES ARE NOT INTERCHANGEABLE
------------------------------------------------
:data:`TOL_QG_ZERO` and :data:`TOL_MC_NAN` guard different things and must
not be collapsed — see their own docstrings below.

WHAT THIS FUNCTION REFUSES TO DO
--------------------------------
It does NOT zero ``v`` at ``G = 0`` when ``q != 0``.  Only the exact
``q = G = 0`` lattice slot (``|K|^2 < zero_tol``) is zeroed; the finite
``G = 0`` term at every other q is part of the body.  Zeroing the whole
G=0 column is the natural-looking tidy-up and it is wrong: measured, it
moves the BSE makeVq-vs-disk residual from ~1e-9 to 0.33
(``bse/vq_interp.py:325-328``).  ``tests/test_coulomb_kernel.py::
test_G0_at_finite_q_is_NOT_zeroed`` is the committed guard.

It does NOT choose a volume convention for you: ``units`` is a required
keyword.  The GW q=0 head keeps bare and divides at injection; the BSE
``eval_vq`` head divides by ``celvol`` to match its stored tile; the
per-sphere builders are per-volume.  A silent default would let two
conventions meet in one expression and off by Omega_cell is invisible in
a plot.
"""
from __future__ import annotations

import numpy as _np

#: Identifies the exact ``q = G = 0`` RECIPROCAL-LATTICE slot, where
#: ``8pi/|K|^2`` is the true Coulomb divergence and the head is injected
#: separately as a rank-1 term.  ``|K|^2`` there is 0 up to the rounding of
#: ``bvec.T @ (q + G)``; 1e-12 (1/bohr^2) sits far below the smallest
#: nonzero ``|q+G|^2`` on any physical grid and far above that rounding.
TOL_QG_ZERO = 1e-12

#: A 0/0 NaN guard on MONTE-CARLO draws, and nothing more.  Mini-BZ
#: samples land at arbitrary ``|K|``, including arbitrarily small ones, and
#: those near-singular samples carry REAL WEIGHT in the estimator — the
#: integrand is integrable and they are where the mass is.  Zeroing them at
#: 1e-12 like a lattice slot would bias the average low.  1e-24 is set just
#: high enough to stop a draw that hits the pole exactly from producing inf.
#: Never raise this to :data:`TOL_QG_ZERO`.
TOL_MC_NAN = 1e-24

_UNITS = ("bare", "per_volume")
_CHANNELS = ("full", "lr", "sr")


def v_qG(K, *, axis, sys_dim, units, channel="full", celvol=None,
         alpha=None, zc=None, zero_tol=TOL_QG_ZERO, xp=_np):
    """Bare (optionally slab-truncated, optionally SR/LR-split) ``v(q+G)``.

    Parameters
    ----------
    K : array
        CARTESIAN ``q + G`` in 1/bohr.  Both layouts in the tree are
        supported through ``axis``; nothing is transposed or copied.
    axis : int
        Which axis of ``K`` is the 3-vector.  ``0`` for the components-first
        ``(3, nG)`` layout (``gw.compute_vcoul``, ``bse.vq_interp``), ``1``
        (or ``-1``) for the points-first ``(N, 3)`` layout (the MC samplers).
    sys_dim : int
        ``3`` bulk (no truncation) or ``2`` slab (Ismail-Beigi along c).
        ``0`` (Wigner-Seitz box) is NOT here: it is a real-space FFT, a
        different algorithm — see :func:`gw.compute_vcoul_0d.compute_vcoul_box`.
    units : {'bare', 'per_volume'}
        REQUIRED.  ``bare`` = ``8pi[...]/|K|^2``; ``per_volume`` multiplies
        by ``1/celvol``.  No default — see the module docstring.
    channel : {'full', 'lr', 'sr'}
        Coulomb channel.  ``lr``/``sr`` need ``alpha``.
    celvol : float, optional
        Omega_cell (bohr^3).  Required iff ``units == 'per_volume'``.
    alpha : float, optional
        b26p Gaussian split parameter (1/bohr).  Required for lr/sr.
    zc : float, optional
        Slab truncation half-length ``pi/b_zz`` (bohr).  Required for
        ``sys_dim == 2``.
    zero_tol : float
        :data:`TOL_QG_ZERO` on a lattice G-sphere, :data:`TOL_MC_NAN` on MC
        draws.  Entries with ``|K|^2 < zero_tol`` come back exactly 0.
    xp : module
        ``numpy`` (host) or ``jax.numpy`` (traced/jitted).  ONE source line
        serves both; nothing here is jitted or sharded — the caller owns
        that (``bse.vq_interp.make_eval_vq`` keeps its ``out_shardings`` and
        ``with_sharding_constraint`` structure around this call).

    Returns
    -------
    array
        Same shape as ``K`` with ``axis`` removed, same array module as
        ``K``, float64.

    Notes
    -----
    The order of operations is load-bearing, not cosmetic: ``(8pi/|K|^2)``
    then truncation then channel then volume, with the volume as
    ``* (1/celvol)``.  That spelling is byte-identical to
    ``gw.compute_vcoul.compute_v_q_per_G`` and to
    ``gw.coulomb.base._minibz_kernel_bare``, and 2 ULP from
    ``bse.vq_interp.v_slab_on_set`` (which spells the volume ``/ celvol``);
    measured on the Si 4x4x4 production ``(q+G)`` table, Frontera job
    7890613.  Reordering it silently re-tiers every parity gate in
    ``tests/test_coulomb_kernel.py``.
    """
    if units not in _UNITS:
        raise ValueError(
            f"v_qG: units must be one of {_UNITS}; got {units!r}.  The GW "
            f"q=0 head is 'bare'; per-sphere builders are 'per_volume'.")
    if channel not in _CHANNELS:
        raise ValueError(f"v_qG: channel must be one of {_CHANNELS}; "
                         f"got {channel!r}")
    if sys_dim not in (2, 3):
        raise ValueError(
            f"v_qG: sys_dim must be 2 (slab) or 3 (bulk); got {sys_dim!r}.  "
            f"sys_dim=0 is the Wigner-Seitz box FFT "
            f"(gw.compute_vcoul_0d.compute_vcoul_box), not this formula.")
    if units == "per_volume" and celvol is None:
        raise ValueError("v_qG: units='per_volume' requires celvol")
    if channel in ("lr", "sr") and alpha is None:
        raise ValueError(f"v_qG: channel={channel!r} requires alpha")
    if sys_dim == 2 and zc is None:
        raise ValueError("v_qG: sys_dim=2 requires zc (= pi / bvec[2, 2])")

    K2 = xp.sum(K * K, axis=axis)
    zero = K2 < zero_tol
    K2s = xp.where(zero, 1.0, K2)
    v = 8.0 * xp.pi / K2s

    if sys_dim == 2:
        if axis == 0:
            kx, ky, kz = K[0], K[1], K[2]
        else:
            kx, ky, kz = K[..., 0], K[..., 1], K[..., 2]
        # Ismail-Beigi slab truncation; sqrt(x^2 + y^2) is bit-identical to
        # linalg.norm on float64 pairs (measured), so both incumbents match.
        f2d = 1.0 - xp.exp(-zc * xp.sqrt(kx ** 2 + ky ** 2)) * xp.cos(kz * zc)
        v = v * f2d

    if channel == "lr":
        # Gaussian factor takes the RAW K2, never the clamped K2s.
        v = v * xp.exp(-K2 / (4.0 * alpha ** 2))
    elif channel == "sr":
        v = v * (-xp.expm1(-K2 / (4.0 * alpha ** 2)))

    if units == "per_volume":
        v = v * (1.0 / float(celvol))

    return xp.where(zero, 0.0, v)


__all__ = ["v_qG", "TOL_QG_ZERO", "TOL_MC_NAN"]
