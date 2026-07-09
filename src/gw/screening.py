"""Screening-frequency planner + executor.

Each Σ scheme declares the (ω, role) pairs at which W must be available;
a generic :func:`compute_screening` turns that declaration into a
``{role → W_q}`` dict that the Σ build consumes.  Decouples *which* W's
are needed from *how* Σ is built so adding a new self-energy scheme is
a two-line change:

    1. Extend :func:`screening_requests_for` with the (ω, role) tuples
       the new scheme needs.
    2. Add a dispatch case in :func:`gw.sigma_dispatch.compute_sigma_xc`
       that reads ``W_by_role[role]`` and produces a
       :class:`~gw.sigma_dispatch.SigmaResult`.

No retrofitting of the SC iteration map or main driver is required.

Conventional roles
------------------
``"static"``  W(ω = 0).  Universal — every screened scheme needs it.
``"probe"``   Probe-frequency W for two-point fits:

              * GN-PPM:  ω = i · ω_p   (imag axis)
              * HL-PPM:  Ω = ω_p       (real axis, above all transitions)

Future schemes will introduce their own role labels (e.g.
``"imag_<i>"`` for full-imaginary-axis CD, or ``"real_grid"`` for direct
real-axis Σ on a frequency mesh).  Roles are symbolic — the Σ builder
looks them up by string, so no enum/registry retrofit is needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax
from jax.sharding import NamedSharding, PartitionSpec as P

from common import jax_profile
import common.timing as timing
from .gw_config import ComputeMode


# ---------------------------------------------------------------------------
# Request type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreeningRequest:
    """One W evaluation a Σ scheme needs.

    Attributes
    ----------
    omega_ry : complex
        Frequency at which W must be available, in Rydberg.  Pure real
        (``omega_ry.imag == 0``) lands on the real axis;
        pure imaginary on the Matsubara/Wick-rotated axis.  Both branches
        are supported by :func:`compute_screening` via the existing
        ``build_real_quadrature`` / ``build_imag_quadrature`` helpers in
        :mod:`gw.w_isdf`.
    role : str
        Symbolic label the Σ builder uses to look up this W in the
        screening output dict.  See module docstring for conventions.
    """

    omega_ry: complex
    role: str


# ---------------------------------------------------------------------------
# Per-scheme declarations
# ---------------------------------------------------------------------------

def screening_requests_for(
    mode: ComputeMode,
    config,
) -> list[ScreeningRequest]:
    """Single source of truth for which W's each Σ scheme needs.

    Returns an empty list for unscreened schemes (``X_ONLY``); a single
    static request for COHSEX; static + probe for the PPM schemes.

    To add a new scheme, extend the dispatch here AND add a
    corresponding case to ``compute_sigma_xc`` that reads the W's by
    role label.
    """
    if mode is ComputeMode.X_ONLY:
        return []  # bare exchange — no screening
    static = ScreeningRequest(omega_ry=0.0 + 0.0j, role="static")
    if mode is ComputeMode.COHSEX:
        return [static]
    if mode is ComputeMode.GN_PPM:
        return [static, ScreeningRequest(
            omega_ry=1j * float(config.ppm.omega_p), role="probe")]
    if mode is ComputeMode.HL_PPM:
        return [static, ScreeningRequest(
            omega_ry=complex(float(config.ppm.omega_p), 0.0), role="probe")]
    raise ValueError(
        f"screening_requests_for: unknown compute mode {mode!r}")


# ---------------------------------------------------------------------------
# Static W with the IBZ fast path
# ---------------------------------------------------------------------------

def compute_static_w(
    wfns,
    V_q: jax.Array,
    quad,
    *,
    e_ref: float,
    sym,
    centroid_indices,
    config,
    meta,
    mesh_xy,
) -> jax.Array:
    """W(ω=0) = (1 − Vχ₀)⁻¹V on the full BZ, solved on the IBZ wedge.

    IBZ cascade for the Dyson solve: V_q and χ₀_q are sliced to IBZ rows
    before :func:`gw.w_isdf.solve_w` so the per-q Cholesky/LU factor runs
    only on ``n_q_ibz`` blocks; W_q comes out at IBZ shape and is unfolded
    back to the full BZ via the SAME helper V_q uses (same physics — W is
    bilinear in centroids and rotates by centroid double-permute + L-phase
    + TRS conj under sym).  Explicit-full-BZ debug bypass
    (``LORRAX_FORCE_FULL_BZ=1``) matches the V_q gate; IBZ activation
    otherwise depends only on orbit-closure of the centroid set (checked
    downstream in ``_resolve_ibz_q_list``).

    Parameters
    ----------
    wfns
        ``Wavefunctions`` bundle (DFT or current-QP basis).
    V_q : (nq, μ, μ) jax.Array
        Bare Coulomb in flat-q ISDF basis, ``P(None, 'x', 'y')``.
    quad, e_ref
        Static minimax quadrature from ``w_isdf.build_static_quadrature``.
    sym, centroid_indices
        Symmetry tables + ISDF centroid set for the IBZ resolve.
    config, meta, mesh_xy
        Standard driver scaffolding.

    Returns
    -------
    W_q : (nq_full, μ, μ) jax.Array
        Static screened Coulomb on the full BZ, ``P(None, 'x', 'y')``.
    """
    from .w_isdf import (
        compute_chi0,
        precompile_chi0,
        precompile_solve_w,
        solve_w,
    )

    use_ibz_w_requested = not bool(
        int(os.environ.get('LORRAX_FORCE_FULL_BZ', '0')))
    if use_ibz_w_requested and getattr(sym, 'q_irr_full_idx', None) is not None:
        from .v_q_g_flat import _resolve_ibz_q_list
        (_, q_irr_frac, full_to_irr_idx, full_to_irr_sym,
         sym_perm, L_table, use_ibz_w) = _resolve_ibz_q_list(
            sym=sym, centroid_indices=centroid_indices,
            kgrid=tuple(meta.kgrid),
            fft_grid=tuple(meta.fft_grid),
            verbose=False)
    else:
        use_ibz_w = False

    with timing.section("gw_jax.chi0_W"):
        with jax_profile.trace_section("chi0_W"):
            # Split compile vs exec for χ₀ and W.  Each section's
            # wall time is read off the end-of-run timing report
            # under ``gw_jax.chi0_W.{chi,W}.{compile,exec}``.  The
            # explicit ``block_until_ready`` inside the exec sections
            # is load-bearing: it (a) pins chi.exec / W.exec wall time
            # to the actual dispatched compute (not just the host
            # dispatch), and (b) drops the last Python reference to
            # χ₀ before the W-solve call so XLA can donate that
            # buffer.  Do NOT use ``_chi_sec.watch(...)`` here — it
            # keeps a bound ``block_until_ready`` method alive on the
            # section object past W-solve, which blocks donation.
            with timing.section("chi.compile"):
                precompile_chi0(wfns, quad, meta, mesh_xy,
                                energy_reference=e_ref)
            with timing.section("chi.exec"):
                chi0_q = compute_chi0(wfns, quad, meta, mesh_xy,
                                      energy_reference=e_ref)
                chi0_q.block_until_ready()
            # IBZ slice on V_q and χ₀_q.  Both retain the canonical
            # ``P(None, 'x', 'y')`` sharding; the helper locks it in.
            if use_ibz_w:
                from common.symmetry_maps import slice_q_full_to_ibz
                _nat = NamedSharding(mesh_xy, P(None, 'x', 'y'))
                with timing.section("W.slice_to_ibz"):
                    V_q_solve = slice_q_full_to_ibz(
                        V_q, sym.q_irr_full_idx, out_sharding=_nat)
                    chi0_q_solve = slice_q_full_to_ibz(
                        chi0_q, sym.q_irr_full_idx, out_sharding=_nat)
                    del chi0_q
                    chi0_q_solve.block_until_ready()
            else:
                V_q_solve = V_q
                chi0_q_solve = chi0_q
            with timing.section("W.compile"):
                precompile_solve_w(V_q_solve, chi0_q_solve, meta, mesh_xy,
                                   solver=config.backend.screening_solver)
            with timing.section("W.exec"):
                W_q_solve = solve_w(V_q_solve, chi0_q_solve, meta, mesh_xy,
                              solver=config.backend.screening_solver)
                # χ₀ is donated inside solve_w — the reference is
                # now invalid.  Do NOT touch ``chi0_q_solve`` after this.
                del chi0_q_solve
                W_q_solve.block_until_ready()
            # IBZ → full-BZ unfold (centroid double-permute + L-phase
            # + TRS conj) — same helper V_q uses.  Σ_COH/SX still
            # iterate over the full BZ in the k-q sums.
            if use_ibz_w:
                from common.symmetry_maps import unfold_v_q
                with timing.section("W.unfold_to_full_bz"):
                    n_sym_spatial = int(
                        np.asarray(sym_perm).shape[0]) // 2
                    W_q = unfold_v_q(
                        W_q_solve,
                        irr_idx=full_to_irr_idx,
                        sym_idx=full_to_irr_sym,
                        sym_perm=sym_perm, L_table=L_table,
                        q_irr_frac=q_irr_frac,
                        mesh_xy=mesh_xy,
                        n_sym_spatial=n_sym_spatial)
                    del W_q_solve
                    W_q.block_until_ready()
            else:
                W_q = W_q_solve
    return W_q


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def compute_screening(
    wfns,
    V_q: jax.Array,
    requests: list[ScreeningRequest],
    *,
    quad,                     # static minimax quad from build_static_quadrature
    e_ref: float,
    config,
    meta,
    mesh_xy,
    print_fn: Callable = print,
) -> dict[str, jax.Array]:
    """Evaluate W at each requested frequency.

    Returns ``{role: W_q}``.  The static role uses the prebuilt minimax
    quadrature ``quad``; non-static roles build a single-frequency
    quadrature on the fly using the existing
    :func:`gw.w_isdf.build_imag_quadrature` /
    :func:`gw.w_isdf.build_real_quadrature` helpers (chosen by whether
    ``omega_ry`` is on the imag or real axis).

    The static minimax interval ``[x_min, x_max]`` is reused for both
    branches — both probe-quad builders take the same ``quad`` argument
    as the interval source.

    Caller is responsible for matching roles to its Σ build's
    expectations; an unrequested role lookup is a KeyError.
    """
    from .w_isdf import (
        build_imag_quadrature,
        build_real_quadrature,
        compute_chi0,
        solve_w,
    )

    W_by_role: dict[str, jax.Array] = {}
    for req in requests:
        if req.role == "static":
            quad_used = quad
        else:
            # Pick imag or real axis by which component of ω is non-zero.
            on_imag = abs(req.omega_ry.imag) > 0.0
            on_real = abs(req.omega_ry.real) > 0.0
            if on_imag and on_real:
                raise ValueError(
                    f"compute_screening: complex-axis ω={req.omega_ry!r} "
                    f"not supported — ω must be pure real or pure imag.")
            if on_imag:
                quad_used = build_imag_quadrature(
                    quad, abs(req.omega_ry.imag),
                    config.minimax_config, print_fn=print_fn)
            else:
                quad_used = build_real_quadrature(
                    quad, abs(req.omega_ry.real),
                    config.minimax_config, print_fn=print_fn)

        chi0 = compute_chi0(
            wfns, quad_used, meta, mesh_xy, energy_reference=e_ref)
        chi0.block_until_ready()
        W = solve_w(
            V_q, chi0, meta, mesh_xy,
            solver=config.backend.screening_solver)
        del chi0
        W.block_until_ready()
        W_by_role[req.role] = W

    return W_by_role


__all__ = [
    "ScreeningRequest",
    "screening_requests_for",
    "compute_static_w",
    "compute_screening",
]
