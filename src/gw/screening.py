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

from dataclasses import dataclass
from typing import Callable

import jax

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
    "compute_screening",
]
