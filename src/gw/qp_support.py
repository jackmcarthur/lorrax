"""Fixed sampled Sigma support, in eV relative to the Sigma chemical potential.

Only requested protected bands set the interval. The optional deck endpoints
can enlarge it. The Z stencil and one 2 eV outer pad are included once;
iterates clamp and only a failed convergence check may rebuild once.
"""
from __future__ import annotations
import numpy as np

SUPPORT_PAD_EV = 2.0


def read_halfwidth_ev():
    from .eqp_bgw import Z_FINITE_DIFFERENCE_EV
    return float(Z_FINITE_DIFFERENCE_EV)


def quasiparticle_mask(z_kn):
    """``Z ∈ (0, 1]`` per state: a quasiparticle.

    The upper bound is read on the build grid (``sigma_box_plan.snap_outward``,
    1e-4 cells): a flat-Σ state sits at Z = 1 to within its finite-difference
    noise, and an exact ``z <= 1`` lets round-off drop it (Fe 4^3 SC: 76 vs 75
    tail samples).
    """
    z = np.asarray(z_kn, dtype=np.float64)
    finite = np.where(np.isfinite(z) & (z > 0.0), z, 0.0)
    z_down = np.floor(finite * 1e4) / 1e4
    return np.isfinite(z) & (z > 0.0) & (z_down <= 1.0)


def requested_states(sigma, frozen_core_bands, energy_relative_ev, required_kn,
                     active_n=None, quasiparticle_kn=None):
    """Protected identities; omega endpoints and previous Z never select bands."""
    required = np.array(np.broadcast_to(np.asarray(required_kn, bool),
                                       np.shape(energy_relative_ev)))
    required[:, :int(frozen_core_bands)] = False
    return required


def plan_support_ev(sigma, deck_grid_ev, energy_relative_ev, requested_kn, plan_index=0):
    """Return the single contiguous support and its unrounded envelope."""
    e = np.asarray(energy_relative_ev, float)
    p = np.broadcast_to(np.asarray(requested_kn, bool), e.shape)
    if not np.any(p) or not np.isfinite(e[p]).all():
        raise ValueError("Sigma support needs finite protected-state energies")
    pad = SUPPORT_PAD_EV + read_halfwidth_ev()
    envelope = (float(e[p].min()) - pad, float(e[p].max()) + pad)
    step = float(sigma.omega_step_ev)
    deck = np.asarray(deck_grid_ev, float)
    lo = np.floor(min(envelope[0], float(deck.min())) / step) * step
    hi = np.ceil(max(envelope[1], float(deck.max())) / step) * step
    grid = lo + step * np.arange(int(round((hi-lo)/step)) + 1)
    return grid, envelope


def clamped_reads(energy_relative_ev, protected_kn, grid_ev):
    """Mask of protected states whose energy or Z stencil leaves the support."""
    e = np.asarray(energy_relative_ev, float)
    h = read_halfwidth_ev()
    return np.asarray(protected_kn, bool) & ((e-h < grid_ev[0]) | (e+h > grid_ev[-1]))


def check_fixed_point(session):
    """Accept the fixed plan, or clear it for its sole convergence rebuild.

    Returns True only on a rebuild. Rule dictionaries are owned by the Sigma
    consumers; the small recursive walk visits sessions, never physical axes.
    """
    if session is None:
        return False
    reasons = []
    def collect(d):
        if d.get("outside_plan"):
            reasons.extend(d["outside_plan"])
        for value in d.values():
            if isinstance(value, dict):
                collect(value)
    collect(session)
    if not reasons:
        return False
    if session.get("convergence_rebuilds", 0):
        raise ValueError("GATE sigma_plan_fixed_point: support failed after its one "
                         "rebuild; enlarge nval/ncond or sigma_omega_min_ev/max_ev. "
                         + "; ".join(reasons))
    session.clear()
    session["convergence_rebuilds"] = 1
    return True
