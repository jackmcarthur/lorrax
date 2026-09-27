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


def requested_states(energy_relative_ev, required_kn):
    """Protected identities; omega endpoints and previous Z never select bands."""
    return np.array(np.broadcast_to(np.asarray(required_kn, bool),
                                    np.shape(energy_relative_ev)))


def plan_support_ev(sigma, energy_relative_ev, requested_kn, *,
                    outer_pad_ev=SUPPORT_PAD_EV, support_floor_ev=()):
    """Return the single contiguous support and its unrounded envelope."""
    e = np.asarray(energy_relative_ev, float)
    p = np.broadcast_to(np.asarray(requested_kn, bool), e.shape)
    if not np.any(p) or not np.isfinite(e[p]).all():
        raise ValueError("Sigma support needs finite protected-state energies")
    pad = float(outer_pad_ev) + read_halfwidth_ev()
    envelope = (float(e[p].min()) - pad, float(e[p].max()) + pad)
    step = float(sigma.omega_step_ev)
    lo, hi = envelope
    if sigma.omega_min_ev is not None:
        lo = min(lo, float(sigma.omega_min_ev))
    if sigma.omega_max_ev is not None:
        hi = max(hi, float(sigma.omega_max_ev))
    # Only a previous physical support is a floor. The config's temporary
    # near-zero grid is not a user request when either endpoint is absent.
    floor = np.asarray(support_floor_ev, float)
    if floor.size:
        if floor.shape != (2,) or not np.isfinite(floor).all() or floor[0] >= floor[1]:
            raise ValueError("Sigma rebuild floor must be a finite ordered interval")
        lo, hi = min(lo, float(floor[0])), max(hi, float(floor[1]))
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step
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
    old_grid = session.get("omega_grid_ev", ())
    floor = () if not len(old_grid) else (old_grid[0], old_grid[-1])
    session.clear()
    session["convergence_rebuilds"] = 1
    session["rebuild_floor_ev"] = floor
    return True


#: Broadening of the rotating-band far patches (eV), above and below E_F.
#: The P-R coupling needs Sigma_io(E_o) only to modest accuracy and a patch's
#: node count scales as E_bw/eta. Replays (CLASSMIX round 2): Si conduction
#: endpoints at 1 eV keep 0.7-0.8 meV; Fe semicore endpoints at 2 eV match
#: the exact read (1.98 vs 1.95 meV) at half the 1 eV price.
FAR_PATCH_ETA_EV = 0.5
FAR_PATCH_ETA_BELOW_EV = 2.0
#: Far-patch sampling step (eV): eta/2 resolves the broadened Sigma.
FAR_PATCH_STEP_EV = 0.5
#: Outer pad of each far patch about its rotating DFT energies (eV).
FAR_PATCH_PAD_EV = 2.0


def far_patches_ev(energy_rel_ev, rotating_kn, near_support_ev):
    """Contiguous far patches covering every rotating DFT energy outside the near support.

    ``energy_rel_ev`` (nk, nb) is about the Sigma frame's E_F. Energies are
    padded by FAR_PATCH_PAD_EV, merged where padded intervals overlap, clipped
    against the near support and snapped outward to FAR_PATCH_STEP_EV.
    Returns ((lo, hi), ...) ascending; empty when no rotating state lies outside.
    """
    e = np.asarray(energy_rel_ev, float)[np.asarray(rotating_kn, bool)]
    lo_near, hi_near = float(near_support_ev[0]), float(near_support_ev[1])
    e = np.sort(e[(e < lo_near) | (e > hi_near)])
    if e.size == 0:
        return ()
    step, pad = FAR_PATCH_STEP_EV, FAR_PATCH_PAD_EV
    breaks = np.nonzero(np.diff(e) > 2.0 * pad)[0]
    starts = np.concatenate(([0], breaks + 1)); stops = np.concatenate((breaks, [e.size - 1]))
    out = []
    for a, b in zip(starts, stops):
        lo, hi = e[a] - pad, e[b] + pad
        if hi > hi_near and lo < hi_near:
            lo = hi_near + step
        if lo < lo_near and hi > lo_near:
            hi = lo_near - step
        lo, hi = float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)
        if out and lo - out[-1][1] <= 2.0 * pad:      # a short hole costs more than it saves
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return tuple(out)


def far_patch_eta_ev(patch):
    """A patch wholly below E_F takes the broad semicore eta."""
    return FAR_PATCH_ETA_BELOW_EV if float(patch[1]) <= 0.0 else FAR_PATCH_ETA_EV


def far_patch_grid_ev(patch):
    lo, hi = float(patch[0]), float(patch[1])
    step = 0.5 * far_patch_eta_ev(patch)
    n = int(np.ceil((hi - lo) / step - 1e-9)) + 1
    return lo + step * np.arange(n)
