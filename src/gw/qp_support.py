"""Fixed sampled Sigma support, in eV relative to the Sigma chemical potential.

Only protected bands set the near interval: the requested states whose DFT
energy lies within WINDOW_CLIP_EV of mu, plus every state inside the optional
deck endpoints, which only enlarge. The Z stencil and one 2 eV outer pad are
included once at map 0 and the plan is then held. A protected read that
leaves the held support, or a product window that leaves its held box,
refuses by name (``GATE sigma_plan_escape``): no clamp and no rebuild (owner
ruling Q5, 2026-09-28). Rotating endpoints beyond the near
support read far patches; a rotating state no patch covers takes the side
scissor (ruling Q3).
"""
from __future__ import annotations
import numpy as np

SUPPORT_PAD_EV = 2.0
#: Requested states farther than this from mu are not protected (eV). They
#: rotate, and their energies come from the far patches. The owner's
#: protected-window rule (+-10 eV of E_F) and the pair budgets set it.
WINDOW_CLIP_EV = 10.0


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
                    outer_pad_ev=SUPPORT_PAD_EV):
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
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step
    grid = lo + step * np.arange(int(round((hi-lo)/step)) + 1)
    return grid, envelope


def clamped_reads(energy_relative_ev, protected_kn, grid_ev):
    """Mask of protected states whose energy or Z stencil leaves the support."""
    e = np.asarray(energy_relative_ev, float)
    h = read_halfwidth_ev()
    return np.asarray(protected_kn, bool) & ((e-h < grid_ev[0]) | (e+h > grid_ev[-1]))


#: Broadening of the rotating-band far patches (eV), above and below E_F.
#: The P-R coupling needs Sigma_io(E_o) only to modest accuracy and a patch's
#: node count scales as E_bw/eta. Replays (CLASSMIX round 2): Si conduction
#: endpoints at 1 eV keep 0.7-0.8 meV; Fe semicore endpoints match the exact
#: read at 2 eV (1.98 vs 1.95 meV). The owner approved 1-2 eV for far reads
#: (ruling Q4, 2026-09-28; INVARIANTS 12); protected states keep the deck eta.
FAR_PATCH_ETA_EV = 1.0
FAR_PATCH_ETA_BELOW_EV = 2.0
#: Rule tolerance of the far-patch crossing windows. A far window's node
#: count is set by its short side over eta (the patch top above the lowest
#: state), not by its pole range, so splitting cannot shorten it; the coupling
#: needs only percent accuracy (CLASSMIX round 4: 158 -> 90 nodes at 1e-2).
FAR_PATCH_EPS = 1.0e-2
#: Far-patch sampling step (eV): eta/2 resolves the broadened Sigma.
FAR_PATCH_STEP_EV = 0.5
#: Outer pad of each far patch about its rotating DFT energies (eV).
FAR_PATCH_PAD_EV = 2.0
#: Offset of a far patch's first (last) sample past the near support's top
#: (bottom) edge (eV): the two grids stay strictly ascending and join with
#: no uncovered sliver.
FAR_PATCH_EDGE_EV = 1.0e-3


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
        lo, hi = float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)
        # A patch that reaches the near support starts FAR_PATCH_EDGE_EV past
        # its edge, so no rotating energy falls in a sliver between the two.
        if hi > hi_near and lo <= hi_near:
            lo = hi_near + FAR_PATCH_EDGE_EV
        if lo < lo_near and hi >= lo_near:
            hi = lo_near - FAR_PATCH_EDGE_EV
        if out and lo - out[-1][1] <= 2.0 * pad:      # a short hole costs more than it saves
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return tuple(out)


def far_patch_eta_ev(patch):
    """A patch wholly below E_F takes the broader semicore eta."""
    return FAR_PATCH_ETA_BELOW_EV if float(patch[1]) <= 0.0 else FAR_PATCH_ETA_EV


def far_patch_grid_ev(patch):
    lo, hi = float(patch[0]), float(patch[1])
    step = 0.5 * far_patch_eta_ev(patch)
    n = int(np.ceil((hi - lo) / step - 1e-9)) + 1
    return np.linspace(lo, hi, max(n, 2))      # ends at hi: never enters the near grid


def far_patch_covered(energy_rel_ev, patches):
    """Mask of energies inside some far patch (inclusive), same frame as the patches."""
    e = np.asarray(energy_rel_ev, float)
    covered = np.zeros(e.shape, bool)
    for lo, hi in patches:
        covered |= (e >= float(lo)) & (e <= float(hi))
    return covered
