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
from typing import NamedTuple
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


#: Broadening of the rotating patches (eV): empty states above the protected
#: cut; the P-R coupling needs Sigma_io(E_o) only to modest accuracy
#: (CLASSMIX: Si conduction endpoints at 1 eV keep 0.7-0.8 meV). The owner
#: approved 1-2 eV for these reads (ruling Q4, 2026-09-28; INVARIANTS 12).
FAR_PATCH_ETA_EV = 1.0
#: Semicore patches sample at the deck eta (SEMICORE_ETA_EV = None) on narrow
#: patches: every semicore endpoint, diagonal and coupling, reads at eta.
#: At eta_semi = 1 eV the P-S couplings put the MoS2 3x3 fixed point 8.5 meV
#: off, at 0.5 eV 4.8 meV; protected at eta, 0.14 meV (PARTITION round 2).
SEMICORE_ETA_EV = 0.5
#: How semicore endpoints are read: "patch" (own energy on the SEMICORE_ETA_EV
#: patches) or "sigma0" (Sigma(omega = 0) on the near grid, main's rule for
#: states below E_F - 15 eV; no semicore crossing windows).
SEMICORE_READ = "patch"
#: How states above the protected cut are read (owner scheme, round 5):
#: "scissor" (every one takes E_DFT + beta_above, scissor law A; no far patch,
#: no own-energy read) or "own" (own-energy reads on held far patches, ruling
#: Q3).
ROTATING_READ = "own"
#: Rule tolerance of the far-patch crossing windows. A far window's node
#: count is set by its short side over eta (the patch top above the lowest
#: state), not by its pole range, so splitting cannot shorten it; the coupling
#: needs only percent accuracy (CLASSMIX round 4: 158 -> 90 nodes at 1e-2).
FAR_PATCH_EPS = 1.0e-2
#: Far-patch sampling step (eV): eta/2 resolves the broadened Sigma.
FAR_PATCH_STEP_EV = 0.5
#: The protected cut takes the first all-k gap at least CUT_GAP_ETAS * eta
#: wide above the requested top, searched up to CUT_SEARCH_EV (else the widest
#: gap there): a narrow cut gap leaves protected states within ~eta of
#: rotating ones (Si 4^3: 0.41 eV gap, 7.9 meV at map 0; 1.8 eV gap, 0.25).
CUT_GAP_ETAS = 4.0
CUT_SEARCH_EV = 5.0
#: A global gap wider than this splits semicore from the valence manifold:
#: twice the padded near-support halfwidth (outer pad + Z stencil), so a
#: narrower gap would be sampled by the near support anyway.
SEMICORE_GAP_EV = 2.0 * (SUPPORT_PAD_EV + 0.5)
#: Offset of a far patch's first (last) sample past the near support's top
#: (bottom) edge (eV): the two grids stay strictly ascending and join with
#: no uncovered sliver.
FAR_PATCH_EDGE_EV = 1.0e-3


def derived_pad_ev(e_dft_rel_ev, e_probe_rel_ev, mask_kn):
    """Pad of a class's patches: max |E_map0 - E_DFT| over the class + SUPPORT_PAD_EV.

    ``e_probe_rel_ev`` stacks one or more (nk, nb) map-0 estimates. Without a
    probe (the provisional map-0 plan) the pad is SUPPORT_PAD_EV.
    """
    m = np.asarray(mask_kn, bool)
    if e_probe_rel_ev is None or not m.any():
        return float(SUPPORT_PAD_EV)
    e0 = np.asarray(e_dft_rel_ev, float)
    probe = np.asarray(e_probe_rel_ev, float).reshape((-1,) + e0.shape)
    shift = np.max(np.abs(probe - e0[None])[:, m])
    return float(shift) + float(SUPPORT_PAD_EV)


def far_patches_ev(energy_rel_ev, mask_kn, near_support_ev, *, pad_ev):
    """Contiguous patches covering every masked energy outside the near support.

    ``energy_rel_ev`` (rows, nb) is about the Sigma frame's E_F (probe rows may
    be stacked below the DFT rows). Energies are padded by ``pad_ev``, merged
    where padded intervals overlap or a hole is shorter than 2 pad_ev, clipped
    against the near support and snapped outward to FAR_PATCH_STEP_EV.
    Returns ((lo, hi), ...) ascending; empty when no masked state lies outside.
    """
    e = np.asarray(energy_rel_ev, float)[np.asarray(mask_kn, bool)]
    lo_near, hi_near = float(near_support_ev[0]), float(near_support_ev[1])
    e = np.sort(e[(e < lo_near) | (e > hi_near)])
    if e.size == 0:
        return ()
    step, pad = FAR_PATCH_STEP_EV, float(pad_ev)
    breaks = np.nonzero(np.diff(e) > 2.0 * pad)[0]
    starts = np.concatenate(([0], breaks + 1)); stops = np.concatenate((breaks, [e.size - 1]))
    out = []
    for a, b in zip(starts, stops):
        lo, hi = e[a] - pad, e[b] + pad
        lo, hi = float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)
        # A patch that reaches the near support starts FAR_PATCH_EDGE_EV past
        # its edge, so no energy falls in a sliver between the two.
        if hi > hi_near and lo <= hi_near:
            lo = hi_near + FAR_PATCH_EDGE_EV
        if lo < lo_near and hi >= lo_near:
            hi = lo_near - FAR_PATCH_EDGE_EV
        if out and lo - out[-1][1] <= 2.0 * pad:      # a short hole costs more than it saves
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return tuple(out)


def far_patch_grid_ev(patch, *, step_ev=None):
    """Samples of one (lo, hi, eta) patch at eta/2 (or ``step_ev``), ending exactly at hi."""
    lo, hi, eta = float(patch[0]), float(patch[1]), float(patch[2])
    step = 0.5 * eta if step_ev is None else float(step_ev)
    n = int(np.ceil((hi - lo) / step - 1e-9)) + 1
    return np.linspace(lo, hi, max(n, 2))      # ends at hi: never enters the near grid


def far_patch_covered(energy_rel_ev, patches):
    """Mask of energies inside some patch (inclusive), same frame as the patches."""
    e = np.asarray(energy_rel_ev, float)
    covered = np.zeros(e.shape, bool)
    for patch in patches:
        covered |= (e >= float(patch[0])) & (e <= float(patch[1]))
    return covered


class SigmaPlan(NamedTuple):
    """The one Sigma plan of an SC run, made at map 0 and held; eV about the Sigma frame.

    ``grid_ev`` samples the protected (near) support at the deck eta;
    ``protected_support_ev`` is its [lo, hi], the 2 eV outer pad and the Z
    stencil included. ``far_patches_ev``/``far_eta_ev`` are the rotating
    patches (empty states above the protected cut), ``semicore_ev``/
    ``semicore_eta_ev`` the semicore patches; each patch pad is derived
    (``derived_pad_ev``). The W sampling ladder reads this one object
    (``SCSupport.plan``; the SC session's ``"sigma_plan"``).
    """
    grid_ev: np.ndarray
    envelope_ev: tuple
    near_eta_ev: float
    far_patches_ev: tuple
    far_eta_ev: tuple
    semicore_ev: tuple = ()
    semicore_eta_ev: tuple = ()
    far_pad_ev: float = SUPPORT_PAD_EV
    semicore_pad_ev: float = SUPPORT_PAD_EV

    @property
    def protected_support_ev(self):
        return (float(self.grid_ev[0]), float(self.grid_ev[-1]))

    @property
    def patches(self):
        """Every held patch as (lo, hi, eta), ascending."""
        rows = ([(a, b, e) for (a, b), e in zip(self.far_patches_ev, self.far_eta_ev)]
                + [(a, b, e) for (a, b), e in zip(self.semicore_ev, self.semicore_eta_ev)])
        return tuple(sorted(rows))


def plan_sigma_windows(sigma, energy_rel_ev, protected_kn, *, rotating_energy_rel_ev=None,
                       rotating_kn=None, semicore_kn=None, far_pad_ev=SUPPORT_PAD_EV,
                       semicore_pad_ev=SUPPORT_PAD_EV, outer_pad_ev=SUPPORT_PAD_EV):
    """THE one Sigma plan: protected support, rotating and semicore patches, in one call.

    ``energy_rel_ev``/``protected_kn`` set the near support
    (:func:`plan_support_ev`); ``rotating_energy_rel_ev`` with
    ``rotating_kn`` / ``semicore_kn`` (None: a route without patches) set the
    rotating and semicore patches (:func:`far_patches_ev`). Extra rows (map-0
    probe estimates) may be stacked below either energy array with the masks
    tiled to match.
    """
    grid, envelope = plan_support_ev(sigma, energy_rel_ev, protected_kn,
                                     outer_pad_ev=outer_pad_ev)
    near = (float(grid[0]), float(grid[-1]))
    far = (() if rotating_kn is None else
           far_patches_ev(rotating_energy_rel_ev, rotating_kn, near, pad_ev=far_pad_ev))
    semi = (() if semicore_kn is None else
            far_patches_ev(rotating_energy_rel_ev, semicore_kn, near, pad_ev=semicore_pad_ev))
    if semi:
        # ONE coarse semicore window (owner, round 5): from the deepest semicore
        # energy minus its pad up to just above the highest semicore band.
        semi = ((semi[0][0], semi[-1][1]),)
    # A semicore patch never overlaps a rotating one (they sit on opposite
    # sides of E_F); refuse rather than merge silently if they ever do.
    rows = sorted([(a, b) for a, b in far] + [(a, b) for a, b in semi])
    if any(r[0] <= q[1] for q, r in zip(rows[:-1], rows[1:])):
        raise ValueError("GATE sigma_far_patch_order: rotating and semicore patches overlap")
    eta = float(sigma.regularization_ev)
    eta_semi = eta if SEMICORE_ETA_EV is None else float(SEMICORE_ETA_EV)
    return SigmaPlan(grid, envelope, eta, tuple(far),
                     tuple(FAR_PATCH_ETA_EV for _ in far), tuple(semi),
                     tuple(eta_semi for _ in semi), float(far_pad_ev),
                     float(semicore_pad_ev))
