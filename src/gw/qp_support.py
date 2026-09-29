"""The sampled Σ(ω) support: the one owner of where the dynamic Σ grid reaches.

Invariant (owner 2026-09-27). The support the one-shot and every SC map sample
stays inside

    D  ∪  [ min_{n∈R} E_in,n − P ,  max_{n∈R} E_in,n + P ]

* ``D`` is the deck's explicit request (``sigma_omega_min_ev`` /
  ``sigma_omega_max_ev``, or the patch list): a fixed interval that never grows.
  Unset edges give the sample next to E_F.
* ``R`` is the requested states: the QP-window identities (nval/ncond) the W
  model treats as active (``shared_pole_recipe.active_band_mask``), outside
  ``sc_frozen_core_bands``, and quasiparticles at the previous map
  (:func:`quasiparticle_mask`).  Under ``clamp``/``static`` only those inside
  the padded requested window (``scissor.sc_padded_window_ev``).
* ``E_in`` is each state's input energy for the map: DFT at the one-shot and SC
  map 0, the carried QP eigenvalue after.  It is never a root: eqp0, eqp1, Z
  and a fixed-point solve do not enter this module.
* ``P`` is flat: :data:`SUPPORT_PAD_EV` = 2 eV at the one plan (the one-shot
  and SC map 0; owner 2026-09-28: one plan, held).  Later maps hold the
  support and move an edge only when a requested state's read support
  [E − h, E + h] (h = :func:`read_halfwidth_ev`, the Z stencil) leaves it, to
  E ± P, the plan's own pad around that state.  The envelope is the union
  over the maps since the plan.

A requested state with Z ∉ (0, 1] at the previous map has no quasiparticle
(UNIFY §2.4: its energy sits within ~η of a pole cluster of Σ_n).  It leaves
``R``: its energy never moves the support, and while it lies off the support
it reads the out-of-grid rule of ``qsgw_utils.sigma_eval_omega`` and is named
in the log.  This is what keeps a runaway state (Na 8^3 b63, Z ≈ −382;
Fe 4^3 k4 b26, Z = 2.82) from growing the grid.

:func:`assert_support_in_envelope` refuses a support that leaves the
invariant; every caller checks the grid it is about to sample.
"""

from __future__ import annotations

import numpy as np

#: Flat support pad of the one plan (one-shot, SC map 0) and of every
#: extension, in eV (owner 2026-09-27: "highest requested state + 2 eV";
#: 2026-09-28: plan once, hold, rebuild by the same rule on an escape).  An
#: extension triggers when a requested state's read support E ± h (h = 0.5 eV)
#: leaves the grid, and moves that edge to E ± 2 eV, the edge the plan would
#: set for that state: 1.5 eV of free motion before it can trigger again.
#: Tempting, and why not: the map-1 re-plan at 1 eV.  It changed the grid,
#: and so every Sigma executable's frequency extent, on every SC run (a
#: recompile at map 1), while the held rules already paid for the map-0 grid.
SUPPORT_PAD_EV = 2.0


def read_halfwidth_ev():
    """Half-width of the Σ(ω) samples one state's evaluation reads.

    Σ(E) is read at E and the Z stencil at E ± dE
    (``eqp_bgw.compute_z_factor_from_omega_grid``), so a state is on its grid
    while [E − dE, E + dE] is.
    """
    from .eqp_bgw import Z_FINITE_DIFFERENCE_EV
    return float(Z_FINITE_DIFFERENCE_EV)


def quasiparticle_mask(z_kn):
    """``Z ∈ (0, 1]`` per state: a quasiparticle.

    The upper bound is read on the build grid (``sigma_box_plan.snap_outward``,
    1e-4 cells): a flat-Σ state sits at Z = 1 to within its finite-difference
    noise, and an exact ``z <= 1`` lets round-off drop it (Fe 4^3 SC: 76 vs 75
    tail samples).
    """
    from .sigma_box_plan import snap_outward
    z = np.asarray(z_kn, dtype=np.float64)
    finite = np.where(np.isfinite(z) & (z > 0.0), z, 0.0)
    z_down = np.vectorize(lambda x: snap_outward(x, 1., -1))(finite)
    return np.isfinite(z) & (z > 0.0) & (z_down <= 1.0)


def requested_states(sigma, frozen_core_bands, energy_relative_ev, required_kn,
                     active_n=None, quasiparticle_kn=None):
    """The requested set ``R`` of this map (module docstring), ``(nk, nb)`` bool."""
    energy = np.asarray(energy_relative_ev, dtype=np.float64)
    required = np.array(np.broadcast_to(
        np.asarray(required_kn, dtype=bool), energy.shape))
    # Frozen core blocks stay at DFT in the SC Hamiltonian; they never
    # require a Sigma sample.
    required[:, :int(frozen_core_bands)] = False
    if sigma.out_of_grid == "cover":
        if active_n is not None:
            required &= np.asarray(active_n, dtype=bool)[None, :]
    else:
        from .gw_config import sigma_classification_window_ev
        from .scissor import sc_padded_window_ev
        win_lo, win_hi = sc_padded_window_ev(*sigma_classification_window_ev(sigma))
        required &= (energy >= win_lo) & (energy <= win_hi)
    if quasiparticle_kn is not None:
        required &= np.asarray(quasiparticle_kn, dtype=bool)
    return required


def support_envelope_ev(energy_relative_ev, requested_kn, pad_ev):
    """``[min_R E − pad, max_R E + pad]`` in eV, or None when ``R`` is empty."""
    energy = np.asarray(energy_relative_ev, dtype=np.float64)
    requested = np.broadcast_to(np.asarray(requested_kn, dtype=bool), energy.shape)
    if not requested.any():
        return None
    return (float(np.min(energy[requested])) - float(pad_ev),
            float(np.max(energy[requested])) + float(pad_ev))


def union_envelope(first, second):
    """The smallest interval holding both envelopes (None is empty)."""
    if first is None:
        return second
    if second is None:
        return first
    return (min(first[0], second[0]), max(first[1], second[1]))


def assert_support_in_envelope(grid_ev, deck_grid_ev, envelope_ev, step_ev, *,
                               context):
    """Refuse a sampled support that leaves ``D ∪ envelope`` (module docstring).

    A grown edge lands on the step lattice, so it may overshoot its envelope
    edge by less than one step; one step is the tolerance.
    """
    grid = np.asarray(grid_ev, dtype=np.float64)
    deck = np.asarray(deck_grid_ev, dtype=np.float64)
    lo, hi = float(deck[0]), float(deck[-1])
    if envelope_ev is not None:
        lo, hi = min(lo, float(envelope_ev[0])), max(hi, float(envelope_ev[1]))
    tol = float(step_ev) * (1.0 + 1e-9)
    if grid[0] < lo - tol or grid[-1] > hi + tol:
        raise ValueError(
            f"GATE sigma_support_envelope ({context}): the sampled Sigma support "
            f"[{grid[0]:+.4f}, {grid[-1]:+.4f}] eV leaves the deck request "
            f"[{deck[0]:+.4f}, {deck[-1]:+.4f}] eV joined with the requested-state "
            f"envelope {envelope_ev} eV (E_in of the requested quasiparticles "
            f"+/- the plan pad).  FALSE case: every sampled frequency is the deck's "
            f"or within a pad of a requested state's input energy.  A root, a "
            f"no-quasiparticle state or a state outside the Sigma window may not "
            f"grow the grid; see gw/qp_support.py.")


def grow_support_ev(grid_ev, energy_relative_ev, requested_kn, step_ev, *,
                    pad_ev, trigger_ev):
    """Extend only the outer samples of ``grid_ev`` over the triggering states.

    A requested state triggers when E ∓ ``trigger_ev`` leaves the grid; the
    crossed edge moves to E ∓ ``pad_ev`` on the ``step_ev`` lattice.  Every
    old sample is kept, and an interior hole of a patched grid keeps its
    refusal (``qsgw_utils.assert_omega_grid_covers``).
    """
    from common.units import RYD_TO_EV
    from .qsgw_utils import assert_omega_grid_covers

    grid = np.asarray(grid_ev, dtype=np.float64)
    energy = np.asarray(energy_relative_ev, dtype=np.float64)
    requested = np.asarray(requested_kn, dtype=bool)
    step = float(step_ev)
    if (grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all()
            or np.any(np.diff(grid) <= 0.0)
            or not np.isfinite(step) or step <= 0.0):
        raise ValueError("Sigma support requires an ascending finite grid and positive step")
    if energy.ndim != 2 or requested.shape not in ((energy.shape[1],), energy.shape):
        raise ValueError("Sigma support: energies and requested identity masks disagree")
    requested = np.broadcast_to(requested, energy.shape)
    if not np.isfinite(energy[requested]).all():
        raise ValueError("Sigma support: requested energies must be finite")
    trigger, pad = float(trigger_ev), float(pad_ev)
    if not (np.isfinite(trigger) and trigger >= 0.0 and np.isfinite(pad) and pad >= trigger):
        raise ValueError("Sigma support: need 0 <= trigger_ev <= pad_ev")
    assert_omega_grid_covers(
        energy / RYD_TO_EV, requested, grid / RYD_TO_EV,
        context="Sigma requested-state support")
    below = requested & (energy - trigger < grid[0])
    above = requested & (energy + trigger > grid[-1])
    if not (below.any() or above.any()):
        return grid
    lower = float(np.min(energy[below])) - pad if below.any() else float(grid[0])
    upper = float(np.max(energy[above])) + pad if above.any() else float(grid[-1])
    n_lower = int(np.ceil((grid[0] - lower) / step))
    n_upper = int(np.ceil((upper - grid[-1]) / step))
    return np.concatenate((
        grid[0] - step * np.arange(n_lower, 0, -1), grid,
        grid[-1] + step * np.arange(1, n_upper + 1)))


def plan_support_ev(sigma, deck_grid_ev, energy_relative_ev, requested_kn):
    """The plan: ``D ∪ [min_R E − P, max_R E + P]`` with P = SUPPORT_PAD_EV.

    Returns ``(grid, envelope)``.
    """
    pad = SUPPORT_PAD_EV
    grid = grow_support_ev(deck_grid_ev, energy_relative_ev, requested_kn,
                           float(sigma.omega_step_ev), pad_ev=pad, trigger_ev=pad)
    envelope = support_envelope_ev(energy_relative_ev, requested_kn, pad)
    assert_support_in_envelope(grid, deck_grid_ev, envelope, sigma.omega_step_ev,
                               context="plan")
    return grid, envelope


def hold_support_ev(sigma, deck_grid_ev, held_grid_ev, held_envelope,
                    energy_relative_ev, requested_kn):
    """A held map: keep ``held_grid_ev`` unless a requested read support leaves it.

    Returns ``(grid, envelope, event)`` with event ``"hold"`` or ``"extend"``;
    the envelope is the running union since the last plan.
    """
    grid = grow_support_ev(held_grid_ev, energy_relative_ev, requested_kn,
                           float(sigma.omega_step_ev), pad_ev=SUPPORT_PAD_EV,
                           trigger_ev=read_halfwidth_ev())
    envelope = union_envelope(held_envelope, support_envelope_ev(
        energy_relative_ev, requested_kn, SUPPORT_PAD_EV))
    assert_support_in_envelope(grid, deck_grid_ev, envelope, sigma.omega_step_ev,
                               context="held map")
    event = "hold" if grid.size == np.asarray(held_grid_ev).size else "extend"
    return grid, envelope, event


# ---------------------------------------------------------------------------
# The semicore patch (owner 2026-09-28/29)
# ---------------------------------------------------------------------------
# Coarse ("semicore") states (``band_partition.semicore_floor``: occupied, below
# the minimum energy of the lowest requested valence band) stay in the QP
# matrix with their full Sigma
# rows, but are read at their own energy on ONE held patch of the grid, below
# the near support and at SEMICORE_ETA_EV, instead of stretching the near grid
# (and its crossing windows) down to them at the deck eta.  The patch is
# planned at SC map 0 over the semicore DFT energies with the plan's pad and
# held; a semicore read support that leaves it extends it by the same rule
# (never shrinks), as the near support does.  A semicore state that rises into
# the near grid reads the near grid.

#: Broadening of the automatic coarse (semicore) windows (eV).  Owner
#: 2026-09-29: "i don't care if the broadening is like, 8 eV or something.
#: there should be some broadening that works".  Converged fixed points at
#: coarse eps 3e-3, far windows at 1 eV (claim 2960), E_F +- 1 eV std/max
#: against the smallest converged eta: Fe 4^3 charge (ref 1 eV) 1.2/8.0 meV at
#: 3 eV, 3.6/15.9 at 5, 8.2/26.7 at 8; MoS2 3x3 (ref: deck eta) 2.9/20.0 at 5,
#: 4.6/31.8 at 8.  5 eV is the largest with a few meV std on both; the coarse
#: windows then add 47 (Fe) and 37 (MoS2) tau pairs per map, and every
#: coarse Z stays in (0, 1].
SEMICORE_ETA_EV = 5.0
#: Patch sampling step (eV): eta_semi / 2 resolves the broadened Sigma.
SEMICORE_PATCH_STEP_EV = 0.5 * SEMICORE_ETA_EV
#: A patch's top sample stays this far (eV) below the near grid's bottom (and
#: an automatic patch this far outside a user window), so the joined grid
#: ascends strictly with no uncovered sliver.
SEMICORE_PATCH_EDGE_EV = 1.0e-3
#: Certificate tolerance of the coarse windows (peak-relative, never tighter
#: than the deck's sigma_quadrature_eps; owner 2026-09-29: "maybe like 3e-3 for
#: that error is a good place to start").  The crossing count falls with
#: ln(1/eps): the Fe 4^3 coarse window takes 188 nodes at 3e-3 against 253 at
#: 1e-4, MoS2 3x3 125 against 186 (closed-form law, eta 1 eV).  Its semicore QP
#: bias and protected feedback are measured in claim 2960 (WINSPLIT).
SEMICORE_EPS = 3.0e-3


def _snap(lo, hi, step):
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


def semicore_patches_ev(energy_rel_ev, semicore_kn, near_lo_ev, *, pad_ev=SUPPORT_PAD_EV,
                        previous=()):
    """The held automatic patches, one per coarse manifold: ``((lo, hi, eta), ...)``.

    Each masked energy below the near grid is padded by ``pad_ev``; padded
    intervals that touch merge, so the manifolds are the coarse levels
    separated by global gaps wider than twice the pad.  Each is snapped
    outward to SEMICORE_PATCH_STEP_EV, joined with ``previous`` (a patch only
    grows) and its top kept SEMICORE_PATCH_EDGE_EV under ``near_lo_ev``.
    How manifolds share Sigma rule windows is the planner's choice
    (``sigma_box_plan.plan_sigma_windows``, the closed-form node law).
    """
    e = np.asarray(energy_rel_ev, float)[np.asarray(semicore_kn, bool)]
    e = np.sort(e[e < float(near_lo_ev)])
    step, pad = SEMICORE_PATCH_STEP_EV, float(pad_ev)
    spans = [_snap(x - pad, x + pad, step) for x in e] + [
        (float(p[0]), float(p[1])) for p in (previous or ())]
    merged = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    top = float(near_lo_ev) - SEMICORE_PATCH_EDGE_EV
    return tuple((lo, min(hi, top), float(SEMICORE_ETA_EV)) for lo, hi in merged if lo < top)


# ---------------------------------------------------------------------------
# Far conduction (owner 2026-09-29, STACK): no material constant
# ---------------------------------------------------------------------------
# A protected conduction state n is read at eta_n = max(eta_deck, Gamma_n),
# Gamma_n = |Im Sigma_nn(E_n)| from the map-0 Sigma (read on the near grid at
# the deck eta).  The far class is every protected state above the highest
# protected conduction state with Gamma_n <= eta_deck; it leaves the near grid
# from map 1 (one planned re-plan) and is read on held windows above it.  Each
# window's eta is a step of the envelope min_{m: E_m >= E} eta_m, rounded down
# to eta_deck * 2^j, so a window never reads a state broader than its own
# Gamma; the plan groups windows by the node law.  The protected end of every
# off-diagonal Hermitian average is still read on the near grid at the deck
# eta, so the mixing of far states with protected ones is kept.


def far_class_kn(energy_rel_ev, gamma_kn, candidates_kn, eta_deck_ev):
    """The far-conduction identities: protected states above mu and above the
    highest candidate whose Gamma_n <= eta_deck (none when no state above mu
    has Gamma_n > eta_deck).  ``gamma_kn`` from the map-0 Sigma, eV."""
    e = np.asarray(energy_rel_ev, float)
    g = np.asarray(gamma_kn, float)
    cand = np.asarray(candidates_kn, bool) & (e > 0.0) & np.isfinite(g)
    if not cand.any():
        return np.zeros(e.shape, bool)
    sharp = cand & (g <= float(eta_deck_ev))
    floor = float(e[sharp].max()) if sharp.any() else 0.0
    return cand & (e > floor)


def far_windows_ev(energy_rel_ev, far_kn, gamma_kn, near_hi_ev, eta_deck_ev, *,
                   pad_ev=SUPPORT_PAD_EV):
    """The held far windows above the near grid: ``((lo, hi, eta), ...)`` ascending,
    tiling [near_hi, top + pad] with no gap; eta per window as in the header."""
    e = np.asarray(energy_rel_ev, float)[np.asarray(far_kn, bool)]
    g = np.asarray(gamma_kn, float)[np.asarray(far_kn, bool)]
    if not e.size:
        return ()
    order = np.argsort(e)
    e, g = e[order], np.maximum(g[order], float(eta_deck_ev))
    env = np.minimum.accumulate(g[::-1])[::-1]              # min eta over states at or above
    level = float(eta_deck_ev) * 2.0 ** np.floor(np.log2(env / float(eta_deck_ev)) + 1e-12)
    bottom = float(near_hi_ev) + SEMICORE_PATCH_EDGE_EV
    out, start = [], 0
    for i in range(1, e.size + 1):
        if i == e.size or level[i] != level[start]:
            out.append([None, float(e[i - 1]), float(level[start])])
            start = i
    # tile: each window runs from the previous one's top to its last state's
    # energy (the top window to its last state + pad), sampled at its eta/2
    lo = bottom
    windows = []
    for k, (_, top, eta) in enumerate(out):
        hi = top + float(pad_ev) if k == len(out) - 1 else top
        step = 0.5 * eta
        hi = float(np.ceil(hi / step) * step)
        if hi <= lo:
            continue
        windows.append((lo, hi, eta))
        lo = hi + SEMICORE_PATCH_EDGE_EV
    return tuple(windows)


def extend_far_windows_ev(windows, energy_rel_ev, far_kn, *, pad_ev=SUPPORT_PAD_EV):
    """A held far set whose top state's read leaves the top window: grow that
    window's top (windows tile below it, and a state below the bottom reads
    the near grid)."""
    if not windows:
        return windows
    e = np.asarray(energy_rel_ev, float)[np.asarray(far_kn, bool)]
    lo, hi, eta = windows[-1]
    need = float(e.max()) + float(pad_ev) if e.size else hi
    step = 0.5 * float(eta)
    hi = max(float(hi), float(np.ceil(need / step) * step))
    return tuple(windows[:-1]) + ((float(lo), hi, float(eta)),)


def far_patch_escapes(energy_rel_ev, far_kn, near_hi_ev, patches):
    """Far states above the near grid whose read support [E-h, E+h] leaves the
    held far windows; a window bottom that abuts the near grid is open."""
    e = np.asarray(energy_rel_ev, float)
    h = read_halfwidth_ev()
    above = np.asarray(far_kn, bool) & (e > float(near_hi_ev))
    ok = np.zeros(e.shape, bool)
    for lo, hi, _ in patches or ():
        abuts = float(lo) <= float(near_hi_ev) + 2.0 * SEMICORE_PATCH_EDGE_EV
        ok |= (e + h <= float(hi)) & ((e - h >= float(lo)) | (abuts & (e >= float(lo))))
    return above & ~ok


def _inside_any(e, windows):
    inside = np.zeros(np.shape(e), bool)
    for w in windows or ():
        inside |= (e >= float(w[0])) & (e <= float(w[1]))
    return inside


def semicore_patch_escapes(energy_rel_ev, semicore_kn, near_lo_ev, patches, user_windows=()):
    """Coarse states below the near grid, outside every user window, whose read
    support [E-h, E+h] leaves the automatic patches.  A patch top that abuts
    the near grid is open: its stencil reads the near grid."""
    e = np.asarray(energy_rel_ev, float)
    h = read_halfwidth_ev()
    below = (np.asarray(semicore_kn, bool) & (e < float(near_lo_ev))
             & ~_inside_any(e, user_windows))
    ok = np.zeros(e.shape, bool)
    for lo, hi, _ in patches or ():
        abuts = float(hi) >= float(near_lo_ev) - 2.0 * SEMICORE_PATCH_EDGE_EV
        ok |= (e - h >= float(lo)) & ((e + h <= float(hi)) | (abuts & (e <= float(hi))))
    return below & ~ok


def patch_on_grid_ev(patch, near_lo_ev):
    """A held window as sampled on this map: its top clipped below the near grid."""
    return (float(patch[0]), min(float(patch[1]), float(near_lo_ev) - SEMICORE_PATCH_EDGE_EV),
            float(patch[2]))


def coarse_windows_ev(near_lo_ev, patches, user_windows=()):
    """The coarse windows of this map, ascending: ``((lo, hi, eta, fixed), ...)``.

    User windows (``sigma_omega_patches_ev`` triples) are fixed; an automatic
    patch is clipped below the near grid and outside every user window, and
    may split around one.  ``fixed`` windows are never grouped with another.
    """
    edge = SEMICORE_PATCH_EDGE_EV
    users = sorted((float(u[0]), float(u[1]), float(u[2])) for u in (user_windows or ()))
    out = [(lo, min(hi, float(near_lo_ev) - edge), eta, True) for lo, hi, eta in users
           if lo < float(near_lo_ev) - edge]
    for p in patches or ():
        pieces = [patch_on_grid_ev(p, near_lo_ev)[:2]]
        for u_lo, u_hi, _ in users:
            cut = []
            for lo, hi in pieces:
                if hi < u_lo - edge or lo > u_hi + edge:
                    cut.append((lo, hi))
                    continue
                if lo < u_lo - edge:
                    cut.append((lo, u_lo - edge))
                if hi > u_hi + edge:
                    cut.append((u_hi + edge, hi))
            pieces = cut
        out += [(lo, hi, float(p[2]), False) for lo, hi in pieces if hi > lo]
    return tuple(sorted(out))


def window_grid_ev(window):
    """One coarse window's samples at eta/2, ending exactly at its top."""
    lo, hi, eta = float(window[0]), float(window[1]), float(window[2])
    n = int(np.ceil((hi - lo) / (0.5 * eta) - 1e-9)) + 1
    return np.linspace(lo, hi, max(n, 2))


def joined_grid_ev(near_grid_ev, windows):
    """The sampled Sigma support, ascending: the coarse windows below the near
    grid, the near grid, then the far-conduction windows above it."""
    near = np.asarray(near_grid_ev, dtype=np.float64)
    if not windows:
        return near
    below = [window_grid_ev(w) for w in windows if float(w[1]) < near[0]]
    above = [window_grid_ev(w) for w in sorted(windows) if float(w[0]) > near[-1]]
    return np.concatenate(below + [near] + above)


def window_labels(omega_ev, windows, eta_ev):
    """Per sample: its broadening (eV) and its coarse window index (-1 = near grid)."""
    w = np.asarray(omega_ev, dtype=np.float64)
    eta = np.full(w.shape, float(eta_ev))
    group = np.full(w.shape, -1, dtype=np.int64)
    for g, win in enumerate(windows or ()):
        inside = (w >= float(win[0]) - 1e-9) & (w <= float(win[1]) + 1e-9)
        eta[inside], group[inside] = float(win[2]), g
    return eta, group


def semicore_patch_route(compute_mode, wfns_transverse):
    """THE one predicate for a Sigma route that reads the semicore patch.

    The scalar MPA/shared-pole Sigma (no transverse wavefunctions).  A sector
    (bispinor) route reads the near grid under main's rule.
    """
    from .gw_config import ComputeMode
    return wfns_transverse is None and compute_mode is ComputeMode.MPA


def assert_semicore_patch_route(patch, compute_mode, wfns_transverse):
    """Refuse a semicore patch on a route that does not read it (GATE semicore_patch_route)."""
    if patch is not None and not semicore_patch_route(compute_mode, wfns_transverse):
        raise ValueError("GATE semicore_patch_route: the semicore patch serves the scalar "
                         "MPA/shared-pole Sigma only; a sector route reads the near grid")
