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
* ``P`` is flat: :data:`SUPPORT_PAD_EV` = 2 eV at the first plan (the one-shot
  and SC map 0) and 1 eV at the map-1 plan; later maps hold the support and
  move an edge only when a requested state's read support [E − h, E + h]
  (h = :func:`read_halfwidth_ev`, the Z stencil) leaves it, to
  E ± :data:`SUPPORT_BUFFER_EV`.  Between plans the envelope is the union over
  the maps since the last plan.

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

#: Flat support pad of the first plan (one-shot, SC map 0) and of the map-1
#: plan, in eV (owner 2026-09-27: "highest requested state + 2 eV for the
#: first window, new highest + 1 eV for the second window").
SUPPORT_PAD_EV = (2.0, 1.0)

#: Pad of an extension on a held map, in eV.  An extension triggers when a
#: requested state's read support E ± h (h = 0.5 eV) leaves the grid; a
#: buffer of 2h leaves that state h of free motion before it can trigger
#: again, and equals the map-1 pad so a map-1 plan and a later extension set
#: the same edge for the same state.
SUPPORT_BUFFER_EV = 1.0


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


def plan_support_ev(sigma, deck_grid_ev, energy_relative_ev, requested_kn, plan_index):
    """A plan: ``D ∪ [min_R E − P, max_R E + P]`` with P = SUPPORT_PAD_EV[plan_index].

    Returns ``(grid, envelope)``.  The plan starts from the deck grid, so a
    map-1 plan may shrink against map 0.
    """
    pad = SUPPORT_PAD_EV[min(int(plan_index), len(SUPPORT_PAD_EV) - 1)]
    grid = grow_support_ev(deck_grid_ev, energy_relative_ev, requested_kn,
                           float(sigma.omega_step_ev), pad_ev=pad, trigger_ev=pad)
    envelope = support_envelope_ev(energy_relative_ev, requested_kn, pad)
    assert_support_in_envelope(grid, deck_grid_ev, envelope, sigma.omega_step_ev,
                               context=f"plan {int(plan_index)}")
    return grid, envelope


def hold_support_ev(sigma, deck_grid_ev, held_grid_ev, held_envelope,
                    energy_relative_ev, requested_kn):
    """A held map: keep ``held_grid_ev`` unless a requested read support leaves it.

    Returns ``(grid, envelope, event)`` with event ``"hold"`` or ``"extend"``;
    the envelope is the running union since the last plan.
    """
    grid = grow_support_ev(held_grid_ev, energy_relative_ev, requested_kn,
                           float(sigma.omega_step_ev), pad_ev=SUPPORT_BUFFER_EV,
                           trigger_ev=read_halfwidth_ev())
    envelope = union_envelope(held_envelope, support_envelope_ev(
        energy_relative_ev, requested_kn, SUPPORT_BUFFER_EV))
    assert_support_in_envelope(grid, deck_grid_ev, envelope, sigma.omega_step_ev,
                               context="held map")
    event = "hold" if grid.size == np.asarray(held_grid_ev).size else "extend"
    return grid, envelope, event
