"""The width split's termination, and the 230 GB of masks it cost to learn.

THE MEASUREMENT THESE CELLS ENCODE (2026-08-09, memwatch profile at
/pscratch/sd/j/jackm/mpa_oom_0809/, steps 56532188.{22..39}): the first
end-to-end MPA Sigma dispatch OOM-killed a 230 GB Perlmutter node eleven
times at MaxRSS 218-235 GB, invariant under n_p, sigma_omega_layout,
batch size and the Sigma window, in 6-11 minutes each.  The instrumented
re-run pinned >95 % of RSS to ONE site: ``_clause_safe_width_split``'s
accumulated leaf masks -- full-size ``(n_q, N_mu, N_mu)`` booleans,
81.4 MB each on the production deck, ~2 800 live at the 185 GB abort.
The termination clause compared ``Gamma_hi`` against ``beta_max *
x_min`` with ``x_min = max(e_lo + a_lo - omega_max, 1e-12)``, and on
that deck ``e_lo + a_lo < omega_max`` put ``x_min`` ON THE FLOOR, where
no amount of splitting satisfies the clause and the recursion bisects to
its depth-16 cap -- 2^16 leaves demanded, 5.3 TB of masks, the node dead
first.  The "fixed ~220 GB allocation" of the eleven attempts was the
OOM killer's position, not a buffer's size.

THE FIX IS ALIGNMENT, in two halves, both derived from the window
builder's own arithmetic rather than from a new heuristic:

* a CROSSING-capable branch never runs the split at all, because every
  Laplace window ``_mpa_groups_for_bucket`` builds there floors its
  ``x_min`` at ``z_edge = edge_factor * Gamma_hi``, so ``beta <=
  1/edge_factor`` structurally and the clause the split serves cannot
  fire (``refuse_edge_factor_below_envelope`` guards the divisor);
* a NON-crossing branch splits with the ``single`` window's OWN
  ``x_min`` -- ``e_lo + a_lo`` with no ``omega_max`` subtraction --
  under which the floor is unreachable and the ``Gamma <= a``
  termination argument actually applies.

Plus one backstop with a name: ``MAX_WIDTH_SPLIT_LEAVES``, for the field
nobody has met yet, converting a divergence into a one-line refusal.
"""

from __future__ import annotations

import numpy as np
import pytest

from gw.mpa import sigma_pass
from gw.mpa import sigma_routing as R


def _failure_geometry(n_modes=512, seed=3):
    """The profiled deck's numbers, miniaturised.

    ``e_lo = 0.0255 Ry`` and ``omega_max = 0.147 Ry`` are read off the
    profiled run's own window print (``E_A=[0.0255, 4.1424] Ry``, the
    +/- 2 eV Sigma grid); ``a`` spans two decades and ``Gamma = ratio *
    a`` with ``ratio <= 1`` respects the fitter's fourth guard; ``xi``
    is the production crossing regularization floor (0.476 eV), so the
    narrow/wide split routes sub-xi widths to the legacy machinery
    exactly as the real pass loop does.  What makes it THE failure
    geometry is only ``e_lo + min(a) < omega_max`` -- the floored
    predicate -- which these numbers satisfy exactly as the deck did.
    """
    rng = np.random.default_rng(seed)
    a = np.exp(rng.uniform(np.log(0.02), np.log(2.0), size=n_modes))
    ratio = np.exp(rng.uniform(np.log(1e-4), np.log(1.0), size=n_modes))
    g = ratio * a
    E_A = np.linspace(0.0255, 4.1424, 32)
    return dict(
        a_ry=a, gamma_ry=g, live_mask=np.ones(n_modes, dtype=bool),
        E_A_host=E_A, base_mask_A_host=np.ones(E_A.size, dtype=bool),
        omega_nonneg_ry=np.linspace(0.0, 0.147, 5),
        xi_ry=0.476 / 13.6056980659, edge_factor=4.0)


def test_the_old_predicate_diverges_on_the_measured_geometry():
    """THE MECHANISM, REPRODUCED -- the cell that says why the node died.

    Calling the split the way the call site used to (``omega_max`` of
    the Sigma grid subtracted inside the predicate) on the failure
    geometry floors ``x_min`` and bisects until leaves are single-width:
    with 512 modes that is hundreds of leaves, and on the production
    field's 81 million modes it was the depth-16 cap and a dead node.
    Each leaf here is a mask over 512 modes; each leaf there was
    81.4 MB.  This cell is the profile's finding kept executable, so the
    next person who "simplifies" the call site back re-discovers the
    OOM in milliseconds instead of on a node.
    """
    geo = _failure_geometry()
    e_lo = float(np.min(geo["E_A_host"]))
    assert e_lo + float(np.min(geo["a_ry"])) < 0.147, "not the failure geometry"
    leaves = sigma_pass._clause_safe_width_split(
        geo["a_ry"], geo["gamma_ry"], geo["live_mask"],
        e_lo=e_lo, omega_max=0.147, beta_max=R.SHIPPED_WIDTH_BETA_MAX)
    assert len(leaves) > sigma_pass.MAX_WIDTH_SPLIT_LEAVES, (
        f"the floored predicate no longer diverges ({len(leaves)} leaves) "
        f"-- if _clause_safe_width_split changed, this cell and the "
        f"aligned call site must move together")
    # The leaves PARTITION the live modes -- the fact that makes a leaf
    # count a memory statement: n_leaves * mask_bytes is the live set.
    total = np.zeros(geo["live_mask"].shape, dtype=int)
    for m in leaves:
        total += m.astype(int)
    assert np.array_equal(total, geo["live_mask"].astype(int))


def test_the_aligned_predicate_terminates_on_the_same_geometry():
    """The same field, the non-crossing builder's own x_min: a handful of
    leaves, still a partition, every one satisfying the clause it will
    actually pay (``beta <= beta_max`` at ``x_min = e_lo + a_lo``)."""
    geo = _failure_geometry()
    e_lo = float(np.min(geo["E_A_host"]))
    leaves = sigma_pass._clause_safe_width_split(
        geo["a_ry"], geo["gamma_ry"], geo["live_mask"],
        e_lo=e_lo, omega_max=0.0, beta_max=R.SHIPPED_WIDTH_BETA_MAX)
    assert 1 <= len(leaves) <= sigma_pass.MAX_WIDTH_SPLIT_LEAVES
    total = np.zeros(geo["live_mask"].shape, dtype=int)
    for m in leaves:
        total += m.astype(int)
        a_lo = float(np.min(geo["a_ry"], where=m, initial=np.inf))
        g_hi = float(np.max(geo["gamma_ry"], where=m, initial=0.0))
        assert g_hi <= R.SHIPPED_WIDTH_BETA_MAX * (e_lo + a_lo) * (1 + 1e-12)
    assert np.array_equal(total, geo["live_mask"].astype(int))


@pytest.mark.parametrize("space,neg_half,crossing", [
    ("cond", False, True),
    ("val", True, True),
    ("cond", True, False),
    ("val", False, False),
])
def test_planning_the_measured_geometry_is_bounded_on_every_branch(
        space, neg_half, crossing):
    """END TO END: ``plan_branch_groups`` on the failure geometry returns
    a bounded handful of groups on all four branch identities.

    This is the call that was six minutes from killing a node; here it
    must come back in well under a second with every group's windows
    BUILT -- i.e. the no-split crossing route really does pass
    ``_refuse_width_clause`` on every window, which is the structural
    claim (``beta <= 1/edge_factor``) the fix rests on.  The group count
    ceiling is generous on purpose: the claim is boundedness, not a
    frozen partition.
    """
    assert R.denominator_can_cross(space, neg_half) is crossing
    geo = _failure_geometry()
    groups, stats = sigma_pass.plan_branch_groups(
        a_ry=geo["a_ry"], gamma_ry=geo["gamma_ry"],
        live_mask=geo["live_mask"], E_A_host=geo["E_A_host"],
        base_mask_A_host=geo["base_mask_A_host"],
        omega_nonneg_ry=geo["omega_nonneg_ry"], space=space,
        neg_omega_half=neg_half, xi_ry=geo["xi_ry"],
        edge_factor=geo["edge_factor"],
        print_fn=lambda *a, **k: None)
    assert stats["n_wide"] > 0 and stats["n_narrow"] > 0, (
        "the geometry spans the xi split by design; both routes must fire")
    assert 1 <= len(groups) <= 40, [g.name for g in groups]
    # Union of the group masks is the live set: nothing dropped, nothing
    # double-counted (the failure mode of a bucketing rewrite is a mode
    # summed twice, which no finiteness check would see).
    total = np.zeros(geo["live_mask"].shape, dtype=int)
    for grp in groups:
        total += np.asarray(grp.mask_B, dtype=int)
    assert np.array_equal(total, geo["live_mask"].astype(int))


def test_the_leaf_ceiling_refuses_by_name_on_a_field_outside_the_argument():
    """The backstop's red twin: a field that DEFEATS the termination
    argument (widths far above their own Laplace edges, violating the
    fitter's ``Gamma <= a`` guard on purpose) must refuse with the
    ceiling, the mask statistics and the word 'out-of-memory' -- not
    accumulate masks.  Constructed through the call-site guard directly,
    because ``plan_branch_groups`` can only reach it through a
    non-crossing branch whose field escaped the fitter's guards."""
    n = 4096
    rng = np.random.default_rng(9)
    a = np.full(n, 1.0e-11)                      # Laplace edge ~ the floor
    g = np.exp(rng.uniform(np.log(1e-8), np.log(1e-1), size=n))
    live = np.ones(n, dtype=bool)
    leaves = sigma_pass._clause_safe_width_split(
        a, g, live, e_lo=0.0, omega_max=0.0,
        beta_max=R.SHIPPED_WIDTH_BETA_MAX)
    with pytest.raises(RuntimeError) as exc:
        sigma_pass._refuse_width_split_explosion(len(leaves), live, g)
    msg = str(exc.value)
    assert str(sigma_pass.MAX_WIDTH_SPLIT_LEAVES) in msg
    assert "out-of-memory" in msg
    assert "4096 modes" in msg


def test_the_crossing_branchs_windows_cannot_trip_the_width_clause():
    """The structural half of the no-split rule, checked as arithmetic:
    every Laplace window a crossing bucket builds floors x_min at
    z_edge = edge_factor * Gamma_hi, so beta = Gamma/x_min <= 1/4 at the
    deck default -- below SHIPPED_WIDTH_BETA_MAX with 4x to spare.  If a
    future edit removes the z_edge floor from ``_mpa_groups_for_bucket``,
    this fails before any node does."""
    geo = _failure_geometry()
    # The WIDE subset, as the call site's xi split guarantees: the
    # crossing core's own A = f_max/Gamma_lo gate is a different (and
    # legitimate) refusal, served in production by legacy-routing every
    # Gamma < xi -- so this cell hands the bucket what the split would.
    wide = geo["gamma_ry"] >= geo["xi_ry"]
    groups = sigma_pass._mpa_groups_for_bucket(
        a_ry=geo["a_ry"], gamma_ry=geo["gamma_ry"],
        bucket_mask=wide, E_A_host=geo["E_A_host"],
        base_mask_A_host=geo["base_mask_A_host"],
        omega_max=float(np.max(geo["omega_nonneg_ry"])),
        space="cond", neg_omega_half=False,
        edge_factor=geo["edge_factor"], rel_tol=1.0e-8,
        max_nodes=R.DEFAULT_MAX_CROSSING_NODES)
    assert groups, "the bucket must produce windows"
    for _names, wins, _mask in groups:
        assert wins
