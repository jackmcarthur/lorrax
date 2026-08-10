"""The window plan as an artifact: written once, loaded by every leg.

WHAT THESE CELLS ARE FOR.  The farm's per-leg fixed term was ~90-110 s of a
~450 s leg and ~65 s of it was the planner, re-run per leg for a plan that
does not depend on the split (§9.5 of MPA_16GPU_PLAN.md).  Serializing it
is only safe if a loaded plan is the plan the planner computes, to the last
bit, and if a plan that does NOT match its inputs cannot be loaded at all.
Those are the two claims; these are the cells that hold them.

The pole fields below are small on purpose.  Nothing about the artifact
depends on the field's size — it depends on the group and window structure,
and the geometry here produces the same structure the production deck does:
a legacy-routed group with three windows (a crossing core, a stripe and a
slab) and several MPA-routed groups whose operand is the complex pole field.
"""

from __future__ import annotations

import numpy as np
import pytest

from gw.mpa import plan_store as PS
from gw.mpa import sigma_pass as SP
from gw.mpa import window_farm as WF


# ---------------------------------------------------------------------------
#  A field with both routes in it
# ---------------------------------------------------------------------------

def _field():
    """``(a, gamma, live, E_A, omega_abs)`` covering narrow and wide poles.

    The narrow poles (Gamma < xi) go down the legacy two-point route and
    produce the multi-window group; the wide ones bucket into MPA groups.
    Both must survive a round trip, and they exercise different fields of
    the window dataclass — the legacy windows carry a ``mask_B_mode`` and a
    threshold, the MPA ones do not.
    """
    rng = np.random.default_rng(20260810)
    a = np.concatenate([
        rng.uniform(0.05, 0.35, size=40),        # narrow, legacy-routed
        rng.uniform(0.4, 3.0, size=60),          # wide, MPA-routed
    ]).astype(np.float64)
    g = np.concatenate([
        np.full(40, 1.0e-4),
        rng.uniform(0.02, 0.30, size=60),
    ]).astype(np.float64)
    live = np.ones(a.shape, dtype=bool)
    E_A = np.linspace(0.03, 0.9, 24).reshape(4, 6).astype(np.float64)
    omega_abs = np.linspace(0.0, 0.4, 5).astype(np.float64)
    return a, g, live, E_A, omega_abs


_XI = 0.02
#: The production knobs, so the rules come from SHIPPED tables.  Asking for
#: an error the catalog does not carry sends the door to an in-process
#: SciPy solve — minutes of wall, a warning about reproducibility across
#: hosts, and a plan whose bytes depend on this machine's LAPACK, which is
#: the last thing a round-trip cell should be comparing.
_SCALARS = dict(
    xi_ry=_XI, edge_factor=1.5, rel_tol=1.0e-8, laplace_ratio_max=8.0,
    target_error=1.0e-6, laplace_max_nodes=64, crossing_eps_q=1.0e-3,
    crossing_max_nodes=200, use_shipped_minimax_tables=True,
)


def _plan(a, g, live, E_A, omega_abs, *, space="cond", neg=False):
    return SP.plan_branch_groups(
        a_ry=a, gamma_ry=g, live_mask=live, E_A_host=E_A,
        base_mask_A_host=np.ones(E_A.shape, dtype=bool),
        omega_nonneg_ry=omega_abs, space=space, neg_omega_half=neg,
        xi_ry=_SCALARS["xi_ry"], edge_factor=_SCALARS["edge_factor"],
        b_abs=np.abs(a) + 0.5, rel_tol=_SCALARS["rel_tol"],
        laplace_ratio_max=_SCALARS["laplace_ratio_max"],
        target_error=_SCALARS["target_error"],
        laplace_max_nodes=_SCALARS["laplace_max_nodes"],
        crossing_eps_q=_SCALARS["crossing_eps_q"],
        crossing_max_nodes=_SCALARS["crossing_max_nodes"],
        use_shipped_minimax_tables=_SCALARS["use_shipped_minimax_tables"],
        log_tag="", print_fn=lambda *a, **k: None)


def _address(a, g, live, E_A, omega_abs, *, space="cond", neg=False,
             sha="testsha", store="/store.h5", pole=3, bkey="pos_cond",
             **override):
    scalars = dict(_SCALARS, space=space, neg_omega_half=neg)
    scalars.update(override)
    return PS.branch_address(
        source_sha=sha, fit_store=store, n_p=8, pole=pole, bkey=bkey,
        slab=PS.slab_digest(a_ry=a, gamma_ry=g, live_mask=live,
                            b_abs=np.abs(a) + 0.5),
        arrays={"E_A_host": E_A,
                "base_mask_A_host": np.ones(E_A.shape, dtype=bool),
                "omega_nonneg_ry": omega_abs},
        scalars=scalars)


@pytest.fixture(scope="module")
def planned():
    a, g, live, E_A, omega_abs = _field()
    groups, stats = _plan(a, g, live, E_A, omega_abs)
    assert len(groups) >= 3, "the fixture geometry stopped producing groups"
    assert any(g_.name == "legacy" for g_ in groups), \
        "the fixture lost its legacy-routed group, which is the one that " \
        "carries the multi-window structure the artifact has to survive"
    return a, g, live, E_A, omega_abs, groups, stats


# ---------------------------------------------------------------------------
#  THE CLAIM: a loaded plan is the computed plan, bit for bit
# ---------------------------------------------------------------------------

def test_a_loaded_plan_is_bit_identical_to_the_computed_one(planned, tmp_path):
    """The gate the whole lane rests on, at unit scale.

    ``full_plan_digest`` covers every number the integrator reads out of a
    plan: each group's index set and its two counts, and each window's tau
    nodes, weights, A-mask, both reference energies, sign, projection,
    prefactor, B-mask mode and threshold, crossing kind, achieved error and
    provenance.  Equality of that digest is the statement that the farm's
    arithmetic cannot notice which route produced its groups.
    """
    a, g, live, E_A, omega_abs, groups, stats = planned
    addr = _address(a, g, live, E_A, omega_abs)
    path = PS.plan_path(tmp_path, pole=3, bkey="pos_cond", address=addr)
    PS.write_branch_plan(
        path, groups, address=addr, pole=3, bkey="pos_cond",
        source_sha="testsha", fit_store="/store.h5", n_p=8, stats=stats,
        a_ry=a, omega_complex=a - 1j * g, print_fn=lambda *x, **k: None)

    back = PS.read_branch_plan(path, lo=0, hi=len(groups), a_ry=a,
                               omega_complex=a - 1j * g)
    assert WF.full_plan_digest(back) == WF.full_plan_digest(groups)
    assert WF.group_plan_digest(back) == WF.group_plan_digest(groups)

    # And field by field, so a failure says WHICH field moved rather than
    # only that some digest did.
    for got, want in zip(back, groups):
        assert got.name == want.name
        assert got.n_modes == want.n_modes
        assert got.b_mass == want.b_mass
        assert got.field_shape == want.field_shape
        assert got.provenance == want.provenance
        np.testing.assert_array_equal(got.idx_B, want.idx_B)
        assert got.idx_B.dtype == want.idx_B.dtype
        np.testing.assert_array_equal(
            np.asarray(got.omega_operand), np.asarray(want.omega_operand))
        assert len(got.windows) == len(want.windows)
        for wg, ww in zip(got.windows, want.windows):
            np.testing.assert_array_equal(
                np.asarray(wg.nodes.t), np.asarray(ww.nodes.t))
            np.testing.assert_array_equal(
                np.asarray(wg.nodes.alpha), np.asarray(ww.nodes.alpha))
            np.testing.assert_array_equal(wg.mask_A, np.asarray(ww.mask_A))
            assert (wg.E_ref_A, wg.E_ref_B) == (ww.E_ref_A, ww.E_ref_B)
            assert wg.omega_sign == ww.omega_sign
            assert wg.project == ww.project
            assert wg.prefactor == ww.prefactor
            assert wg.mask_B_mode == ww.mask_B_mode
            assert wg.mask_B_threshold == ww.mask_B_threshold
            assert wg.crossing_kind == ww.crossing_kind
            assert wg.max_error == ww.max_error
            assert wg.provenance == ww.provenance


def test_a_leg_can_load_only_the_groups_it_owns(planned, tmp_path):
    """A slice off disk equals the same slice of the computed plan.

    This is what makes the artifact worth having at farm scale: sixteen
    legs between them read each group's index set once, not sixteen times.
    """
    a, g, live, E_A, omega_abs, groups, stats = planned
    addr = _address(a, g, live, E_A, omega_abs)
    path = PS.plan_path(tmp_path, pole=3, bkey="pos_cond", address=addr)
    PS.write_branch_plan(
        path, groups, address=addr, pole=3, bkey="pos_cond",
        source_sha="testsha", fit_store="/store.h5", n_p=8, stats=stats,
        a_ry=a, omega_complex=a - 1j * g, print_fn=lambda *x, **k: None)
    lo, hi = 1, min(3, len(groups))
    part = PS.read_branch_plan(path, lo=lo, hi=hi, a_ry=a,
                               omega_complex=a - 1j * g)
    assert WF.full_plan_digest(part) == WF.full_plan_digest(groups[lo:hi])


def test_the_header_carries_the_partition_without_the_index_sets(planned,
                                                                 tmp_path):
    """The cheap check runs off the header, which is why it is cheap.

    A leg checks the partition it was handed against the census BEFORE it
    reads any index set — the group count and the census digest come out of
    a few kilobytes of attributes.  If that stopped being true the leg
    would have to read hundreds of megabytes to find out it should not.
    """
    a, g, live, E_A, omega_abs, groups, stats = planned
    addr = _address(a, g, live, E_A, omega_abs)
    path = PS.plan_path(tmp_path, pole=3, bkey="pos_cond", address=addr)
    PS.write_branch_plan(
        path, groups, address=addr, pole=3, bkey="pos_cond",
        source_sha="testsha", fit_store="/store.h5", n_p=8, stats=stats,
        a_ry=a, omega_complex=a - 1j * g, print_fn=lambda *x, **k: None)
    head = PS.read_plan_header(path)
    assert head["n_groups"] == len(groups)
    assert head["address"] == addr
    assert WF.group_plan_digest_from_rows(head["rows"]) == \
        WF.group_plan_digest(groups)
    assert head["stats"]["n_narrow"] == stats["n_narrow"]
    assert head["stats"]["n_tau"] == stats["n_tau"]


# ---------------------------------------------------------------------------
#  THE OTHER CLAIM: a plan built from other inputs is not addressable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("what", [
    "pole_field", "widths", "live", "E_A", "omega_grid", "sha", "store",
    "pole", "branch", "target_error", "edge_factor", "xi",
])
def test_every_planner_input_moves_the_address(planned, what):
    """Staleness is not detected, it is inexpressible — one input at a time.

    Each parametrization moves exactly one thing the planner reads and
    asserts the address moves with it.  A summary-based address (a store
    path, an mtime, a shape) would pass several of these while naming the
    same file, and the failure that lets through is a leg integrating
    certified rules built for a different calculation: finite, smooth, and
    wrong by whatever moved.
    """
    a, g, live, E_A, omega_abs, _groups, _stats = planned
    base = _address(a, g, live, E_A, omega_abs)
    if what == "pole_field":
        a2 = a.copy()
        a2[7] = np.nextafter(a2[7], np.inf)      # ONE ulp of ONE pole
        got = _address(a2, g, live, E_A, omega_abs)
    elif what == "widths":
        g2 = g.copy()
        g2[50] = np.nextafter(g2[50], np.inf)
        got = _address(a, g2, live, E_A, omega_abs)
    elif what == "live":
        live2 = live.copy()
        live2[3] = False
        got = _address(a, g, live2, E_A, omega_abs)
    elif what == "E_A":
        E2 = E_A.copy()
        E2[1, 1] = np.nextafter(E2[1, 1], np.inf)
        got = _address(a, g, live, E2, omega_abs)
    elif what == "omega_grid":
        got = _address(a, g, live, E_A, omega_abs * 1.0000001)
    elif what == "sha":
        got = _address(a, g, live, E_A, omega_abs, sha="othersha")
    elif what == "store":
        got = _address(a, g, live, E_A, omega_abs, store="/other.h5")
    elif what == "pole":
        got = _address(a, g, live, E_A, omega_abs, pole=4)
    elif what == "branch":
        got = _address(a, g, live, E_A, omega_abs, bkey="neg_val")
    elif what == "target_error":
        got = _address(a, g, live, E_A, omega_abs, target_error=1.0e-7)
    elif what == "edge_factor":
        got = _address(a, g, live, E_A, omega_abs, edge_factor=1.6)
    else:
        got = _address(a, g, live, E_A, omega_abs, xi_ry=_XI * 1.01)
    assert got != base, f"the address did not move when {what} did"


def test_the_same_inputs_give_the_same_address_every_time(planned):
    """Determinism, without which the address is a random file name."""
    a, g, live, E_A, omega_abs, _groups, _stats = planned
    assert (_address(a, g, live, E_A, omega_abs)
            == _address(a.copy(), g.copy(), live.copy(), E_A.copy(),
                        omega_abs.copy()))


def test_a_missing_plan_is_refused_by_name_and_never_replanned(tmp_path):
    """RED TWIN.  Absence is a refusal, and it says which case it is.

    Two cases, and the message tells them apart: nothing for this
    (pole, branch) at all, versus a plan at a DIFFERENT address, which
    means the planner's inputs moved.  The second is the dangerous one
    and it is the one a fallback-to-planning would have hidden.
    """
    with pytest.raises(PS.PlanMissing) as exc:
        PS.refuse_missing_plan(tmp_path, pole=3, bkey="pos_cond",
                               address="deadbeef")
    assert "pole 3" in str(exc.value) and "pos_cond" in str(exc.value)
    assert "no plan for this (pole, branch) at any address" in str(exc.value)

    stale = PS.plan_path(tmp_path, pole=3, bkey="pos_cond",
                         address="0123456789abcdef")
    with open(stale, "wb") as f:
        f.write(b"\0" * 16)
    with pytest.raises(PS.PlanMissing) as exc:
        PS.refuse_missing_plan(tmp_path, pole=3, bkey="pos_cond",
                               address="deadbeef")
    assert "other addresses" in str(exc.value)
    assert "0123456789abcdef" in str(exc.value)
    assert "will NOT plan one for itself" in str(exc.value)


def test_a_window_field_this_module_does_not_know_is_refused(monkeypatch):
    """RED TWIN.  A dropped field must be a failure, not a silent omission.

    The plan store writes a fixed list of ``_SigmaWindow`` fields.  If the
    dataclass gains one, a plan silently loses it and every leg that loads
    that plan integrates a window that differs from the planner's in a way
    no shape or Hermiticity check can see.  So the field set is asserted at
    the write.
    """
    monkeypatch.setattr(PS, "_WINDOW_FIELDS",
                        tuple(x for x in PS._WINDOW_FIELDS
                              if x != "mask_B_threshold"))
    with pytest.raises(ValueError, match="_SigmaWindow carries fields"):
        PS._window_fields()


def test_a_group_field_this_module_does_not_know_is_refused(monkeypatch):
    """The same twin for ``WindowGroup``."""
    monkeypatch.setattr(PS, "_GROUP_FIELDS",
                        PS._GROUP_FIELDS + ("a_field_that_is_not_there",))
    with pytest.raises(ValueError, match="WindowGroup carries fields"):
        PS._group_fields()


def test_an_operand_that_is_neither_slab_form_is_refused(planned):
    """RED TWIN.  The operand is rebuilt, so it must be identifiable.

    A group whose ``Omega_q`` is neither ``Re Omega`` nor
    ``Re Omega - i Gamma`` cannot be rebuilt from the leg's slab, and
    guessing would put a different pole field under a certified rule.
    """
    a, g, live, E_A, omega_abs, groups, _stats = planned
    victim = groups[0]
    stand_in = SP.WindowGroup(
        name=victim.name, windows=victim.windows, idx_B=victim.idx_B,
        field_shape=victim.field_shape, omega_operand=(a * 2.0),
        n_modes=victim.n_modes, b_mass=victim.b_mass,
        provenance=victim.provenance)
    with pytest.raises(ValueError, match="neither this pole's Re Omega"):
        PS._operand_tag(stand_in, a_ry=a, omega_complex=a - 1j * g)


# ---------------------------------------------------------------------------
#  The farm's completeness discipline, extended to what the farm READS
# ---------------------------------------------------------------------------

def test_the_manifest_declares_the_plans_and_refuses_when_one_is_gone(
        tmp_path):
    """RED TWIN.  A declared input that is gone refuses the merge, by name.

    The same mechanism as a leg that did not land, and for the same
    reason: what is on disk has to be able to say what was computed.  A
    farm whose plans have been deleted has cubes whose partition cannot be
    re-derived from anything written down.
    """
    plans = {}
    for key in ("0.pos_cond", "0.pos_val"):
        p = tmp_path / f"plan_{key}.h5"
        p.write_bytes(b"\0" * 8192)
        plans[key] = {"path": str(p), "address": "abc123"}
    legs = [{"id": "leg00", "kind": "pass", "n_tau": 1,
             "range_label": "pole 0 pos_cond[0,1) of 1"}]
    out = tmp_path / "cube_dir"
    out.mkdir()
    (out / "leg00.h5").write_bytes(b"\0" * 8192)
    man_path = tmp_path / "manifest.json"
    WF.write_manifest(man_path, legs, kind="pass", fit_store="/store.h5",
                      out_dir=str(out),
                      inputs=PS.declared_plan_inputs(plans))
    man = WF.read_manifest(man_path)
    assert len(man["inputs"]) == 2
    WF.refuse_incomplete(man, print_fn=lambda *a, **k: None)

    (tmp_path / "plan_0.pos_val.h5").unlink()
    with pytest.raises(WF.FarmIncomplete) as exc:
        WF.refuse_incomplete(man, print_fn=lambda *a, **k: None)
    assert "plan 0.pos_val" in str(exc.value)
    assert "abc123" in str(exc.value)
    assert "do not merge cubes whose plan is gone" in str(exc.value)


def test_two_censuses_with_different_plan_addresses_are_refused(tmp_path):
    """RED TWIN.  One (pole, branch) cannot have two plans.

    Two census files that name different plan artifacts for the same
    (pole, branch) were taken against different inputs, so a balance
    struck from them would name ranges into two different partitions.
    """
    common = dict(fit_store="/store.h5", n_p=8, sha="s1")
    rows_a = [{"pole": 0, "branch": "pos_cond", "branch_tag": "t", "index": 0,
               "name": "g", "n_modes": 2, "n_tau": 3, "n_windows": 1}]
    rows_b = [{"pole": 1, "branch": "pos_cond", "branch_tag": "t", "index": 0,
               "name": "g", "n_modes": 2, "n_tau": 3, "n_windows": 1}]
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    WF.write_census(pa, rows_a, extra={
        "plans": {"0.pos_cond": {"path": "/a.h5", "address": "aaa"}}},
        **common)
    WF.write_census(pb, rows_b, extra={
        "plans": {"0.pos_cond": {"path": "/b.h5", "address": "bbb"}}},
        **common)
    with pytest.raises(ValueError, match="two plan artifacts"):
        WF.merge_census_files([str(pa), str(pb)])


def test_the_partition_check_is_one_function_for_both_routes():
    """The refusals a loading leg gets are the refusals a planning leg gets.

    The load route is newer, so it is the one that would have drifted into
    checking something slightly different.  It does not, because there is
    one function and both call it.
    """
    with pytest.raises(ValueError, match="balanced from a DIFFERENT"):
        WF.check_partition(n_groups=11, digest_got="x", pole=3,
                           bkey="pos_val", total=12, lo=0, hi=5)
    with pytest.raises(ValueError, match="Same count, different partition"):
        WF.check_partition(n_groups=12, digest_got="x", pole=3,
                           bkey="pos_val", total=12, lo=0, hi=5, digest="y",
                           source="the stored plan")
    WF.check_partition(n_groups=12, digest_got="y", pole=3, bkey="pos_val",
                       total=12, lo=0, hi=5, digest="y")
