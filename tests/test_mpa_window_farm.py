"""The window-group farm: the split covers, the balance balances, the merge refuses.

WHAT THESE CELLS ARE FOR.  The farm's whole risk is a COVERAGE error —
a window group integrated twice or never — because that is the one
failure on this path that produces a finite, smooth, plausible Σ_c of
exactly the right shape, dtype and units.  The 2026-08-10 fit farm
demonstrated the same failure one stage upstream: a leg died at
placement, its q window [48, 52) was never fitted, and nothing
downstream noticed, because a fit store missing four q looks exactly
like a fit store.

So every cell below is either "the partition covers exactly once" or
"the refusal fires and names the thing that is missing".  The
performance claim is measured on a cluster and is not testable here;
what is testable here is that the mechanism cannot lose work quietly.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from gw.mpa import window_farm as WF


# ---------------------------------------------------------------------------
#  A census shaped like the production deck, without a production deck
# ---------------------------------------------------------------------------

def _census(n_poles=8, n_groups=29, pole_tau=None):
    """A census with the production deck's pole skew and group counts."""
    pole_tau = pole_tau or [10027, 7207, 4869, 4765, 3652, 3343, 2539, 3537]
    rng = np.random.default_rng(20260810)
    rows = []
    for p in range(n_poles):
        per_branch = [pole_tau[p] // 4] * 4
        per_branch[0] += pole_tau[p] - sum(per_branch)
        for bkey, budget in zip(WF.BRANCH_KEYS, per_branch):
            cuts = np.sort(rng.choice(np.arange(1, budget), n_groups - 1,
                                      replace=False))
            edges = np.concatenate([[0], cuts, [budget]])
            for i, w in enumerate(np.diff(edges)):
                rows.append({"pole": p, "branch": bkey, "branch_tag": bkey,
                             "index": i, "name": f"g{p}{bkey}{i}",
                             "n_modes": 100, "n_tau": int(w), "n_windows": 1})
    return {"format": WF.CENSUS_FORMAT, "fit_store": "/store/fit.h5",
            "n_p": n_poles, "sha": "deadbeef", "digests": {}, "rows": rows}


class _FakeWindow:
    def __init__(self, n):
        self.n_tau = n


class _FakeGroup:
    def __init__(self, name, n_tau, n_modes=7):
        self.name = name
        self.windows = [_FakeWindow(n_tau)]
        self.n_modes = n_modes


# ---------------------------------------------------------------------------
#  The address: a leg spec parses, round-trips and refuses what it must
# ---------------------------------------------------------------------------

def test_group_subset_parses_and_round_trips():
    spec = WF.parse_group_subset("3.pos_val:2-9/40, 3.neg_cond:0-5/12")
    assert spec == {(3, "pos_val"): (2, 9, 40), (3, "neg_cond"): (0, 5, 12)}
    assert WF.format_group_subset(spec) == "3.pos_val:2-9/40,3.neg_cond:0-5/12"
    assert WF.parse_group_subset("") is None
    assert WF.parse_group_subset(None) is None


def test_group_subset_refuses_an_empty_range_and_an_unknown_branch():
    # An empty leg is a leg that reports success having integrated nothing,
    # which is the shape of every failure this module exists to catch.
    with pytest.raises(ValueError, match="not a non-empty range"):
        WF.parse_group_subset("0.pos_cond:5-5/9")
    with pytest.raises(ValueError, match="not a non-empty range"):
        WF.parse_group_subset("0.pos_cond:0-10/9")
    with pytest.raises(ValueError, match="not a Σ branch|not a group range"):
        WF.parse_group_subset("0.sideways:0-1/9")


def test_group_subset_refuses_the_same_branch_twice_in_one_leg():
    with pytest.raises(ValueError, match="appears twice"):
        WF.parse_group_subset("0.pos_cond:0-2/9,0.pos_cond:3-5/9")


def test_branch_key_is_the_branchs_physical_identity():
    assert WF.branch_key("cond", False) == "pos_cond"
    assert WF.branch_key("val", True) == "neg_val"
    with pytest.raises(ValueError, match="not one of the four"):
        WF.branch_key("spin", False)


# ---------------------------------------------------------------------------
#  Selection: the planner is sliced, never re-planned
# ---------------------------------------------------------------------------

def test_no_subset_selects_the_whole_planned_walk_unchanged():
    groups = [_FakeGroup(f"g{i}", 10) for i in range(6)]
    sel, lo = WF.select_branch_groups(groups, pole=0, bkey="pos_cond",
                                      spec=None)
    assert sel == groups and lo == 0


def test_a_branch_this_leg_does_not_own_selects_nothing():
    groups = [_FakeGroup(f"g{i}", 10) for i in range(6)]
    spec = WF.parse_group_subset("0.pos_val:0-3/6")
    sel, lo = WF.select_branch_groups(groups, pole=0, bkey="pos_cond",
                                      spec=spec)
    assert sel == [] and lo == 0


def test_selection_is_the_planner_order_sliced():
    groups = [_FakeGroup(f"g{i}", 10) for i in range(6)]
    spec = WF.parse_group_subset("0.pos_cond:2-5/6")
    sel, lo = WF.select_branch_groups(groups, pole=0, bkey="pos_cond",
                                      spec=spec)
    assert [g.name for g in sel] == ["g2", "g3", "g4"] and lo == 2


def test_selection_refuses_a_manifest_that_counted_different_groups():
    # THE STALE-CENSUS FAILURE.  Same key, different partition: the leg
    # would integrate a range that no longer names the groups the balance
    # meant, and every leg would still succeed.
    groups = [_FakeGroup(f"g{i}", 10) for i in range(5)]
    spec = WF.parse_group_subset("0.pos_cond:2-5/6")
    with pytest.raises(ValueError, match="balanced from a DIFFERENT"):
        WF.select_branch_groups(groups, pole=0, bkey="pos_cond", spec=spec)


def test_selection_refuses_the_same_count_with_a_different_partition():
    # The count matches and the partition does not — which a count alone
    # cannot see, and which the digest is for.
    a = [_FakeGroup(f"g{i}", 10) for i in range(6)]
    b = [_FakeGroup(f"g{i}", 11) for i in range(6)]
    dig = WF.group_plan_digest(a)
    spec = WF.parse_group_subset("0.pos_cond:0-3/6")
    WF.select_branch_groups(a, pole=0, bkey="pos_cond", spec=spec, digest=dig)
    with pytest.raises(ValueError, match="fingerprint"):
        WF.select_branch_groups(b, pole=0, bkey="pos_cond", spec=spec,
                                digest=dig)


# ---------------------------------------------------------------------------
#  The balance: exactly once, and actually balanced
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_legs", [1, 2, 8, 16, 32, 64])
def test_the_balance_covers_every_group_exactly_once(n_legs):
    census = _census()
    legs = WF.balance_legs(census, n_legs)
    assert len(legs) == n_legs
    seen = {}
    for leg in legs:
        for key, (lo, hi, total) in WF.parse_group_subset(
                leg["group_subset"]).items():
            assert total == WF.universe_from_census(census)[key]
            for g in range(lo, hi):
                assert (key, g) not in seen, "a group in two legs"
                seen[(key, g)] = leg["id"]
    assert len(seen) == len(census["rows"])


def test_sixteen_legs_beat_the_pole_skew_the_pole_farm_could_not():
    # THE LEVER, as an inequality rather than a wall clock.  Eight pole
    # legs are bounded below by the dearest pole, which on this deck is
    # 3.95x the cheapest; sixteen group legs are not, and the balance has
    # to demonstrate that rather than assert it.
    census = _census()
    per_pole = {}
    for r in census["rows"]:
        per_pole[r["pole"]] = per_pole.get(r["pole"], 0) + r["n_tau"]
    total = sum(per_pole.values())
    pole_farm_max = max(per_pole.values())
    legs = WF.balance_legs(census, 16)
    farm_max = max(leg["n_tau"] for leg in legs)
    assert pole_farm_max / (total / 8) > 1.9        # the skew is real
    assert farm_max / (total / 16) < 1.05           # and the split removes it
    assert farm_max < pole_farm_max / 3.5


def test_the_balance_is_deterministic_for_a_given_census():
    census = _census()
    assert (WF.balance_legs(census, 16) == WF.balance_legs(census, 16))


def test_more_legs_than_groups_is_refused_rather_than_padded():
    census = _census(n_poles=1, n_groups=2, pole_tau=[100])
    with pytest.raises(ValueError, match="cannot have more legs"):
        WF.balance_legs(census, 999)


# ---------------------------------------------------------------------------
#  The census fold
# ---------------------------------------------------------------------------

def test_census_rows_carry_the_dispatch_count_that_the_balance_weighs(tmp_path):
    groups = [_FakeGroup("a", 12), _FakeGroup("b", 30)]
    groups[1].windows.append(_FakeWindow(5))
    rows = WF.census_rows_from_groups(groups, pole=2, bkey="neg_val",
                                      branch_tag="ω<E_F val")
    assert [r["n_tau"] for r in rows] == [12, 35]
    assert [r["index"] for r in rows] == [0, 1]
    p = tmp_path / "c.json"
    WF.write_census(p, rows, fit_store="/s.h5", n_p=4, sha="abc")
    back = WF.read_census(p)
    assert back["rows"] == rows and back["n_p"] == 4


def test_two_censuses_of_the_same_pole_are_refused(tmp_path):
    rows = WF.census_rows_from_groups([_FakeGroup("a", 3)], pole=0,
                                      bkey="pos_cond", branch_tag="t")
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    for p in (a, b):
        WF.write_census(p, rows, fit_store="/s.h5", n_p=2, sha="abc")
    with pytest.raises(ValueError, match="claimed by both"):
        WF.merge_census_files([a, b])


def test_censuses_from_two_shas_are_refused(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    WF.write_census(a, WF.census_rows_from_groups(
        [_FakeGroup("a", 3)], pole=0, bkey="pos_cond", branch_tag="t"),
        fit_store="/s.h5", n_p=2, sha="aaa")
    WF.write_census(b, WF.census_rows_from_groups(
        [_FakeGroup("a", 3)], pole=1, bkey="pos_cond", branch_tag="t"),
        fit_store="/s.h5", n_p=2, sha="bbb")
    with pytest.raises(ValueError, match="different shas"):
        WF.merge_census_files([a, b])


# ---------------------------------------------------------------------------
#  The manifest and its refusal — the integrity half
# ---------------------------------------------------------------------------

def _landed(path):
    with open(path, "wb") as f:
        f.write(b"\0" * 8192)


def test_the_merge_refuses_a_farm_that_lost_a_leg_and_names_it(tmp_path):
    census = _census()
    legs = WF.balance_legs(census, 16)
    out = tmp_path / "parts"
    out.mkdir()
    man = tmp_path / "manifest.json"
    WF.write_manifest(man, legs, kind="pass", fit_store="/store/fit.h5",
                      n_p=8, sha="deadbeef", out_dir=str(out), census=census)
    manifest = WF.read_manifest(man)
    for leg in manifest["legs"]:
        _landed(leg["output"])
    WF.refuse_incomplete(manifest, print_fn=lambda *a: None)   # green

    victim = manifest["legs"][12]
    os.remove(victim["output"])
    with pytest.raises(WF.FarmIncomplete) as exc:
        WF.refuse_incomplete(manifest, print_fn=lambda *a: None)
    assert victim["id"] in str(exc.value)
    assert victim["range_label"] in str(exc.value)
    assert victim["output"] in str(exc.value)


def test_a_leg_that_landed_as_a_stub_is_not_a_leg_that_landed(tmp_path):
    # $HOME filling is a measured failure on this fleet and it produces
    # files that exist, are nonzero and contain nothing.
    legs = WF.plan_fit_legs(64, 16)
    out = tmp_path / "fit"
    out.mkdir()
    man = tmp_path / "fit_manifest.json"
    WF.write_manifest(man, legs, kind="fit", fit_store="(none)",
                      out_dir=str(out))
    manifest = WF.read_manifest(man)
    for leg in manifest["legs"]:
        _landed(leg["output"])
    with open(manifest["legs"][3]["output"], "wb") as f:
        f.write(b"\0" * 38)
    with pytest.raises(WF.FarmIncomplete, match="38 bytes"):
        WF.refuse_incomplete(manifest, print_fn=lambda *a: None)


def test_the_fit_farms_missing_leg_is_named_by_its_q_range(tmp_path):
    # THE 2026-08-10 FAILURE, reproduced.  Leg 12 of the sixteen-way fit
    # farm died at placement and q [48, 52) was never fitted; the farm
    # wall printed, fifteen cost reports printed, and a fit store missing
    # four q looks exactly like a fit store.
    legs = WF.plan_fit_legs(64, 16)
    assert legs[12]["q_lo"] == 48 and legs[12]["q_hi"] == 52
    out = tmp_path / "fit"
    out.mkdir()
    man = tmp_path / "fit_manifest.json"
    WF.write_manifest(man, legs, kind="fit", fit_store="(none)",
                      out_dir=str(out))
    manifest = WF.read_manifest(man)
    for leg in manifest["legs"]:
        _landed(leg["output"])
    os.remove(manifest["legs"][12]["output"])
    with pytest.raises(WF.FarmIncomplete, match=r"q \[48, 52\)"):
        WF.refuse_incomplete(manifest, print_fn=lambda *a: None)


def test_there_is_no_allow_partial_on_the_completeness_check():
    # Stated as a cell because the pressure to add one arrives at 2 a.m.
    # with fifteen of sixteen legs green and a pool about to expire.
    import inspect

    sig = inspect.signature(WF.refuse_incomplete)
    assert set(sig.parameters) == {"manifest", "probe", "print_fn"}


def test_duplicate_leg_ids_are_refused_because_they_share_an_output(tmp_path):
    legs = WF.plan_fit_legs(8, 4)
    legs[2]["id"] = legs[1]["id"]
    with pytest.raises(ValueError, match="not unique"):
        WF.write_manifest(tmp_path / "m.json", legs, kind="fit",
                          fit_store="x", out_dir=str(tmp_path))


def test_the_manifest_records_the_universe_the_farm_must_cover(tmp_path):
    census = _census()
    legs = WF.balance_legs(census, 16)
    man = tmp_path / "m.json"
    WF.write_manifest(man, legs, kind="pass", fit_store="/s.h5", n_p=8,
                      sha="x", out_dir=str(tmp_path), census=census)
    doc = json.loads(open(man).read())
    assert len(doc["universe"]) == 32                # 8 poles x 4 branches
    assert all(v == 29 for v in doc["universe"].values())
    assert doc["total_n_tau"] == sum(r["n_tau"] for r in census["rows"])
