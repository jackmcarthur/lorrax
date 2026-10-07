"""Shared-pole line supports placed over the requested reads (CPU only, seconds).

1. ``qp_support.requested_reads_ev`` pads each requested state by +-P on the Sigma step
   and ignores states outside the mask.
2. The strip-law count is held between today's floor and the 40-sample cap.
3. A far requested state moves the line top out to it; a fixed +-5 eV window did not
   (the defect of claims 2868/3268).
4. The Sigma planner names a sample group that reads outside the support, and warns
   only.
5. With the recipe's 5 eV read pad the line top is never below the former rule's
   (states within +-5 eV of mu at +-5 eV offsets) when those states are requested.
6. An SC map whose line count changes re-plans with a finite, hashable receipt; a smaller
   count keeps the held one (the count only grows).
"""
import numpy as np


def test_requested_reads_pad_and_mask():
    from gw.qp_support import requested_reads_ev
    energy = np.array([[-1.0, 0.5, 30.0]])
    reads = requested_reads_ev(energy, np.array([[True, True, False]]), 0.25, pad_ev=2.0)
    assert reads.size == 2 * 17
    assert np.isclose(reads.min(), -3.0) and np.isclose(reads.max(), 2.5)


def test_line_count_strip_law_clamped():
    from gw.shared_pole_recipe import support_line_count
    n_near, rec = support_line_count(np.array([-3.0, 3.0]), 2.6, 4, 4)
    assert n_near == rec["floor"] == 14
    n_far, rec = support_line_count(np.array([-3.0, 60.0]), 2.6, 4, 4)
    assert n_far == rec["cap"] == 32 and rec["strip"] > 32
    n_mid, rec = support_line_count(np.array([-12.0, 20.8]), 2.6, 4, 4)
    assert n_mid == rec["strip"] == 19


def test_far_state_moves_the_line_top():
    from gw.qp_support import requested_reads_ev
    from gw.shared_pole_recipe import support_line_count, support_rule_line_sites
    rng = np.random.default_rng(0)
    levels = np.sort(np.concatenate([rng.uniform(-10.0, 3.0, 400), rng.uniform(3.0, 30.0, 200)]))
    near = np.abs(levels) <= 5.0
    reads_near = requested_reads_ev(levels, near, 0.25)
    reads_all = requested_reads_ev(levels, np.ones(levels.shape, bool), 0.25)
    top_near = support_rule_line_sites(levels, 0.0, 0.25, 2.6, reads_near, 14)[-1]
    n, _ = support_line_count(reads_all, 2.6, 4, 4)
    sites = support_rule_line_sites(levels, 0.0, 0.25, 2.6, reads_all, n)
    assert top_near < 11.0 and sites[-1] > 25.0
    assert np.all(np.diff(sites) > 0.0) and np.isclose(sites[0], 2.6, atol=0.01)


def test_planner_names_reads_past_support():
    from gw.sigma_box_plan import support_reads_past
    from common.units import RYD_TO_EV
    omega = np.array([-20.0, -10.0, 0.0, 10.0]) / RYD_TO_EV
    group = np.array([0, -1, -1, -1])
    past = support_reads_past(omega, group, (-12.0 / RYD_TO_EV, 12.0 / RYD_TO_EV))
    assert [p["group"] for p in past] == [0]
    assert support_reads_past(omega, group, (-25.0 / RYD_TO_EV, 12.0 / RYD_TO_EV)) == []


def test_read_pad_keeps_the_former_reach():
    from gw.qp_support import requested_reads_ev
    from gw.shared_pole_recipe import SUPPORT_READ_PAD_EV, support_rule_line_sites
    rng = np.random.default_rng(1)
    levels = np.sort(rng.uniform(-6.0, 6.0, 300))
    old_reads = (levels[np.abs(levels) <= 5.0][:, None] + np.arange(-5.0, 5.125, 0.25)[None, :]).ravel()
    new_reads = requested_reads_ev(levels, np.abs(levels) <= 5.0, 0.25, pad_ev=SUPPORT_READ_PAD_EV)
    old_top = support_rule_line_sites(levels, 0.0, 0.25, 2.6, old_reads, 14)[-1]
    new_top = support_rule_line_sites(levels, 0.0, 0.25, 2.6, new_reads, 14)[-1]
    assert SUPPORT_READ_PAD_EV == 5.0 and new_top >= old_top - 1e-9


def test_line_count_change_between_maps_keeps_a_finite_hashable_recipe():
    # An SC map whose line count differs from the held one re-plans with a finite
    # (or null) drift: the receipt is hashed with the recipe under allow_nan=False.
    from gw.shared_pole_recipe import _held_line_sites, table_hash
    session = {"epoch": 0}
    sites = lambda n: np.linspace(2.6, 20.0, n)  # noqa: E731
    rule = lambda n: {"reach_ev": 20.0, "strip": n, "floor": 14, "cap": 32, "count": n}  # noqa: E731
    first = {"status": "initial"}
    line, _ = _held_line_sites(sites, 14, rule(14), first, session)
    assert line.size == 14 and session["line_count"] == 14
    grow = {"status": "hit"}
    line, _ = _held_line_sites(sites, 16, rule(16), grow, session)
    assert line.size == 16 and grow["status"] == "expanded"
    assert grow["line_site_drift_ev"] is None and grow["line_site_count"] == [14, 16]
    table_hash(grow)
    shrink = {"status": "hit"}
    line, used = _held_line_sites(sites, 15, rule(15), shrink, session)
    assert line.size == 16 and used["count"] == 16 and shrink["status"] == "hit"
    assert shrink["line_site_drift_ev"] == 0.0
    table_hash(shrink)
    reference = {"status": "initial_reference"}
    line, _ = _held_line_sites(sites, 15, rule(15), reference, session)
    assert line.size == 15
