"""SC map 0 plans W's frequency sites and later maps hold them; current W stays live."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from common.units import RYD_TO_EV
from gw.shared_pole_recipe import bind_shared_pole_census, resolve_shared_pole_recipe
from gw.shared_pole_recipe import _sector_treatment_ceiling
from gw.qp_support import SigmaPlan


def inputs(gap=.7, *, eta=.25, tier="production", top=20.):
    config = NS(sigma=NS(w_model="shared_pole", w_accuracy=tier,
                         regularization_ev=eta), memory=NS(per_device_gb=30.),
                bispinor=False)
    wfns = NS(enk=np.array([[-20., -gap/2, gap/2, 999.]]*2)/RYD_TO_EV,
              occ=np.array([[1., 1., 0., 0.]]*2),
              slices=NS(b0=0, b4_logical=3, val=slice(0, 2),
                        cond=slice(2, 3), cond_all_logical=slice(2, 3)))
    volume = 4*np.pi*2 / (((top-3.5)/RYD_TO_EV/2)**2)
    meta = NS(nspin=1, nspinor=1, n_rmu=17, nk_tot=2, cell_volume=volume)
    rebind(wfns, meta)
    return config, wfns, meta


def rebind(wfns, meta):
    bind_shared_pole_census(wfns, meta, occupation_state=None, trs_allowed=True,
                           state_capacity=2., kweights=[.5, .5])


def plan(window=(-6., 4.), patches=(), etas=(), eta=.25):
    return SigmaPlan(np.asarray(window, float), tuple(window), eta, tuple(patches), tuple(etas))


def resolve(args, session=None, window=(-6., 4.), patches=(), etas=()):
    return resolve_shared_pole_recipe(
        *args, mesh_xy=NS(shape={"x": 2, "y": 2}), print_fn=lambda *_: None,
        sigma_plan=plan(window, patches, etas), support_session=session)


GEOMETRY = ("line_ev", "imaginary_ev", "held_line_ev", "held_imaginary_ev",
            "z_ry", "role", "distinct_id", "held", "support_pair", "fit_ids", "held_ids")


def assert_same_geometry(a, b):
    for key in GEOMETRY:
        assert a[key].dtype == b[key].dtype
        assert a[key].tobytes() == b[key].tobytes(), key
    assert a["recipe_hash"] == b["recipe_hash"]


def test_map0_plans_and_later_maps_hold_the_sites_but_rebind_current_state():
    args = inputs()
    session = {}
    first = resolve(args, session)
    assert first["support_plan"]["status"] == "plan"
    assert_same_geometry(first, resolve(args))
    ledger = args[2].shared_pole_capacity
    for gap in (1.48, 2., .4):          # the QP gap opens, then closes below the map-0 gap
        args[1].enk[:, 1:3] = np.array([-gap/2, gap/2])/RYD_TO_EV
        rebind(args[1], args[2])
        current = resolve(args, session)
        assert current["support_plan"]["status"] == "held"
        assert_same_geometry(first, current)
        assert current["census"]["gap_ev"] == pytest.approx(gap)
        assert current["support_plan"]["required"]["u_min_ev"] == pytest.approx(max(gap, 1.))
    assert args[2].shared_pole_capacity is not ledger
    assert set(session) == {"key", "plan"}


def test_line_ladder_reaches_the_map0_sigma_support_plus_the_pad():
    first = resolve(inputs(), {}, window=(-6., 4.))
    np.testing.assert_array_equal(first["line_ev"], np.arange(1., 9.))   # 6 + 2 eV at 4 eta = 1 eV
    assert first["top_ev"] == 8.


def test_a_later_plan_with_other_segments_refuses_by_name():
    args = inputs()
    session = {}
    first = resolve(args, session, window=(-6., 4.))
    assert_same_geometry(first, resolve(args, session, window=(-5.5, 5.5)))  # same lattice top
    with pytest.raises(ValueError, match="GATE shared_pole_line_coverage"):
        resolve(args, session, window=(-6.5, 4.))
    with pytest.raises(ValueError, match="GATE shared_pole_line_coverage"):
        resolve(args, session, window=(-6., 4.), patches=((4.001, 14.),), etas=(1.,))


def test_far_patches_add_coarse_segments_at_four_times_their_eta():
    r = resolve(inputs(), window=(-6., 4.), patches=((-30., -20.), (4.001, 14.)), etas=(2., 1.))
    # fine to 8 eV at 1 eV; eta_far 1 eV above E_F -> 4 eV to 16 (14 + 2);
    # eta_far 2 eV below -> 8 eV spacing from 16 to 32 (30 + 2).
    assert r["line_segments_ev"] == ((0., 8., 1.), (8., 16., 4.), (16., 32., 8.))
    np.testing.assert_array_equal(r["line_ev"], np.r_[np.arange(1., 9.), 12., 16., 24., 32.])
    np.testing.assert_array_equal(r["line_height_ev"], np.r_[np.ones(8), 4., 4., 8., 8.])
    heights = r["z_ry"].imag[r["role"] == 0] * RYD_TO_EV
    np.testing.assert_allclose(heights, r["line_height_ev"])
    held = r["z_ry"][r["role"] == 3]
    assert np.all(np.isin(np.round(held.imag * RYD_TO_EV, 9), [1., 4., 8.]))
    assert r["recipe_hash"] != resolve(inputs(), window=(-6., 4.))["recipe_hash"]


def test_recipe_hash_binds_the_ladder():
    a = resolve(inputs(), window=(-6., 4.))
    b = resolve(inputs(), window=(-7., 4.))
    assert a["line_count"] + 1 == b["line_count"]
    assert a["recipe_hash"] != b["recipe_hash"]
    c = resolve(inputs(top=30.))
    assert c["imaginary_ev"].tobytes() != a["imaginary_ev"].tobytes()
    assert c["recipe_hash"] != a["recipe_hash"]
    assert a["response_group_tolerance"] == 1e-9


def test_sector_treatment_ceiling_freezes_map0_and_masks_after_span_growth():
    from gw.shared_pole_gates import shared_pole_treatment_mask

    session = {}
    first = _sector_treatment_ceiling(4.0, session)
    assert first['status'] == 'initialized_map0'
    assert first['ceiling_ry'] == 8.0
    current = _sector_treatment_ceiling(3.5, session)
    assert current['status'] == 'reused_map0'
    assert current['ceiling_ry'] == 8.0
    expanded = _sector_treatment_ceiling(4.1, session)
    assert expanded['ceiling_ry'] == 8.0
    assert expanded['current_candidate_ceiling_ry'] == 8.2
    mask, row = shared_pole_treatment_mask(
        np.array([[4.0 ** 2, 9.0 ** 2]]), np.array([[True, True]]),
        ceiling_ry=expanded['ceiling_ry'])
    assert np.asarray(mask).tolist() == [[True, False]]
    assert np.asarray(row['retained_omega_max_ry']).tolist() == [4.0]


@pytest.mark.parametrize("changed", ["eta", "tier", "basis"])
def test_policy_or_basis_change_refuses_the_held_plan(changed):
    args = inputs()
    session = {}
    resolve(args, session)
    if changed == "eta":
        args[0].sigma.regularization_ev = .5
    elif changed == "tier":
        args[0].sigma.w_accuracy = "relaxed"
    else:
        args[2].n_rmu = 25
    with pytest.raises(ValueError, match="GATE shared_pole_held_plan"):
        resolve(args, session)


def test_current_census_is_required_even_with_a_held_plan():
    args = inputs()
    session = {}
    resolve(args, session)
    args[1].enk[0, 0] += .1
    with pytest.raises(ValueError, match="stale energies"):
        resolve(args, session)


def test_session_does_not_bypass_invalid_current_interval():
    session = {}
    resolve(inputs(), session)
    with pytest.raises(ValueError, match="GATE shared_pole_interval"):
        resolve(inputs(eta=5.), session)


def test_held_recipe_preserves_distinct_causal_training_and_holdout_points():
    session = {}
    resolve(inputs(), session)
    current = resolve(inputs(2.), session)
    assert np.all(current["z_ry"].imag > 0)
    assert not set(current["fit_ids"]) & set(current["held_ids"])
    for sample in set(current["distinct_id"]):
        rows = current["distinct_id"] == sample
        assert np.unique(current["z_ry"][rows]).size == 1
        assert np.unique(current["held"][rows]).size == 1


def test_independent_resolution_remains_current_and_has_no_session_receipt():
    first = resolve(inputs())
    current = resolve(inputs(2.))
    assert "support_plan" not in first and "support_plan" not in current
    assert first["imaginary_count"] == 4 and current["imaginary_count"] == 3


def test_failed_reference_validation_does_not_advance_session():
    session = {}
    args = inputs()
    args[1].enk[0, 0] += .1
    with pytest.raises(ValueError, match="stale energies"):
        resolve(args, session)
    assert session == {}
