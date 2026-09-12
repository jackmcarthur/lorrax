"""SC support geometry encloses current requirements; current W stays live."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from common.units import RYD_TO_EV
from gw.shared_pole_recipe import bind_shared_pole_census, resolve_shared_pole_recipe


def inputs(gap=.7, *, eta=.25, tier="production", plasma_seed=10.):
    """``plasma_seed`` is omega_p of the shallow occupied band alone, in eV.

    It stays under 14.1, so the 20 eV band never joins the active set and the
    geometry these tests exercise is the shallow manifold's, not a fixed-point
    flip (that transition is covered in test_shared_pole_inputs).
    """
    config = NS(sigma=NS(w_model="shared_pole", w_accuracy=tier,
                         regularization_ev=eta, omega_min_ev=-5., omega_max_ev=5.,
                         parsed_omega_patches_ev=list),
                memory=NS(per_device_gb=30.))
    wfns = NS(enk=np.array([[-20., -gap/2, gap/2, 999.]]*2)/RYD_TO_EV,
              occ=np.array([[1., 1., 0., 0.]]*2),
              slices=NS(b0=0, b4_logical=3, val=slice(0, 2),
                        cond_all_logical=slice(2, 3)))
    volume = 4*np.pi*2 / ((plasma_seed/RYD_TO_EV/2)**2)
    meta = NS(nspin=1, nspinor=1, n_rmu=17, nk_tot=2, cell_volume=volume,
              b_id_4_chi_user=3)
    rebind(wfns, meta)
    return config, wfns, meta


def rebind(wfns, meta):
    bind_shared_pole_census(wfns, meta, occupation_state=None, trs_allowed=True,
                           state_capacity=2., kweights=[.5, .5])


def resolve(args, session=None):
    return resolve_shared_pole_recipe(
        *args, mesh_xy=NS(shape={"x": 2, "y": 2}), print_fn=lambda *_: None,
        support_session=session)


def assert_same_geometry(a, b):
    for key in ("line_ev", "imaginary_ev", "held_line_ev", "held_imaginary_ev",
                "z_ry", "role", "distinct_id", "held", "support_pair",
                "fit_ids", "held_ids"):
        assert a[key].dtype == b[key].dtype
        assert a[key].tobytes() == b[key].tobytes(), key


def interacting_session(args):
    session = {}
    reference = resolve(args, session)
    assert reference["support_envelope"]["status"] == "initial_reference"
    assert "envelope" not in session
    return session, resolve(args, session)


def test_gap_growth_keeps_points_roles_but_rebinds_current_state():
    args = inputs()
    session = {}
    reference = resolve(args, session)
    assert reference["imaginary_count"] == 4
    assert_same_geometry(reference, resolve(args))
    args[1].enk[:, 1:3] = np.array([-.8, .8])/RYD_TO_EV
    rebind(args[1], args[2])
    first = resolve(args, session)
    assert first["support_envelope"]["status"] == "initial"
    assert first["support_envelope"]["epoch"] == 0
    ledger = args[2].shared_pole_capacity
    args[1].enk[:, 1:3] = np.array([-1., 1.])/RYD_TO_EV
    rebind(args[1], args[2])
    current = resolve(args, session)
    assert_same_geometry(first, current)
    assert first["imaginary_count"] == current["imaginary_count"] == 3
    assert current["census"]["gap_ev"] == pytest.approx(2.)
    assert first["census"]["energy_sha256"] != current["census"]["energy_sha256"]
    assert args[2].shared_pole_capacity is not ledger
    assert not np.array_equal(current["imaginary_ev"], resolve(args)["imaginary_ev"])
    assert current["support_envelope"]["status"] == "hit"
    assert current["support_envelope"]["epoch"] == 0
    assert set(session) == {"reference_complete", "key", "envelope", "epoch"}
    assert all(isinstance(v, float) for v in session["envelope"].values())


def test_gap_shrink_expands_once_then_growth_stays_enclosed():
    args = inputs(3.)
    session, first = interacting_session(args)
    args[1].enk[:, 1:3] = np.array([-.6, .6])/RYD_TO_EV
    rebind(args[1], args[2])
    expanded = resolve(args, session)
    assert expanded["support_envelope"]["status"] == "expanded"
    assert expanded["support_envelope"]["epoch"] == 1
    assert expanded["u_min_ev"] == pytest.approx(1.2)
    assert expanded["kappa"] > first["kappa"]
    assert not np.array_equal(first["imaginary_ev"], expanded["imaginary_ev"])
    args[1].enk[:, 1:3] = np.array([-.8, .8])/RYD_TO_EV
    rebind(args[1], args[2])
    assert_same_geometry(expanded, resolve(args, session))


def test_plasma_interval_expands_and_never_shrinks():
    session, first = interacting_session(inputs(plasma_seed=10.))
    expanded = resolve(inputs(plasma_seed=12.), session)
    assert expanded["support_envelope"]["status"] == "expanded"
    assert expanded["top_ev"] > first["top_ev"]
    assert expanded["u_max_ev"] > first["u_max_ev"]
    contracted = resolve(inputs(plasma_seed=9.), session)
    assert_same_geometry(expanded, contracted)
    assert contracted["plasma_ev"] < expanded["plasma_ev"]
    assert contracted["support_envelope"]["required"]["line_top_ev"] < contracted["top_ev"]


@pytest.mark.parametrize("changed", ["eta", "tier", "basis"])
def test_policy_or_basis_change_starts_a_new_envelope(changed):
    args = inputs()
    session, first = interacting_session(args)
    if changed == "eta":
        args[0].sigma.regularization_ev = .5
    elif changed == "tier":
        args[0].sigma.w_accuracy = "relaxed"
    else:
        args[2].n_rmu = 25
    current = resolve(args, session)
    assert current["support_envelope"]["status"] == "policy_changed"
    assert current["support_envelope"]["epoch"] == 1
    assert_same_geometry(current, resolve(args))
    if changed == "eta":
        assert current["height_ev"] == 2*first["height_ev"]
        assert current["u_min_ev"] >= current["height_ev"]


def test_current_census_is_required_even_with_an_existing_envelope():
    args = inputs()
    session, _ = interacting_session(args)
    args[1].enk[0, 0] += .1
    with pytest.raises(ValueError, match="stale energies"):
        resolve(args, session)


def test_session_does_not_bypass_invalid_current_interval():
    session, _ = interacting_session(inputs())
    with pytest.raises(ValueError, match="GATE shared_pole_interval"):
        resolve(inputs(eta=7.), session)


def test_enclosed_recipe_preserves_distinct_causal_training_and_holdout_points():
    session, _ = interacting_session(inputs())
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
    assert "support_envelope" not in first and "support_envelope" not in current
    assert first["imaginary_count"] == 4 and current["imaginary_count"] == 3


def test_reference_interval_is_not_retained_by_first_interacting_map():
    session = {}
    reference = resolve(inputs(.7, plasma_seed=12.), session)
    assert reference["support_envelope"]["status"] == "initial_reference"
    assert reference["support_envelope"]["epoch"] == -1
    assert session == {"reference_complete": True}
    current = resolve(inputs(1.5, plasma_seed=10.), session)
    assert_same_geometry(current, resolve(inputs(1.5, plasma_seed=10.)))
    assert current["top_ev"] < reference["top_ev"]
    assert current["u_min_ev"] > reference["u_min_ev"]
    assert current["support_envelope"]["status"] == "initial"
    assert current["support_envelope"]["epoch"] == 0


def test_failed_reference_validation_does_not_advance_session():
    session = {}
    args = inputs()
    args[1].enk[0, 0] += .1
    with pytest.raises(ValueError, match="stale energies"):
        resolve(args, session)
    assert session == {}
