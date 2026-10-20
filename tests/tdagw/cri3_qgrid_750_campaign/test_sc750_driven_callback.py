"""Host-only planning/parser/prerequisite oracles, NOT native TD execution."""
import ast
import copy
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("sc750_driven_callback", Path(__file__).with_name("sc750_driven_callback.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_bounded_plan_and_actual_calibration_selection_no_field_extrapolation():
    plan = driver.example_plan()
    assert driver.validate_plan(plan) is plan
    rows = [dict(tag="measured_low", completed=True, amplitude_v_m=3.e8, carrier_peak=.012),
            dict(tag="measured_high", completed=True, amplitude_v_m=8.e9, carrier_peak=.11)]
    chosen = driver.choose_calibrations(rows)
    assert [r["amplitude_v_m"] for r in chosen] == [3.e8, 8.e9]
    with pytest.raises(ValueError, match="did not reach"):
        driver.choose_calibrations(rows[:1])
    bad = copy.deepcopy(plan)
    bad["amplitude_candidates_v_m"] *= 2
    with pytest.raises(ValueError, match="at most six"):
        driver.validate_plan(bad)
    bad = copy.deepcopy(plan)
    bad["degeneracy_mode"] = "off"
    with pytest.raises(ValueError, match="multiplet"):
        driver.validate_plan(bad)


def test_native_parsed_B_C_decks_identical_except_W_update_and_A_is_truly_off():
    plan = driver.example_plan()
    source = dict(input_deck="authenticated_source.in", input_dir="private_native_source")
    decks = [driver.make_deck(plan, source, "same_output", slice(100, 144), 2.731, 4.e9, arm)
             for arm in driver.ARMS]
    a, b, c = decks
    assert a.values["lorrax.collision"] == "off"
    assert b.values["lorrax.collision"] == c.values["lorrax.collision"] == "raw"
    changes = {k for k in b.values if b.values[k] != c.values[k]}
    assert changes == {"lorrax.collision_screening"}
    assert b.values["lorrax.collision_screening"] == "joint_frozen"
    assert c.values["lorrax.collision_screening"] == "joint"
    assert (b.values["system.ibndmin"], b.values["system.ibndmax"], b.values["system.ivbm"]) == (101, 144, 130)
    assert b.trials == c.trials
    field = b.trials[0]["E_x"]
    sample = lambda t: eval(field, {"np": np, "HBAR_IN_EV_AS": 658.2119569}, {"t": t})
    assert sample(0.) == 0. and sample(1000.) == 0. and abs(sample(5000.)) < 1.e-19
    assert abs(sample(2400.)) > 1.e6
    assert a.measurements[-1][2]["brange1"] == [101, 145]


def test_matching_zero_decks_are_short_same_window_and_not_disabled_by_control_flag():
    plan = dict(driver.example_plan(), run_zero_controls=False)
    source = dict(input_deck="actual.in", input_dir="private_source")
    for arm in driver.ARMS[1:]:
        deck = driver.make_deck(plan, source, "matching_zero", slice(100, 144), 2.7, 0., arm, zero_gate=True)
        duration = (deck.values["control.timesteps.tf"]-deck.values["control.timesteps.t0"])*driver.FS_PER_AU
        assert duration == pytest.approx(.6)
        assert (deck.values["system.ibndmin"], deck.values["system.ibndmax"]) == (101, 144)
        assert all(deck.trials[0]["E_"+axis] == 0. for axis in "xyz")
        assert deck.values["lorrax.collision"] == "raw"
    with pytest.raises(ValueError, match="exactly zero"):
        driver.make_deck(plan, source, "bad", slice(100, 144), 2.7, 1., driver.ARMS[1], zero_gate=True)


def test_matching_zero_refresh_must_be_strictly_inside_gate_before_native_work():
    plan = dict(driver.example_plan(), execute_mgkba=True, refresh_fs=.5)
    assert driver.validate_plan(plan) is plan
    for cadence in (.6, 1.):
        with pytest.raises(ValueError, match="zero gate needs"):
            driver.validate_plan(dict(plan, refresh_fs=cadence))
    assert driver.validate_plan(dict(plan, execute_mgkba=False, refresh_fs=1.))["refresh_fs"] == 1.


def test_calibration_stops_on_first_actual_valid_distinct_target_pair():
    # .075 already satisfies the explicitly allowed factor-two .1 target;
    # do not run a stronger candidate merely to cross .1 literally.
    observations = [.008, .075, .15, .3]
    rows, tried = [], []
    for index, actual in enumerate(observations):
        tried.append(index)
        rows.append(dict(completed=True, carrier_peak=actual, amplitude_v_m=10.**(index+1), tag=str(index)))
        try:
            driver.choose_calibrations(rows, (.01, .1), 2.)
            break
        except ValueError:
            pass
    assert tried == [0, 1]
    assert [row["actual_A_carrier_peak"] for row in driver.choose_calibrations(rows)] == [.008, .075]
    tree = ast.parse(Path(driver.__file__).read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_td_probe")
    loop = next(n for n in run.body if isinstance(n, ast.For) and isinstance(n.iter, ast.Call)
                and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "enumerate")
    assert isinstance(loop.body[-1], ast.Try)
    assert any(isinstance(n, ast.Break) for n in loop.body[-1].body)


def test_four_band_or_wrong_policy_zero_never_admits_matching_44_gate():
    policy = dict(response_qgrid=[4, 4, 1], response_window="fejer")
    good = dict(completed=True, zero_gate=True, amplitude_v_m=0., duration_fs=.6,
        actual_active=[100, 144], response_policy=policy, accepted_refreshes=1,
        fresh_snapshot_times_fs=[0., .5])
    assert driver.require_matching_zero_result(good, slice(100, 144), policy) is good
    for field, value in (("actual_active", [128, 132]), ("zero_gate", False),
                         ("duration_fs", 10.), ("accepted_refreshes", 0),
                         ("fresh_snapshot_times_fs", [0.]),
                         ("response_policy", dict(policy, response_window="sharp"))):
        with pytest.raises(ValueError, match="matching-active"):
            driver.require_matching_zero_result(dict(good, **{field: value}), slice(100, 144), policy)


def test_both_mandatory_zero_results_precede_any_driven_MGKBA_loop():
    tree = ast.parse(Path(driver.__file__).read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_td_probe")
    execution = next(n for n in run.body if isinstance(n, ast.If)
        and isinstance(n.test, ast.Subscript) and ast.literal_eval(n.test.slice) == "execute_mgkba")
    loops = [n for n in execution.body if isinstance(n, ast.For)]
    assert len(loops) == 2
    calls = [n for n in ast.walk(loops[0]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "run_arm"]
    assert len(calls) == 1 and any(k.arg == "zero_gate" and ast.literal_eval(k.value) is True
                                  for k in calls[0].keywords)
    assert not any(isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
                   and n.slice.value == "run_zero_controls" for n in ast.walk(execution))


def _prerequisites(tmp_path):
    frame = dict(frame="actual_accepted_input", input_h_sha256="H", wfn_fingerprint="WFN")
    plan = driver.example_plan()
    plan["execute_mgkba"] = True
    plan["bse_report"] = str(tmp_path/"bse.json")
    zero = tmp_path/"zero"
    zero.mkdir()
    plan["zero_directory"] = str(zero)
    bse = dict(frame=frame, target_root_index=0, bright_subgap_root_indices=[0],
        energies_ev=[2.731], residual_norm_ev=[.0002], gap_ev=3.849, nval=4, ncond=4,
        strength_xyz_bohr2_cell=[[.1, 0., 0.]])
    (tmp_path/"bse.json").write_text(json.dumps(bse))
    (zero/"reference.json").write_text(json.dumps(dict(frame, sigma_eta_ev=.25, response_gamma_ev=.05)))
    (zero/"complete.json").write_text(json.dumps(dict(frame=frame["frame"], accepted_refreshes=1,
        status="SHORT_ZERO_FIELD_PROBE_COMPLETE_NOT_DRIVEN_ADMISSION")))
    bd = dict(weighted_trace_rate_ry=2.e-16, per_k=dict(empty_count=[1], filled_count=[1],
        empty_min_injection=[1.e-5], filled_min_blocking=[3.e-5]))
    for n in range(2):
        (zero/f"snapshot_{n:04d}_boundary.json").write_text(json.dumps(bd))
    with h5py.File(zero/"tdgw.h5", "w") as f:
        f["tdgw_data/checkpoint/it_next"] = 3
        f["tdgw_data/time_domain/carrier_density/data"] = np.array([[0., 0.], [.0001, .0001], [.0002, .0002], [999., 999.]])
    return plan, frame, zero


def test_actual_prerequisite_records_checkpoint_crop_and_fresh_number_gate(tmp_path):
    plan, frame, zero = _prerequisites(tmp_path)
    value = driver.validate_material_prerequisites(plan, frame)
    assert value["target_energy_ev"] == 2.731
    assert value["zero_carrier_max"] == .0002  # preallocated unwritten sentinel excluded
    path = zero/"snapshot_0001_boundary.json"
    bad = json.loads(path.read_text())
    bad["weighted_trace_rate_ry"] = 1.e-5
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="fresh number flux"):
        driver.validate_material_prerequisites(plan, frame)


@pytest.mark.parametrize("zero_case", ["missing", "invalid", "not_named"])
def test_A_only_valid_BSE_can_calibrate_without_raw_zero_artifacts(tmp_path, zero_case):
    plan, frame, zero = _prerequisites(tmp_path)
    plan["execute_mgkba"] = False
    if zero_case == "missing":
        plan["zero_directory"] = str(tmp_path/"does_not_exist")
    elif zero_case == "invalid":
        (zero/"complete.json").write_text("not valid JSON: raw zero refused")
    else:
        del plan["zero_directory"]
    result = driver.validate_material_prerequisites(plan, frame)
    assert result["target_energy_ev"] == 2.731
    assert result["raw_zero_status"] == "NOT_REQUIRED_STATIC_A_ONLY"
    assert result["zero_evidence"] is None and result["MGKBA_execution"] == "NOT_ATTEMPTED"
    assert "zero_complete_sha256" not in result
    # The same missing/refused artifact must still stop any B/C execution.
    with pytest.raises((FileNotFoundError, json.JSONDecodeError, KeyError)):
        driver.validate_material_prerequisites(dict(plan, execute_mgkba=True), frame)


def test_A_only_still_refuses_bad_BSE_and_live_frame_or_coarse_policy(tmp_path):
    plan, frame, _ = _prerequisites(tmp_path)
    plan["execute_mgkba"] = False
    with pytest.raises(ValueError, match="identities differ"):
        driver.validate_material_prerequisites(plan, dict(frame, input_h_sha256="wrong"))
    policy = dict(response_qgrid=[4, 4, 1], response_window="fejer")
    with pytest.raises(ValueError, match="BSE response grid/window"):
        driver.validate_material_prerequisites(plan, frame, response_policy=policy)
    path = Path(plan["bse_report"])
    data = json.loads(path.read_text())
    data["target_root_index"] = None
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="no actual resolved bright"):
        driver.validate_material_prerequisites(plan, frame)


def test_coarse_driven_requires_same_response_policy_in_BSE_and_zero(tmp_path):
    plan, frame, zero = _prerequisites(tmp_path)
    policy = dict(response_qgrid=[4, 4, 1], response_window="fejer")
    with pytest.raises(ValueError, match="response grid/window"):
        driver.validate_material_prerequisites(plan, frame, response_policy=policy)
    for path in (Path(plan["bse_report"]), zero/"reference.json"):
        record = json.loads(path.read_text())
        path.write_text(json.dumps(dict(record, **policy)))
    assert driver.validate_material_prerequisites(plan, frame, response_policy=policy)["target_energy_ev"] == 2.731
    path = Path(plan["bse_report"])
    record = json.loads(path.read_text())
    record["response_window"] = "sharp"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="response grid/window"):
        driver.validate_material_prerequisites(plan, frame, response_policy=policy)


def test_unresolved_or_mismatched_material_target_is_not_a_pulse_choice(tmp_path):
    plan, frame, _ = _prerequisites(tmp_path)
    with pytest.raises(ValueError, match="identities differ"):
        driver.validate_material_prerequisites(plan, dict(frame, input_h_sha256="other"))
    path = Path(plan["bse_report"])
    bse = json.loads(path.read_text())
    bse["target_root_index"] = None
    path.write_text(json.dumps(bse))
    with pytest.raises(ValueError, match="no actual"):
        driver.validate_material_prerequisites(plan, frame)


def test_actual_axis_brightness_not_inferred_from_total_inplane_strength(tmp_path):
    plan, frame, _ = _prerequisites(tmp_path)
    path = Path(plan["bse_report"])
    bse = json.loads(path.read_text())
    bse["strength_xyz_bohr2_cell"] = [[0., .2, 0.]]
    path.write_text(json.dumps(bse))
    with pytest.raises(ValueError, match="selected Cartesian axis"):
        driver.validate_material_prerequisites(plan, frame)
    plan["pulse"]["polarization"] = [0., -1., 0.]
    result = driver.validate_material_prerequisites(plan, frame)
    assert result["target_energy_ev"] == 2.731 and result["target_axis"] == "y"
    plan["pulse"]["polarization"] = [2**-.5, 2**-.5, 0.]
    with pytest.raises(ValueError, match="Cartesian axis"):
        driver.validate_plan(plan)


def test_strict_zero_callback_unchanged_and_no_exception_path_collective_save():
    tree = ast.parse(Path(driver.__file__).read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_td_probe")
    handlers = [n for n in ast.walk(run) if isinstance(n, ast.ExceptHandler)]
    assert all(isinstance(n.type, ast.Name) and n.type.id in ("OSError", "ValueError") for n in handlers)
    assert not any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "save"
                   for h in handlers for x in ast.walk(h))
    assert "sc750_zero_joint_0p6fs.toml" in Path(driver.__file__).read_text()
    # Existing strict-zero parser remains its separate entry point.
    zero = driver.sibling("sc750_td_callback")
    assert zero.validate_probe_deck is not driver.validate_plan


def test_plot_payload_preserves_exact_time_complex_P_and_crops_unwritten_rows(tmp_path):
    path = tmp_path/"tdgw.h5"
    with h5py.File(path, "w") as f:
        f["tdgw_data/checkpoint/it_next"] = 3
        f["tdgw_data/checkpoint/tchkp"] = 10./driver.FS_PER_AU
        for key, data in (("polarization", np.full((4, 3), 2.+.1j)),
                          ("Efield", np.ones((4, 3))),
                          ("carrier_density", np.array([[0., 0.], [.01, .01], [.1, .1], [999., 999.]]))):
            f["tdgw_data/time_domain/"+key+"/times"] = np.array([0., 1., 2., 999.])
            f["tdgw_data/time_domain/"+key+"/data"] = data
    data, receipt = driver.completed_observables(path)
    assert receipt["carrier_peak"] == .1 and receipt["checkpoint_fs"] == 10.
    assert receipt["completed_samples"] == 3
    np.testing.assert_array_equal(data["polarization_au"].imag, np.full((3, 3), .1))
    np.testing.assert_array_equal(data["time_fs"], np.array([0., 1., 2.])*driver.FS_PER_AU)
    with h5py.File(path, "a") as f:
        f["tdgw_data/time_domain/Efield/times"][1] = 1.01
    with pytest.raises(ValueError, match="times differ"):
        driver.completed_observables(path)


@pytest.mark.parametrize("defect", ["beyond_length", "nonmonotonic", "checkpoint", "incomplete"])
def test_completed_payload_refuses_truncated_or_inconsistent_native_coverage(tmp_path, defect):
    path = tmp_path/"tdgw.h5"
    times = np.arange(4)*.05/driver.FS_PER_AU
    with h5py.File(path, "w") as f:
        f["tdgw_data/checkpoint/it_next"] = 5 if defect == "beyond_length" else 4
        f["tdgw_data/checkpoint/tchkp"] = (.1 if defect == "checkpoint" else .2)/driver.FS_PER_AU
        for key, shape in (("polarization", (4, 3)), ("Efield", (4, 3)), ("carrier_density", (4, 2))):
            t = times.copy()
            if defect == "nonmonotonic":
                t[2] = t[1]
            f["tdgw_data/time_domain/"+key+"/times"] = t
            f["tdgw_data/time_domain/"+key+"/data"] = np.zeros(shape)
    with pytest.raises(ValueError, match="driven_samples"):
        driver.completed_observables(path, duration_fs=10. if defect == "incomplete" else .2,
                                     measure_interval_fs=.05)


def test_all_three_arm_labels_and_equilibrium_shared_handle_are_explicit():
    assert driver.ARMS == ("A_STATIC_TDAGW_NO_COLLISION", "B_FROZEN_W_MGKBA", "C_UPDATED_W_MGKBA")
    tree = ast.parse(Path(driver.__file__).read_text())
    observe = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "observe")
    text = ast.unparse(observe)
    assert "cache.update(equilibrium=shared_equilibrium['handle']" in text
    assert "np.array_equal" in text
    assert "value = shared_equilibrium['row']" in text


def test_window_gaps_and_mixing_do_not_count_degenerate_gauge_rotation():
    e = np.array([[0., 0., 1., 2.], [.1, .1, .9, 2.]])
    U = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    U[:, :2, :2] = np.array([[1., 1.], [-1., 1.]])/np.sqrt(2)
    result = driver.gap_mixing_summary(U, e, e, 2)
    assert result["minimum_direct_gap_ry"] == pytest.approx(.8)
    assert result["fundamental_gap_ry"] == pytest.approx(.8)
    assert result["mixing_outside_reference_degenerate_blocks_max"] == 0.
    U = U[:, :, [2, 1, 0, 3]]
    result = driver.gap_mixing_summary(U, e, e, 2)
    assert result["mixing_outside_reference_degenerate_blocks_max"] == 1.
    assert "NOT occupation/full750" in result["scope"]
