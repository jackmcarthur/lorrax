"""Host-only actual-sample/scale-separation checks; no material data generated."""
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("compare750", Path(__file__).with_name("compare_sc750_driven.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_scale_change_is_distinguished_from_waveform_shape_change():
    x = np.array([[1., 0., 0.], [0., 1., 0.]])
    change = module.waveform_difference(x, 1.03*x)
    assert change["relative_l2"] == pytest.approx(.03)
    assert change["fitted_real_scale"] == pytest.approx(1.03)
    assert change["shape_l2_after_scale"] < 1.e-15
    distorted = 1.03*x+np.array([[0., .1, 0.], [.1, 0., 0.]])
    assert module.waveform_difference(x, distorted)["shape_l2_after_scale"] == pytest.approx(.1)


def test_exact_axes_and_drive_required_without_interpolation():
    a = dict(time_fs=np.array([0., .05]), field=np.ones((2, 3)), weights=np.ones(64)/64,
             kpts=np.zeros((64, 3)))
    b = {key: value.copy() for key, value in a.items()}
    module.exact_common([a, b])
    b["time_fs"][1] += 1.e-12
    with pytest.raises(ValueError, match="time_fs mismatch"):
        module.exact_common([a, b])
    b["time_fs"] = a["time_fs"].copy()
    b["field"][0, 0] += 1.e-12
    with pytest.raises(ValueError, match="field mismatch"):
        module.exact_common([a, b])


def test_undefined_or_nonfinite_relative_waveforms_refuse():
    with pytest.raises(ValueError, match="zero reference"):
        module.waveform_difference(np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="finite matching"):
        module.waveform_difference(np.ones(3), np.full(3, np.nan))


def test_complete_h5_requires_actual_sample_coverage_and_matching_arm(tmp_path):
    """Synthetic tiny observable fixture, not a generated material result."""
    arm = module.ARMS[0]
    (tmp_path/"complete.json").write_text(json.dumps(dict(completed=True, arm=arm, amplitude_v_m=1.e8)))
    with h5py.File(tmp_path/"tdgw.h5", "w") as f:
        cp = f.create_group("tdgw_data/checkpoint")
        cp["it_next"], cp["tchkp"] = 3, .15/module.FS_PER_AU
        head = f.create_group("tdgw_header/wfn_header")
        head["w_k"], head["kpts"] = np.ones(64)/64, np.zeros((64, 3))
        bands = head.create_group("bands")
        for name, value in zip(("start", "stop", "step"), (101, 145, 1)):
            bands[name] = value
        td = f["tdgw_data"].create_group("time_domain")
        for name, cols in (("polarization", 3), ("Efield", 3), ("carrier_density", 2)):
            group = td.create_group(name)
            group["times"] = np.arange(3)*.05/module.FS_PER_AU
            group["data"] = np.ones((3, cols), complex)
    result = module.read_run(tmp_path, arm, 1.e8, [100, 144], .15, 50.)
    assert result["polarization"].shape == (3, 3)
    with pytest.raises(ValueError, match="completed-arm"):
        module.read_run(tmp_path, module.ARMS[1], 1.e8, [100, 144], .15, 50.)
    with h5py.File(tmp_path/"tdgw.h5", "a") as f:
        f["tdgw_data/checkpoint/it_next"][...] = 2
    with pytest.raises(ValueError, match="invalid actual"):
        module.read_run(tmp_path, arm, 1.e8, [100, 144], .15, 50.)


def _calibration_campaign(root, amplitudes=(1.e8, 4.e8)):
    """Synthetic small native-format samples only; not material results."""
    policy = dict(response_qgrid=[4, 4, 1], response_window="fejer")
    manifest = dict(execution_revision="synthetic_test", actual_active=[100, 144],
        response_policy=policy, plan=dict(duration_fs=10., measure_as=50.,
            carrier_targets=[.01, .1], pulse=dict(polarization=[1., 0., 0.], end_fs=5.)))
    (root/"manifest.json").write_text(json.dumps(manifest))
    t = np.arange(200)*.05
    selected = []
    for i, amp in enumerate(amplitudes):
        tag = f"A_calibration_{i:02d}"
        directory = root/tag
        directory.mkdir()
        shape = np.column_stack((np.sin(t), np.zeros((200, 2))))
        pol, field = shape*amp*1.e-9, shape*amp*1.e-12
        carrier = np.column_stack((t*.001*(i+1), t*.0009*(i+1)))
        row = dict(tag=tag, completed=True, arm=module.ARMS[0], amplitude_v_m=amp,
            zero_gate=False, actual_active=[100, 144], response_policy=policy, duration_fs=10.,
            completed_samples=200, checkpoint_fs=10.-1.e-14,
            carrier_peak=float(carrier[:, 0].max()), hole_peak=float(carrier[:, 1].max()))
        for path in (root/f"calibration_{i:02d}.json", directory/"complete.json"):
            path.write_text(json.dumps(row))
        with h5py.File(directory/"tdgw.h5", "w") as f:
            cp = f.create_group("tdgw_data/checkpoint")
            cp["it_next"], cp["tchkp"] = 200, 10./module.FS_PER_AU
            header = f.create_group("tdgw_header/wfn_header")
            header["w_k"], header["kpts"] = np.ones(64)/64, np.zeros((64, 3))
            bands = header.create_group("bands")
            for name, value in zip(("start", "stop", "step"), (101, 145, 1)):
                bands[name] = value
            td = f["tdgw_data"].create_group("time_domain")
            for name, value in (("polarization", pol), ("Efield", field), ("carrier_density", carrier)):
                group = td.create_group(name)
                group["times"], group["data"] = t/module.FS_PER_AU, value.astype(complex)
        # Match the producer's native→fs expression exactly.
        np.savez(directory/"plot_data.npz", time_fs=(t/module.FS_PER_AU)*module.FS_PER_AU,
            polarization_au=pol.astype(complex), field_au=field.astype(complex),
            conduction_electrons_per_cell=carrier[:, 0], valence_holes_per_cell=carrier[:, 1])
        selected.append(dict(target_carriers=manifest["plan"]["carrier_targets"][i%2],
            amplitude_v_m=amp, actual_A_carrier_peak=row["carrier_peak"], calibration_tag=tag))
    return manifest, selected


def test_calibration_mode_accepts_only_completed_legs_of_running_campaign(tmp_path):
    _calibration_campaign(tmp_path)
    (tmp_path/"A_calibration_02").mkdir()  # An uncommitted running leg is NOT plotted.
    _, rows, items, report = module.calibration_inputs(tmp_path)
    assert len(rows) == len(items) == 2
    assert report["selected"] is None
    assert report["campaign_status"] == "CAMPAIGN_NOT_COMPLETED_COMPLETED_A_LEGS_ONLY"
    assert report["scope"] == "STATIC_A_CALIBRATION_ONLY_NO_BC_NO_DAMPING_INFERENCE"
    assert report["amplitude_normalized_nonlinearity"][1]["P_per_amplitude_versus_reference"]["relative_l2"] < 1.e-14


def test_calibration_mode_rejects_npz_and_native_checkpoint_mismatches(tmp_path):
    _calibration_campaign(tmp_path, (1.e8,))
    path = tmp_path/"A_calibration_00/plot_data.npz"
    with np.load(path) as f:
        payload = dict(f)
    original = payload["polarization_au"].copy()
    payload["polarization_au"][0, 0] += .01
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="native H5/plot NPZ mismatch"):
        module.calibration_inputs(tmp_path)
    payload["polarization_au"] = original
    np.savez(path, **payload)
    with h5py.File(tmp_path/"A_calibration_00/tdgw.h5", "a") as f:
        f["tdgw_data/checkpoint/it_next"][...] = 199
    with pytest.raises(ValueError, match="invalid actual"):
        module.calibration_inputs(tmp_path)


def test_calibration_mode_requires_exact_gauge_and_proportional_actual_drive(tmp_path):
    _calibration_campaign(tmp_path)
    native = tmp_path/"A_calibration_01/tdgw.h5"
    with h5py.File(native, "a") as f:
        f["tdgw_header/wfn_header/kpts"][0, 0] = 1.e-12
    with pytest.raises(ValueError, match="exact kpts"):
        module.calibration_inputs(tmp_path)
    with h5py.File(native, "a") as f:
        f["tdgw_header/wfn_header/kpts"][0, 0] = 0.
        field = f["tdgw_data/time_domain/Efield/data"][:]
        field[0, 0] += .01
        f["tdgw_data/time_domain/Efield/data"][...] = field
    path = tmp_path/"A_calibration_01/plot_data.npz"
    with np.load(path) as f:
        payload = dict(f)
    payload["field_au"] = field
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="normalized applied field"):
        module.calibration_inputs(tmp_path)


def test_calibration_selection_must_join_actual_amplitude_and_peak(tmp_path):
    _, selected = _calibration_campaign(tmp_path)
    path = tmp_path/"calibration_selected.json"
    path.write_text(json.dumps(dict(arm=module.ARMS[0], selected=selected)))
    assert module.calibration_inputs(tmp_path)[-1]["selected"] == selected
    selected[0]["actual_A_carrier_peak"] += .01
    path.write_text(json.dumps(dict(arm=module.ARMS[0], selected=selected)))
    with pytest.raises(ValueError, match="does not join actual"):
        module.calibration_inputs(tmp_path)


def test_explicit_calibration_subset_does_not_require_unrequested_payloads(tmp_path):
    _calibration_campaign(tmp_path, (1.e8,))
    # Only a metadata receipt has been copied for another completed leg.
    (tmp_path/"calibration_01.json").write_text(json.dumps(dict(tag="A_calibration_01")))
    with pytest.raises(FileNotFoundError):
        module.calibration_inputs(tmp_path)
    report = module.calibration_inputs(tmp_path, tags=["A_calibration_00"])[-1]
    assert report["requested_calibration_tags"] == ["A_calibration_00"]
    assert report["unplotted_committed_tags"] == ["A_calibration_01"]
    assert report["selected"] is None
    with pytest.raises(ValueError, match="distinct committed"):
        module.calibration_inputs(tmp_path, tags=["A_calibration_99"])


def test_strict_default_still_refuses_calibration_only_campaign(tmp_path):
    _calibration_campaign(tmp_path, (1.e8,))
    (tmp_path/"complete.json").write_text(json.dumps(
        dict(status="A_CALIBRATION_ONLY_COMPLETE_BC_NOT_ATTEMPTED")))
    with pytest.raises(ValueError, match="completed three-arm"):
        module.main([str(tmp_path), "--out", str(tmp_path/"plots")])
    assert not (tmp_path/"plots").exists()
