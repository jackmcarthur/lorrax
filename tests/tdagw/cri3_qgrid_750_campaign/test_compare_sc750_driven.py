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
