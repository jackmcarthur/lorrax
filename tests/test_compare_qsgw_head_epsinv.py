import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "tools" / "compare_qsgw_head_epsinv.py"
SPEC = importlib.util.spec_from_file_location("compare_qsgw_head_epsinv", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_lorrax_schema_and_epsinvdyn_real_branch(tmp_path):
    lorrax_path = tmp_path / "lorrax_head.dat"
    lorrax_path.write_text(
        "# source: synthetic QSGW head\n"
        "# broadening_ev: 0.10\n"
        "# columns: omega_ev Re_epsinv00 Im_epsinv00 Re_chi00 Im_chi00\n"
        "0.0 1.0 0.0 2.0 0.0\n"
        "1.0 2.0 -0.5 3.0 -0.25\n",
    )
    bgw_path = tmp_path / "EpsInvDyn"
    bgw_path.write_text(
        "# q= 0.25 0.0 0.0 nmtx= 1\n"
        "0.0 9.0 0.0\n"
        "1.0 9.0 0.0\n"
        "# q= 0.0 0.0 0.125 nmtx= 1\n"
        "0.0 1.0 0.0\n"
        "1.0 2.0 -0.5\n"
        "0.0 4.0 1.0\n",
    )

    lorrax = MODULE.read_lorrax_head(lorrax_path)
    bgw = MODULE.read_epsinvdyn(bgw_path, broadening_ev=0.10)
    result = MODULE.align_and_compare(lorrax, bgw)

    assert "chi00" in lorrax
    assert bgw["q_index"] == 1
    assert bgw["ignored_nonreal_rows"] == 1
    assert result["alignment"] == "direct"
    np.testing.assert_allclose(result["difference"], 0.0)


def test_interpolation_is_explicit_and_extrapolation_is_rejected(tmp_path):
    lorrax_path = tmp_path / "lorrax_head.dat"
    lorrax_path.write_text(
        "# broadening_ev: 0.2\n"
        "0.0 1.0 0.0\n"
        "1.0 2.0 1.0\n"
        "2.0 3.0 2.0\n",
    )
    lorrax = MODULE.read_lorrax_head(lorrax_path)
    bgw = {
        "omega_ev": np.array([0.0, 2.0]),
        "epsinv00": np.array([1.0 + 0.0j, 3.0 + 2.0j]),
        "broadening_ev": 0.2,
    }

    result = MODULE.align_and_compare(lorrax, bgw)
    assert result["alignment"] == "linear_interpolation"
    assert result["interpolated_count"] == 1
    np.testing.assert_allclose(result["difference"], 0.0)

    outside = dict(lorrax)
    outside["omega_ev"] = np.array([0.0, 3.0])
    outside["epsinv00"] = np.zeros(2, dtype=complex)
    with pytest.raises(ValueError, match="extrapolation is disabled"):
        MODULE.align_and_compare(outside, bgw)


def test_broadening_must_be_known_and_match(tmp_path):
    lorrax_path = tmp_path / "lorrax_head.dat"
    lorrax_path.write_text("0.0 1.0 0.0\n")
    with pytest.raises(ValueError, match="no broadening_ev metadata"):
        MODULE.read_lorrax_head(lorrax_path)

    lorrax = MODULE.read_lorrax_head(lorrax_path, broadening_override_ev=0.1)
    bgw = {
        "omega_ev": np.array([0.0]),
        "epsinv00": np.array([1.0 + 0.0j]),
        "broadening_ev": 0.2,
    }
    with pytest.raises(ValueError, match="broadenings do not match"):
        MODULE.align_and_compare(lorrax, bgw)


def test_eps0_h5_real_axis_head(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "eps0mat.h5"
    with h5py.File(path, "w") as handle:
        handle["eps_header/params/matrix_type"] = 0
        handle["eps_header/qpoints/nq"] = 1
        handle["eps_header/qpoints/qpts"] = np.array([[0.0, 0.0, 0.125]])
        handle["eps_header/freqs/nfreq"] = 3
        handle["eps_header/freqs/nfreq_imag"] = 1
        handle["eps_header/freqs/freqs"] = np.array(
            [[0.0, 0.1], [1.0, 0.1], [0.0, 2.0]],
        )
        matrix = np.zeros((1, 1, 3, 1, 1, 2))
        matrix[0, 0, :, 0, 0, 0] = [1.0, 2.0, 9.0]
        matrix[0, 0, :, 0, 0, 1] = [0.0, -0.5, 9.0]
        handle["mats/matrix"] = matrix

    result = MODULE.read_eps0_h5(path)
    np.testing.assert_allclose(result["omega_ev"], [0.0, 1.0])
    np.testing.assert_allclose(result["epsinv00"], [1.0, 2.0 - 0.5j])
    assert result["broadening_ev"] == pytest.approx(0.1)
    assert result["ignored_nonreal_rows"] == 1
