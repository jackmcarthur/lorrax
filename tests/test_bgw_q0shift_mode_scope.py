"""Mode scoping for BerkeleyGW's indivisible metallic q0 treatment."""
from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from gw.gw_config import LorraxConfig


BASE = """\
[cohsex]
nval = 2
ncond = 2
nband = 4
sys_dim = 3
memory_per_device_gb = 4.0
bgw_metal_q0_treatment = bgw_q0shift
compute_mode = {mode}
"""


def _config(tmp_path, mode):
    path = tmp_path / f"q0_{mode}.in"
    path.write_text(BASE.format(mode=mode))
    file_io = types.ModuleType("file_io")
    file_io.__path__ = []
    file_io.resolve_input_paths = lambda params, input_dir: None
    qp_wfn = types.ModuleType("file_io.qp_wfn")
    qp_wfn.QP_ROTATIONS_K_STORAGE = ("auto", "full", "ibz")
    with mock.patch.dict(
            sys.modules, {
                "file_io": file_io,
                "file_io.qp_wfn": qp_wfn,
            }):
        return LorraxConfig.from_input_file(
            str(path), print_fn=lambda *args, **kwargs: None)


@pytest.mark.parametrize("mode", ["x_only", "cohsex", "gn_ppm", "hl_ppm"])
def test_bgw_q0shift_refuses_every_non_mpa_compute_mode(tmp_path, mode):
    with pytest.raises(
            ValueError,
            match=(
                rf"bgw_metal_q0_treatment = bgw_q0shift is refused with "
                rf"compute_mode = {mode}")):
        _config(tmp_path, mode)


def test_bgw_q0shift_remains_available_to_mpa(tmp_path):
    config = _config(tmp_path, "mpa")
    assert config.head.uses_bgw_metal_q0shift
    assert config.compute_mode.value == "mpa"
