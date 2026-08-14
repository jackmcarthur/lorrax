"""Input-controlled MPA sample geometry, independent of metal physics."""

from __future__ import annotations

import numpy as np
import pytest

from gw.gw_config import LorraxConfig
from gw.mpa import model, sample_plan


_BASE = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""


def _config(tmp_path, extra=""):
    path = tmp_path / "mpa_sampling.in"
    path.write_text(_BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *args, **kwargs: None)


def test_metal_sampling_flags_build_the_configured_grid(tmp_path):
    config = _config(
        tmp_path,
        "mpa_n_poles = 4\n"
        "mpa_material_class = metal\n"
        "mpa_sampling_alpha = 2\n"
        "mpa_varpi_near_ry = 0.15\n"
        "mpa_varpi_far_ry = 1.5\n",
    )

    plan = config.mpa.sample_plan(8.0)
    np.testing.assert_array_equal(
        sample_plan.plan_z(plan),
        np.asarray([
            2.0e-5j, 0.5 + 0.15j, 2.0 + 0.15j, 8.0 + 0.15j,
            1.5j, 0.5 + 1.5j, 2.0 + 1.5j, 8.0 + 1.5j,
        ], dtype=np.complex128),
    )


def test_unknown_material_class_is_a_parse_error(tmp_path):
    with pytest.raises(ValueError, match="mpa_material_class"):
        _config(tmp_path, "mpa_material_class = semimetal\n")


def test_metal_evaluator_refuses_before_creating_output(tmp_path):
    config = _config(tmp_path, "mpa_material_class = metal\n")
    run_dir = tmp_path / "must_not_exist"
    with pytest.raises(
            NotImplementedError, match="mpa_metal_evaluator_unavailable"):
        model.build_mpa_fit(
            run_dir, "metal", wfns=None, V_q=None, quad=None, sym=None,
            centroid_indices=None, head_resolver=None, config=config,
            meta=None, mesh_xy=None)
    assert not run_dir.exists()
