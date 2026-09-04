"""CPU-safe typed configuration seams used by ``lx doctor --deck``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gw.gw_config import LorraxConfig, infer_material_class


def test_hardware_free_config_keeps_auto_memory_and_gpu_request(tmp_path):
    deck = tmp_path / "cohsex.in"
    deck.write_text(
        "[cohsex]\n"
        "memory_per_device_gb = 0\n"
        "linalg = distributed\n"
    )
    config = LorraxConfig.from_input_file(
        str(deck),
        runtime_platform="gpu",
        resolve_hardware=False,
        print_fn=lambda *_: None,
    )
    assert config.memory.per_device_gb == 0.0
    assert config.backend.distributed_lu == "cusolvermp"


@pytest.mark.parametrize(
    ("occupations", "expected"),
    [
        (np.array([0.0, 1.0, 1.0 - 5.0e-7]), "insulator"),
        (np.array([0.0, 1.0, 0.75]), "metal"),
    ],
)
def test_material_class_uses_the_driver_tolerance(occupations, expected):
    assert infer_material_class(occupations) == expected


@pytest.mark.parametrize("occupations", [np.array([]), np.array([np.nan])])
def test_material_class_refuses_missing_or_nonfinite_occupations(occupations):
    with pytest.raises(ValueError, match="finite and nonempty"):
        infer_material_class(occupations)


def test_gw_driver_uses_shared_material_classifier():
    driver = Path(__file__).resolve().parents[1] / "src/gw/gw_jax.py"
    source = driver.read_text()
    assert "material_class = infer_material_class(wfn.occs)" in source
    assert "_infer_material_class" not in source
