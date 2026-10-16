"""The Σ window's two-node pairing is a per-leg row (gw.ppm_accumulators.WINDOW_PAIRING)."""
import numpy as np
import jax
import pytest
from jax.sharding import Mesh

import gw.ppm_accumulators as acc
import lxkit


def test_pairing_rows():
    assert acc.WINDOW_PAIRING == {"CUDA": True, "ROCM": False}


@pytest.mark.parametrize("vendor,expected", [("cuda", True), ("rocm", False), ("cpu", False)])
def test_window_pairing_reads_the_leg_row(monkeypatch, vendor, expected):
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    monkeypatch.setattr(lxkit, "device_vendor", lambda d: vendor)
    assert acc.window_pairing(mesh) is expected
