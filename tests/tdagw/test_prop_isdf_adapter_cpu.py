"""The propagator's ISDF adapter matches the TD-aGW kernel (CPU host mesh, seconds).

``tdagw.self_energy.isdf_model`` wraps TDMODEL's ``build_td_agw_apply``:
it hands the model Delta rho = rho - rho_eq in the model's layout (bands
padded and sharded, rows in the model's C-order k), takes the rows back in
the propagator's k order and converts Ry to Ha.  On a synthetic model (3x2
k, 4 bands, 2 spinor components, 7 centroids; TDMODEL's synthetic operands)
with the propagator's k list a random permutation of the grid shifted by
lattice vectors, two trials, k padded 6 -> 8 over a 2x2 host mesh:

G1 adapter dSigma == 1/2 K (rho - rho_eq) for both terms, SEX, Hartree,
   K the explicit-sum kernel, rel <= 1e-12; k pads stay exactly zero
R1 the same apply without the Ry -> Ha halving fails G1
R2 rho_eq = 0 (the model sees rho, not Delta rho) fails G1
"""
import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def result():
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=4",
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, os.path.join(_HERE, "_isdf_adapter_case.py")],
                         capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stderr[-4000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[7:])


@pytest.mark.parametrize("terms", ["both", "sex", "hartree"])
def test_adapter_matches_kernel(result, terms):
    r = result[f"{terms}:gate"]
    assert result["mesh"] == [2, 2] and result["nk_pad"] == 8
    assert r["rel"] <= 1e-12 and r["pad"] == 0.0, r


@pytest.mark.parametrize("twin", ["twin_no_halving", "twin_rho_eq_zero"])
def test_red_twin_fails(result, twin):
    assert not result[f"both:{twin}"]["rel"] <= 1e-12, result[f"both:{twin}"]


def test_tda_matches_the_projected_kernel(result):
    """tda_n_val = 2: only drho_cv drives and only dSigma_cv (+ its conjugate) is kept."""
    r = result["both:tda"]
    assert r["rel"] <= 1e-12 and r["pad"] == 0.0, r
    assert not result["both:twin_tda_full"]["rel"] <= 1e-12, result["both:twin_tda_full"]
