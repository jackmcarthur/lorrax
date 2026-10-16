"""TD-aGW propagation reproduces TDGW's own output (CPU host mesh, ~30 s).

``fixtures/syn2v2c`` is TDGW 0.3.0 (@41a5c34) run by
``fixtures/make_tdgw_fixture.py`` on a synthetic 2v2c slab: 3x3 k, a COHSEX
TDA kernel, a 5e8 V/m Gaussian pulse along (2, 1, 0) (the intraband term is
live), depopulation = dephasing = 0.05 eV, ``solve_ivp`` RK45 at rtol 1e-6,
atol 1e-11, samples every 10 as to 1000 as.  The header (``tdgw_header``)
is TDGW's; the operators are rebuilt from it by ``tdagw.tdgw_io``.

The propagator runs in a child process on a 2x2 host mesh (k padded 9 -> 12
over four devices) and, for G3, on one device.

G1 dp54 takes TDGW's steps (RHS count equal) and every measurement
   (Efield, polarization, current, rho_cv) and the checkpoint lessG agree
   to <= 1e-12 relative; it_next and tchkp equal
R1 intraband commutator off      -> G1's checker fails
R2 kernel sign flipped (-K)      -> G1's checker fails (sign of dSigma)
G2 cfm4 and etdrk4 at dt = 1 as reach P(t) within 1e-5 of TDGW
R3 magnus2 at dt = 5 as misses that tolerance (the checker resolves order)
G3 cfm4 / etdrk4 on 2x2 devices == on one device (P(t) <= 1e-12 relative)
R4 cfm4 vs etdrk4 on the same mesh differ by more than G3's tolerance
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(_HERE, "_tdgw_case.py")
_TOL = 1e-12


def _spawn(tmp, ndev, cases):
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS=f"--xla_force_host_platform_device_count={ndev}",
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, _RUNNER, str(tmp), json.dumps(cases)],
                         capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stderr[-4000:]
    rows = [json.loads(l[7:]) for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    return {r["tag"]: r for r in rows}


@pytest.fixture(scope="module")
def four(tmp_path_factory):
    return _spawn(tmp_path_factory.mktemp("p4"), 4, [
        dict(tag="base"), dict(tag="intra_off", intraband=False),
        dict(tag="sign_flip", kernel_sign=-1),
        dict(tag="cfm4", method="cfm4", dt=1.0), dict(tag="etdrk4", method="etdrk4", dt=1.0),
        dict(tag="magnus2", method="magnus2", dt=5.0, n_corr=2)])


@pytest.fixture(scope="module")
def one(tmp_path_factory):
    return _spawn(tmp_path_factory.mktemp("p1"), 1, [
        dict(tag="cfm4", method="cfm4", dt=1.0), dict(tag="etdrk4", method="etdrk4", dt=1.0)])


def _matches_tdgw(r):
    return (r["dsigma"] == r["tdgw_nfev"] and max(r["rel"].values()) <= _TOL
            and r["lessG"] <= _TOL and r["it_next"][0] == r["it_next"][1]
            and r["tchkp"][0] == r["tchkp"][1])


def _p(r):
    a = np.asarray(r["series"]["polarization"])
    return a[:, 0] + 1j * a[:, 1]


def test_dp54_matches_tdgw(four):
    r = four["base"]
    assert r["mesh"] == [2, 2]
    assert _matches_tdgw(r), {k: r[k] for k in ("dsigma", "tdgw_nfev", "rel", "lessG")}


@pytest.mark.parametrize("tag", ["intra_off", "sign_flip"])
def test_red_twin_wrong_physics_fails_parity(four, tag):
    assert not _matches_tdgw(four[tag]), four[tag]["rel"]


@pytest.mark.parametrize("tag", ["cfm4", "etdrk4"])
def test_fixed_step_methods_reach_tdgw(four, tag):
    assert four[tag]["rel"]["polarization"] <= 1e-5, four[tag]["rel"]


def test_red_twin_second_order_misses_tolerance(four):
    assert not four["magnus2"]["rel"]["polarization"] <= 1e-5, four["magnus2"]["rel"]


@pytest.mark.parametrize("tag", ["cfm4", "etdrk4"])
def test_four_devices_equal_one(four, one, tag):
    a, b = _p(four[tag]), _p(one[tag])
    assert one[tag]["mesh"] == [1, 1]
    assert np.max(np.abs(a - b)) <= _TOL * np.max(np.abs(b))


def test_red_twin_mesh_checker_resolves_methods(four):
    a, b = _p(four["cfm4"]), _p(four["etdrk4"])
    assert not np.max(np.abs(a - b)) <= _TOL * np.max(np.abs(b))
