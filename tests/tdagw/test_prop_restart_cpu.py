"""Exact checkpoint restart of the TD-aGW propagator (CPU host mesh, ~1 min).

On the TDGW fixture (``fixtures/syn2v2c``: 3x3 k, 2v2c, COHSEX TDA kernel,
5e8 V/m, Gamma 0.05 eV and the intraband term on, 100 samples, chunks of
20), for every integrator: run A uninterrupted, run B stopped after its
first chunk, run C restarted from B's checkpoint into a new outdir (k
padded 9 -> 12 over a 2x2 host mesh).

G1 every time-domain dataset, the final lessG, it_next and tchkp of C equal
   A's exactly (bitwise), for dp54, rk4, etdrk4, magnus2, cfm4; C took only
   the steps after the checkpoint; C's file keeps TDGW's layout (tdgw-pp)
R1 cfm4 / magnus2 resumed with the dSigma history zeroed, and dp54 resumed
   with a fresh controller (TDGW's restart: the state only), fail G1
G2 a restart with other settings (n_corr) refuses td_restart_settings; a
   TDGW checkpoint (no LORRAX carry) refuses td_restart_carry
"""
import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(_HERE, "_restart_case.py")
_METHODS = [("dp54", None), ("rk4", 2.5), ("etdrk4", 2.5), ("magnus2", 5.0), ("cfm4", 2.5)]


def _spawn(tmp, cases):
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=4",
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, _RUNNER, str(tmp), json.dumps(cases)],
                         capture_output=True, text=True, env=env, timeout=1800)
    assert out.returncode == 0, out.stderr[-4000:]
    rows = [json.loads(l[7:]) for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    return {r["tag"]: r for r in rows}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    cases = [dict(tag=m, method=m, dt=dt) for m, dt in _METHODS]
    cases += [dict(tag="twin_cfm4", method="cfm4", dt=2.5, twin="zero_history"),
              dict(tag="twin_magnus2", method="magnus2", dt=5.0, twin="zero_history"),
              dict(tag="twin_dp54", method="dp54", twin="fresh_controller"),
              dict(tag="mismatch", method="cfm4", dt=2.5, mutate=("lorrax.n_corr", 3)),
              dict(tag="tdgw_ckpt", method="cfm4", dt=2.5, from_tdgw=True)]
    return _spawn(tmp_path_factory.mktemp("restart"), cases)


def _exact(r):
    keys = [k for k in r if k not in ("tag", "mesh", "it_next", "tchkp", "resumed_at",
                                      "steps_a", "steps_c", "tdgw_layout", "refused")]
    return (all(r[k] == 0.0 for k in keys) and r["it_next"][0] == r["it_next"][1]
            and r["tchkp"][0] == r["tchkp"][1])


@pytest.mark.parametrize("method", [m for m, _ in _METHODS])
def test_restart_continues_bitwise(results, method):
    r = results[method]
    assert r["mesh"] == [2, 2] and r["resumed_at"] == 20 and r["tdgw_layout"], r
    assert _exact(r), r
    assert 0 < r["steps_c"] < r["steps_a"], r


@pytest.mark.parametrize("tag", ["twin_cfm4", "twin_magnus2", "twin_dp54"])
def test_red_twin_inexact_restart_fails(results, tag):
    assert not _exact(results[tag]), results[tag]


def test_restart_refuses_other_settings(results):
    assert results["mismatch"].get("refused") == "GATE td_restart_settings", results["mismatch"]


def test_restart_refuses_a_tdgw_checkpoint(results):
    assert results["tdgw_ckpt"].get("refused") == "GATE td_restart_carry", results["tdgw_ckpt"]
