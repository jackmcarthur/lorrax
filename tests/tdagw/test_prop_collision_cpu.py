"""The held collision term in the TD-aGW propagator (CPU host mesh, ~3 min).

docs/theory/td_agw_dynamical.md §1.5 and §7.1.  On the TDGW fixture
(``fixtures/syn2v2c``: 3x3 k, 2v2c, k padded 9 -> 12 over a 2x2 host mesh) with
a synthetic collision term (random PSD rates G~>, G~< per k, ~5e-3 Ha, the
production ``tdagw.propagate.collision_program`` and the production refresh
path of ``run_propagation``; ``_collision_case.py``).  The exact references
are augmented matrix exponentials per k (shared with nothing in the module).

G1 subtracted form, no field (COHSEX kernel, Gamma and intraband on): rho_eq is
   a fixed point exactly (every sample, bitwise) for dp54, rk4, etdrk4,
   magnus2 (interaction frame) and cfm4; magnus2 whole_rho at its rounding floor
G2 raw form drifts by C[rho_eq] (> 1e-2), the drift the subtraction removes
G3 IP, no field, Gamma 0, raw: the exact solution of rho' = -i[H0, rho] + C[rho];
   measured orders cfm4 4, magnus2 2 (both frames), rk4 4, etdrk4 4; dp54
   within its tolerance.  (The collision flow is the Magnus remainder: the
   triple jump keeps cfm4 fourth order.)
G4 the fixture's drive and model: every method == rk4 at dt / 10 within its
   own error; collision on vs off differs by > 1e-2
G5 four refreshes (rates x (1 + 0.2 n), held Delta Sigma^h = n c) on the order
   problem: every method == the exact piecewise solution at its own refresh
   times (dp54 refreshes where its step lands)
G6 no recompile across refreshes: one chunk program, one traced
   specialization, and no compile at the 2nd..4th refresh (jax_log_compiles)
G7 the Magnus history rebase: with one correction (n_corr = 1) dropping it
   costs > 10x the error; with two the effect is below 1e-8 (measured)
G8 k-locality: the compiled chunk program has the same collectives with the
   collision on as off (cfm4, rk4)
R1 rk4 refreshed without the carry rebase (a stale first-same-as-last
   derivative) misses G4's reference by > 1e-6
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(_HERE, "_collision_case.py")
_EQ = np.diag([1.0, 1.0, 0.0, 0.0])


def _cases():
    c = []
    for m, dt, fr in [("dp54", None, None), ("rk4", 2.5, None), ("etdrk4", 2.5, None),
                      ("magnus2", 5.0, None), ("magnus2", 5.0, "interaction"), ("cfm4", 2.5, None)]:
        c.append(dict(tag=f"fixed_{m}" + ("_ip" if fr else ""), kind="fixed", method=m, dt=dt,
                      frame=fr))
    c.append(dict(tag="fixed_raw_cfm4", kind="fixed", method="cfm4", dt=2.5, form="raw"))
    for m, dts, fr in [("cfm4", (5.0, 2.5), None), ("magnus2", (5.0, 2.5), "interaction"),
                       ("magnus2", (5.0, 2.5), None), ("rk4", (2.0, 1.0), None),
                       ("etdrk4", (5.0, 2.5), None)]:
        for dt in dts:
            c.append(dict(tag=f"order_{m}" + ("_ip" if fr else "") + f"_{dt}", kind="order",
                          method=m, dt=dt, frame=fr, form="raw"))
    c.append(dict(tag="order_dp54", kind="order", method="dp54", form="raw"))
    c.append(dict(tag="driven_ref", kind="driven", method="rk4", dt=0.25))
    for m, dt in [("rk4", 2.5), ("etdrk4", 2.5), ("dp54", None), ("magnus2", 2.5), ("cfm4", 2.5)]:
        c.append(dict(tag=f"driven_{m}", kind="driven", method=m, dt=dt))
    c.append(dict(tag="driven_off_cfm4", kind="driven", method="cfm4", dt=2.5, form="off"))
    for m, dt, fr in [("cfm4", 2.5, None), ("magnus2", 2.5, "interaction"), ("magnus2", 2.5, None),
                      ("rk4", 1.0, None), ("etdrk4", 2.5, None), ("dp54", None, None)]:
        c.append(dict(tag=f"orefresh_{m}" + ("_ip" if fr else ""), kind="order", method=m, dt=dt,
                      frame=fr, form="raw", refresh=True))
    c.append(dict(tag="refresh_ref", kind="driven", method="rk4", dt=0.25, refresh=True))
    c.append(dict(tag="refresh_rk4", kind="driven", method="rk4", dt=2.5, refresh=True))
    c.append(dict(tag="refresh_rk4_stale", kind="driven", method="rk4", dt=2.5, refresh=True,
                  stale=True))
    for nc, dt in ((1, 1.0), (2, 2.5)):
        for nr in (False, True):
            c.append(dict(tag=f"refresh_cfm4_nc{nc}" + ("_norebase" if nr else ""), kind="driven",
                          method="cfm4", dt=dt, n_corr=nc, refresh=True, no_rebase=nr))
    for m in ("cfm4", "rk4"):
        for form in ("subtracted", "off"):
            c.append(dict(tag=f"hlo_{form}_{m}", kind="hlo", method=m, dt=2.5, form=form))
    return [{k: v for k, v in x.items() if v is not None} for x in c]


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("collision")
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=4",
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, _RUNNER, str(tmp), json.dumps(_cases())],
                         capture_output=True, text=True, env=env, timeout=3000)
    assert out.returncode == 0, out.stderr[-4000:]
    rows = {}
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            assert "error" not in r, (r["tag"], r.get("trace"))
            r["rho_t"] = np.load(os.path.join(str(tmp), r["tag"], "rho_t.npy"))
            rows[r["tag"]] = r
    assert len(rows) == len(_cases())
    return rows


def _rel(rows, a, b):
    x, y = rows[a]["rho_t"], rows[b]["rho_t"]
    return float(np.max(np.abs(x - y)) / np.max(np.abs(y - _EQ)))


@pytest.mark.parametrize("tag", ["fixed_dp54", "fixed_rk4", "fixed_etdrk4", "fixed_magnus2_ip",
                                 "fixed_cfm4"])
def test_equilibrium_is_an_exact_fixed_point(results, tag):
    assert results[tag]["max_dev"] == 0.0


def test_whole_rho_magnus_fixed_point_at_rounding(results):
    assert results["fixed_magnus2"]["max_dev"] <= 1e-13


def test_raw_form_drifts(results):
    assert results["fixed_raw_cfm4"]["max_dev"] > 1e-2


@pytest.mark.parametrize("pair,order", [
    (("order_cfm4_5.0", "order_cfm4_2.5"), 4), (("order_magnus2_ip_5.0", "order_magnus2_ip_2.5"), 2),
    (("order_magnus2_5.0", "order_magnus2_2.5"), 2), (("order_rk4_2.0", "order_rk4_1.0"), 4),
    (("order_etdrk4_5.0", "order_etdrk4_2.5"), 4)])
def test_order_against_the_exact_solution(results, pair, order):
    ea, eb = (results[t]["err_exact"] for t in pair)
    assert abs(np.log2(ea / eb) - order) <= 0.3, (ea, eb)
    assert eb <= (1e-10 if order == 4 else 5e-6)
    assert results[pair[1]]["dev_exact"] > 0.1                # the collision dynamics is O(1)


def test_dp54_against_the_exact_solution(results):
    assert results["order_dp54"]["err_exact"] <= 1e-9


@pytest.mark.parametrize("tag,tol", [("driven_rk4", 1e-7), ("driven_etdrk4", 1e-7),
                                     ("driven_dp54", 1e-8), ("driven_cfm4", 1e-7),
                                     ("driven_magnus2", 1e-3)])
def test_methods_agree_on_the_driven_problem(results, tag, tol):
    assert _rel(results, tag, "driven_ref") <= tol


def test_collision_changes_the_driven_run(results):
    assert _rel(results, "driven_off_cfm4", "driven_ref") > 1e-2


@pytest.mark.parametrize("tag,tol", [("orefresh_cfm4", 1e-10), ("orefresh_rk4", 1e-10),
                                     ("orefresh_etdrk4", 1e-10), ("orefresh_dp54", 1e-10),
                                     ("orefresh_magnus2", 5e-6), ("orefresh_magnus2_ip", 5e-6)])
def test_refreshes_follow_the_exact_piecewise_solution(results, tag, tol):
    r = results[tag]
    assert r["refreshes"] == 4 and len(r["refresh_t"]) == 4
    assert r["err_exact"] <= tol


@pytest.mark.parametrize("tag,twin", [("orefresh_cfm4", "order_cfm4_2.5"),
                                      ("orefresh_rk4", "order_rk4_1.0"),
                                      ("orefresh_dp54", "order_dp54"), ("refresh_rk4", "driven_rk4"),
                                      ("refresh_cfm4_nc2", "driven_cfm4")])
def test_no_recompile_across_refreshes(results, tag, twin):
    r, t = results[tag], results[twin]
    assert (r["programs"], r["jit_cache"]) == (t["programs"], t["jit_cache"])   # no retrace
    if r["method"] != "dp54":
        assert r["programs"] == r["jit_cache"] == 1
    at = r["refresh_compiles"]                                # compiles done before each refresh
    assert len(at) == 4 and at[1] == at[2] == at[3], at       # the 2nd..4th refresh compile nothing


def test_red_twin_stale_derivative(results):
    assert _rel(results, "refresh_rk4", "refresh_ref") <= 1e-9
    assert _rel(results, "refresh_rk4_stale", "refresh_ref") > 1e-6


def test_magnus_history_rebase(results):
    err = lambda t: _rel(results, t, "refresh_ref")
    assert err("refresh_cfm4_nc1_norebase") > 10.0 * err("refresh_cfm4_nc1")
    assert _rel(results, "refresh_cfm4_nc2_norebase", "refresh_cfm4_nc2") <= 1e-8


@pytest.mark.parametrize("method", ["cfm4", "rk4"])
def test_collision_adds_no_collective(results, method):
    on, off = (results[f"hlo_{f}_{method}"]["collectives"] for f in ("subtracted", "off"))
    assert on == off and sum(off.values()) > 0
