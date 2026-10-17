"""The stiff split of the cfm4 step (CPU, ~2 min).

dSigma splits exactly as dSigma[x] = (dSigma[x] - Σ_j <u_j, x> b_j) + Σ_j <u_j, x>
b_j, b_j = dSigma[u_j] (``solvers.time_integrators.FastPart``); the step keeps
the first part on its predictor-corrector and solves the second implicitly at
the Gauss nodes (``ip_magnus_step(..., fast=)``).  Checked on the six-level
commutator problem of ``test_prop_integrators_cpu.py`` (B = g W o (rho -
rho_eq): L is diagonal in the matrix elements, so the two Hermitian vectors
of a cv pair span an invariant subspace) and, through the driver's stages, on
``_stiff_case.py``'s ISDF-form window with a strong local-field Hartree term.

G1 split CF4 is fourth order against DOP853 (rtol 1e-13) with the field on,
   its closing solve converged at every step
G2 the same at a strong field (amp 3.0: |x| ~ 1, the commutator nonlinearity
   of the same size as the linear term)
G3 Pi = 0 (a zero basis through the split path) reproduces the plain step to
   rounding (1e-13 relative)
G4 through stages, gate active: the plain cfm4 refuses 100 as (sigma test);
   the split (4 stiff modes) runs it, P(t) within 1e-4 of a 10 as plain
   reference and fourth order between 100 and 50 as; dt = 'auto' of the split
   is >= 2.5 x the plain one, each with its probe passed
G5 the zero-basis split through stages matches the plain run to rounding
G6 a split run resumed from its first chunk's checkpoint (the carry: slow and
   coefficient histories, closing-solve counters) continues bitwise
R1 an unreachable closing tolerance (probe bypassed) refuses by name
   (td_stiff_solve) at a chunk end, before its samples are written
R2 stiff_modes with a method other than cfm4 refuses at the deck (td_stiff_method)
"""
import json
import os
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from solvers import time_integrators as ti   # noqa: E402
from test_prop_integrators_cpu import (N, NOCC, T_END, _parts, _problem, _rho0,   # noqa: E402
                                       _scipy_reference)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _stiff_pairs(p, n_pairs):
    """The cv pairs (n conduction, m valence) with the largest coupling g W_nm."""
    w = np.asarray(p["w"])
    cand = sorted(((w[n, m], n, m) for n in range(NOCC, N) for m in range(NOCC)), reverse=True)
    return [(n, m) for _, n, m in cand[:n_pairs]]


def _fast(p, f, pairs, h, *, zero=False, tol=1e-12, max_iter=8):
    basis = []
    for n, m in pairs:
        e1 = np.zeros((N, N), complex)
        e1[n, m] = e1[m, n] = 2 ** -0.5
        e2 = np.zeros((N, N), complex)
        e2[n, m], e2[m, n] = 1j * 2 ** -0.5, -1j * 2 ** -0.5
        basis += [e1, e2]
    u = jnp.asarray(np.stack(basis))
    b = jnp.stack([f["state_part"](0.0, ui) for ui in u])
    if zero:
        u, b = 0.0 * u, 0.0 * b
    occ = jnp.asarray((np.arange(N) < NOCC).astype(float))
    gap = (occ[None, :] - occ[:, None]).astype(complex)
    fast = ti.FastPart(project=lambda x: jnp.real(jnp.einsum("jab,...ab->...j", jnp.conj(u), x)),
                       combine=lambda a: jnp.einsum("...j,jab->...ab", a.astype(complex), b),
                       reference=lambda m: -1j * m * gap, newton=None, tol=tol, max_iter=max_iter)
    jac = ti.fast_jacobian(fast, f["free"], h, ti.CFM4_NODES, int(u.shape[0]))
    return fast._replace(newton=jnp.linalg.inv(jnp.eye(jac.shape[0]) - jac))


def _run(p, r0, h, pairs=None, zero=False):
    """CF4 to T_END at step h, plain (pairs None) or split; (final rho, worst closing residual, max evaluations)."""
    f = _parts(p)
    nodes = ti.CFM4_NODES
    req = f["rho_eq"]
    inner = lambda gen, t, x, tau: ti.ip_cf4_flow(gen, f["free"], f["conj_dev"], t, x, tau)
    fast = None if pairs is None else _fast(p, f, pairs, h, zero=zero)
    n = int(round(T_END / h))

    def step(c, j):
        x, worst, most = c
        t = j * h
        if fast is None:
            guess = tuple(f["free"](f["state_part"](t, x), -cc * h) for cc in nodes)
            x1, _ = ti.ip_magnus_step(f["field"], f["state_part"], inner, f["free"], t, x, h, guess,
                                      nodes=nodes, n_corr=3)
            return (x1, worst, most), None
        a0 = fast.project(x)
        guess = (tuple(f["free"](f["state_part"](t, x) - fast.combine(a0), -cc * h) for cc in nodes),
                 jnp.stack([a0] * len(nodes), axis=-2))
        x1, _, _, (it, rel, ok) = ti.ip_magnus_step(f["field"], f["state_part"], inner, f["free"], t,
                                                    x, h, guess, nodes=nodes, n_corr=3, fast=fast)
        rel = jnp.where(ok, rel, jnp.inf)
        return (x1, jnp.maximum(worst, rel), jnp.maximum(most, it)), None

    @jax.jit
    def go(x0):
        return jax.lax.scan(step, (x0, jnp.float64(0.0), jnp.int32(0)),
                            jnp.arange(n, dtype=jnp.float64))[0]
    x, worst, most = go(jnp.asarray(r0) - req)
    return np.asarray(x + req), float(worst), int(most)


@pytest.fixture(scope="module", params=[0.4, 3.0], ids=["field0.4", "field3.0"])
def driven(request):
    p = _problem(gamma=0.0, amp=request.param)
    r0 = _rho0(p)
    ref = _scipy_reference(p, r0).y[:, -1].reshape(N, N)
    return p, r0, ref, _stiff_pairs(p, 2)


def test_split_is_fourth_order(driven):
    """G1 / G2: order 4 with the field on (nominal and strong), closing solve converged."""
    p, r0, ref, pairs = driven
    r1, w1, m1 = _run(p, r0, 0.1, pairs)
    r2, w2, m2 = _run(p, r0, 0.05, pairs)
    e1, e2 = np.linalg.norm(r1 - ref), np.linalg.norm(r2 - ref)
    order = np.log2(e1 / e2)
    assert abs(order - 4.0) < 0.3, (order, e1, e2)
    assert max(w1, w2) <= 1e-12 and max(m1, m2) <= 8, (w1, w2, m1, m2)
    # the plain step at the same h has the same order and an error of the same size
    q1 = np.linalg.norm(_run(p, r0, 0.1)[0] - ref)
    assert 0.1 * q1 <= e1 <= 10.0 * q1, (e1, q1)


def test_zero_basis_reproduces_the_plain_step():
    """G3: Pi = 0 through the split path = the plain step, to rounding."""
    p = _problem(gamma=0.0, amp=0.4)
    r0 = _rho0(p)
    plain = _run(p, r0, 0.1)[0]
    zero, worst, _ = _run(p, r0, 0.1, _stiff_pairs(p, 2), zero=True)
    assert np.max(np.abs(zero - plain)) <= 1e-13 * np.max(np.abs(plain - p["rho_eq"])), \
        np.max(np.abs(zero - plain))


@pytest.fixture(scope="module")
def stages_case(tmp_path_factory):
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=4",
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, os.path.join(_HERE, "_stiff_case.py"),
                          str(tmp_path_factory.mktemp("stiff"))],
                         capture_output=True, text=True, env=env, timeout=2400)
    assert out.returncode == 0, out.stderr[-4000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[7:])


def test_split_runs_where_the_plain_step_refuses(stages_case):
    """G4: 100 as refused plain, run split; accuracy and order against the 10 as reference."""
    c = stages_case
    assert c["plain_100"].get("refused") == "GATE td_magnus_stability", c["plain_100"]
    s100, s50 = c["split_100"], c["split_50"]
    assert "refused" not in s100 and "refused" not in s50, (s100, s50)
    assert s100["rel_err"] <= 1e-4 and s100["rel_err"] / s50["rel_err"] >= 10.0, (s100, s50)
    assert s100["fast"]["unconverged"] == 0 and s100["probes"][-1]["growth"] <= 1.002, s100
    assert c["sigma_slow_ev"] < 0.5 * c["sigma_full_ev"], (c["sigma_slow_ev"], c["sigma_full_ev"])


def test_auto_step_of_the_split(stages_case):
    """G4: dt = 'auto' with the split >= 2.5 x without, each probe passed."""
    pa, sa = stages_case["plain_auto"], stages_case["split_auto"]
    assert pa["probes"][-1]["growth"] <= 1.002 and sa["probes"][-1]["growth"] <= 1.002, (pa, sa)
    assert not sa["probes"][-1].get("fast_unconverged"), sa
    assert sa["dt_as"] >= 2.5 * pa["dt_as"], (sa["dt_as"], pa["dt_as"])


def test_zero_basis_through_stages(stages_case):
    """G5: the split path with u = b = 0 matches the plain run to rounding."""
    z = stages_case["zero_vs_plain"]
    assert z["max_abs"] <= 1e-12 * z["max_p"], z


def test_red_twin_unreachable_tolerance_refuses(stages_case):
    """R1: the chunk-end refusal names the closing solve."""
    r = stages_case["unreachable_tol"]
    assert r.get("refused") == "GATE td_stiff_solve" and "closing solve" in r["message"], r


def test_red_twin_wrong_method_refuses(stages_case):
    """R2: stiff_modes needs cfm4."""
    assert stages_case["bad_method"].get("refused") == "GATE td_stiff_method", stages_case["bad_method"]


def test_split_restart_is_bitwise(stages_case):
    """G6: checkpoint after the first chunk, resume, compare with the uninterrupted run."""
    r = stages_case["restart"]
    assert r["bitwise"] and r["fast_tail"]["unconverged"] == 0, r
