"""The step limit of the corrected Magnus steps (CPU, ~1 min).

The predictor-corrector Magnus step, linearized for a coupling linear in
the deviation, is the recurrence ``time_integrators.magnus_amplification``;
on the imaginary axis it is stable for h sigma below
``magnus_stability_limit``, sigma the spectral radius of the coupling map
L(x) = -i [dSigma[x], rho_eq] (``propagate.coupling_radius``, a power
iteration on the model).  The run refuses a step above the limit, or one
whose probe at 1.25 dt grows (``td_magnus_stability``), or picks one
(``dt = 'auto'``).

G1 the recurrence's spectral radius equals the growth per step of a mode of
   the real ip_magnus_step on a 2-level TDA toy to 1e-3 (cubic and linear
   predictors, 1-3 corrections, h kappa 0.3-1.3; stable, weakly and
   strongly unstable cases whose dominant root is clear)
R1 the recurrence with the predictor dropped (zero history) misses it
G2 on the TDGW fixture with its kernel scaled x40: the power iteration's
   sigma is within 2 % below the dense spectrum's; cfm4 at 2 dt_max refuses
   (the sigma test), at dt_max / 2 runs bounded, 'auto' runs at a dt <=
   dt_max whose probe passed
G3 the step probe (the scheme's own step, field off, from a random
   coherence; ``propagate.step_growth``) of the plain step stays at 1 at
   dt_max / 2 and grows at 2 dt_max; at dt_max itself the probe at 1.25 dt_max
   grows, so the gate refuses dt_max although the sigma test passes it (on
   the 30x30 df714 CrI3 model the real limit lies below the static one)
R2 the 2 dt_max run with the gate bypassed is unbounded over 5 fs (|P|
   grows past 100x the stable run's or overflows): the refusal is a real
   instability
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

_HERE = os.path.dirname(os.path.abspath(__file__))


def _toy_growth(y, n_corr, history, nsteps=70):
    """Growth per step of |x_cv| of the real step on u' = -i kappa u (h = 1, kappa = y)."""
    eps = jnp.asarray([0.0, 0.1])
    dlt = eps[:, None] - eps[None, :]
    f = jnp.asarray([1.0, 0.0])
    free = lambda m, s: jnp.exp(-1j * s * dlt) * m

    def conj_dev(w, x, tau):
        w = 0.5 * (w + w.conj().T)
        e, v = jnp.linalg.eigh(w)
        vh = v.conj().T
        ph = jnp.exp(-1j * tau * e)
        em1 = -2.0 * jnp.sin(0.5 * tau * e) ** 2 - 1j * jnp.sin(tau * e)
        g = v @ (em1[:, None] * vh)
        ux = v @ ((vh @ x @ v) * (ph[:, None] * jnp.conj(ph)[None, :])) @ vh
        gf = g * f[None, :]
        return ux + gf + gf.conj().T + gf @ g.conj().T

    def b_of(t, x):
        out = jnp.zeros((2, 2), complex).at[1, 0].set(y * x[1, 0])
        return out.at[0, 1].set(jnp.conj(out[1, 0]))
    nodes = ti.CFM4_NODES
    w = jnp.asarray(ti.extrapolation_weights(history, list(nodes)))
    inner = lambda gen, t, x, tau: ti.ip_cf4_flow(gen, free, conj_dev, t, x, tau)
    field = lambda t: jnp.zeros((2, 2), complex)

    def step(c, j):
        x, hist = c
        x1, g = ti.ip_magnus_step(field, b_of, inner, free, j, x, 1.0,
                                  tuple(jnp.tensordot(w, hist, axes=1)), nodes=nodes,
                                  n_corr=n_corr)
        hist = free(jnp.concatenate([hist, jnp.stack(g)])[-len(history):], 1.0)
        return (x1, hist), jnp.abs(x1[1, 0])
    x0 = jnp.zeros((2, 2), complex).at[1, 0].set(1e-60).at[0, 1].set(1e-60)
    seed = jnp.asarray(np.random.default_rng(1).normal(size=(len(history), 1, 1)))
    _, a = jax.lax.scan(step, (x0, jnp.stack([b_of(0.0, x0)] * len(history)) * seed),
                        jnp.arange(nsteps, dtype=jnp.float64))
    a = np.asarray(a)
    return float(np.exp(np.log(a[60] / a[40]) / 20.0))


#: Cases whose dominant root leads the next by >= 15 % (a clean 20-step measurement):
#: stable, weakly unstable (linear, 3 corrections, 0.6: 1.00019) and strongly unstable.
_TOY = [("cubic", 1, 0.3), ("cubic", 1, 1.3), ("cubic", 2, 0.3), ("cubic", 2, 0.6),
        ("cubic", 2, 1.3), ("cubic", 3, 0.6), ("cubic", 3, 1.3), ("linear", 1, 0.3),
        ("linear", 1, 1.3), ("linear", 2, 0.6), ("linear", 2, 0.9), ("linear", 2, 1.3),
        ("linear", 3, 0.6), ("linear", 3, 1.3)]


def _history(name):
    return ti.CFM4_HISTORY if name == "cubic" else ti.CFM4_HISTORY[2:]


@pytest.mark.parametrize("hist,n_corr,y", _TOY, ids=[f"{h}-{n}-{y}" for h, n, y in _TOY])
def test_recurrence_is_the_linearized_step(hist, n_corr, y):
    rho = np.max(np.abs(np.linalg.eigvals(
        ti.magnus_amplification(ti.CFM4_NODES, _history(hist), n_corr, 1j * y))))
    got = _toy_growth(y, n_corr, _history(hist))
    assert abs(got - rho) <= 1e-3 * rho, (hist, n_corr, y, rho, got)


def test_red_twin_recurrence_without_predictor():
    """Zero history weights: the corrections alone (Picard from 0) - not the step."""
    y, n_corr = 1.3, 2
    a, b = ti._node_quadrature(ti.CFM4_NODES)
    g = np.zeros(2, complex)
    for _ in range(n_corr):
        g = 1j * y * np.ones(2) + 1j * y * (a @ g)
    rho_twin = abs(1 + b @ g)
    got = _toy_growth(y, n_corr, ti.CFM4_HISTORY)
    assert not abs(got - rho_twin) <= 1e-3 * rho_twin, (rho_twin, got)


@pytest.fixture(scope="module")
def gate(tmp_path_factory):
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=4",
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, os.path.join(_HERE, "_stability_case.py"),
                          str(tmp_path_factory.mktemp("stab")), "40"],
                         capture_output=True, text=True, env=env, timeout=1800)
    assert out.returncode == 0, out.stderr[-4000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[7:])


def test_power_iteration_finds_the_radius(gate):
    assert 0.98 * gate["exact_ev"] <= gate["sigma_ev"] <= gate["exact_ev"] * (1 + 1e-9), gate


def test_gate_refuses_above_and_runs_below(gate):
    o = gate["over"]
    assert o.get("refused") == "GATE td_magnus_stability" and "h*sigma" in o["message"], o
    u = gate["under"]
    assert u.get("finite") and "refused" not in u, u
    a = gate["auto"]
    assert a.get("finite") and a["dt_as"] <= gate["dt_max_as"], a
    assert a["probes"] and a["probes"][-1]["growth"] <= 1.0 + 2e-3, a
    assert abs(a["max_p"] - u["max_p"]) <= 1e-2 * u["max_p"], (a, u)


def test_probe_refuses_the_static_limit(gate):
    """At dt_max the scheme's own step at 1.25 dt_max grows: the probe refuses it."""
    r = gate["at_limit"]
    assert r.get("refused") == "GATE td_magnus_stability", r
    assert "passes the sigma limit" in r["message"], r


def test_probe_tracks_the_recurrence(gate):
    """Field off, from a random coherence, the plain step: bounded at dt_max / 2, growing at
    2 dt_max.  Not equal to the recurrence's growth: with several transition energies the
    coupling is time dependent in the interaction picture even for a TDA kernel (the
    recurrence is exact for one pair, G1), which is why the gate measures the step."""
    lo, hi = gate["probe"]["0.5"], gate["probe"]["2.0"]
    assert abs(lo["growth"] - 1.0) <= 2e-3, lo
    assert hi["growth"] > 1.1 and hi["recurrence"] > 1.1, hi


def test_red_twin_ungated_step_is_unstable(gate):
    t, u = gate["over_ungated"], gate["under"]
    assert (not t["finite"]) or t["max_p"] > 100.0 * u["max_p"], (t, u)


@pytest.fixture(scope="module")
def zero_gate(tmp_path_factory):
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=4",
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, os.path.join(_HERE, "_zero_coupling_case.py"),
                          str(tmp_path_factory.mktemp("zero"))],
                         capture_output=True, text=True, env=env, timeout=1800)
    assert out.returncode == 0, out.stderr[-4000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[7:])


def test_zero_map_has_zero_radius(zero_gate):
    """No coupling (dSigma = 0): sigma = 0 exactly and every estimate finite, never 0/0."""
    assert zero_gate["sigma"] == 0.0 and zero_gate["est_finite"], zero_gate
    assert zero_gate["est_max"] == 0.0, zero_gate


def test_gate_accepts_a_model_without_coupling(zero_gate):
    """An IP deck (no interaction), cfm4 in the interaction frame at an explicit 20 as, runs:
    the gate names the case and runs no probe (TDVAL's rung-1 Rabi decks)."""
    r = zero_gate["ip"]
    assert "refused" not in r and r["finite"] and abs(r["dt_as"] - 20.0) < 1e-6, r
    assert r["probes"] == [] and any("no coupling" in l for l in r["log"]), r


def test_red_twin_nan_map_refuses_by_name(zero_gate):
    r = zero_gate["nan"]
    assert r.get("refused") == "GATE td_magnus_stability", r
    assert "coupling radius sigma = nan" in r["message"], r


def test_red_twin_no_probe_refuses_by_name(zero_gate):
    """A finite radius with no usable sigma limit: refused by name, not an IndexError."""
    r = zero_gate["no_probe"]
    assert r.get("refused") == "GATE td_magnus_stability" and "no probe ran" in r["message"], r
