"""Stability-optimized predictors and relaxed corrections of the cfm4 step (CPU).

P1 each named predictor of ti.CFM4_PREDICTORS reproduces polynomials of its
   degree at the new Gauss nodes from its history offsets (the order condition)
P2 the documented imaginary-axis limits of the linearized step (recurrence)
P3 the real ip_magnus_step on the 2-level toy grows as the recurrence with the
   named weights and with relaxed corrections (below and above the limit)
P4 order 4 on the driven 6-level stand-in of the TD-aGW equation (DOP853
   reference): stab1 with n_corr = 1, stab2 with n_corr = 2, relaxed cubic
   n_corr 2 (theta 0.8 on the last sweep), next to cubic n_corr 2
R1 the deck refuses relaxations it cannot use (GATE td_corr_relax) and unknown
   predictors (GATE td_predictor); accepts the named ones
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)

from solvers import time_integrators as ti   # noqa: E402
from test_prop_implicit_cpu import _toy, _conj_dev, N, _deck_raw   # noqa: E402

#: name -> (polynomial degree reproduced, {n_corr: documented limit})
_NAMED = {"stab1": (3, {1: 0.645, 2: 0.654, 3: 0.299}),
          "stab2": (2, {1: 0.124, 2: 1.296, 3: 1.284}),
          "cubic": (3, {1: 0.117, 2: 0.730, 3: 1.144})}


@pytest.mark.parametrize("name", sorted(_NAMED))
def test_named_predictor_is_exact_to_its_degree(name):
    offs, w = ti.predictor_weights(name)
    tau = np.asarray(offs)
    deg = _NAMED[name][0]
    scale = np.max(np.abs(tau)) ** deg
    for j, c in enumerate(ti.CFM4_NODES):
        for q in range(deg + 1):
            assert abs(w[j] @ tau ** q - c ** q) <= 1e-13 * scale, (name, j, q)


@pytest.mark.parametrize("name", sorted(_NAMED))
def test_named_predictor_limits(name):
    offs, w = ti.predictor_weights(name)
    for nc, lim in _NAMED[name][1].items():
        got = ti.magnus_stability_limit(ti.CFM4_NODES, offs, nc, weights=w)
        assert abs(got - lim) <= 1.5e-3, (name, nc, got, lim)


def _toy_growth(y, n_corr, offs, w, relax=None, nsteps=90):
    """Growth per step of |x_cv| of the real step on u' = -i kappa u (h = 1, kappa = y)."""
    eps = jnp.asarray([0.0, 0.1])
    dlt = eps[:, None] - eps[None, :]
    free = lambda m, s: jnp.exp(-1j * s * dlt) * m
    conj_dev = _conj_dev(jnp.asarray([1.0, 0.0]))

    def b_of(t, x):
        out = jnp.zeros_like(x).at[:, 1, 0].set(y * x[:, 1, 0])
        return out.at[:, 0, 1].set(jnp.conj(out[:, 1, 0]))
    nodes = ti.CFM4_NODES
    wj = jnp.asarray(w)
    inner = lambda gen, t, x, tau: ti.ip_cf4_flow(gen, free, conj_dev, t, x, tau)
    field = lambda t: jnp.zeros((2, 2), complex)
    nh = len(offs)

    def step(c, j):
        x, hist = c
        x1, g = ti.ip_magnus_step(field, b_of, inner, free, j, x, 1.0,
                                  tuple(jnp.tensordot(wj, hist, axes=1)), nodes=nodes,
                                  n_corr=n_corr, relax=relax)
        hist = free(jnp.concatenate([hist, jnp.stack(g)])[-nh:], 1.0)
        return (x1, hist), jnp.abs(x1[0, 1, 0])
    x0 = jnp.zeros((1, 2, 2), complex).at[0, 1, 0].set(1e-60).at[0, 0, 1].set(1e-60)
    seed = jnp.asarray(np.random.default_rng(1).normal(size=(nh, 1, 1, 1)))
    _, a = jax.lax.scan(step, (x0, jnp.stack([b_of(0.0, x0)] * nh) * seed),
                        jnp.arange(nsteps, dtype=jnp.float64))
    a = np.asarray(a)
    return float(np.exp(np.log(a[80] / a[60]) / 20.0))


#: (predictor, n_corr, relax, y): stable inside the limit, growing outside; points
#: whose dominant root leads the next by >= 15 % (a clean 20-step measurement).
#: cubic with one relaxed correction (0.75) grows at every y (from (1 - theta) times
#: the Lagrange weights: unstable as y -> 0).
_TWIN = [("stab1", 1, None, 0.3), ("stab1", 1, None, 0.9), ("stab2", 2, None, 0.8),
         ("stab2", 2, None, 1.8), ("cubic", 2, (0.8,), 0.3), ("cubic", 2, (0.8,), 1.8),
         ("cubic", 1, (0.75,), 0.3)]


@pytest.mark.parametrize("name,n_corr,relax,y", _TWIN, ids=[f"{a}-{b}-{c}-{d}" for a, b, c, d in _TWIN])
def test_real_step_grows_as_the_recurrence(name, n_corr, relax, y):
    offs, w = ti.predictor_weights(name)
    rho = ti.magnus_growth(ti.CFM4_NODES, offs, n_corr, y, weights=w, relax=relax)
    got = _toy_growth(y, n_corr, offs, w, relax)
    assert abs(got - rho) <= 2e-3 * rho, (name, n_corr, relax, y, rho, got)


def _run(p, h, t_end, name, n_corr, relax=None):
    offs, w = ti.predictor_weights(name)
    nodes, nh = ti.CFM4_NODES, len(offs)
    wj = jnp.asarray(w)
    x0 = jnp.zeros((1, N, N), complex)
    hist0 = jnp.stack([p["state_part"](0.0, x0)] * nh)
    n_boot = -(-nh // 2)

    def step(c, j):
        x, hist = c
        nc = jnp.where(j < n_boot, n_corr + 3, n_corr)          # the propagator's bootstrap
        x1, g = ti.ip_magnus_step(p["field"], p["state_part"], p["inner"], p["free"], j * h, x, h,
                                  tuple(jnp.tensordot(wj, hist, axes=1)), nodes=nodes, n_corr=nc,
                                  relax=relax)
        return (x1, p["free"](jnp.concatenate([hist[2:], jnp.stack(g)]), h)), None
    n = int(round(t_end / h))
    (x, _), _ = jax.jit(lambda c: jax.lax.scan(step, c, jnp.arange(n, dtype=jnp.float64)))((x0, hist0))
    return np.asarray(x[0] + p["R"])


@pytest.fixture(scope="module")
def toy_ref():
    p = _toy()
    t_end = 2.0
    ref = solve_ivp(lambda t, y: np.asarray(p["rhs"](t, jnp.asarray(y.reshape(N, N)))).ravel(),
                    (0.0, t_end), p["rho0"].ravel(), method="DOP853", rtol=1e-13, atol=1e-15)
    return p, t_end, ref.y[:, -1].reshape(N, N)


_ORDER = [("cubic", 2, None), ("stab1", 1, None), ("stab2", 2, None), ("cubic", 2, (0.8,))]


@pytest.mark.parametrize("name,n_corr,relax", _ORDER, ids=[f"{a}-{b}-{c}" for a, b, c in _ORDER])
def test_order_4_on_the_driven_toy(toy_ref, name, n_corr, relax):
    p, t_end, yref = toy_ref
    errs = [float(np.max(np.abs(_run(p, h, t_end, name, n_corr, relax) - yref)))
            for h in (0.1, 0.05, 0.025)]
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert all(3.6 <= o <= 4.8 for o in orders), (name, errs, orders)
    base = [float(np.max(np.abs(_run(p, h, t_end, "cubic", 2) - yref))) for h in (0.05,)]
    assert errs[1] <= 4.0 * base[0], (name, errs[1], base[0])


@pytest.mark.parametrize("lor,gate", [
    (dict(corr_relax=[0.8, 0.9, 1.0]), "td_corr_relax"),           # more than n_corr = 2
    (dict(corr_relax=[0.0]), "td_corr_relax"),
    (dict(corr_relax=[2.5]), "td_corr_relax"),
    (dict(corr_relax=["a"]), "td_corr_relax"),
    (dict(corr_relax=[0.8], node_solver="anderson"), "td_corr_relax"),
    (dict(predictor="stab9"), "td_predictor"),
    (dict(method="magnus2", predictor="stab2"), "td_predictor")])
def test_deck_refuses_bad_predictor_settings(lor, gate):
    from tdagw.td_config import parse_td_deck
    with pytest.raises(ValueError, match=f"GATE {gate}"):
        parse_td_deck(_deck_raw(**lor))


def test_deck_accepts_named_predictors():
    from tdagw.td_config import parse_td_deck
    d = parse_td_deck(_deck_raw(predictor="stab1", n_corr=1))
    assert d.values["lorrax.predictor"] == "stab1" and d.values["lorrax.corr_relax"] == []
    d = parse_td_deck(_deck_raw(predictor="stab2", corr_relax=[0.9]))
    assert d.values["lorrax.corr_relax"] == [0.9]
