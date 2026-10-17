"""Converged Gauss nodes for cfm4 (node_solver = 'anderson'), CPU (~3 min).

The truncated cfm4 step applies n_corr sweeps of its node map F (one
correction: the step's exact nonlinear flow to each Gauss node and dSigma
there) to a cubic prediction; its stable step is bounded by h sigma <= 0.730
(two corrections).  With converged nodes the step solves g = F(g) per step to
a relative residual (``time_integrators.anderson_fixed_point``, Anderson type
II with real coefficients) and refuses by name when it cannot.

G1 the converged recurrence (magnus_amplification, n_corr=None) has spectral
   radius 1 to 1e-12 on the imaginary axis up to h sigma = 8; R1 the
   truncated two-correction recurrence grows there (> 1.4 at h sigma = 1)
G2 anderson_fixed_point on an affine contraction of two complex "nodes" with
   two trials of very different scale: each trial converges to its own fixed
   point within tol (relative), Hermitian iterates stay Hermitian, the
   evaluation count is reported; R2 max_iter = 3 reports converged = False
   with the residual above tol
G3 on a 6-level commutator toy the converged step equals the truncated step
   with 40 corrections (the same fixed point) to 1e-13, and its error against
   DOP853 falls with order 4 (h, h/2, h/4)
G4 on the 2-level TDA toy of test_prop_stability_cpu.py, where the truncated
   step grows (h kappa = 1.3, 2.0, 3.0), the converged step's mode keeps its
   modulus to 1e-9 per step
G5 on the TDGW fixture with its kernel x40: at 2 dt_max of the truncated step
   (which refuses there) the converged run passes the gate (probe growth <= 1
   + 2e-3, every probe solve converged), runs finite with dSigma counted from
   the solves, and agrees with the truncated run at dt_max / 4 to 1e-2
   relative; the converged and truncated runs at dt_max / 4 agree to 1e-6
R3 node_max_iter = 2 at the same step: the probe refuses (GATE
   td_magnus_stability, naming the unconverged solves); with the probe
   bypassed the run itself refuses (GATE td_node_solve) and writes no chunk
G6 the deck refuses anderson outside cfm4 (td_node_solver), dt = 'auto'
   with it (td_dt), and bad solve settings (td_node_solve_settings)
G7 a converged-node run restarted from its first checkpoint continues
   bitwise; R4 a restart with another node_tol refuses (td_restart_settings)
"""
import json
import os
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)

from solvers import time_integrators as ti   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def _env():
    return dict(os.environ, JAX_PLATFORMS="cpu",
                XLA_FLAGS="--xla_force_host_platform_device_count=4",
                PYTHONPATH=os.pathsep.join(p for p in sys.path if p))


# ---------------------------------------------------------------- G1 / R1

def test_converged_recurrence_is_unitary_on_the_axis():
    for y in np.linspace(0.05, 8.0, 41):
        g = ti.magnus_growth(ti.CFM4_NODES, ti.CFM4_HISTORY, None, y)
        assert abs(g - 1.0) <= 1e-12, (y, g)
    assert ti.magnus_stability_limit(ti.CFM4_NODES, ti.CFM4_HISTORY, None) == 8.0


def test_red_twin_truncated_recurrence_grows():
    assert ti.magnus_growth(ti.CFM4_NODES, ti.CFM4_HISTORY, 2, 1.0) > 1.4


# ---------------------------------------------------------------- G2 / R2

def _affine_problem():
    rng = np.random.default_rng(5)
    n = 5
    def herm(a):
        return 0.5 * (a + np.conj(np.swapaxes(a, -1, -2)))
    # g = M(g) + c on Hermitian pairs, M real-linear: M(g)_i = sum_j a_ij (S g_j S^H)
    s = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    s = 0.45 * s / np.linalg.norm(s, 2)
    a = np.array([[0.25, 0.25 - np.sqrt(3) / 6], [0.25 + np.sqrt(3) / 6, 0.25]])
    c = [herm(rng.normal(size=(2, n, n)) + 1j * rng.normal(size=(2, n, n))) for _ in range(2)]
    scale = np.array([1.0, 1e-9])[:, None, None]          # trial 1 is 1e9 smaller
    c = [ci * scale for ci in c]
    S = jnp.asarray(s)

    def fmap(g):
        tg = [S @ gi @ jnp.conj(S.T) for gi in g]
        return tuple(jnp.asarray(c[i]) + 1.7 * (a[i, 0] * tg[0] + a[i, 1] * tg[1]) for i in range(2))
    return fmap, c


def _exact_fixed_point(fmap, c):
    """Solve the affine system densely (vectorized real-linear map)."""
    shp = c[0].shape
    nvar = 2 * int(np.prod(shp))
    zero = tuple(jnp.zeros(shp, complex) for _ in range(2))
    f0 = [np.asarray(v) for v in fmap(zero)]
    cols = []
    for k in range(2 * nvar):
        e = np.zeros(2 * nvar)
        e[k] = 1.0
        z = (e[:nvar] + 1j * e[nvar:]).reshape((2,) + shp)
        fz = fmap((jnp.asarray(z[0]), jnp.asarray(z[1])))
        d = np.concatenate([np.asarray(fz[i]) - f0[i] for i in range(2)]).ravel()
        cols.append(np.concatenate([d.real, d.imag]))
    m = np.array(cols).T
    b = np.concatenate([np.concatenate(f0).ravel().real, np.concatenate(f0).ravel().imag])
    x = np.linalg.solve(np.eye(2 * nvar) - m, b)
    z = (x[:nvar] + 1j * x[nvar:]).reshape((2,) + shp)
    return z


def test_anderson_converges_per_trial_and_keeps_hermiticity():
    fmap, c = _affine_problem()
    g0 = tuple(jnp.zeros_like(jnp.asarray(ci)) for ci in c)
    tol = 1e-12
    g, info = jax.jit(lambda g: ti.anderson_fixed_point(fmap, g, tol=tol, max_iter=60, memory=5))(g0)
    assert bool(info.converged) and float(info.residual) <= tol, info
    exact = _exact_fixed_point(fmap, c)
    for tr in range(2):                       # each trial to its own scale
        err = max(float(np.max(np.abs(np.asarray(g[i])[tr] - exact[i][tr]))) for i in range(2))
        size = max(float(np.max(np.abs(exact[i][tr]))) for i in range(2))
        assert err <= 1e-9 * size, (tr, err, size)
    for gi in g:
        gi = np.asarray(gi)
        assert np.max(np.abs(gi - np.conj(np.swapaxes(gi, -1, -2)))) <= 1e-12 * np.max(np.abs(gi))
    assert 2 <= int(info.n_eval) <= 40, info


def test_red_twin_anderson_starved_reports_unconverged():
    fmap, c = _affine_problem()
    g0 = tuple(jnp.zeros_like(jnp.asarray(ci)) for ci in c)
    _, info = ti.anderson_fixed_point(fmap, g0, tol=1e-12, max_iter=3, memory=5)
    assert (not bool(info.converged)) and float(info.residual) > 1e-12 and int(info.n_eval) == 3, info


# ---------------------------------------------------------------- G3

N, NOCC = 6, 3
_H = lambda a: jnp.conj(jnp.swapaxes(a, -1, -2))   # noqa: E731


def _conj_dev(f):
    """The deviation conjugation of propagate.make_conj_dev, batched over leading axes."""
    def conj_dev(wm, x, tau):
        wm = 0.5 * (wm + _H(wm))
        ev, v = jnp.linalg.eigh(wm)
        vh = _H(v)
        ph = jnp.exp(-1j * tau * ev)
        em1 = -2.0 * jnp.sin(0.5 * tau * ev) ** 2 - 1j * jnp.sin(tau * ev)
        g = v @ (em1[..., :, None] * vh)
        ux = v @ ((vh @ x @ v) * (ph[..., :, None] * jnp.conj(ph)[..., None, :])) @ vh
        gf = g * f[None, :]
        return ux + gf + _H(gf) + gf @ _H(g)
    return conj_dev


def _toy():
    """A 6-level stand-in of the TD-aGW equation, states (1, N, N) (one trial)."""
    rng = np.random.default_rng(7)
    eps = np.sort(rng.uniform(0.0, 3.0, N))
    d = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    d = 0.5 * (d + d.conj().T)
    np.fill_diagonal(d, 0.0)
    w = rng.uniform(0.2, 1.0, (N, N))
    w = 0.6 * 0.5 * (w + w.T)
    req = np.diag((np.arange(N) < NOCC).astype(float)).astype(complex)
    e = jnp.asarray(eps)
    dlt = e[:, None] - e[None, :]
    D, Wm, R = jnp.asarray(d), jnp.asarray(w), jnp.asarray(req)
    fld = lambda t: 0.4 * jnp.sin(1.3 * t) * jnp.exp(-0.3 * (t - 1.0) ** 2)
    field = lambda t: fld(t) * D
    state_part = lambda t, x: Wm * x
    free = lambda m, s: jnp.exp(-1j * s * dlt) * m
    conj_dev = _conj_dev(jnp.real(jnp.diag(R)))
    h0 = jnp.diag(e.astype(complex))
    comm = lambda a, r: a @ r - r @ a
    rhs = lambda t, r: -1j * comm(h0 + field(t) + Wm * (r - R), r)
    inner = lambda gen, t, x, tau: ti.ip_cf4_flow(gen, free, conj_dev, t, x, tau)
    return dict(field=field, state_part=state_part, free=free, inner=inner, rhs=rhs, R=R,
                rho0=np.asarray(req))


def _run_toy(p, h, t_end, solver, n_corr=2):
    nodes = ti.CFM4_NODES
    w = jnp.asarray(ti.extrapolation_weights(ti.CFM4_HISTORY, list(nodes)))
    x0 = jnp.zeros((1, N, N), complex)
    hist0 = jnp.stack([p["state_part"](0.0, x0)] * 4)
    nodes_fn = lambda ts, xs: tuple(p["state_part"](tt, xx) for tt, xx in zip(ts, xs))

    def step(c, j):
        x, hist = c
        t = j * h
        guess = tuple(jnp.tensordot(w, hist, axes=1))
        if solver == "converged":
            x1, g, info = ti.ip_magnus_implicit_step(p["field"], nodes_fn, p["inner"], p["free"], t, x, h,
                                                     guess, nodes=nodes, tol=1e-14, max_iter=60, memory=6)
            ok = info.converged
        else:
            x1, g = ti.ip_magnus_step(p["field"], p["state_part"], p["inner"], p["free"], t, x, h,
                                      guess, nodes=nodes, n_corr=n_corr)
            ok = jnp.asarray(True)
        hist = p["free"](jnp.concatenate([hist[2:], jnp.stack(g)]), h)
        return (x1, hist), ok
    n = int(round(t_end / h))
    (x, _), ok = jax.jit(lambda c: jax.lax.scan(step, c, jnp.arange(n, dtype=jnp.float64)))((x0, hist0))
    return np.asarray(x[0] + p["R"]), bool(np.all(np.asarray(ok)))


def test_converged_step_is_the_fixed_point_and_order_4():
    p = _toy()
    t_end = 2.0
    ref = solve_ivp(lambda t, y: np.asarray(p["rhs"](t, jnp.asarray(y.reshape(N, N)))).ravel(),
                    (0.0, t_end), p["rho0"].ravel(), method="DOP853", rtol=1e-13, atol=1e-15)
    yref = ref.y[:, -1].reshape(N, N)
    conv, ok = _run_toy(p, 0.1, t_end, "converged")
    many, _ = _run_toy(p, 0.1, t_end, "truncated", n_corr=40)
    assert ok
    assert np.max(np.abs(conv - many)) <= 1e-13, np.max(np.abs(conv - many))
    errs = []
    for h in (0.2, 0.1, 0.05):
        y, ok = _run_toy(p, h, t_end, "converged")
        assert ok
        errs.append(float(np.max(np.abs(y - yref))))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert all(3.6 <= o <= 4.6 for o in orders), (errs, orders)


# ---------------------------------------------------------------- G4

def _toy2_growth(y, solver, nsteps=70):
    """Growth per step of |x_cv| on the 2-level TDA toy u' = -i kappa u (h = 1, kappa = y)."""
    eps = jnp.asarray([0.0, 0.1])
    dlt = eps[:, None] - eps[None, :]
    free = lambda m, s: jnp.exp(-1j * s * dlt) * m
    conj_dev = _conj_dev(jnp.asarray([1.0, 0.0]))

    def b_of(t, x):
        out = jnp.zeros_like(x).at[:, 1, 0].set(y * x[:, 1, 0])
        return out.at[:, 0, 1].set(jnp.conj(out[:, 1, 0]))
    nodes = ti.CFM4_NODES
    w = jnp.asarray(ti.extrapolation_weights(ti.CFM4_HISTORY, list(nodes)))
    inner = lambda gen, t, x, tau: ti.ip_cf4_flow(gen, free, conj_dev, t, x, tau)
    field = lambda t: jnp.zeros((2, 2), complex)
    nodes_fn = lambda ts, xs: tuple(b_of(tt, xx) for tt, xx in zip(ts, xs))

    def step(c, j):
        x, hist = c
        guess = tuple(jnp.tensordot(w, hist, axes=1))
        if solver == "converged":
            x1, g, _ = ti.ip_magnus_implicit_step(field, nodes_fn, inner, free, j, x, 1.0, guess,
                                                  nodes=nodes, tol=1e-13, max_iter=80, memory=6)
        else:
            x1, g = ti.ip_magnus_step(field, b_of, inner, free, j, x, 1.0, guess, nodes=nodes,
                                      n_corr=2)
        hist = free(jnp.concatenate([hist, jnp.stack(g)])[-4:], 1.0)
        return (x1, hist), jnp.abs(x1[0, 1, 0])
    x0 = jnp.zeros((1, 2, 2), complex).at[0, 1, 0].set(1e-60).at[0, 0, 1].set(1e-60)
    seed = jnp.asarray(np.random.default_rng(1).normal(size=(4, 1, 1, 1)))
    _, a = jax.lax.scan(step, (x0, jnp.stack([b_of(0.0, x0)] * 4) * seed),
                        jnp.arange(nsteps, dtype=jnp.float64))
    a = np.asarray(a)
    return float(np.exp(np.log(a[60] / a[40]) / 20.0))


@pytest.mark.parametrize("y", [1.3, 2.0, 3.0])
def test_converged_step_keeps_the_mode_where_truncated_grows(y):
    assert _toy2_growth(y, "truncated") > 1.05, y
    g = _toy2_growth(y, "converged")
    assert abs(g - 1.0) <= 1e-9, (y, g)


# ---------------------------------------------------------------- G5 / R3

@pytest.fixture(scope="module")
def fixture_runs(tmp_path_factory):
    out = subprocess.run([sys.executable, os.path.join(_HERE, "_implicit_case.py"),
                          str(tmp_path_factory.mktemp("impl")), "40"],
                         capture_output=True, text=True, env=_env(), timeout=1800)
    assert out.returncode == 0, out.stderr[-4000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[7:])


def _p(r):
    return np.asarray(r["p_re"]) + 1j * np.asarray(r["p_im"])


def test_converged_nodes_run_where_truncated_refuses(fixture_runs):
    r = fixture_runs
    assert r["trunc_over"].get("refused") == "GATE td_magnus_stability", r["trunc_over"]
    c = r["conv_over"]
    assert "refused" not in c and c["finite"], c
    q = c["probes"][-1]
    assert q["growth"] <= 1.0 + 2e-3 and q["n_fail"] == 0 and q["n_eval"] > 0, q
    nd = c["nodes"]
    assert nd["n_eval"] >= 2 * c["steps"] and nd["res_max"] <= 1e-11, nd
    assert c["dsigma"] == 2 * nd["n_eval"] + 1, c
    ref = _p(r["ref"])
    err = np.linalg.norm(_p(c) - ref) / np.linalg.norm(ref)
    assert err <= 1e-2, err
    small = np.linalg.norm(_p(r["conv_ref"]) - ref) / np.linalg.norm(ref)
    assert small <= 1e-6, small


def test_red_twin_starved_solve_refuses_by_name(fixture_runs):
    s = fixture_runs["starved"]
    assert s.get("refused") == "GATE td_magnus_stability" and "did not converge" in s["message"], s
    u = fixture_runs["starved_unprobed"]
    assert u.get("refused") == "GATE td_node_solve" and "residual" in u["message"], u
    assert u["it_next"] <= 0, u                   # no chunk of the failing run was written


# ---------------------------------------------------------------- G6

def _deck_raw(**lor):
    meta = json.load(open(os.path.join(_HERE, "fixtures", "syn2v2c", "deck_tdgw.json")))
    raw = json.loads(json.dumps(meta["deck"]))
    raw["control"]["files"]["load_header_from"] = "x"
    raw["lorrax"] = dict({"method": "cfm4", "dt": 5.0}, **lor)
    return raw


@pytest.mark.parametrize("lor,gate", [
    (dict(method="rk4", node_solver="anderson"), "td_node_solver"),
    (dict(node_solver="newton"), "td_node_solver"),
    (dict(node_solver="anderson", dt="auto"), "td_dt"),
    (dict(node_solver="anderson", node_tol=0.0), "td_node_solve_settings"),
    (dict(node_solver="anderson", node_max_iter=1), "td_node_solve_settings"),
    (dict(node_solver="anderson", node_memory=0), "td_node_solve_settings")])
def test_deck_refuses_bad_node_settings(lor, gate):
    from tdagw.td_config import parse_td_deck
    with pytest.raises(ValueError, match=f"GATE {gate}"):
        parse_td_deck(_deck_raw(**lor))


def test_deck_accepts_anderson():
    from tdagw.td_config import parse_td_deck
    d = parse_td_deck(_deck_raw(node_solver="anderson", node_tol=1e-8, node_memory=4))
    assert d.values["lorrax.node_solver"] == "anderson" and d.values["lorrax.node_tol"] == 1e-8


# ---------------------------------------------------------------- G7 / R4

@pytest.fixture(scope="module")
def restart(tmp_path_factory):
    conv = {"node_solver": "anderson", "node_tol": 1e-12}
    cases = [dict(tag="conv", method="cfm4", dt=2.5, lorrax=conv),
             dict(tag="conv_mismatch", method="cfm4", dt=2.5, lorrax=conv,
                  mutate=("lorrax.node_tol", 1e-9))]
    out = subprocess.run([sys.executable, os.path.join(_HERE, "_restart_case.py"),
                          str(tmp_path_factory.mktemp("rst")), json.dumps(cases)],
                         capture_output=True, text=True, env=_env(), timeout=1800)
    assert out.returncode == 0, out.stderr[-4000:]
    return {json.loads(l[7:])["tag"]: json.loads(l[7:]) for l in out.stdout.splitlines()
            if l.startswith("RESULT ")}


def test_converged_restart_continues_bitwise(restart):
    r = restart["conv"]
    assert r["mesh"] == [2, 2] and r["resumed_at"] == 20, r
    diffs = {k: v for k, v in r.items() if isinstance(v, float)}
    assert diffs and all(v == 0.0 for v in diffs.values()), r
    assert 0 < r["steps_c"] < r["steps_a"], r


def test_red_twin_converged_restart_refuses_other_tol(restart):
    assert restart["conv_mismatch"].get("refused") == "GATE td_restart_settings", restart["conv_mismatch"]
