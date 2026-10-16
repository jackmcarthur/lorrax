"""The TD-aGW time integrators on a small commutator equation (CPU, seconds).

``solvers.time_integrators`` is checked on

    rho' = -i [H0 + E(t) D + B(rho), rho] - Gamma o (rho - rho_eq),
    B(rho) = g W o (rho - rho_eq)        (W real symmetric: B is Hermitian),

a six-level stand-in for the TD-aGW equation of motion (Hartree/SEX enter
only through a Hermitian B linear in rho - rho_eq).  The reference is SciPy's
DOP853 at rtol 1e-13, code shared with nothing here.

Gates (G) and red twins (R); each twin feeds a deliberately wrong scheme to
the same checker as the gate it shadows and must fail it.

G1 dp54_advance == scipy solve_ivp RK45: same RHS count, samples <= 1e-12
R1 safety factor 0.8 instead of SciPy's 0.9 -> RHS count differs
G2 measured orders: rk4 4, etdrk4 4, magnus2 (n_iter 2) 2, cfm4 (n_corr 3) 4
R2 cfm4 with its two exponential weights swapped -> order 2
G3 Magnus keeps rho a projector (||rho^2 - rho|| <= 1e-12) at h = 0.25
R3 rk4 at the same step drifts (> 1e-8)
G4 strang(cfm4, exact relaxation) has order 2 with Gamma != 0
R4 Lie splitting (whole relaxation after the step) has order 1
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)

from solvers import time_integrators as ti   # noqa: E402

N, NOCC, T_END = 6, 3, 2.0


def _problem(gamma=0.0):
    rng = np.random.default_rng(7)
    eps = np.sort(rng.uniform(0.0, 3.0, N))
    d = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    d = 0.5 * (d + d.conj().T)
    w = rng.uniform(0.2, 1.0, (N, N))
    w = 0.5 * (w + w.T)
    rho_eq = np.diag((np.arange(N) < NOCC).astype(float)).astype(complex)
    gam = np.full((N, N), gamma) * (1 - np.eye(N)) + gamma * np.eye(N)
    return dict(eps=eps, d=d, w=0.3 * w, rho_eq=rho_eq, gam=gam)


def _field(t):
    return 0.4 * jnp.sin(1.3 * t) * jnp.exp(-0.3 * (t - 1.0) ** 2)


def _parts(p):
    h0 = jnp.diag(jnp.asarray(p["eps"], jnp.complex128))
    d, w, req = (jnp.asarray(p[k]) for k in ("d", "w", "rho_eq"))
    gam = jnp.asarray(p["gam"])
    explicit = lambda t: h0 + _field(t) * d
    state_part = lambda t, r: w * (r - req)
    comm = lambda a, r: a @ r - r @ a
    rhs = lambda t, r: -1j * comm(explicit(t) + state_part(t, r), r) - gam * (r - req)
    nonlin = lambda t, r: -1j * comm(_field(t) * d + state_part(t, r), r)
    lam = -1j * (jnp.asarray(p["eps"])[:, None] - jnp.asarray(p["eps"])[None, :]) - gam

    def conjugate(hm, r, h):
        e, v = jnp.linalg.eigh(0.5 * (hm + hm.conj().T))
        u = (v * jnp.exp(-1j * h * e)[None, :]) @ v.conj().T
        return u @ r @ u.conj().T

    def relax(t, r, tau):
        return req + jnp.exp(-gam * tau) * (r - req)

    return dict(explicit=explicit, state_part=state_part, rhs=rhs, nonlin=nonlin,
                lam=lam, conjugate=conjugate, relax=relax, rho_eq=req)


def _rho0(p):
    """A pure state rotated off equilibrium, so the commutator is live at t=0."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    q, _ = np.linalg.qr(np.eye(N) + 0.2 * a)
    return q @ p["rho_eq"] @ q.conj().T


def _scipy_reference(p, r0, method="DOP853", rtol=1e-13, atol=1e-15, **kw):
    f = _parts(p)
    fun = lambda t, y: np.asarray(f["rhs"](t, jnp.asarray(y.reshape(N, N)))).ravel()
    return solve_ivp(fun, (0.0, T_END), r0.ravel(), method=method, rtol=rtol, atol=atol, **kw)


def _fixed(scheme, p, r0, h):
    """Integrate to T_END with a fixed step h; returns the final rho."""
    f = _parts(p)
    n = int(round(T_END / h))

    def run(r):
        if scheme == "rk4":
            step = lambda t, y: ti.rk4_step(f["rhs"], t, y, h)
        elif scheme == "etdrk4":
            coef = ti.etdrk4_coefficients(f["lam"], h)
            step = lambda t, y: ti.etdrk4_step(f["nonlin"], coef, t, y, h)
        elif scheme == "magnus2":
            step = lambda t, y: ti.magnus_midpoint_step(
                f["explicit"], f["state_part"], f["conjugate"], t, y, h,
                f["state_part"](t, y), n_iter=2)[0]
        elif scheme == "cfm4":
            step = lambda t, y: ti.cfm4_step(
                f["explicit"], f["state_part"], f["conjugate"], t, y, h,
                (f["state_part"](t, y),) * 2, n_corr=3)[0]
        elif scheme == "strang":
            core = lambda t, y, hh: ti.cfm4_step(
                f["explicit"], f["state_part"], f["conjugate"], t, y, hh,
                (f["state_part"](t, y),) * 2, n_corr=3)[0]
            step = lambda t, y: ti.strang_step(core, f["relax"], t, y, h)
        elif scheme == "lie":
            core = lambda t, y, hh: ti.cfm4_step(
                f["explicit"], f["state_part"], f["conjugate"], t, y, hh,
                (f["state_part"](t, y),) * 2, n_corr=3)[0]
            step = lambda t, y: f["relax"](t, core(t, y, h), h)
        else:
            raise ValueError(scheme)
        out, _ = ti.fixed_advance(lambda t, c, hh: step(t, c), r, t_start=0.0, h=h,
                                  n_sub=n, n_samples=1, sample_fn=lambda t, c: ())
        return out

    return np.asarray(jax.jit(run)(jnp.asarray(r0)))


def _order(scheme, p, r0, ref, h=0.1):
    e1 = np.linalg.norm(_fixed(scheme, p, r0, h) - ref)
    e2 = np.linalg.norm(_fixed(scheme, p, r0, h / 2) - ref)
    return np.log2(e1 / e2)


# ---------------------------------------------------------------- G1 / R1 ----

def _dp54_vs_scipy():
    p = _problem(gamma=0.02)
    f = _parts(p)
    r0 = _rho0(p)
    times = np.linspace(0.0, T_END, 21)
    ref = _scipy_reference(p, r0, method="RK45", rtol=1e-6, atol=1e-9,
                           t_eval=times, first_step=0.01)

    @jax.jit
    def run(y0):
        st = ti.dp54_initial_state(f["rhs"], 0.0, y0, 0.01)
        return ti.dp54_advance(f["rhs"], st, t_bound=T_END, max_step=np.inf, atol=1e-9,
                               rtol=1e-6, count=N * N, sample_times=jnp.asarray(times),
                               sample_fn=lambda t, y: y)

    st, samples = run(jnp.asarray(r0))
    got = np.asarray(samples).reshape(len(times), -1)
    rel = np.max(np.abs(got - ref.y.T)) / np.max(np.abs(ref.y))
    return int(st.n_rhs), int(ref.nfev), rel


def test_dp54_reproduces_scipy_rk45():
    n_rhs, nfev, rel = _dp54_vs_scipy()
    assert n_rhs == nfev and rel <= 1e-12, (n_rhs, nfev, rel)


def test_red_twin_dp54_wrong_safety_factor(monkeypatch):
    monkeypatch.setattr(ti, "DP54_SAFETY", 0.8)
    n_rhs, nfev, rel = _dp54_vs_scipy()
    assert not (n_rhs == nfev and rel <= 1e-12), (n_rhs, nfev, rel)


# ---------------------------------------------------------------- G2 / R2 ----

@pytest.fixture(scope="module")
def conservative():
    p = _problem(gamma=0.0)
    r0 = _rho0(p)
    ref = _scipy_reference(p, r0).y[:, -1].reshape(N, N)
    return p, r0, ref


@pytest.mark.parametrize("scheme,expected", [("rk4", 4.0), ("etdrk4", 4.0),
                                             ("magnus2", 2.0), ("cfm4", 4.0)])
def test_convergence_order(conservative, scheme, expected):
    p, r0, ref = conservative
    order = _order(scheme, p, r0, ref)
    assert abs(order - expected) < 0.3, (scheme, order)


def test_red_twin_cfm4_swapped_weights(conservative, monkeypatch):
    p, r0, ref = conservative
    al1, al2 = ti.CFM4_WEIGHTS
    monkeypatch.setattr(ti, "CFM4_WEIGHTS", (al2, al1))
    order = _order("cfm4", p, r0, ref)
    assert not abs(order - 4.0) < 0.3, order


# ---------------------------------------------------------------- G3 / R3 ----

def _projector_defect(scheme):
    p = _problem(gamma=0.0)
    r = _fixed(scheme, p, _rho0(p), 0.25)
    return np.linalg.norm(r @ r - r)


@pytest.mark.parametrize("scheme", ["magnus2", "cfm4"])
def test_magnus_keeps_a_projector(scheme):
    assert _projector_defect(scheme) <= 1e-12


def test_red_twin_rk4_is_not_unitary():
    assert not _projector_defect("rk4") <= 1e-12


# ---------------------------------------------------------------- G4 / R4 ----

@pytest.fixture(scope="module")
def dissipative():
    p = _problem(gamma=0.4)
    r0 = _rho0(p)
    ref = _scipy_reference(p, r0).y[:, -1].reshape(N, N)
    return p, r0, ref


def test_strang_splitting_is_second_order(dissipative):
    p, r0, ref = dissipative
    order = _order("strang", p, r0, ref, h=0.05)
    assert abs(order - 2.0) < 0.3, order


def test_red_twin_lie_splitting_is_first_order(dissipative):
    p, r0, ref = dissipative
    order = _order("lie", p, r0, ref, h=0.05)
    assert not abs(order - 2.0) < 0.3, order
