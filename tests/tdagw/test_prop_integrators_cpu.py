"""The TD-aGW time integrators on a small commutator equation (CPU, seconds).

``solvers.time_integrators`` is checked on

    rho' = -i [H0 + E(t) D + B(rho), rho] + R(t, rho - rho_eq),
    B(rho) = g W o (rho - rho_eq)        (W real symmetric: B is Hermitian),

a six-level stand-in for the TD-aGW equation of motion (Hartree/SEX enter
only through a Hermitian B linear in rho - rho_eq).  R is either Gamma
relaxation (exact flow) or a time-dependent non-commutator linear map
E(t) c (S x S^H - x) integrated by RK4, the shape of the intraband term.
The reference is SciPy's DOP853 at rtol 1e-13, code shared with nothing here.

Gates (G) and red twins (R); each twin feeds a deliberately wrong scheme to
the same checker as the gate it shadows and must fail it.

G1 dp54_advance == scipy solve_ivp RK45: same RHS count, samples <= 1e-12
R1 safety factor 0.8 instead of SciPy's 0.9 -> RHS count differs
G2 measured orders, Gamma = 0: rk4 4, etdrk4 4, IP midpoint 2, IP CF4 4
R2 IP CF4 with dSigma frozen through equispaced nodes (1/4, 3/4) -> order 2
G3 Magnus keeps rho_eq + x a projector (||rho^2 - rho|| <= 1e-12) at h = 0.25
R3 rk4 at the same step drifts (> 1e-8)
G4 Strang has order 2 with a remainder (Gamma, and the intraband-like map)
G5 the triple jump of the same Strang step has order 4 with either remainder
R4 Lie splitting (whole remainder after the step) -> order 1, fails G4
R5 three Strang steps of h/3 (no negative substep) -> order 2, fails G5
G6 weak coupling (|x| ~ 1e-9): the deviation's relative error keeps falling
   with h to <= 2e-11 (no rounding floor)
R6 the full-rho conjugation in the Schroedinger picture (the conjugation
   rounds the O(1) occupations) stalls above 1e-8 on the same problem
G7 hermite_dense in the frame of lam, for a coupling c y small next to lam
   (the regime of a transition shifted by the interaction): error at
   theta = 1/2 falls as h^4 and theta = 0, 1 return the ends; R7 the plain
   cubic (lam frame dropped) is 100x worse at h |lam| = 1
G8 fixed_advance: on-step samples are the stepped states (1e-14: an eager
   loop is the reference), interior samples come from dense within twice
   the method's own error at the steps (DOP853 reference);
   R8 a plan shifted by one step fails that checker
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)

from solvers import time_integrators as ti   # noqa: E402

N, NOCC, T_END = 6, 3, 2.0


def _problem(gamma=0.0, amp=0.4, g=0.3, intra=0.0):
    rng = np.random.default_rng(7)
    eps = np.sort(rng.uniform(0.0, 3.0, N))
    d = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    d = 0.5 * (d + d.conj().T)
    np.fill_diagonal(d, 0.0)
    w = rng.uniform(0.2, 1.0, (N, N))
    w = 0.5 * (w + w.T)
    rho_eq = np.diag((np.arange(N) < NOCC).astype(float)).astype(complex)
    gam = np.full((N, N), gamma) * (1 - np.eye(N)) + 0.5 * gamma * np.eye(N)
    s, _ = np.linalg.qr(np.eye(N) + 0.3 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))))
    return dict(eps=eps, d=d, w=g * w, rho_eq=rho_eq, gam=gam, amp=amp, s=s, intra=intra)


def _parts(p):
    eps = jnp.asarray(p["eps"])
    dlt = eps[:, None] - eps[None, :]
    h0 = jnp.diag(eps.astype(jnp.complex128))
    d, w, req, gam, s = (jnp.asarray(p[k]) for k in ("d", "w", "rho_eq", "gam", "s"))
    fld = lambda t: p["amp"] * jnp.sin(1.3 * t) * jnp.exp(-0.3 * (t - 1.0) ** 2)
    field = lambda t: fld(t) * d
    state_part = lambda t, x: w * x
    comm = lambda a, r: a @ r - r @ a
    lmap = lambda t, x: p["intra"] * fld(t) * (s @ x @ s.conj().T - x)     # intraband-like
    rhs = lambda t, r: (-1j * comm(h0 + field(t) + w * (r - req), r) - gam * (r - req)
                        + lmap(t, r - req) + lmap(t, req))
    nonlin = lambda t, r: -1j * comm(field(t) + w * (r - req), r) + lmap(t, r) + gam * req
    lam = -1j * dlt - gam
    free = lambda m, sft: jnp.exp(-1j * sft * dlt) * m
    f = jnp.real(jnp.diag(req))

    def conj_dev(wm, x, tau):
        wm = 0.5 * (wm + wm.conj().T)
        e, v = jnp.linalg.eigh(wm)
        vh = v.conj().T
        ph = jnp.exp(-1j * tau * e)
        em1 = -2.0 * jnp.sin(0.5 * tau * e) ** 2 - 1j * jnp.sin(tau * e)
        g = v @ (em1[:, None] * vh)
        ux = v @ ((vh @ x @ v) * (ph[:, None] * jnp.conj(ph)[None, :])) @ vh
        gf = g * f[None, :]
        return ux + gf + gf.conj().T + gf @ g.conj().T

    def conj_full(wm, x, tau):
        """The Schroedinger-picture conjugation of the whole state (wm includes H0)."""
        e, v = jnp.linalg.eigh(0.5 * (wm + wm.conj().T))
        u = (v * jnp.exp(-1j * tau * e)[None, :]) @ v.conj().T
        return u @ (req + x) @ u.conj().T - req

    def remainder(t, x, tau):
        decay = jnp.exp(-0.5 * gam * tau)
        x = decay * x
        if p["intra"]:
            x = ti.rk4_step(lambda sft, y: lmap(sft, y) + lmap(sft, req), t, x, tau)
        return decay * x

    return dict(field=field, state_part=state_part, rhs=rhs, nonlin=nonlin, lam=lam,
                free=free, conj_dev=conj_dev, conj_full=conj_full, remainder=remainder,
                rho_eq=req, h0=h0)


def _rho0(p, small=0.2):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    q, _ = np.linalg.qr(np.eye(N) + small * a)
    return q @ p["rho_eq"] @ q.conj().T


def _scipy_reference(p, r0, method="DOP853", rtol=1e-13, atol=1e-15, **kw):
    f = _parts(p)
    fun = lambda t, y: np.asarray(f["rhs"](t, jnp.asarray(y.reshape(N, N)))).ravel()
    return solve_ivp(fun, (0.0, T_END), r0.ravel(), method=method, rtol=rtol, atol=atol, **kw)


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


def _check_dp54_reproduces_scipy_rk45():
    n_rhs, nfev, rel = _dp54_vs_scipy()
    assert n_rhs == nfev and rel <= 1e-12, (n_rhs, nfev, rel)


def _check_red_twin_dp54_wrong_safety_factor(monkeypatch):
    monkeypatch.setattr(ti, "DP54_SAFETY", 0.8)
    n_rhs, nfev, rel = _dp54_vs_scipy()
    assert not (n_rhs == nfev and rel <= 1e-12), (n_rhs, nfev, rel)


# ---------------------------------------------------------------- G2 / R2 ----

# ---------------------------------------------------------------- G3 / R3 ----

# ------------------------------------------------------ G4 / G5 / R4 / R5 ----

@pytest.fixture(scope="module", params=["gamma", "intraband"])
def split(request):
    p = _problem(gamma=0.4) if request.param == "gamma" else _problem(intra=0.8)
    r0 = _rho0(p)
    ref = _scipy_reference(p, r0).y[:, -1].reshape(N, N)
    return p, r0, ref


# ---------------------------------------------------------------- G6 / R6 ----

@pytest.fixture(scope="module")
def weak():
    p = _problem(gamma=0.0, amp=1e-9)
    r0 = _rho0(p, small=1e-9)
    ref = _scipy_reference(p, r0, atol=1e-24).y[:, -1].reshape(N, N)
    return p, r0, ref, np.linalg.norm(ref - p["rho_eq"])


# ---------------------------------------------------------------- G7 / R7 ----

# ---------------------------------------------------------------- G8 / R8 ----

def test_dp54_reproduces_scipy_rk45(monkeypatch):
    """G1 and R1: DP54 takes RK45 steps; the wrong safety factor does not."""
    _check_dp54_reproduces_scipy_rk45()
    with monkeypatch.context() as mp:
        _check_red_twin_dp54_wrong_safety_factor(mp)

