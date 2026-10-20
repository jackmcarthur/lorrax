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


def _magnus(f, scheme, h, nodes, fractions, picture="ip"):
    """step(t, x) of an IP Magnus scheme; picture 'full' is the Schroedinger twin."""
    if picture == "ip":
        free, conj = f["free"], f["conj_dev"]
        field = f["field"]
    else:
        free, conj = (lambda m, sft: m), f["conj_full"]
        field = lambda t: f["h0"] + f["field"](t)
    flow = ti.ip_cf4_flow if scheme == "cf4" else ti.ip_midpoint_flow

    def inner(gen, t, x, tau):
        core = lambda a, y, sft: flow(gen, free, conj, a, y, sft)
        return core(t, x, tau) if fractions is None else ti.compose_strang(
            core, f["remainder"], t, x, tau, fractions)

    def step(t, x):
        guess = tuple(free(f["state_part"](t, x), -c * h) for c in nodes)
        return ti.ip_magnus_step(field, f["state_part"], inner, free, t, x, h, guess,
                                 nodes=nodes, n_corr=3)[0]
    return step


def _fixed(scheme, p, r0, h, fractions=None, nodes=None, picture="ip"):
    """Integrate to T_END with a fixed step h; returns the final rho."""
    f = _parts(p)
    n = int(round(T_END / h))
    req = f["rho_eq"]
    if scheme == "rk4":
        step = lambda t, y: ti.rk4_step(f["rhs"], t, y, h)
    elif scheme == "etdrk4":
        coef = ti.etdrk4_coefficients(f["lam"], h)
        step = lambda t, y: ti.etdrk4_step(f["nonlin"], coef, t, y, h)
    elif scheme in ("cf4", "midpoint"):
        nd = nodes or (ti.CFM4_NODES if scheme == "cf4" else ti.MIDPOINT_NODES)
        mstep = _magnus(f, scheme, h, nd, fractions, picture)
        step = lambda t, y: req + mstep(t, y - req)
    elif scheme == "lie":
        mstep = _magnus(f, "cf4", h, ti.CFM4_NODES, None)
        step = lambda t, y: req + f["remainder"](t, mstep(t, y - req), h)
    else:
        raise ValueError(scheme)

    @jax.jit
    def run(r):
        def body(c, _):
            y, j = c
            return (step(j * h, y), j + 1.0), None
        (y, _), _ = jax.lax.scan(body, (r, 0.0), None, length=n, unroll=1)
        return y

    return np.asarray(run(jnp.asarray(r0)))


def _order(scheme, p, r0, ref, h=0.1, **kw):
    e1 = np.linalg.norm(_fixed(scheme, p, r0, h, **kw) - ref)
    e2 = np.linalg.norm(_fixed(scheme, p, r0, h / 2, **kw) - ref)
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
                                             ("midpoint", 2.0), ("cf4", 4.0)])
def test_convergence_order(conservative, scheme, expected):
    p, r0, ref = conservative
    order = _order(scheme, p, r0, ref)
    assert abs(order - expected) < 0.3, (scheme, order)


def test_red_twin_cf4_equispaced_nodes(conservative):
    p, r0, ref = conservative
    order = _order("cf4", p, r0, ref, nodes=(0.25, 0.75))
    assert not abs(order - 4.0) < 0.3, order


# ---------------------------------------------------------------- G3 / R3 ----

def _projector_defect(scheme):
    p = _problem(gamma=0.0)
    r = _fixed(scheme, p, _rho0(p), 0.25)
    return np.linalg.norm(r @ r - r)


@pytest.mark.parametrize("scheme", ["midpoint", "cf4"])
def test_magnus_keeps_a_projector(scheme):
    assert _projector_defect(scheme) <= 1e-12


def test_red_twin_rk4_is_not_unitary():
    assert not _projector_defect("rk4") <= 1e-12


# ------------------------------------------------------ G4 / G5 / R4 / R5 ----

@pytest.fixture(scope="module", params=["gamma", "intraband"])
def split(request):
    p = _problem(gamma=0.4) if request.param == "gamma" else _problem(intra=0.8)
    r0 = _rho0(p)
    ref = _scipy_reference(p, r0).y[:, -1].reshape(N, N)
    return p, r0, ref


def test_strang_splitting_is_second_order(split):
    p, r0, ref = split
    order = _order("cf4", p, r0, ref, h=0.05, fractions=(1.0,))
    assert abs(order - 2.0) < 0.3, order


def test_triple_jump_is_fourth_order(split):
    p, r0, ref = split
    order = _order("cf4", p, r0, ref, h=0.1, fractions=ti.TRIPLE_JUMP)
    assert abs(order - 4.0) < 0.3, order


def test_red_twin_lie_splitting_is_first_order(split):
    p, r0, ref = split
    order = _order("lie", p, r0, ref, h=0.05)
    assert not abs(order - 2.0) < 0.3, order


def test_red_twin_equal_thirds_is_not_fourth_order(split):
    p, r0, ref = split
    order = _order("cf4", p, r0, ref, h=0.1, fractions=(1 / 3, 1 / 3, 1 / 3))
    assert not abs(order - 4.0) < 0.3, order


# ---------------------------------------------------------------- G6 / R6 ----

@pytest.fixture(scope="module")
def weak():
    p = _problem(gamma=0.0, amp=1e-9)
    r0 = _rho0(p, small=1e-9)
    ref = _scipy_reference(p, r0, atol=1e-24).y[:, -1].reshape(N, N)
    return p, r0, ref, np.linalg.norm(ref - p["rho_eq"])


def _weak_errors(weak, picture):
    p, r0, ref, dev = weak
    return [np.linalg.norm(_fixed("cf4", p, r0, h, picture=picture) - ref) / dev
            for h in (0.05, 0.01, 0.005)]


def test_no_rounding_floor_in_weak_coupling(weak):
    e = _weak_errors(weak, "ip")
    assert e[-1] <= 2e-11 and e[0] > e[1] > e[2], e


def test_red_twin_full_conjugation_has_a_floor(weak):
    e = _weak_errors(weak, "full")
    assert not (e[-1] <= 2e-11 and e[0] > e[1] > e[2]), e
    assert e[-1] > 1e-8, e


# ---------------------------------------------------------------- G7 / R7 ----

def _dense_error(h, frame=True):
    """y' = lam y + n with n = c y (a small coupling: the exciton shift of a free
    transition), y = y0 e^{(lam + c) t} known; error at theta = 1/2 of one step."""
    lam = jnp.asarray([-1.0j, -0.7j - 0.05])
    c = jnp.asarray([-0.1j - 0.05, 0.08j])
    y0 = jnp.asarray([0.3 + 0.1j, -0.2j])
    exact = lambda t: y0 * jnp.exp((lam + c) * t)
    y1 = exact(h)
    lf = lam if frame else jnp.zeros_like(lam)
    d0 = (lam + c - lf) * y0                 # the derivative minus lf * y at each end
    d1 = (lam + c - lf) * y1
    got = {th: ti.hermite_dense(lf, y0, d0, y1, d1, h, th) for th in (0.0, 0.5, 1.0)}
    return (float(jnp.max(jnp.abs(got[0.5] - exact(0.5 * h)))),
            float(jnp.max(jnp.abs(got[0.0] - y0))), float(jnp.max(jnp.abs(got[1.0] - y1))))


def test_hermite_dense_is_fourth_order_in_the_frame():
    e1, a0, a1 = _dense_error(1.0)
    e2, _, _ = _dense_error(0.5)
    assert abs(np.log2(e1 / e2) - 4.0) < 0.5 and a0 <= 1e-15 and a1 <= 1e-15, (e1, e2, a0, a1)


def test_red_twin_plain_cubic_is_worse():
    e_frame = _dense_error(1.0)[0]
    e_plain = _dense_error(1.0, frame=False)[0]
    assert e_plain > 100 * e_frame, (e_plain, e_frame)


# ---------------------------------------------------------------- G8 / R8 ----

def _sampled(shift=0):
    p = _problem(gamma=0.1)
    f = _parts(p)
    h = 0.1
    times = np.asarray([0.0, 0.2, 0.25, 0.47, 0.6])         # on-step, interior, ...
    r = times / h
    on = np.abs(r - np.rint(r)) < 1e-9
    sstep = np.where(on, np.rint(r), np.floor(r)).astype(np.int64) + shift
    sth = np.where(on, 0.0, r - np.floor(r))

    def step(t, c, hh, t_next):
        y = ti.rk4_step(f["rhs"], t, c[0], hh, k1=c[1])
        return (y, f["rhs"](t_next, y)), ()

    lam = f["lam"]
    dense = lambda t, c0, c1, aux, th: ti.hermite_dense(
        lam, c0[0], c0[1] - lam * c0[0], c1[0], c1[1] - lam * c1[0], h, th)
    r0 = jnp.asarray(_rho0(p))

    @jax.jit
    def run(y0):
        c = (y0, f["rhs"](0.0, y0))
        return ti.fixed_advance(step, dense, lambda c: c[0], c, t_start=0.0, h=h, step0=0,
                                n_steps=7, sample_step=jnp.asarray(sstep),
                                sample_theta=jnp.asarray(sth),
                                sample_times=jnp.asarray(times),
                                sample_fn=lambda t, y: y)

    _, got = run(r0)
    # reference: a plain loop of the same steps, and DOP853 for the interior samples
    y, states = r0, [r0]
    for j in range(7):
        y = ti.rk4_step(f["rhs"], j * h, y, h)
        states.append(y)
    t_all = np.concatenate([times, h * np.arange(8)])
    t_eval, inv = np.unique(np.round(t_all, 12), return_inverse=True)
    sol = _scipy_reference(p, np.asarray(r0), t_eval=t_eval).y.T.reshape(-1, N, N)[inv]
    step_err = np.max(np.abs(np.asarray(states) - sol[len(times):]))
    return np.asarray(got), np.asarray(states), sol[:len(times)], on, step_err


def _plan_matches(got, states, ref, on, step_err):
    """On-step samples = the stepped states (to rounding: eager vs jitted loop);
    interior samples within twice RK4's own error at the steps (vs DOP853)."""
    on_err = max(np.max(np.abs(got[i] - states[j])) for i, j in ((0, 0), (1, 2), (4, 6)))
    in_err = np.max(np.abs(got[~on] - ref[~on]))
    return on_err <= 1e-14 and in_err <= 2.0 * step_err, (on_err, in_err, step_err)


def test_fixed_advance_samples_on_and_between_steps():
    ok, errs = _plan_matches(*_sampled())
    assert ok, errs


def test_red_twin_shifted_plan_does_not_match():
    ok, errs = _plan_matches(*_sampled(shift=1))
    assert not ok, errs
