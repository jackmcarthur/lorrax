"""The on-shell collision term ``tdagw.collision`` against NumPy / SciPy references (CPU, seconds).

docs/theory/td_agw_dynamical.md §1.5, Eq. (8).  Random problem: leading axes
(trial 2, k 3), 5 bands; pair tables of a particle and a hole branch with
positive widths (retarded particle, time-ordered hole), a Hermitian state ρ
and an equilibrium ρ_eq = diag(f).  References are written here from the
equations, sharing no code with the module.

G1 one-sided branches: R_p = cond[1], R_h = val[0]^H; Gamma, D and M_H
   equal an independent positive-time transform reference                              rel <= 1e-13
G2 C_raw equals -i[M_H, ρ] - 1/2 {Gamma, ρ} + D; Hermitian for Hermitian ρ     rel <= 1e-13
G3 flow(τ) == the affine generator's exact solution, expm of the vectorized
   [[L, s], [0, 0]] (L x = -1/2 {G~, x}), τ in {1e-3, 0.7, 5, -0.3}           rel <= 1e-12
G4 rate == -1/2 {G~, x} + S and == d flow/dτ at 0 (central difference)
G5 phi1: phi1(0) = 1; the series and the closed form agree at the switch;
   both == a 30-term series below 1e-2 and expm1 above                       rel <= 1e-15
G6 subtracted at the equilibrium rates: S = 0 and flow(0, τ) = 0 exactly;
   raw: flow(0, τ) == τ phi1(z) S' in the eigenbasis (== G3's reference)
G7 trace_rates: out + in == Tr C[ρ]; subtracted: == Tr (C[ρ] - C_eq[ρ_eq])
G8 hermitian_update: full, dynamic (static change removed), Hermitian
G9 refresh_due: cadence, population tolerance, both off
G10 flow(τ1 + τ2) == flow(τ2) ∘ flow(τ1) (a semigroup: exact, not a scheme)
G11 D >= 0 and Gamma-D >= 0 keep an initially physical full matrix in [0, 1]
R1 G~ ρ + ρ G~ replaced by 2 G~ ρ: C[ρ] no longer Hermitian, fails G2
R2 the source integrated as τ S (phi1 -> 1): fails G3
"""
import numpy as np
import jax
import pytest
from scipy.linalg import expm

jax.config.update("jax_enable_x64", True)

from tdagw import collision as C   # noqa: E402

NTR, NK, N = 2, 3, 5


def _herm(a):
    return 0.5 * (a + np.conj(np.swapaxes(a, -1, -2)))


def _psd(rng, scale):
    a = rng.normal(size=(NTR, NK, N, N)) + 1j * rng.normal(size=(NTR, NK, N, N))
    return scale * (a @ np.conj(np.swapaxes(a, -1, -2))) / N


@pytest.fixture(scope="module")
def prob():
    rng = np.random.default_rng(11)
    # Pair tables whose on-shell widths are the PSD matrices g_p, g_h: particle
    # S~_p = H_p - i g_p / 2 (retarded), hole S~_h = H_h + i g_h / 2 (time ordered).
    g_p, g_h = _psd(rng, 0.02), _psd(rng, 0.015)
    h_p = _herm(rng.normal(size=(NTR, NK, N, N)) + 1j * rng.normal(size=(NTR, NK, N, N)))
    h_h = _herm(rng.normal(size=(NTR, NK, N, N)) + 1j * rng.normal(size=(NTR, NK, N, N)))
    skew = lambda: 0.01 * (rng.normal(size=(NTR, NK, N, N)) + 1j * rng.normal(size=(NTR, NK, N, N)))
    d_p, d_h = skew(), skew()                      # T1 - T2 differences (cancel in S~)
    s_p, s_h = h_p - 0.5j * g_p, h_h + 0.5j * g_h
    pair_p = np.stack([s_p + d_p, s_p - d_p])
    pair_h = np.stack([s_h + d_h, s_h - d_h])
    f = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    rho_eq = np.broadcast_to(np.diag(f).astype(complex), (NTR, NK, N, N)).copy()
    x = 0.05 * _herm(rng.normal(size=(NTR, NK, N, N)) + 1j * rng.normal(size=(NTR, NK, N, N)))
    return dict(pair_p=pair_p, pair_h=pair_h, g_p=g_p, g_h=g_h, s_p=s_p, s_h=s_h,
                rho_eq=rho_eq, rho=rho_eq + x, x=x, rng=rng)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _c_ref(g_out, g_in, rho):
    one = np.eye(N)
    return (-0.5 * (g_out @ rho + rho @ g_out)
            + 0.5 * (g_in @ (one - rho) + (one - rho) @ g_in))


def _affine_ref(gam, src, x0, tau):
    """x(τ) of x' = -1/2 (G x + x G) + S by expm of the augmented (n^2 + 1) matrix, per index."""
    out = np.empty_like(x0)
    one = np.eye(N)
    for idx in np.ndindex(x0.shape[:-2]):
        g = gam[idx]
        L = -0.5 * (np.kron(g, one) + np.kron(one, g.T))    # row-major vec: (A X B) = (A ⊗ B^T)
        M = np.zeros((N * N + 1, N * N + 1), complex)
        M[:N * N, :N * N] = L
        M[:N * N, N * N] = src[idx].reshape(-1)
        v = np.concatenate([x0[idx].reshape(-1), [1.0]])
        out[idx] = (expm(tau * M) @ v)[:N * N].reshape(N, N)
    return out


def test_one_sided_branch_operators(prob):
    dag = lambda a: np.conj(np.swapaxes(a, -1, -2))
    pp, ph = prob["pair_p"], prob["pair_h"]
    static = prob["s_p"] + dag(prob["s_p"])
    gamma, injection, m_h = map(np.asarray, C.branch_operators(pp, ph, static))
    rp, rh = pp[1], dag(ph[0])
    m = rp + rh + static
    assert _rel(gamma, 1j * (rp + rh - dag(rp + rh))) <= 1e-15
    assert _rel(injection, 1j * rh + dag(1j * rh)) <= 1e-15
    assert _rel(m_h, _herm(m)) <= 1e-15
    # The advanced row, not advanced column, becomes the retarded hole column.
    wrong_rh = dag(ph[1])
    assert _rel(injection, 1j * wrong_rh + dag(1j * wrong_rh)) > 1e-2
    # Sum of the split coherent and remainder terms equals the literal C_raw.
    rho = prob["rho"]
    ain, aout, one = 1j * rh, 1j * rp, np.eye(N)
    raw = ain @ (one - rho) + (one - rho) @ dag(ain) - aout @ rho - rho @ dag(aout)
    split = -1j * (_herm(rp + rh) @ rho - rho @ _herm(rp + rh))
    split += np.asarray(C.collision(gamma, injection, rho))
    assert _rel(split, raw) <= 1e-13


def test_positive_time_transform_reference():
    """Column extraction equals direct ∫_0^∞ exp(i E_b t) Sigma</>(t) dt."""
    rng = np.random.default_rng(37)
    e = np.linspace(-0.8, 1.1, N)
    eta, omega = 0.23, 0.41
    greater_amp = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    lesser_amp = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))

    # Sigma(t) = amp exp[-(eta + i omega)t]. The two tables sample the
    # transform at the row or column energy independently.
    transform = lambda amp, energy: amp / (eta + 1j * (omega - energy))
    sigma_g_col = transform(greater_amp, e[None, :])
    sigma_l_col = transform(lesser_amp, e[None, :])
    rp, rh = sigma_g_col, -sigma_l_col
    cond = np.stack([transform(greater_amp, e[:, None]), rp])
    # Construct the advanced table directly from Sigma^<(-t)=-Sigma^<(t)^H.
    # Keep slot 1 nonzero so selecting the advanced column would be detected.
    advanced_amp = -np.conj(lesser_amp.T)
    advanced = lambda energy: advanced_amp / (eta - 1j * (omega - energy))
    val = np.stack([advanced(e[:, None]), advanced(e[None, :])])

    gamma, injection, m_h = map(np.asarray, C.branch_operators(cond, val))
    dag = lambda a: np.conj(a.T)
    ain, aout, m = -1j * sigma_l_col, 1j * sigma_g_col, rp + rh
    assert _rel(gamma, 1j * (m - dag(m))) <= 1e-13
    assert _rel(injection, ain + dag(ain)) <= 1e-13
    assert _rel(m_h, _herm(m)) <= 1e-13

    rho = _herm(rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    one = np.eye(N)
    literal = ain @ (one - rho) + (one - rho) @ dag(ain)
    literal -= aout @ rho + rho @ dag(aout)
    split = -1j * (m_h @ rho - rho @ m_h) + np.asarray(
        C.collision(gamma, injection, rho))
    assert _rel(split, literal) <= 1e-13


def test_pv_column_terms_survive(prob):
    pp, ph = prob["pair_p"], prob["pair_h"]
    got = tuple(map(np.asarray, C.branch_operators(pp, ph)))
    # Zero arithmetic-average change, nonzero one-sided change: the old
    # on-shell averaging would miss all of this perturbation.
    delta = 0.02j * np.ones_like(pp[0])
    altered = ph + np.stack([delta, -delta])
    changed = tuple(map(np.asarray, C.branch_operators(pp, altered)))
    assert _rel(changed[1], got[1]) > 1e-2
    assert _rel(np.asarray(C.collision(changed[0], changed[1], prob["rho"])),
                np.asarray(C.collision(got[0], got[1], prob["rho"]))) > 1e-2


@pytest.mark.parametrize("tau", [0.7, -0.3])
def test_column_sampled_flow(prob, tau):
    gamma, injection, _ = map(np.asarray, C.branch_operators(prob["pair_p"], prob["pair_h"]))
    src = -0.5 * (gamma @ prob["rho_eq"] + prob["rho_eq"] @ gamma) + injection
    d = C.dissipator(gamma, src)
    assert _rel(np.asarray(C.flow(d, prob["x"], tau)),
                _affine_ref(gamma, src, prob["x"], tau)) <= 1e-12


def test_positive_held_generator_preserves_pauli_interval():
    """The exact affine flow respects both matrix boundary inequalities."""
    rng = np.random.default_rng(311)
    a = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    b = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    injection = a @ np.conj(a.T) / (20 * N)
    blocking = b @ np.conj(b.T) / (20 * N)
    gamma = injection + blocking
    u, _ = np.linalg.qr(rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    rho = u @ np.diag(np.linspace(0.05, 0.95, N)) @ np.conj(u.T)
    d = C.dissipator(gamma, injection)
    for tau in (0.0, 0.2, 2.0, 20.0):
        evolved = _herm(np.asarray(C.flow(d, rho, tau)))
        eig = np.linalg.eigvalsh(evolved)
        assert eig.min() >= -2e-13
        assert eig.max() <= 1.0 + 2e-13


def test_degenerate_unitary_covariance(prob):
    rng = np.random.default_rng(107)
    u, _ = np.linalg.qr(rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    uh = np.conj(u.T)
    pair_p, pair_h = prob["pair_p"], prob["pair_h"]
    ops = tuple(map(np.asarray, C.branch_operators(pair_p, pair_h)))
    rotated = tuple(map(np.asarray, C.branch_operators(uh @ pair_p @ u, uh @ pair_h @ u)))
    for a, b in zip(ops, rotated):
        assert _rel(uh @ a @ u, b) <= 1e-13
    rho = prob["rho"]
    assert _rel(uh @ np.asarray(C.collision(ops[0], ops[1], rho)) @ u,
                np.asarray(C.collision(rotated[0], rotated[1], uh @ rho @ u))) <= 1e-13


def test_collision_operator(prob):
    g_out, g_in = prob["g_p"], prob["g_h"]
    c = np.asarray(C.collision(g_out + g_in, g_in, prob["rho"]))
    assert _rel(c, _c_ref(g_out, g_in, prob["rho"])) <= 1e-14
    assert float(np.max(np.abs(c - np.conj(np.swapaxes(c, -1, -2))))) <= 1e-15


def test_red_twin_unsymmetrized_anticommutator(prob):
    g_out, g_in = prob["g_p"], prob["g_h"]
    gam = g_out + g_in
    bad = -gam @ prob["rho"] + g_in                       # 2 G~ ρ / 2 in place of {G~, ρ} / 2
    assert float(np.max(np.abs(bad - np.conj(np.swapaxes(bad, -1, -2))))) > 1e-3
    assert _rel(bad, _c_ref(g_out, g_in, prob["rho"])) > 1e-2


@pytest.mark.parametrize("tau", [1e-3, 0.7, 5.0, -0.3])
def test_flow_is_the_exact_affine_solution(prob, tau):
    g_out, g_in = prob["g_p"], prob["g_h"]
    src = np.asarray(C.source(g_out + g_in, g_in, prob["rho_eq"]))          # raw: S = C[ρ_eq]
    d = C.dissipator(g_out + g_in, src)
    got = np.asarray(C.flow(d, prob["x"], tau))
    ref = _affine_ref(g_out + g_in, src, prob["x"], tau)
    assert _rel(got, ref) <= 1e-12


def test_red_twin_source_without_phi1(prob):
    g_out, g_in = prob["g_p"], prob["g_h"]
    src = np.asarray(C.source(g_out + g_in, g_in, prob["rho_eq"]))
    d = C.dissipator(g_out + g_in, src)
    tau = 5.0
    xe = np.conj(np.swapaxes(np.asarray(d.vecs), -1, -2)) @ prob["x"] @ np.asarray(d.vecs)
    g = np.asarray(d.vals)
    z = tau * 0.5 * (g[..., :, None] + g[..., None, :])
    ye = np.exp(-z) * xe + tau * np.asarray(d.source)                # phi1 -> 1
    bad = np.asarray(d.vecs) @ ye @ np.conj(np.swapaxes(np.asarray(d.vecs), -1, -2))
    assert _rel(bad, _affine_ref(g_out + g_in, src, prob["x"], tau)) > 1e-3


def test_rate_is_the_generator(prob):
    g_out, g_in = prob["g_p"], prob["g_h"]
    gam = g_out + g_in
    src = np.asarray(C.source(g_out + g_in, g_in, prob["rho_eq"]))
    d = C.dissipator(gam, src)
    r = np.asarray(C.rate(d, prob["x"]))
    assert _rel(r, -0.5 * (gam @ prob["x"] + prob["x"] @ gam) + src) <= 1e-13
    h = 1e-4
    fd = (np.asarray(C.flow(d, prob["x"], h)) - np.asarray(C.flow(d, prob["x"], -h))) / (2 * h)
    assert _rel(fd, r) <= 1e-7


def test_phi1():
    z = np.array([0.0, 1e-12, 9.99e-6, 1.001e-5, 1e-3, 1e-2, 0.5, 3.0, 40.0])
    got = np.asarray(C.phi1(z))
    series = lambda v: sum((-v) ** n / np.prod(np.arange(1, n + 2, dtype=float)) for n in range(30))
    ref = np.where(z < 1e-2, series(z), -np.expm1(-z) / np.where(z == 0, 1.0, z))
    assert got[0] == 1.0
    assert _rel(got, ref) <= 1e-15
    lo, hi = np.asarray(C.phi1(np.array([1e-5 * (1 - 1e-12), 1e-5 * (1 + 1e-12)])))
    assert abs(lo - hi) <= 1e-15


def test_equilibrium_fixed_point_and_raw_drift(prob):
    g_out, g_in = prob["g_p"], prob["g_h"]
    c_eq = np.asarray(C.collision(g_out + g_in, g_in, prob["rho_eq"]))
    assert float(np.max(np.abs(c_eq))) > 1e-3                         # a nonzero C[ρ_eq]
    src = np.asarray(C.source(g_out + g_in, g_in, prob["rho_eq"], c_eq))
    assert float(np.max(np.abs(src))) == 0.0
    d = C.dissipator(g_out + g_in, src)
    zero = np.zeros_like(prob["x"])
    assert float(np.max(np.abs(np.asarray(C.flow(d, zero, 3.0))))) == 0.0
    d_raw = C.dissipator(g_out + g_in, np.asarray(C.source(g_out + g_in, g_in, prob["rho_eq"])))
    tau = 2.0
    got = np.asarray(C.flow(d_raw, zero, tau))
    assert _rel(got, _affine_ref(g_out + g_in, c_eq, zero, tau)) <= 1e-12


def test_trace_rates(prob):
    g_out, g_in = prob["g_p"], prob["g_h"]
    rho, rho_eq = prob["rho"], prob["rho_eq"]
    out, inn = (np.asarray(v) for v in C.trace_rates(g_out + g_in, g_in, rho))
    tr_c = np.real(np.trace(_c_ref(g_out, g_in, rho), axis1=-2, axis2=-1))
    assert _rel(out + inn, tr_c) <= 1e-13
    assert np.all(out <= 1e-15)                                       # out-scattering drains
    g_out_eq, g_in_eq = 0.7 * g_out, 1.3 * g_in
    out_s, inn_s = (np.asarray(v) for v in C.trace_rates(g_out + g_in, g_in, rho, g_out_eq + g_in_eq, g_in_eq,
                                                         rho_eq))
    tr_s = tr_c - np.real(np.trace(_c_ref(g_out_eq, g_in_eq, rho_eq), axis1=-2, axis2=-1))
    assert _rel(out_s + inn_s, tr_s) <= 1e-13


def test_hermitian_update(prob):
    rng = prob["rng"]
    cplx = lambda: rng.normal(size=(NTR, NK, N, N)) + 1j * rng.normal(size=(NTR, NK, N, N))
    s_t, s_eq, st_t, st_eq = cplx(), cplx(), _herm(cplx()), _herm(cplx())
    full = np.asarray(C.hermitian_update(s_t, s_eq))
    dyn = np.asarray(C.hermitian_update(s_t, s_eq, st_t, st_eq))
    assert _rel(full, _herm(s_t - s_eq)) <= 1e-15
    assert _rel(dyn, _herm(s_t - s_eq) - (st_t - st_eq)) <= 1e-14
    for h in (full, dyn):
        assert float(np.max(np.abs(h - np.conj(np.swapaxes(h, -1, -2))))) <= 1e-15


def test_refresh_policy():
    f0 = np.zeros((1, 2, 3))
    f1 = f0.copy()
    f1[0, 1, 2] = 2e-3
    assert C.refresh_due(C.RefreshPolicy(), 10.0, 0.0, f1, f0)[0] is False
    assert C.refresh_due(C.RefreshPolicy(df=1e-3), 1.0, 0.0, f1, f0)[:3:2] == (True, "df")
    assert C.refresh_due(C.RefreshPolicy(df=5e-3), 1.0, 0.0, f1, f0)[0] is False
    assert C.refresh_due(C.RefreshPolicy(every=2.0), 1.0, 0.0, f1, f0)[0] is False
    assert C.refresh_due(C.RefreshPolicy(every=2.0), 2.0, 0.0, f1, f0)[:3:2] == (True, "cadence")
    due, df, _ = C.refresh_due(C.RefreshPolicy(every=2.0), 2.0, 0.0, f1, f0)
    assert df == pytest.approx(2e-3)


def test_flow_semigroup(prob):
    g_out, g_in = prob["g_p"], prob["g_h"]
    src = np.asarray(C.source(g_out + g_in, g_in, prob["rho_eq"]))
    d = C.dissipator(g_out + g_in, src)
    a = np.asarray(C.flow(d, np.asarray(C.flow(d, prob["x"], 0.4)), 1.1))
    b = np.asarray(C.flow(d, prob["x"], 1.5))
    assert _rel(a, b) <= 1e-13
