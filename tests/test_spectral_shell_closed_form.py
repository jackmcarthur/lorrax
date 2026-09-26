"""spectral_shell at production size: closed-form Weyl tail, β to a tolerance.

The 2026-09-23 rewrite of ``BandLadder.log_moment`` (Euler–Maclaurin for the
k-independent Weyl segment) and ``solve_shell_exponents`` (a bracketed find
stopped at a tolerance, not 80 fixed halvings) must not move a number:

  * the closed form against the explicit sum it replaced;
  * the whole ``fit_band_extrapolation_spectral`` against the pre-rewrite
    algorithm, written out below and swapped in;
  * the CrI3 16x16 shape, whose tail shell took 119 GB per rank
    (58783428.7): one timing / memory line.
"""
from __future__ import annotations

import resource
import time
import tracemalloc

import numpy as np

import gw.band_extrapolation as bx


def _ladder(nk, n_dft, n_target, seed=0):
    rng = np.random.default_rng(seed)
    n = np.arange(1, n_dft + 1)
    e = -0.45 + 0.02 * (n + 3.0) ** (2 / 3)
    enk_ry = np.sort(e + 0.01 * rng.standard_normal((nk, n_dft)), axis=1)
    return bx.build_band_ladder(enk_ry=enk_ry, n_target=n_target,
                                kweights=rng.uniform(1.0, 6.0, nk))


def _explicit_log_moment(lad, lo, hi, beta):
    """The pre-rewrite ``log_moment``: every band one term, Weyl bands too."""
    lo_d, hi_d = min(lo, lad.n_dft), min(hi, lad.n_dft)
    e_w = lad.e_weyl_ev[max(lo, lad.n_dft) - lad.n_dft:
                        max(hi, lad.n_dft) - lad.n_dft]
    x = np.concatenate([lad.e_dft_ev[lo_d:hi_d].reshape(-1), e_w])
    x = (x - lad.e0_ev) / lad.estar_ev
    lw = np.zeros(x.size)
    lw[:(hi_d - lo_d) * lad.w_k.size] = np.tile(np.log(lad.w_k), hi_d - lo_d)
    b = np.asarray(beta, dtype=np.float64)
    lg = -b.reshape(-1, 1) * np.log(x) + lw
    m = lg.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(lg - m).sum(axis=1, keepdims=True))
            ).reshape(b.shape)


def _bisection_solve(ladder, shell2, shell3, ratio):
    """The pre-rewrite ``solve_shell_exponents``: 80 fixed halvings."""
    flat = np.asarray(ratio, dtype=np.float64).ravel()
    beta = np.full(flat.shape, np.nan)
    code = np.full(flat.shape, bx.SHELL_FAIL_ZERO)
    live = np.isfinite(flat) & (flat > 0.0)
    logr = np.log(flat[live])
    f = lambda b: (ladder.log_moment(*shell3, b)            # noqa: E731
                   - ladder.log_moment(*shell2, b) - logr)
    lo_b, hi_b = bx.SHELL_EXPONENT_BRACKET
    lo, hi = np.full(logr.shape, lo_b), np.full(logr.shape, hi_b)
    flo, fhi = f(lo), f(hi)
    ok = np.isfinite(flo) & np.isfinite(fhi) & (flo * fhi <= 0.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        ok &= np.isfinite(fm)
        take_hi = flo * fm <= 0.0
        hi, lo = np.where(take_hi, mid, hi), np.where(take_hi, lo, mid)
        flo = np.where(take_hi, flo, fm)
    b = 0.5 * (lo + hi)
    edge = np.minimum(b - lo_b, hi_b - b) <= bx.SHELL_EXPONENT_EDGE_TOL
    sub = np.where(~ok, bx.SHELL_FAIL_NO_ROOT,
                   np.where(edge, bx.SHELL_FAIL_EDGE, bx.SHELL_OK))
    beta[live] = np.where(sub == bx.SHELL_OK, b, np.nan)
    code[live] = sub
    return beta.reshape(np.shape(ratio)), code.reshape(np.shape(ratio))


def _planted_sums(lad, counts, beta_true, rng):
    """S(N₁..₃) whose shell ratio is exactly I₃/I₂ at ``beta_true``."""
    a1, a2, a3 = (lad.absolute(c) for c in counts)
    i2 = np.stack([lad.moment(a1, a2, b) for b in beta_true])
    i3 = np.stack([lad.moment(a2, a3, b) for b in beta_true])
    amp = rng.uniform(0.05, 1.0, beta_true.shape) / i2.mean()
    amp *= rng.choice([-1.0, 1.0], beta_true.shape)
    s1 = 2.0 + rng.standard_normal(beta_true.shape) * (1 + 1j) * 0.1
    return np.stack([s1, s1 + amp * i2, s1 + amp * (i2 + i3)])


def test_weyl_closed_form_matches_the_explicit_sum():
    beta = np.concatenate([np.linspace(0.05, 40.0, 81), [1.5, 1.5 + 1e-9]])
    worst = 0.0
    for nk, n_dft, n_target in ((4, 120, 20000), (2, 12, 3000)):
        lad = _ladder(nk, n_dft, n_target)
        n_em = int(np.ceil(bx._WEYL_EM_FROM - lad.n0))
        assert lad.n_dft < n_em < lad.n_target   # both Weyl segments live
        for lo, hi in ((lad.n_dft, lad.n_target), (n_dft - 5, lad.n_target),
                       (lad.n_dft + 3, n_em + 7), (n_em - 3, n_em + 2),
                       (n_target // 2, n_target // 2 + 1),
                       (lad.n_target - 3, lad.n_target), (0, lad.n_dft)):
            got = lad.log_moment(lo, hi, beta)
            want = _explicit_log_moment(lad, lo, hi, beta)
            worst = max(worst, float(np.max(np.abs(np.expm1(got - want)))))
    assert worst <= 1e-12, worst
    # And the sum by itself, from u = 256 to 150k terms, up to u ~ N_T.
    s = 2.0 * beta / 3.0
    for u_a in (256.0, 1001.0, 152910.0):
        for n in (1, 2, 17, 150000):
            u = u_a + np.arange(n)
            lg = -s[:, None] * np.log(u)
            m = lg.max(axis=1)
            want = m + np.log(np.exp(lg - m[:, None]).sum(axis=1))
            got = bx._log_weyl_power_sum(s, u_a, n)
            np.testing.assert_allclose(np.exp(got - want), 1.0, rtol=1e-12,
                                       atol=0)


def test_fit_matches_the_80_bisection_explicit_sum_estimator(monkeypatch):
    rng = np.random.default_rng(3)
    lad = _ladder(nk=4, n_dft=120, n_target=20000, seed=1)
    counts = (80, 100, 120)
    # 1.5 + 1e-6, not 1.5: an exponent AT the summability boundary
    # (SHELL_SUMMABLE_BETA) would be classified by the last ulp of each
    # solver.  s = 2beta/3 = 1 itself is pinned by the log-moment test above.
    beta_true = np.concatenate([rng.uniform(0.3, 12.0, 40),
                                [1.5 + 1e-6, 39.0, 0.05 + 1e-6]]).reshape(1, -1)
    S = _planted_sums(lad, counts, beta_true, rng)
    S[1, 0, 0] = S[0, 0, 0] - (S[2, 0, 0] - S[1, 0, 0])   # D2, D3 opposite
    S[1, 0, 1] = S[0, 0, 1]                               # D2 = 0
    S[2, 0, 2] = S[1, 0, 2] + 1e6 * (S[1, 0, 2] - S[0, 0, 2])   # no root
    new = bx.fit_band_extrapolation_spectral(counts, S, lad)
    monkeypatch.setattr(bx.BandLadder, "log_moment", _explicit_log_moment)
    monkeypatch.setattr(bx, "solve_shell_exponents", _bisection_solve)
    old = bx.fit_band_extrapolation_spectral(counts, S, lad)

    np.testing.assert_array_equal(new.failure, old.failure)
    assert list(new.failure[0, :3]) == [bx.SHELL_FAIL_SIGN, bx.SHELL_FAIL_ZERO,
                                        bx.SHELL_FAIL_NO_ROOT]
    assert new.failure[0, -1] == bx.SHELL_FAIL_EDGE
    good = new.failure == bx.SHELL_OK
    ordinary = beta_true[0, 3:-1]          # states 0-2 and the last are planted
    assert good.sum() == int((ordinary > bx.SHELL_SUMMABLE_BETA).sum())
    assert np.all(new.failure[0, 3:-1][ordinary <= bx.SHELL_SUMMABLE_BETA]
                  == bx.SHELL_FAIL_NOT_SUMMABLE)
    for name in ("beta", "tail_ratio", "s_inf"):
        a, b = getattr(new, name), getattr(old, name)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_allclose(a[good], b[good], rtol=1e-12, atol=0,
                                   err_msg=name)


def test_cri3_16x16_shape_timing_and_memory():
    """WFN 30 irr k x 900 bands, N_T = 76456 x 2, Σ states 256 x 183.

    Before the rewrite this fit took 150 s and 2.6 GB peak RSS on a login
    node, and 119 GB unchunked; the bounds below fail on either.
    """
    rng = np.random.default_rng(0)
    lad = _ladder(nk=30, n_dft=900, n_target=152912)
    counts = (385, 433, 481)
    S = _planted_sums(lad, counts, rng.uniform(2.5, 6.0, (256, 183)), rng)
    tracemalloc.start()
    t0 = time.perf_counter()
    fit = bx.fit_band_extrapolation_spectral(counts, S, lad)
    wall = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"\nspectral_shell at the CrI3 16x16 shape ({fit.n_states} states, "
          f"N_T = {lad.n_target}): {wall:.2f} s, traced peak "
          f"{peak / 2**20:.0f} MiB, process VmHWM {hwm:.2f} GiB")
    assert fit.n_failed == 0
    assert wall < 30.0
    assert peak < 256 * 2**20
