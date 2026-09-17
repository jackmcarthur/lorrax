"""Independent analytic checks for the finite-temperature Matsubara rule door."""
import numpy as np
import pytest

import minimax


def _pair_transform(rule, k, e_m, e_n, beta):
    """sum_l W h(t_l) + conj(W) h(beta - t_l) for h = f_m (1-f_n) exp(-(e_n - e_m) tau), mu = 0.

    Each one-particle factor is evaluated in log form (as a consumer must): the
    naive product f_m (1-f_n) exp(-x (beta - t)) of a large negative-x pair
    multiplies an underflowed prefactor by an overflowing exponential.
    """
    def lower(e, tau):
        return np.exp(e * tau - np.logaddexp(0.0, beta * e))

    def upper(e, tau):
        return np.exp(-e * tau - np.logaddexp(0.0, -beta * e))

    f_m = np.exp(-np.logaddexp(0.0, beta * e_m))
    f_n = np.exp(-np.logaddexp(0.0, beta * e_n))
    t, W = rule["t"], rule["weights"][k]
    got = (W @ (lower(e_m, t) * upper(e_n, t))
           + np.conj(W) @ (lower(e_m, beta - t) * upper(e_n, beta - t)))
    return got, f_m - f_n, e_n - e_m


@pytest.mark.parametrize("beta,delta,indices,tol", [
    (100.0, 11.0, (0, 1, 2), 1e-6),
    (50.0, 3.0, (0, 1, 3), 1e-8),
    (4.0, 2.5, (0, 2), 1e-9),
])
def test_pairs_of_either_sign_reach_the_lindhard_weight(beta, delta, indices, tol):
    rule = minimax.matsubara_response_rule(beta, delta, indices, rel_tol=tol)
    cert = rule["certificate"]
    assert cert["status"] == "PASS" and np.all(rule["t"] > 0) and np.all(rule["t"] <= beta / 2)
    assert max(cert["even_error"] + cert["odd_error"]) <= tol
    rng = np.random.default_rng(20260917)
    energies = np.sort(rng.uniform(-delta / 2, delta / 2, size=40))
    energies[[3, 4]] = energies[3]                 # a degenerate pair: -df/de at nu_0
    for k, nu in zip(range(len(indices)), rule["nu_ry"]):
        peak = beta if nu == 0.0 else 1.0 / nu
        worst = 0.0
        for e_m in energies:
            for e_n in energies:
                got, df, x = _pair_transform(rule, k, e_m, e_n, beta)
                if nu == 0.0 and abs(x) < 1e-14:
                    f = np.exp(-np.logaddexp(0.0, beta * e_m))
                    exact = beta * f * (1.0 - f)
                else:
                    exact = df / (x - 1j * nu)
                worst = max(worst, abs(got - exact) / peak)
        # Per pair the error is f_m(1-f_n) <= 1 times the certified kernel error; 2x for both mirrored channels.
        assert worst <= 2.0 * tol


def test_a_rule_does_not_serve_a_frequency_it_was_not_built_for():
    beta, delta = 50.0, 3.0
    rule = minimax.matsubara_response_rule(beta, delta, (1,), rel_tol=1e-8)
    wrong = dict(rule, weights=rule["weights"])
    got, df, x = _pair_transform(wrong, 0, -0.3, 0.4, beta)
    nu_2 = 4.0 * np.pi / beta
    assert abs(got - df / (x - 1j * nu_2)) * nu_2 > 1e-3


@pytest.mark.parametrize("kwargs,match", [
    (dict(beta_ry_inv=0.0, delta_max_ry=1.0, n_indices=(0,)), "beta"),
    (dict(beta_ry_inv=10.0, delta_max_ry=-1.0, n_indices=(0,)), "delta_max"),
    (dict(beta_ry_inv=10.0, delta_max_ry=1.0, n_indices=(1, 1)), "distinct"),
    (dict(beta_ry_inv=10.0, delta_max_ry=1.0, n_indices=(-1,)), "distinct"),
])
def test_invalid_requests_refuse(kwargs, match):
    with pytest.raises(ValueError, match=match):
        minimax.matsubara_response_rule(**kwargs)
