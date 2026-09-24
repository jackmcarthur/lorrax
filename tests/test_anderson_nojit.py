"""``anderson_nojit`` -- the SC loop's one-evaluation accelerator.

One contract, on a small stacked Hermitian carry (no devices):
one map evaluation per iteration, and convergence on a map with an
expansive (|J| > 1) and an overshooting (J = -3) direction, where the plain
fixed point diverges.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from mixing.acceleration import anderson_nojit


def _hermitian(a):
    return 0.5 * (a + np.conj(np.swapaxes(a, -1, -2)))


def _problem(seed=0, nk=3, nb=6):
    rng = np.random.default_rng(seed)
    h_star = _hermitian(rng.standard_normal((nk, nb, nb))
                        + 1j * rng.standard_normal((nk, nb, nb)))
    gain = np.zeros((nb, nb))
    gain[0, 0], gain[1, 1] = -3.0, 2.5          # overshooting, expansive
    calls = []

    def residual(x):
        calls.append(1)
        h = _hermitian(np.asarray(x))
        d = h - h_star
        return jnp.asarray(0.2 * d + gain[None] * d + 0.05 * d @ d)

    x0 = jnp.asarray(h_star + 0.01 * _hermitian(rng.standard_normal((nk, nb, nb))))
    return residual, x0, calls


def test_one_evaluation_per_iteration_and_convergence_where_picard_fails():
    residual, x0, calls = _problem()
    result = anderson_nojit(residual, x0, m=20, maxit=40, tol=1e-9)
    assert result.converged
    assert len(calls) == result.iterations + 1      # map 0 + one per iteration
    assert result.iterations <= 14
    # The plain fixed point diverges on the same map (|1 + J| > 1).
    residual_p, x, _ = _problem()
    for _ in range(6):
        x = x + residual_p(x)
    assert float(jnp.max(jnp.abs(residual_p(x)))) > float(
        jnp.max(jnp.abs(residual_p(x0))))
