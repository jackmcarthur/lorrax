"""The rCROP Gram metric: the least squares sees only the trusted block."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from mixing.acceleration import rcrop_nojit


def _small_problem(n_hist=3, maxit=4):
    """A contracting linear map on a 2x2 block; a noisy 'scissored' block."""
    rng = np.random.default_rng(7)
    A = 0.4 * rng.standard_normal((2, 2))
    target = rng.standard_normal((2, 2))
    calls = [0]

    def f_small(x):                      # (1, 2, 2) -> residual
        return jnp.asarray(A @ (x[0] - target) + (target - x[0]))[None]

    def f_full(x):                       # (1, 4, 4): block [0:2,0:2] is the
        calls[0] += 1                    # small problem; [2:4,2:4] is noise
        out = jnp.zeros_like(x)
        out = out.at[:, :2, :2].set(f_small(x[:, :2, :2]))
        noise = 50.0 * np.sin(np.arange(4).reshape(2, 2) + calls[0])
        return out.at[:, 2:, 2:].set(jnp.asarray(noise)[None])
    return f_small, f_full, maxit, n_hist


def test_metric_makes_the_full_problem_follow_the_masked_subproblem():
    f_small, f_full, maxit, m = _small_problem()
    x_small = jnp.zeros((1, 2, 2), dtype=jnp.complex128)
    x_full = jnp.zeros((1, 4, 4), dtype=jnp.complex128)
    metric = np.zeros((1, 4, 4)); metric[0, :2, :2] = 1.0
    small = rcrop_nojit(f_small, x_small, m=m, maxit=maxit, tol=0.0)
    full = rcrop_nojit(f_full, x_full, m=m, maxit=maxit, tol=0.0,
                       metric=jnp.asarray(metric))
    np.testing.assert_allclose(
        np.asarray(full.x[:, :2, :2]), np.asarray(small.x), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(full.residual_norms), np.asarray(small.residual_norms),
        rtol=1e-12, atol=0)


def test_unit_metric_is_the_unweighted_solve_bit_for_bit():
    f_small, _, maxit, m = _small_problem()
    x0 = jnp.zeros((1, 2, 2), dtype=jnp.complex128)
    plain = rcrop_nojit(f_small, x0, m=m, maxit=maxit, tol=0.0)
    unit = rcrop_nojit(f_small, x0, m=m, maxit=maxit, tol=0.0,
                       metric=jnp.ones((1, 2, 2)))
    assert np.array_equal(np.asarray(plain.x), np.asarray(unit.x))
    assert np.array_equal(np.asarray(plain.residual_norms),
                          np.asarray(unit.residual_norms))


def test_without_metric_the_noise_block_steers_the_coefficients():
    f_small, f_full, maxit, m = _small_problem()
    x_small = jnp.zeros((1, 2, 2), dtype=jnp.complex128)
    x_full = jnp.zeros((1, 4, 4), dtype=jnp.complex128)
    small = rcrop_nojit(f_small, x_small, m=m, maxit=maxit, tol=0.0)
    full = rcrop_nojit(f_full, x_full, m=m, maxit=maxit, tol=0.0)
    assert not np.allclose(np.asarray(full.x[:, :2, :2]),
                           np.asarray(small.x), atol=1e-6)
