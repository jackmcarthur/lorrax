"""Independent prefix-product oracle, including clamped final blocks."""
import numpy as np


def test_complex_factor_prefix_matches_numpy_with_ragged_tail_and_holes():
    import jax
    import jax.numpy as jnp
    from common.pivoted_cholesky import _prefix_factor_projection

    kernel = jax.jit(_prefix_factor_projection)
    rng = np.random.default_rng(46171)
    worst_scaled_error = 0.
    for budget, columns in (
        (0, (0,)),
        (5, (0, 1, 5)),
        (1153, (0, 1, 1023, 1024, 1025, 1152, 1153)),
        (3500, (0, 1023, 1024, 1025, 2048, 2049, 3072, 3073, 3499, 3500)),
    ):
        factor = rng.normal(size=(19, budget)) + 1j*rng.normal(size=(19, budget))
        pivot = rng.normal(size=budget) + 1j*rng.normal(size=budget)
        factor[:, 7::11] = 0  # Dependent columns can occur before later pivots.
        factor_j, pivot_j = jnp.asarray(factor), jnp.asarray(pivot)
        for j in columns:
            expected = factor[:, :j] @ np.conj(pivot[:j])
            actual = np.asarray(kernel(factor_j, pivot_j, jnp.int32(j)))
            scale = max(1., float(np.max(np.abs(expected), initial=0)))
            worst_scaled_error = max(
                worst_scaled_error, float(np.max(np.abs(actual-expected), initial=0))/scale)
            np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-14*scale)
        np.testing.assert_array_equal(np.asarray(factor_j), factor)
        np.testing.assert_array_equal(np.asarray(pivot_j), pivot)
    print(f'prefix projection normalized maximum error: {worst_scaled_error:.3e}')
