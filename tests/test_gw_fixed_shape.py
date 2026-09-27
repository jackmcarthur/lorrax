"""Regression tests for stable GW selector and postprocessing executables."""
import jax
import jax.numpy as jnp
import numpy as np

from gw.ppm_tau_kernel import build_shared_w_tau
from gw.ppm_pipeline import _band_count_kernel, _extrapolation_kernel


def test_active_pole_prefix_skips_poisoned_inactive_rows_with_one_executable():
    residue = jnp.asarray([1 + 2j, 3 - 1j, -2 + 0.5j, complex('nan')]).reshape(4, 1, 1, 1)
    omega = jnp.asarray([0.3 - 0.1j, 0.7 - 0.2j, 1.2 - 0.3j, complex('nan')]).reshape(4, 1, 1, 1)
    bounds = jnp.asarray([[0, np.inf, -np.inf, -np.inf, np.inf, np.inf]] * 4)
    args = (residue, omega, jnp.arange(4, dtype=jnp.int32), bounds,
            jnp.zeros(4, dtype=bool), jnp.asarray(0.05), jnp.asarray(0.2 - 0.1j))
    compiled = jax.jit(build_shared_w_tau).lower(*args, jnp.int32(1)).compile()
    for count in (0, 1, 3):
        got = np.asarray(compiled(*args, jnp.int32(count)))
        expected = np.sum(np.asarray(residue)[:count] * np.exp(
            -1j * (np.asarray(omega)[:count] - 0.05) * (0.2 - 0.1j)), axis=0)
        np.testing.assert_allclose(got, expected, rtol=3e-15, atol=3e-15)


def test_postprocessing_executables_accept_changed_indices_and_weights():
    rng = np.random.default_rng(23)
    cube = jnp.asarray(rng.normal(size=(3, 2, 4, 4)) + 1j * rng.normal(size=(3, 2, 4, 4)))
    point = _band_count_kernel(None).lower(cube, jnp.int32(0)).compile()
    for index in (0, 2):
        np.testing.assert_array_equal(point(cube, jnp.int32(index)), np.asarray(cube)[index])
    # Per-state weights (3, nk, nb) on a (3, nw, nk, nb, nb) cube, applied as
    # the symmetric mean 1/2(w_i + w_j).
    cube5 = jnp.asarray(rng.normal(size=(3, 2, 3, 4, 4)) + 1j * rng.normal(size=(3, 2, 3, 4, 4)))
    weights = jnp.asarray(rng.normal(size=(3, 3, 4)))
    combine = _extrapolation_kernel(None).lower(cube5, weights).compile()
    for w in (weights, weights[::-1]):
        wn = np.asarray(w)
        wsym = 0.5 * (wn[:, :, :, None] + wn[:, :, None, :])
        np.testing.assert_allclose(combine(cube5, w), np.sum(wsym[:, None] * np.asarray(cube5), axis=0), rtol=3e-15, atol=3e-15)
    assert _band_count_kernel(None) is _band_count_kernel(None)
    assert _extrapolation_kernel(None) is _extrapolation_kernel(None)
