"""The probe Hermiticity residual of W(i omega_p) (ppm_sigma._probe_antihermitian_residual)."""
import numpy as np
import jax.numpy as jnp

from gw.ppm_sigma import _probe_antihermitian_residual


def test_probe_residual_is_zero_for_hermitian_and_sized_for_antihermitian():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3, 40, 40)) + 1j * rng.normal(size=(3, 40, 40))
    herm = a + np.conj(np.swapaxes(a, 1, 2))
    assert float(_probe_antihermitian_residual(jnp.asarray(herm))) < 1e-13
    anti = a - np.conj(np.swapaxes(a, 1, 2))
    r = float(_probe_antihermitian_residual(jnp.asarray(herm + 1e-3 * anti)))
    assert 1e-4 < r < 1e-2
