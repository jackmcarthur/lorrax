"""PBE adapter contract against compiled libxc energy and finite differences."""
import importlib.util

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from psp.xc import pbe_functional, XCLevel

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax_xc") is None,
    reason="optional jax-xc backend absent; install using config/xc/README.md",
)


def test_pbe_energy_and_density_derivatives():
    from jax_xc.libxc import LibXCFunctional

    eps, level = pbe_functional()
    assert level is XCLevel.GGA
    rho = np.array([1e-3, .01, .1, 1.])
    sigma = np.array([1e-7, 1e-4, .01, .1])
    compiled = [LibXCFunctional(name, 1) for name in ('gga_x_pbe', 'gga_c_pbe')]

    def reference(r, s):
        return 2 * sum(f.compute({'rho': r, 'sigma': s}, do_vxc=False)['zk'].ravel()
                       for f in compiled)

    np.testing.assert_allclose(jax.jit(eps)(rho, sigma), reference(rho, sigma),
                               rtol=1e-12, atol=1e-14)
    def energy(r, s):
        return jnp.sum(r * eps(r, s))

    dr, ds = jax.jit(jax.grad(energy, (0, 1)))(rho, sigma)
    hr, hs = 1e-5 * rho, 1e-5 * sigma
    fd_r = ((rho + hr) * reference(rho + hr, sigma)
            - (rho - hr) * reference(rho - hr, sigma)) / (2 * hr)
    fd_s = rho * (reference(rho, sigma + hs) - reference(rho, sigma - hs)) / (2 * hs)
    np.testing.assert_allclose(dr, fd_r, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(ds, fd_s, rtol=3e-5, atol=2e-7)


def test_pbe_zero_gradient_is_finite_and_preserves_shape():
    eps, _ = pbe_functional()
    rho = jnp.full((2, 3, 4), .1)
    assert eps(rho, 0.).shape == rho.shape
    energy = lambda r, s: jnp.sum(r * eps(r, s))
    derivatives = jax.jit(jax.grad(energy, (0, 1)))(rho, jnp.zeros_like(rho))
    assert all(np.isfinite(d).all() for d in derivatives)
