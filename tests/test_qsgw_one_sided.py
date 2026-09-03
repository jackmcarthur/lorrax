"""Window-edge evaluation rules for the QSGW Hermitianization."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from gw.qsgw_utils import build_qsgw_sigma_xc


def test_one_sided_cross_edge_uses_core_energy():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    # At omega=0 the cross coupling is 2; at omega=1 it is 6.  The ordinary
    # QSGW half-sum is 4, whereas the one-sided rule must use the core
    # state's omega=0 value for both Hermitian partners.
    sigma = np.zeros((2, 1, 2, 2), dtype=np.complex128)
    sigma[0, 0, 0, 1] = sigma[0, 0, 1, 0] = 2.0
    sigma[1, 0, 0, 1] = sigma[1, 0, 1, 0] = 6.0
    sigma = jnp.asarray(sigma)
    sigma_x = jnp.zeros((1, 2, 2), dtype=jnp.complex128)
    energies = np.asarray([[0.0, 1.0]])

    standard, standard_diag = build_qsgw_sigma_xc(
        sigma, sigma_x, np.asarray([0.0, 1.0]), energies, mesh)
    one_sided, edge_diag = build_qsgw_sigma_xc(
        sigma, sigma_x, np.asarray([0.0, 1.0]), energies, mesh,
        one_sided_core_mask=np.asarray([True, False]))

    np.testing.assert_allclose(np.asarray(standard)[0, 0, 1], 4.0)
    np.testing.assert_allclose(np.asarray(one_sided)[0, 0, 1], 2.0)
    np.testing.assert_allclose(np.asarray(one_sided)[0, 1, 0], 2.0)
    assert standard_diag["n_one_sided_edges"] == 0.0
    assert edge_diag["n_one_sided_edges"] == 2.0


def test_one_sided_mask_requires_core_and_buffer():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    sigma = jnp.zeros((2, 1, 2, 2), dtype=jnp.complex128)
    sigma_x = jnp.zeros((1, 2, 2), dtype=jnp.complex128)
    with np.testing.assert_raises_regex(ValueError, "nonempty core and buffer"):
        build_qsgw_sigma_xc(
            sigma, sigma_x, np.asarray([0.0, 1.0]),
            np.asarray([[0.0, 1.0]]), mesh,
            one_sided_core_mask=np.asarray([True, True]))
