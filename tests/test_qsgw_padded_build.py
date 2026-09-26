"""The QSGW Hermitianization on a padded band carrier."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from gw.qsgw_utils import build_qsgw_sigma_xc
from runtime.padding import PaddedAxis, pad_square


def test_padded_sigma_build_equals_logical_build_bit_exactly():
    """A carrier changes no logical QSGW value and is stripped on output."""
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    rng = np.random.default_rng(19)
    logical = (rng.standard_normal((3, 2, 3, 3))
               + 1j * rng.standard_normal((3, 2, 3, 3)))
    sig_x = (rng.standard_normal((2, 3, 3))
             + 1j * rng.standard_normal((2, 3, 3)))
    omega = np.asarray([-1.0, 0.0, 1.0])
    energies = np.asarray([[-0.7, 0.1, 0.8], [-0.6, 0.2, 0.7]])

    want, want_diag = build_qsgw_sigma_xc(
        jnp.asarray(logical), jnp.asarray(sig_x), omega, energies, mesh)
    tag = PaddedAxis("Sigma band window", 3, 4, 1)
    got, got_diag = build_qsgw_sigma_xc(
        pad_square(jnp.asarray(logical), tag), jnp.asarray(sig_x),
        omega, energies, mesh, band_axis=tag)

    assert got.shape == want.shape == (2, 3, 3)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert got_diag == want_diag
