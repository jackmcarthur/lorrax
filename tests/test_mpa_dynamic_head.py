"""Two independent small oracles for the scalar dynamic MPA head."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from gw.head_correction import fold_cartesian_head_wings_sharded


def _complex(rng, shape, scale=1.0):
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def test_cartesian_ywz_fold_matches_dense_bordered_dyson():
    """The reduced 3x3 head has the dense Dyson sign and one 1/V factor."""
    rng = np.random.default_rng(418)
    n_z, n_mu = 2, 5
    cell_volume = 97.0
    lam = 0.05
    qhat = np.asarray([0.31, -0.48, 0.82])
    qhat /= np.linalg.norm(qhat)
    vbare = 8.0 * np.pi / lam**2

    V_body = np.empty((n_z, n_mu, n_mu), dtype=np.complex128)
    chi_body = np.empty_like(V_body)
    W_body = np.empty_like(V_body)
    for iz in range(n_z):
        a = _complex(rng, (n_mu, n_mu), 0.15)
        V_body[iz] = a @ a.conj().T + 0.4 * np.eye(n_mu)
        b = _complex(rng, (n_mu, n_mu), 0.08)
        chi_body[iz] = -(b @ b.conj().T)
        W_body[iz] = np.linalg.solve(
            np.eye(n_mu) - V_body[iz] @ chi_body[iz], V_body[iz])

    S_direct = _complex(rng, (n_z, 3, 3), 2.0e-3)
    Y = _complex(rng, (n_z, 3, n_mu), 8.0e-3)
    Z = _complex(rng, (n_z, n_mu, 3), 8.0e-3)

    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, axis_names=("x", "y"))
    got_S = np.asarray(fold_cartesian_head_wings_sharded(
        jnp.asarray(S_direct), jnp.asarray(Y), jnp.asarray(W_body),
        jnp.asarray(Z), cell_volume, mesh_xy=mesh))

    for iz in range(n_z):
        chi_aug = np.zeros((n_mu + 1, n_mu + 1), dtype=np.complex128)
        # compute_S_omega's direct tensor already owns 1/V_cell; the
        # transition-space bordered matrix uses its unnormalised numerator.
        chi_aug[0, 0] = (
            lam**2 * cell_volume * (qhat @ S_direct[iz] @ qhat))
        chi_aug[0, 1:] = lam * (qhat @ Y[iz])
        chi_aug[1:, 0] = lam * (Z[iz] @ qhat)
        chi_aug[1:, 1:] = chi_body[iz]

        V_aug = np.zeros_like(chi_aug)
        V_aug[0, 0] = vbare / cell_volume
        V_aug[1:, 1:] = V_body[iz]
        W_aug = np.linalg.solve(
            np.eye(n_mu + 1) - V_aug @ chi_aug, V_aug)

        dense_head = cell_volume * W_aug[0, 0]
        reduced_head = vbare / (
            1.0 - vbare * lam**2 * (qhat @ got_S[iz] @ qhat))
        np.testing.assert_allclose(reduced_head, dense_head, rtol=2e-13,
                                   atol=2e-11)
