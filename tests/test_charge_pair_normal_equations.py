"""Focused algebra gate for the charge ordered-pair training domain."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from isdf import complete_ordered_pair_normal_equations
from symmetry_maps import q_negation_index


jax.config.update("jax_enable_x64", True)


def _normal_equations(design, target):
    """Small direct definition used only to discriminate the source helper."""
    c = np.einsum("qpm,qpn->qmn", design.conj(), design)
    z = np.einsum("qpm,qpr->qmr", design.conj(), target)
    return c, z


def _backward_error(c, x, z):
    residual = c @ x - z
    numerator = np.linalg.norm(residual, axis=(-2, -1))
    denominator = (
        np.linalg.norm(c, axis=(-2, -1))
        * np.linalg.norm(x, axis=(-2, -1))
        + np.linalg.norm(z, axis=(-2, -1)))
    return np.divide(
        numerator, denominator,
        out=np.where(numerator == 0.0, 0.0, np.inf),
        where=denominator != 0.0)


def _relative(a, b):
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b)) / scale)


def test_ordered_pair_completion_direct_solve_and_reciprocity():
    """One discriminating case proves the four charge-domain invariants.

    The source-native CCT/ZCT probe recorded in the charge-reciprocity report
    establishes the same LR/RL identity through the production kernels.  This
    collected cell isolates its exact normal-equation completion and runs it
    with the production trailing-axis sharding whenever four devices exist.
    """
    grid = (3, 1, 1)
    neg = q_negation_index(grid)
    nq, npair, nmu, nr = 3, 7, 4, 4
    q, pair, mu = np.indices((nq, npair, nmu))
    design_lr = ((1.0 + (2*q + pair + 3*mu) / 31.0)
                 * np.exp(2j*np.pi*(q + 2*pair + q*mu + 3*mu) / 23.0))
    q, pair, r = np.indices((nq, npair, nr))
    target_lr = ((0.7 + (q + 2*pair + r) / 29.0)
                 * np.exp(2j*np.pi*(2*q + pair*r + 4*r) / 19.0))

    # Relabelling endpoints gives the ordered RL member at each -q.
    design_rl = np.conj(design_lr[neg])
    target_rl = np.conj(target_lr[neg])
    c_lr, z_lr = _normal_equations(design_lr, target_lr)
    c_rl, z_rl = _normal_equations(design_rl, target_rl)
    c_direct, z_direct = _normal_equations(
        np.concatenate((design_lr, design_rl), axis=1),
        np.concatenate((target_lr, target_rl), axis=1))

    ndev = 4 if len(jax.devices()) >= 4 else 1
    mesh = Mesh(np.asarray(jax.devices()[:ndev]).reshape(
        (2, 2) if ndev == 4 else (1, 1)), ("x", "y"))
    shard = NamedSharding(mesh, P(None, "x", "y"))
    c_done = np.asarray(complete_ordered_pair_normal_equations(
        jax.device_put(jnp.asarray(c_lr), shard), neg))
    z_done = np.asarray(complete_ordered_pair_normal_equations(
        jax.device_put(jnp.asarray(z_lr), shard), neg))

    # (1) The helper is exactly the direct concatenated LR+RL normal equation.
    np.testing.assert_allclose(c_done, c_direct, rtol=2e-15, atol=2e-13)
    np.testing.assert_allclose(z_done, z_direct, rtol=2e-15, atol=2e-13)
    np.testing.assert_allclose(c_rl, np.conj(c_lr[neg]), rtol=2e-15, atol=2e-13)
    np.testing.assert_allclose(z_rl, np.conj(z_lr[neg]), rtol=2e-15, atol=2e-13)

    x = np.linalg.solve(c_done, z_done)
    # (2) This is a backward-error measurement, not condition estimation.
    assert float(np.max(_backward_error(c_done, x, z_done))) < 2e-15
    # (3) The sole solve inherits exact q <-> -q conjugation closure.
    assert _relative(x, np.conj(x[neg])) < 2e-13

    # (4) Completion only doubles an already reciprocal domain, so its solve
    # is unchanged.  This also catches a one-sided C-only or Z-only change.
    c_closed = c_lr + np.conj(c_lr[neg])
    z_closed = z_lr + np.conj(z_lr[neg])
    x_before = np.linalg.solve(c_closed, z_closed)
    c_twice = np.asarray(complete_ordered_pair_normal_equations(
        jax.device_put(jnp.asarray(c_closed), shard), neg))
    z_twice = np.asarray(complete_ordered_pair_normal_equations(
        jax.device_put(jnp.asarray(z_closed), shard), neg))
    np.testing.assert_allclose(c_twice, 2*c_closed, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(z_twice, 2*z_closed, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.linalg.solve(c_twice, z_twice), x_before,
        rtol=2e-13, atol=2e-13)
