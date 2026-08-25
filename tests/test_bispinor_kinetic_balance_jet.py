"""Exact algebra gates for the fixed-Psi_L kinetic-balance endpoint jet."""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import jax.numpy as jnp

import common.bispinor_init as bispinor_init
from common.bispinor_init import (
    ALPHA_FS,
    HALFALPHA,
    kinetic_balance_lift_jet,
    lift_to_4spinor,
    sigma_dot_cartesian,
)
from common.gamma_matrices import gamma_apply, gamma_perm_phase


_PAULI = np.asarray([
    [[0.0, 1.0], [1.0, 0.0]],
    [[0.0, -1.0j], [1.0j, 0.0]],
    [[1.0, 0.0], [0.0, -1.0]],
], dtype=np.complex128)


def _large_spinors():
    return np.asarray([
        [[
            [0.70 + 0.10j, -0.20 + 0.30j, 0.40 - 0.10j],
            [0.10 - 0.20j, 0.50 + 0.00j, -0.30 + 0.20j],
        ], [
            [-0.40 + 0.10j, 0.20 + 0.20j, 0.10 - 0.30j],
            [0.30 - 0.20j, 0.40 + 0.10j, -0.20 + 0.00j],
        ]],
        [[
            [0.15 - 0.05j, 0.25 + 0.10j, -0.35 + 0.20j],
            [0.45 + 0.05j, -0.15 + 0.30j, 0.05 - 0.25j],
        ], [
            [0.20 + 0.40j, 0.30 - 0.10j, -0.50 + 0.00j],
            [0.60 + 0.00j, -0.10 + 0.20j, 0.20 + 0.30j],
        ]],
    ], dtype=np.complex128)


def test_sigma_dot_matches_independent_cartesian_pauli_matrices():
    psi = _large_spinors()
    vector = np.asarray([
        [[0.2, -0.3, 0.5], [0.7, 0.1, -0.4], [-0.2, 0.8, 0.6]],
        [[-0.1, 0.4, 0.3], [0.5, -0.6, 0.2], [0.9, 0.2, -0.7]],
    ])
    expected = np.einsum(
        "aij,kga,kbjg->kbig", _PAULI, vector, psi, optimize=True)
    got = sigma_dot_cartesian(jnp.asarray(psi), jnp.asarray(vector))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=2.0e-15)


@pytest.mark.parametrize("at_origin", [False, True])
def test_lift_jet_is_the_cartesian_finite_difference_of_existing_lift(
        at_origin):
    psi = _large_spinors()[:1]
    B = np.asarray([
        [1.10, 0.18, 0.02],
        [0.00, 0.93, 0.16],
        [0.07, 0.00, 1.04],
    ])
    if at_origin:
        G = np.zeros((1, 3, 3), dtype=np.float64)
        k = np.zeros((1, 3), dtype=np.float64)
    else:
        G = np.asarray([[[0, 0, 0], [1, 0, 0], [0, -1, 1]]],
                       dtype=np.float64)
        k = np.asarray([[0.19, -0.17, 0.12]])
    K = (G + k[:, None, :]) @ B
    lifted, derivative = kinetic_balance_lift_jet(
        jnp.asarray(psi), jnp.asarray(K))
    existing = lift_to_4spinor(
        jnp.asarray(psi), jnp.asarray(G), jnp.asarray(k), jnp.asarray(B))
    np.testing.assert_allclose(lifted, existing, rtol=0.0, atol=2.0e-15)
    assert derivative.shape == (3, 1, 2, 4, 3)

    h = 2.0e-6
    B_inv = np.linalg.inv(B)
    for a in range(3):
        step_cart = np.zeros(3)
        step_cart[a] = h
        step_crys = step_cart @ B_inv
        plus = lift_to_4spinor(
            jnp.asarray(psi), jnp.asarray(G), jnp.asarray(k + step_crys),
            jnp.asarray(B))
        minus = lift_to_4spinor(
            jnp.asarray(psi), jnp.asarray(G), jnp.asarray(k - step_crys),
            jnp.asarray(B))
        finite_difference = (plus - minus) / (2.0 * h)
        np.testing.assert_allclose(
            derivative[a], finite_difference, rtol=3.0e-10, atol=3.0e-12)

    # The first derivative is independent of K: this is the exact zero
    # second-derivative contract, without allocating a 3x3 wavefunction jet.
    _, shifted_derivative = kinetic_balance_lift_jet(
        jnp.asarray(psi), jnp.asarray(K + 0.37))
    np.testing.assert_array_equal(derivative, shifted_derivative)


def _alpha_apply(psi_4, direction):
    perm, phase = gamma_perm_phase(direction + 1)
    return gamma_apply(psi_4, perm, phase, axis=-2)


def test_raw_alpha_jet_matches_rydberg_kinetic_velocity_and_contact():
    bra_L = jnp.asarray(_large_spinors()[0, :1])
    ket_L = jnp.asarray(_large_spinors()[1])
    K = jnp.asarray([
        [0.2, -0.3, 0.5], [0.7, 0.1, -0.4], [-0.2, 0.8, 0.6],
    ])
    bra, d_bra = kinetic_balance_lift_jet(bra_L, K)
    ket, d_ket = kinetic_balance_lift_jet(ket_L, K)
    overlap = jnp.einsum(
        "msG,nsG->mn", jnp.conj(bra_L), ket_L, optimize=True)

    raw_current = []
    raw_current_derivative = np.empty((3, 3, 1, 2), np.complex128)
    for i in range(3):
        alpha_ket = _alpha_apply(ket, i)
        raw_current.append(jnp.einsum(
            "msG,nsG->mn", jnp.conj(bra), alpha_ket, optimize=True))
        for a in range(3):
            raw_current_derivative[a, i] = np.asarray(
                jnp.einsum(
                    "msG,nsG->mn", jnp.conj(d_bra[a]), alpha_ket,
                    optimize=True)
                + jnp.einsum(
                    "msG,nsG->mn", jnp.conj(bra),
                    _alpha_apply(d_ket[a], i), optimize=True))
    raw_current = jnp.stack(raw_current, axis=0)

    expected_velocity = 2.0 * jnp.einsum(
        "msG,Gi,nsG->imn", jnp.conj(bra_L), K, ket_L, optimize=True)
    np.testing.assert_allclose(
        (2.0 / ALPHA_FS) * raw_current, expected_velocity,
        rtol=2.0e-13, atol=2.0e-13)

    expected_contact = (
        2.0 * np.eye(3)[:, :, None, None] * np.asarray(overlap)[None, None])
    np.testing.assert_allclose(
        (2.0 / ALPHA_FS) * raw_current_derivative, expected_contact,
        rtol=2.0e-13, atol=2.0e-13)
    assert ALPHA_FS == 2.0 * HALFALPHA


def test_large_component_boundary_and_sigma_dot_reuse_are_explicit():
    psi = jnp.ones((1, 1, 2, 3), dtype=jnp.complex128)
    vector = jnp.ones((1, 3, 3), dtype=jnp.float64)
    with pytest.raises(ValueError, match="explicit Psi_L"):
        sigma_dot_cartesian(jnp.ones((1, 1, 4, 3)), vector)
    with pytest.raises(ValueError, match="paired with Psi_L"):
        sigma_dot_cartesian(psi, jnp.ones((3, 3)))
    with pytest.raises(ValueError, match="explicit Psi_L"):
        lift_to_4spinor(
            jnp.ones((1, 1, 4, 3)), jnp.zeros((1, 3, 3)),
            jnp.zeros((1, 3)), jnp.eye(3))

    tree = ast.parse(inspect.getsource(bispinor_init))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    lift_calls = {
        node.func.id for node in ast.walk(functions["lift_to_4spinor"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "sigma_dot_cartesian" in lift_calls
    sigma_names = {
        node.id for node in ast.walk(functions["sigma_dot_cartesian"])
        if isinstance(node, ast.Name)
    }
    assert "paulis" in sigma_names
    assert not functions["sigma_dot_cartesian"].decorator_list
    assert not functions["kinetic_balance_lift_jet"].decorator_list
