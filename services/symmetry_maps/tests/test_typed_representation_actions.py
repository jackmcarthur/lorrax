"""Discriminating tests for the typed operation/representation interface."""

from __future__ import annotations

import numpy as np
import pytest

from symmetry_maps import SymMaps, project_polar_fft_field


def _typed_sym():
    """Three spatial rows: identity, inversion and a nonsymmetric C4."""
    sym = object.__new__(SymMaps)
    inverse_c4 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    inverse = np.stack((np.eye(3, dtype=int), -np.eye(3, dtype=int), inverse_c4))
    sym.sym_matrices = inverse
    reciprocal = inverse.transpose(0, 2, 1)
    sym.sym_mats_k = np.concatenate((reciprocal, -reciprocal))
    sym.translations = 2.0 * np.pi * np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.5, 0.0]])
    sym.R_cart = np.concatenate((inverse.astype(float), -inverse.astype(float)))
    sym.U_spinor = SymMaps.get_spinor_rotations(None, sym.R_cart[:3])
    return sym


def test_operation_rows_owns_rotation_translation_and_antiunitary_typing():
    sym = _typed_sym()
    rows = np.array([2, 4], dtype=np.int32)
    reciprocal, translation, antiunitary = sym.operation_rows(rows)

    np.testing.assert_array_equal(reciprocal, sym.sym_mats_k[rows])
    np.testing.assert_array_equal(
        translation, sym.translations[[2, 1]])
    np.testing.assert_array_equal(antiunitary, [False, True])

    with pytest.raises(ValueError, match="outside"):
        sym.operation_rows([6])
    with pytest.raises(ValueError, match="integers"):
        sym.operation_rows([1.5])

    carrier = np.array([[0.0, 1.0, 0.0]])
    assert abs(np.imag(sym.reciprocal_phase(2, carrier)[0])) > 0.9
    np.testing.assert_allclose(
        sym.reciprocal_phase(5, carrier),
        np.conj(sym.reciprocal_phase(2, carrier)), atol=1e-15)


def test_fft_pullback_accepts_typed_rows_without_consumer_row_decoding():
    sym = _typed_sym()
    pullback = sym.fft_grid_pullback(
        np.asarray([2, 5], dtype=np.int32), (4, 4, 1))
    assert pullback.shape == (2, 16)
    np.testing.assert_array_equal(pullback[0], pullback[1])
    np.testing.assert_array_equal(np.sort(pullback[0]), np.arange(16))

    with pytest.raises(ValueError, match="rank one"):
        sym.fft_grid_pullback(np.asarray([[2]]), (4, 4, 1))


def test_cartesian_traits_distinguish_forward_inverse_parity_and_time_reversal():
    sym = _typed_sym()
    rows = np.arange(6, dtype=np.int32)
    polar_even = sym.cartesian_action(rows, axial=False, time_odd=False)
    polar_odd = sym.cartesian_action(rows, axial=False, time_odd=True)
    axial_even = sym.cartesian_action(rows, axial=True, time_odd=False)
    axial_odd = sym.cartesian_action(rows, axial=True, time_odd=True)

    forward_c4 = sym.R_cart[2].T
    assert not np.array_equal(forward_c4, sym.R_cart[2])
    np.testing.assert_array_equal(polar_even[2], forward_c4)
    np.testing.assert_array_equal(polar_even[5], forward_c4)
    np.testing.assert_array_equal(polar_odd[5], -forward_c4)

    # Inversion reverses a polar vector but preserves an axial vector.
    np.testing.assert_array_equal(polar_even[1], -np.eye(3))
    np.testing.assert_array_equal(axial_even[1], np.eye(3))
    # An antiunitary inversion reverses a time-odd axial vector exactly once.
    np.testing.assert_array_equal(axial_even[4], np.eye(3))
    np.testing.assert_array_equal(axial_odd[4], -np.eye(3))


def test_spinor_action_uses_the_same_row_typing_and_pauli_representation():
    sym = _typed_sym()
    sigma = np.array((
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ), dtype=np.complex128)

    # The unitary C4 row obeys the axial-vector Pauli sandwich.
    U = sym.spinor_action(2, nspinor=2)
    axial_inverse = sym.cartesian_action(2, axial=True, time_odd=False).T
    lhs = np.einsum("ab,ibc,cd->iad", U.conj().T, sigma, U)
    rhs = np.einsum("ji,jab->iab", axial_inverse, sigma)
    np.testing.assert_allclose(lhs, rhs, atol=1e-14)

    # Its antiunitary partner uses i sigma_y K, not another spatial row.
    theta_u = sym.spinor_action(5, nspinor=2)
    i_sigma_y = np.array([[0, 1], [-1, 0]], dtype=np.complex128)
    np.testing.assert_allclose(theta_u, i_sigma_y @ np.conj(U), atol=1e-14)


def test_complex_polar_field_applies_antiunitary_conjugation_exactly_once():
    sym = object.__new__(SymMaps)
    sym.sym_matrices = np.eye(3, dtype=np.int32)[None]
    sym.sym_mats_k = np.stack((np.eye(3), -np.eye(3))).astype(np.int32)
    sym.translations = np.zeros((1, 3), dtype=np.float64)
    sym.R_cart = np.stack((np.eye(3), -np.eye(3)))
    sym.active_symmetry_rows = np.asarray([0, 1], dtype=np.int32)
    sym.trs_allowed = True

    field = np.asarray([
        [[[1.0 + 2.0j]], [[3.0 - 4.0j]]],
        [[[5.0 + 6.0j]], [[7.0 - 8.0j]]],
        [[[9.0 + 1.0j]], [[2.0 - 3.0j]]],
    ])
    projected = project_polar_fft_field(field, sym).field
    np.testing.assert_allclose(projected, 1j * np.imag(field), atol=1e-15)
    assert np.linalg.norm(projected) > 1.0
