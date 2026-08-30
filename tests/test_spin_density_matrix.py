"""Focused CPU checks for the two-component local spin-density API."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import jax.numpy as jnp  # noqa: E402

from ffi import _services  # noqa: E402

_services.ensure_on_path()

from psp.get_DFT_mtxels import (  # noqa: E402
    density_components_from_psi_r,
    spin_density_matrix_to_pauli_fields,
    valence_density_from_kpoint,
)


@pytest.mark.parametrize(
    ("spinor", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
        ([0.0, 1.0], [1.0, 0.0, 0.0, -1.0]),
        (np.asarray([1.0, 1.0]) / np.sqrt(2.0), [1.0, 1.0, 0.0, 0.0]),
        (np.asarray([1.0, 1.0j]) / np.sqrt(2.0), [1.0, 0.0, 1.0, 0.0]),
    ],
    ids=("up", "down", "plus-x", "plus-y"),
)
def test_pure_spinors_have_expected_pauli_fields(spinor, expected):
    psi = np.asarray(spinor, dtype=np.complex128).reshape(1, 2, 1, 1, 1)
    matrix = np.asarray(density_components_from_psi_r(
        jnp.asarray(psi), return_spin_density_matrix=True))
    fields = np.asarray(spin_density_matrix_to_pauli_fields(matrix))

    assert np.array_equal(matrix[1, 0], np.conj(matrix[0, 1]))
    assert np.isrealobj(fields)
    assert np.allclose(fields[:, 0, 0, 0], expected, atol=2.0e-15)


def test_mixed_occupations_are_hermitian_real_and_match_direct_paulis():
    rng = np.random.default_rng(20260829)
    psi = (rng.standard_normal((3, 2, 2, 3, 1))
           + 1j * rng.standard_normal((3, 2, 2, 3, 1)))
    occupations = np.asarray([0.8, 0.25, -0.04])
    matrix = np.asarray(density_components_from_psi_r(
        jnp.asarray(psi), occupations, return_spin_density_matrix=True))
    fields = np.asarray(spin_density_matrix_to_pauli_fields(matrix))
    charge = np.asarray(density_components_from_psi_r(
        jnp.asarray(psi), occupations))

    identity = np.eye(2, dtype=np.complex128)
    sigma_x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
    direct = np.stack([
        np.real(np.einsum(
            "n,naxyz,ab,nbxyz->xyz", occupations, np.conj(psi), sigma,
            psi, optimize=True))
        for sigma in (identity, sigma_x, sigma_y, sigma_z)
    ])

    # rho_10 is assigned from rho_01, not independently reduced.
    assert np.array_equal(matrix[1, 0], np.conj(matrix[0, 1]))
    assert np.array_equal(np.imag(matrix[0, 0]),
                          np.zeros_like(np.imag(matrix[0, 0])))
    assert np.array_equal(np.imag(matrix[1, 1]),
                          np.zeros_like(np.imag(matrix[1, 1])))
    assert np.isrealobj(fields)
    assert np.allclose(fields, direct, rtol=2.0e-15, atol=2.0e-15)
    assert np.allclose(fields[0], charge, rtol=2.0e-15, atol=2.0e-15)


def test_pauli_helper_is_literal_trace_for_nonhermitian_diagnostic_input():
    matrix = np.asarray(
        [[1.0 + 2.0j, 2.0 + 3.0j], [4.0 + 5.0j, 6.0 - 1.0j]],
        dtype=np.complex128,
    )
    paulis = (
        np.eye(2, dtype=np.complex128),
        np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
        np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
    )
    expected = np.asarray([np.real(np.trace(matrix @ p)) for p in paulis])
    got = np.asarray(spin_density_matrix_to_pauli_fields(matrix))
    assert np.array_equal(got, expected)


def test_kpoint_matrix_trace_matches_charge_and_preserves_all_weights():
    rng = np.random.default_rng(17)
    box = (rng.standard_normal((3, 2, 3, 2, 2))
           + 1j * rng.standard_normal((3, 2, 3, 2, 2)))
    occupations = np.asarray([0.9, 0.35, -0.02])
    kwargs = dict(nocc=None, weight=0.3, cell_volume=19.0,
                  spin_degeneracy=1.5, band_occupations=occupations)
    matrix = np.asarray(valence_density_from_kpoint(
        jnp.asarray(box), return_spin_density_matrix=True, **kwargs))
    charge = np.asarray(valence_density_from_kpoint(jnp.asarray(box), **kwargs))
    fields = np.asarray(spin_density_matrix_to_pauli_fields(matrix))

    assert matrix.shape == (2, 2, 3, 2, 2)
    assert np.array_equal(matrix[1, 0], np.conj(matrix[0, 1]))
    assert np.allclose(fields[0], charge, rtol=3.0e-15, atol=3.0e-15)
    dvol = kwargs["cell_volume"] / float(np.prod(box.shape[-3:]))
    expected_charge = (kwargs["weight"] * kwargs["spin_degeneracy"]
                       * np.einsum("n,nsxyz->", occupations,
                                   np.abs(box) ** 2, optimize=True))
    assert float(np.sum(fields[0])) * dvol == pytest.approx(
        expected_charge, rel=3.0e-15, abs=3.0e-15)


def test_spin_matrix_mode_refuses_non_pauli_and_conflicting_requests():
    scalar = jnp.ones((1, 1, 1, 1, 1), dtype=jnp.complex128)
    with pytest.raises(ValueError, match="exactly two-component"):
        density_components_from_psi_r(
            scalar, return_spin_density_matrix=True)
    with pytest.raises(ValueError, match="exactly two-component"):
        valence_density_from_kpoint(
            scalar, nocc=None, weight=1.0, cell_volume=1.0,
            return_spin_density_matrix=True)

    pauli = jnp.ones((1, 2, 1, 1, 1), dtype=jnp.complex128)
    with pytest.raises(ValueError, match="mutually exclusive"):
        density_components_from_psi_r(
            pauli, include_dirac_current=True,
            return_spin_density_matrix=True)
