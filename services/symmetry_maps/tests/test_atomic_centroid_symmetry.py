"""Atom-only spatial Seitz rows used to close centroid sampling sets."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from symmetry_maps import fft_grid_pullback_perm, recover_atomic_space_group


def _operation_row(Rinv, tau, rotation, translation, atol=1.0e-12):
    rows = [
        row for row in range(Rinv.shape[0])
        if np.array_equal(Rinv[row], rotation)
        and np.allclose(
            (tau[row] - translation) - np.rint(tau[row] - translation),
            0.0, rtol=0.0, atol=atol)
    ]
    assert len(rows) <= 1, "Seitz rows must be deterministically deduplicated"
    return rows[0] if rows else None


def test_species_decoration_breaks_an_uncolored_lattice_symmetry():
    """A spatial operation may not exchange chemically distinct sites."""
    avec = np.eye(3)
    atom_crys = np.array([[0.25, 0.0, 0.0],
                          [0.0, 0.25, 0.0]])
    swap_xy = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]], dtype=np.int32)
    tau_zero = np.zeros(3)

    _, uncolored_Rinv, uncolored_tau = recover_atomic_space_group(
        avec, atom_crys, np.array([1, 1]))
    _, decorated_Rinv, decorated_tau = recover_atomic_space_group(
        avec, atom_crys, np.array([1, 2]))

    assert _operation_row(
        uncolored_Rinv, uncolored_tau, swap_xy, tau_zero) is not None
    assert _operation_row(
        decorated_Rinv, decorated_tau, swap_xy, tau_zero) is None


def test_a_nonsymmorphic_screw_translation_is_recovered_in_pullback_convention():
    """The returned BGW row and tau drive the existing FFT-grid pullback."""
    avec = np.eye(3)
    atom_crys = np.array([[0.1, 0.2, 0.0],
                          [0.9, 0.8, 0.5]])
    atom_types = np.array([14, 14])
    screw_rotation = np.diag([-1, -1, 1]).astype(np.int32)
    screw_tau = np.array([0.0, 0.0, 0.5])

    R, Rinv, tau = recover_atomic_space_group(
        avec, atom_crys, atom_types)
    row = _operation_row(Rinv, tau, screw_rotation, screw_tau)
    assert row is not None, "the 2_1 screw row, including tau, was lost"
    np.testing.assert_array_equal(R[row] @ Rinv[row], np.eye(3, dtype=np.int32))

    fft_grid = np.array([10, 10, 10])
    pullback = fft_grid_pullback_perm(
        R[row:row + 1], 2.0 * np.pi * tau[row:row + 1], fft_grid)
    source = np.array([1, 2, 0])
    destination = np.array([9, 8, 5])
    source_flat = np.ravel_multi_index(tuple(source), tuple(fft_grid))
    destination_flat = np.ravel_multi_index(tuple(destination), tuple(fft_grid))
    assert pullback[0, destination_flat] == source_flat


def test_fft_pullback_refuses_an_off_grid_translation_before_snapping():
    """A half-grid shift can round to a permutation, but it is still wrong."""
    identity = np.eye(3, dtype=np.int32)[None, :, :]
    fft_grid = np.array([4, 4, 4])
    on_grid_tau = np.array([[0.25, 0.0, 0.0]])
    good = fft_grid_pullback_perm(
        identity, 2.0 * np.pi * on_grid_tau, fft_grid)
    assert np.unique(good[0]).size == int(np.prod(fft_grid))

    half_grid_tau = np.array([[0.125, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match="not commensurate with the FFT grid"):
        fft_grid_pullback_perm(
            identity, 2.0 * np.pi * half_grid_tau, fft_grid)


def test_improper_spatial_operations_are_included():
    """The atom-only group is the full spatial group, not its proper half."""
    avec = np.array([[1.00, 0.00, 0.00],
                     [0.17, 1.31, 0.00],
                     [0.23, 0.37, 1.79]])
    atom_crys = np.array([[0.13, 0.21, 0.31]])
    R, Rinv, tau = recover_atomic_space_group(
        avec, atom_crys, np.array([8]))

    np.testing.assert_array_equal(Rinv[0], np.eye(3, dtype=np.int32))
    np.testing.assert_array_equal(R[0], np.eye(3, dtype=np.int32))
    np.testing.assert_array_equal(tau[0], np.zeros(3))
    inversion = -np.eye(3, dtype=np.int32)
    inversion_row = _operation_row(
        Rinv, tau, inversion, np.mod(2.0 * atom_crys[0], 1.0))
    assert inversion_row is not None
    assert round(np.linalg.det(Rinv[inversion_row])) == -1


def test_a_skew_primitive_basis_recovers_large_integer_rows_and_closure():
    """The metric derives its search bound; it does not assume entries ±1."""
    # A unimodular shear of the simple-cubic primitive basis.  The physical
    # lattice still has 48 point operations, but their matrices in these
    # fractional coordinates contain entries with magnitude greater than one.
    avec = np.array([[1.0, 0.0, 0.0],
                     [2.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])
    _, Rinv, tau = recover_atomic_space_group(
        avec, np.zeros((1, 3)), np.array([1]))

    assert Rinv.shape == (48, 3, 3)
    assert np.max(np.abs(Rinv)) > 1
    np.testing.assert_array_equal(tau, np.zeros_like(tau))
    rotation_keys = {rotation.tobytes() for rotation in Rinv}
    for left in Rinv:
        for right in Rinv:
            assert (left @ right).tobytes() in rotation_keys


def test_six_decimal_hexagonal_atoms_do_not_lose_one_inverse_at_the_tolerance():
    """A +1 ulp comparison asymmetry must not turn D3h into eight rows."""
    avec = np.array([[1.0, 0.0, 0.0],
                     [-0.4999987358941589, 0.866025322568208, 0.0],
                     [0.0, 0.0, 3.7923175231615787]])
    atom_crys = np.array([[0.666667, 0.333333, 0.0],
                          [0.333333, 0.666667, 0.131928],
                          [0.333333, 0.666667, -0.131928]])
    atom_types = np.array([42, 16, 16])
    _, Rinv, _ = recover_atomic_space_group(avec, atom_crys, atom_types)

    assert Rinv.shape == (12, 3, 3)
    rotation_keys = {rotation.tobytes() for rotation in Rinv}
    for rotation in Rinv:
        inverse = np.rint(np.linalg.inv(rotation)).astype(np.int32)
        assert inverse.tobytes() in rotation_keys


class _CoordinatesWithElectronicMetadata(np.ndarray):
    """Test carrier proving array attributes cannot influence this policy."""


def test_group_is_deterministic_and_independent_of_electronic_metadata():
    """Density, magnetism, TR, and occupations are outside the API by design."""
    parameters = inspect.signature(recover_atomic_space_group).parameters
    assert tuple(parameters) == ("avec", "atom_crys", "atom_types", "tol")
    assert not ({"charge_density", "magnetism", "time_reversal",
                 "trs_holds", "occupations"} & set(parameters))

    avec = np.eye(3)
    positions = np.array([[0.1, 0.2, 0.0],
                          [0.9, 0.8, 0.5]])
    types = np.array([14, 14])
    arm_a = positions.copy().view(_CoordinatesWithElectronicMetadata)
    arm_a.charge_density = np.arange(8).reshape(2, 2, 2)
    arm_a.magnetism = "ferromagnetic"
    arm_a.time_reversal = False
    arm_a.occupations = np.array([1.0, 0.0])

    reverse = np.array([1, 0])
    arm_b = positions[reverse].copy().view(_CoordinatesWithElectronicMetadata)
    arm_b.charge_density = np.zeros((3, 3, 3))
    arm_b.magnetism = "nonmagnetic"
    arm_b.time_reversal = True
    arm_b.occupations = np.array([0.37, 0.63])

    result_a = recover_atomic_space_group(avec, arm_a, types)
    result_b = recover_atomic_space_group(avec, arm_b, types[reverse])
    for left, right in zip(result_a, result_b):
        np.testing.assert_array_equal(left, right)
