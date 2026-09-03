"""Unit tests for the fixed-index self-consistent update law."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from gw.sc_loop import (
    BandClasses,
    EvaluationPolicy,
    QpHamiltonian,
    SigmaTable,
    effective_sigma,
)


def _table(*, outside_sensitive=True):
    omega = np.array([-1.0, 0.0, 1.0])
    nk, n_p, n_u = 1, 4, 2
    # Deliberately non-Hermitian before the prescribed final operation, and
    # linear in omega so interpolation has an exact answer.
    base = np.arange(n_p * n_p).reshape(n_p, n_p) + 1j * np.arange(
        n_p * n_p).reshape(n_p, n_p)[::-1]
    slope = np.ones((n_p, n_p), dtype=np.complex128)
    if outside_sensitive:
        slope *= 2.0 + 0.5j
    cube = np.stack([base + w * slope for w in omega], axis=0)[:, None]
    kin = np.zeros((nk, n_p + n_u, n_p + n_u), dtype=np.complex128)
    dft = np.diag(np.arange(n_p + n_u, dtype=np.float64))[None].astype(
        np.complex128)
    return SigmaTable(
        omega_ev=omega,
        sigma_c_pp_wkij_ry=cube,
        sigma_x_pp_kij_ry=np.zeros((nk, n_p, n_p), np.complex128),
        v_h_pp_kij_ry=np.zeros((nk, n_p, n_p), np.complex128),
        sigma_xc_pu_fermi_kij_ry=np.full(
            (nk, n_p, n_u), 3.0 + 2.0j, np.complex128),
        v_h_pu_kij_ry=np.full(
            (nk, n_p, n_u), -0.5 + 0.25j, np.complex128),
        kin_ion_qp_kij_ry=kin,
        dft_h_qp_kij_ry=dft,
        e_dft_kn_ry=np.arange(n_p + n_u, dtype=np.float64)[None],
    )


def _classes():
    return BandClasses(
        band_start=0, occupied_stop=2, protected_stop=4, outer_stop=6)


def test_sigma_table_refuses_policy_inside_interpolator():
    table = _table()
    with pytest.raises(ValueError, match="out-of-domain"):
        table.at(np.array([[0.0, 0.0, 0.0, 1.01]]))


def test_fermi_pair_switches_both_ends_to_fermi():
    table = _table()
    classes = _classes()
    energies = np.array([[-0.5, 0.25, 0.75, 1.5, 4.0, 5.0]])
    sigma, diagnostics = effective_sigma(
        table, classes, EvaluationPolicy.FERMI, energies, 0.0)
    sigma = np.asarray(sigma)

    # Band 3 is outside. Every pair touching it reads the whole matrix at
    # E_F, rather than mixing an in-grid endpoint with a clamped edge.
    at_zero = np.asarray(table.at(np.zeros((1, 4))))
    expected = 0.5 * (at_zero + np.conj(at_zero.swapaxes(-1, -2)))
    np.testing.assert_allclose(sigma[:, :4, :4][:, 3, :], expected[:, 3, :])
    assert diagnostics["n_outside"] == 1
    assert diagnostics["policy"] == "fermi"


def test_fermi_membership_is_one_boolean_per_band_over_all_k():
    one = _table()
    table = replace(
        one,
        sigma_c_pp_wkij_ry=np.repeat(
            np.asarray(one.sigma_c_pp_wkij_ry), 2, axis=1),
        sigma_x_pp_kij_ry=np.repeat(
            np.asarray(one.sigma_x_pp_kij_ry), 2, axis=0),
        v_h_pp_kij_ry=np.repeat(
            np.asarray(one.v_h_pp_kij_ry), 2, axis=0),
        sigma_xc_pu_fermi_kij_ry=np.repeat(
            np.asarray(one.sigma_xc_pu_fermi_kij_ry), 2, axis=0),
        v_h_pu_kij_ry=np.repeat(
            np.asarray(one.v_h_pu_kij_ry), 2, axis=0),
        kin_ion_qp_kij_ry=np.repeat(
            np.asarray(one.kin_ion_qp_kij_ry), 2, axis=0),
        dft_h_qp_kij_ry=np.repeat(
            np.asarray(one.dft_h_qp_kij_ry), 2, axis=0),
        e_dft_kn_ry=np.repeat(
            np.asarray(one.e_dft_kn_ry), 2, axis=0),
    )
    energies = np.array([
        [-0.5, 0.25, 0.75, 0.5, 4.0, 5.0],
        [-0.5, 0.25, 0.75, 1.5, 4.0, 5.0],
    ])
    sigma, _ = effective_sigma(table, _classes(), "fermi", energies, 0.0)
    at_zero = np.asarray(table.at(np.zeros((2, 4))))
    expected = 0.5 * (at_zero + np.conj(at_zero.swapaxes(-1, -2)))
    # Band 3 is outside at only k=1, but every k/pair touching band 3 uses
    # E_F.  A per-(k,n) switch would fail this assertion at k=0.
    np.testing.assert_allclose(
        np.asarray(sigma)[:, :4, :4][:, 3, :], expected[:, 3, :])


def test_clamp_is_continuous_at_window_edge():
    table = _table()
    classes = _classes()
    below = np.array([[-0.5, 0.25, 0.75, 1.0 - 1.0e-8, 4.0, 5.0]])
    above = below.copy()
    above[0, 3] = 1.0 + 1.0e-8
    left, _ = effective_sigma(table, classes, "clamp", below, 0.0)
    right, _ = effective_sigma(table, classes, "clamp", above, 0.0)
    np.testing.assert_allclose(left, right, rtol=0.0, atol=3.0e-8)


def test_outer_law_and_hamiltonian_are_hermitian():
    table = _table()
    classes = _classes()
    energies = np.array([[-0.5, 0.25, 0.75, 1.5, 4.0, 5.0]])
    sigma, diagnostics = effective_sigma(
        table, classes, "clamp", energies, 0.0)
    hamiltonian = np.asarray(QpHamiltonian(
        table.kin_ion_qp_kij_ry).build(sigma))

    np.testing.assert_allclose(
        hamiltonian, np.conj(hamiltonian.swapaxes(-1, -2)),
        rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        hamiltonian[:, :4, 4:], 2.5 + 2.25j)
    np.testing.assert_allclose(hamiltonian[:, 4:, 4:], np.diag(
        np.asarray(table.e_dft_kn_ry)[0, 4:] + diagnostics["delta_ry"])[None])


def test_band_classes_ignore_padding_and_require_scissor_samples():
    classes = _classes()
    assert classes.n_protected == 4
    assert classes.n_outer == 2
    assert classes.n_total == 6
    with pytest.raises(ValueError, match="at least two protected conduction"):
        BandClasses(
            band_start=0, occupied_stop=3, protected_stop=4, outer_stop=6)
