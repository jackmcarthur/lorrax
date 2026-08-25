"""First-principles checks for the uniform DFT gauge-vertex owner.

The tests are deterministic and operate directly on the canonical
``dft_operators`` representation.  They compare the public derivatives to
central differences of that same physical Hamiltonian matrix, but do not
construct a WFN reader, FFT path, symmetry map, or frontend.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from psp.dft_operators import (
    VNLChannelData,
    static_gauge_vertices_matrix_k,
    vnl_contact_autodiff,
    vnl_matrix_at_k,
    vnl_matrix_derivatives_from_projector_coefficients,
    vnl_projector_coefficients_k,
    vnl_velocity_autodiff,
    vnl_velocity_from_dZ,
    build_Z_and_dZ,
    build_Z_dZ_d2Z,
)
from psp.vnl_ops import (
    apply_vnl_derivatives_to_ket,
    apply_vnl_velocity_to_ket,
    vnl_matrix,
    vnl_velocity_matrix,
)


def _fixture():
    """A non-orthogonal cell and a fixed Hermitian spin-orbit VNL channel."""
    B = jnp.asarray([
        [1.15, 0.17, 0.03],
        [0.00, 0.91, 0.14],
        [0.08, 0.00, 1.07],
    ], dtype=jnp.float64)
    G = jnp.asarray([
        [0, 0, 0],
        [1, 0, 0],
        [0, -1, 1],
        [-1, 1, 0],
    ], dtype=jnp.int32)
    k = jnp.asarray([0.19, -0.23, 0.11], dtype=jnp.float64)

    # Fixed literals, followed by deterministic orthonormalisation in the
    # flattened spin/G space.  No random fixture can hide a flaky direction.
    raw = np.asarray([
        [1.0 + 0.1j, 0.2 - 0.3j, -0.4 + 0.2j, 0.5 + 0.0j,
         0.1 - 0.2j, -0.3 + 0.4j, 0.6 - 0.1j, 0.2 + 0.3j],
        [0.3 - 0.2j, 0.8 + 0.0j, 0.1 + 0.4j, -0.2 + 0.1j,
         0.5 + 0.2j, 0.1 - 0.5j, -0.3 + 0.0j, 0.7 - 0.1j],
    ], dtype=np.complex128)
    q, _ = np.linalg.qr(raw.T)
    psi = jnp.asarray(q.T.reshape(2, 2, 4), dtype=jnp.complex128)

    dq = 0.02
    qgrid = dq * np.arange(512, dtype=np.float64)
    # A linear reduced radial table has an exact constant derivative under
    # the canonical interpolator, leaving no unrelated table-derivative bias
    # in the finite-difference checks below.
    radial = 1.0 + 0.025 * qgrid
    dradial = np.full_like(radial, 0.025)
    E = np.asarray([
        [[[0.83 + 0.0j]], [[0.12 + 0.07j]]],
        [[[0.12 - 0.07j]], [[-0.31 + 0.0j]]],
    ], dtype=np.complex128)
    channel = VNLChannelData(
        tau=jnp.asarray([[0.13, 0.29, 0.07]], dtype=jnp.float64),
        prefactor=0.41,
        l=0,
        nbeta=1,
        q0=0.0,
        dq=dq,
        reduced_tables=(radial,),
        reduced_dtables=(dradial,),
        E=jnp.asarray(E),
    )
    return psi, G, k, B, [channel]


def _hamiltonian_k_dependent_matrix(psi, G, k, B, channels):
    """Independent matrix value whose derivatives define the public API.

    Only the k-dependent terms appear: kinetic plus the canonical VNL matrix.
    A local multiplicative potential is constant under uniform k/A shifts.
    """
    K_cart = (G.astype(jnp.float64) + k[None, :]) @ B
    kinetic_diag = jnp.sum(K_cart * K_cart, axis=1)
    kinetic = jnp.einsum(
        'msG,G,nsG->mn', jnp.conj(psi), kinetic_diag, psi, optimize=True)
    return kinetic + vnl_matrix_at_k(k, psi, G, B, channels)


def _shift_crystal(k, B, cart_step):
    return k + jnp.asarray(cart_step, dtype=jnp.float64) @ jnp.linalg.inv(B)


def test_kinetic_vertices_are_exact_in_rydberg_units():
    psi, G, k, B, _ = _fixture()
    got = static_gauge_vertices_matrix_k(psi, G, k, B, [])
    K_cart = (G.astype(jnp.float64) + k[None, :]) @ B
    gamma = 2.0 * jnp.einsum(
        'msG,Ga,nsG->amn', jnp.conj(psi), K_cart, psi, optimize=True)
    overlap = jnp.einsum('msG,nsG->mn', jnp.conj(psi), psi, optimize=True)
    contact = (
        2.0 * jnp.eye(3, dtype=overlap.dtype)[:, :, None, None]
        * overlap[None, None])
    np.testing.assert_allclose(got.gamma_cart, gamma, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(got.lambda_cart, contact, rtol=2e-13, atol=2e-13)


def test_vnl_autodiff_velocity_has_public_axis_order_and_matches_dz():
    psi, G, k, B, channels = _fixture()
    autodiff = vnl_velocity_autodiff(k, psi, G, B, channels)
    analytic = vnl_velocity_from_dZ(
        psi, build_Z_and_dZ(k, G, B, channels))
    assert autodiff.shape == analytic.shape == (3, 2, 2)
    np.testing.assert_allclose(autodiff, analytic, rtol=2e-11, atol=2e-11)


def test_persisted_beta_coefficients_close_all_band_matrices_without_G_axis():
    psi, G, k, B, channels = _fixture()
    coefficients = vnl_projector_coefficients_k(k, psi, G, B, channels)
    assert len(coefficients) == 1
    block = coefficients[0]
    assert block.c.shape == (1, 1, 2, 2)
    assert block.dc.shape == (3, 1, 1, 2, 2)
    assert block.d2c.shape == (3, 3, 1, 1, 2, 2)
    assert int(G.shape[0]) not in block.c.shape + block.dc.shape + block.d2c.shape

    closed = vnl_matrix_derivatives_from_projector_coefficients(coefficients)
    Z, dZ, E = build_Z_and_dZ(k, G, B, channels)[0]
    c = jnp.einsum('aqG,ntG->aqtn', jnp.conj(Z), psi, optimize=True)
    Ec = jnp.einsum('strq,aqtn->arsn', E, c, optimize=True)
    applied_G = jnp.einsum('arG,arsn->nsG', Z, Ec, optimize=True)
    reexpanded_reference = jnp.einsum(
        'msG,nsG->mn', jnp.conj(psi), applied_G, optimize=True)
    np.testing.assert_allclose(
        closed.value, reexpanded_reference,
        rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        vnl_matrix_at_k(k, psi, G, B, channels), reexpanded_reference,
        rtol=2e-12, atol=2e-12)
    # The existing flattened production API now shares the same canonical
    # coefficient closure.  Its separate apply-to-ket door is retained only
    # for consumers that genuinely need a G-space operator action.
    Z_flat = Z[0]
    dZ_flat = dZ[:, 0]
    np.testing.assert_allclose(
        vnl_matrix(psi, Z_flat, E), reexpanded_reference,
        rtol=2e-12, atol=2e-12)
    velocity_applied = apply_vnl_velocity_to_ket(
        psi, Z_flat, dZ_flat, E)
    velocity_reexpanded_reference = jnp.einsum(
        'msG,xnsG->xmn', jnp.conj(psi), velocity_applied,
        optimize=True)
    np.testing.assert_allclose(
        vnl_velocity_matrix(psi, Z_flat, dZ_flat, E),
        velocity_reexpanded_reference, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        closed.lambda_cart,
        vnl_contact_autodiff(k, psi, G, B, channels),
        rtol=2e-10, atol=2e-10)
    projector_derivatives = build_Z_dZ_d2Z(k, G, B, channels)[0]
    applied_derivatives = apply_vnl_derivatives_to_ket(
        psi,
        projector_derivatives.Z,
        projector_derivatives.dZ,
        projector_derivatives.d2Z,
        projector_derivatives.E,
    )
    np.testing.assert_allclose(
        jnp.einsum(
            'msG,xnsG->xmn', jnp.conj(psi),
            applied_derivatives.gamma_cart_ket, optimize=True),
        closed.gamma_cart, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        jnp.einsum(
            'msG,xynsG->xymn', jnp.conj(psi),
            applied_derivatives.lambda_cart_ket, optimize=True),
        closed.lambda_cart, rtol=2e-12, atol=2e-12)
    direct = static_gauge_vertices_matrix_k(psi, G, k, B, channels)
    reused = static_gauge_vertices_matrix_k(
        psi, G, k, B, channels,
        vnl_projector_coefficients=coefficients)
    np.testing.assert_array_equal(reused.gamma_cart, direct.gamma_cart)
    np.testing.assert_array_equal(reused.lambda_cart, direct.lambda_cart)


def test_uniform_gamma_and_contact_match_cartesian_finite_differences():
    psi, G, k, B, channels = _fixture()
    got = static_gauge_vertices_matrix_k(psi, G, k, B, channels)
    h = 2.0e-5
    gamma_fd = np.zeros((3, 2, 2), dtype=np.complex128)
    lambda_fd = np.zeros((3, 3, 2, 2), dtype=np.complex128)
    for a in range(3):
        step = np.zeros(3, dtype=np.float64)
        step[a] = h
        kp = _shift_crystal(k, B, step)
        km = _shift_crystal(k, B, -step)
        gamma_fd[a] = np.asarray(
            (_hamiltonian_k_dependent_matrix(psi, G, kp, B, channels)
             - _hamiltonian_k_dependent_matrix(psi, G, km, B, channels))
            / (2.0 * h))
        gp = static_gauge_vertices_matrix_k(psi, G, kp, B, channels).gamma_cart
        gm = static_gauge_vertices_matrix_k(psi, G, km, B, channels).gamma_cart
        lambda_fd[:, a] = np.asarray((gp - gm) / (2.0 * h))

    np.testing.assert_allclose(
        got.gamma_cart, gamma_fd, rtol=3e-8, atol=3e-8)
    np.testing.assert_allclose(
        got.lambda_cart, lambda_fd, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(
        got.lambda_cart, jnp.swapaxes(got.lambda_cart, 0, 1),
        rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        got.lambda_cart,
        jnp.conj(jnp.swapaxes(got.lambda_cart, -1, -2)),
        rtol=2e-11, atol=2e-11)


def test_finite_q_nonlocal_path_refuses_instead_of_guessing():
    psi, G, k, B, channels = _fixture()
    with pytest.raises(
            NotImplementedError, match="EM-VERTEX-FINITE-Q-WILSON"):
        static_gauge_vertices_matrix_k(
            psi, G, k, B, channels,
            q_cart_bohr_inv=(1.0e-4, 0.0, 0.0))


def test_provenance_binds_units_sign_scope_and_soc_coverage():
    psi, G, k, B, channels = _fixture()
    got = static_gauge_vertices_matrix_k(psi, G, k, B, channels)
    prov = got.provenance
    assert prov.q_scope == "exact_q_cart_bohr_inv_zero_only"
    assert prov.gamma_definition.startswith("+dH/dK_cart")
    assert prov.lambda_definition.startswith("+d2H/dK_cart")
    assert prov.gamma_units == "rydberg*bohr"
    assert prov.lambda_units == "rydberg*bohr^2"
    assert prov.nonlocal_finite_q_path == "unbound_and_refused"
    assert "spin-orbit E matrices" in prov.included_terms
