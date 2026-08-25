"""Five deterministic first-principles gates for production VNL contacts."""
from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from psp import vnl_ops
from psp.dft_operators import static_gauge_vertices_matrix_k
from psp.radial_tables import projector_second_deriv_table
from psp.species import SpeciesData
from psp.vnl_ops import (
    VNLSetup,
    apply_uniform_vnl_derivatives_to_ket,
    build_uniform_vnl_kdata,
    build_vnl_kdata_from_kvec,
    vnl_matrix_derivatives_from_projector_coefficients,
    vnl_projector_coefficients_k,
)


def _setup(*, curved: bool) -> VNLSetup:
    dq = 0.002
    q = dq * np.arange(2501, dtype=np.float64)
    if curved:
        radial = 1.2 - 0.20 * q * q
        radial_prime = -0.40 * q
        radial_second = np.full_like(q, -0.40)
        B = np.eye(3)
        tau = np.zeros((1, 3), dtype=np.float64)
    else:
        # Exactly represented by linear interpolation.  Every K is nonzero
        # in the non-Gamma fixture, so this isolates Cartesian chain/product
        # rules from interpolation error.
        radial = 0.9 + 0.07 * q
        radial_prime = np.full_like(q, 0.07)
        radial_second = np.zeros_like(q)
        B = np.asarray([
            [1.10, 0.18, 0.02],
            [0.00, 0.93, 0.16],
            [0.07, 0.00, 1.04],
        ])
        tau = np.asarray([[0.13, 0.27, 0.09]])

    E = np.asarray([
        [[[0.81 + 0.0j]], [[0.12 + 0.05j]]],
        [[[0.12 - 0.05j]], [[-0.23 + 0.0j]]],
    ], dtype=np.complex128)
    return VNLSetup(
        channels=[], dq=dq, n_q=q.size, q_max=float(q[-1]),
        G_table=jnp.asarray(radial[None, :]),
        Gp_table=jnp.asarray(radial_prime[None, :]),
        Gpp_table=jnp.asarray(radial_second[None, :]),
        prefactor=0.67, B=B, cell_volume=1.0,
        total_R=1, nspinor=2, E_super=jnp.asarray(E), l_max=0,
        soc=True,
        row_beta_idx=jnp.asarray([0], dtype=jnp.int32),
        row_l=jnp.asarray([0], dtype=jnp.int32),
        row_m=jnp.asarray([0], dtype=jnp.int32),
        row_tau=jnp.asarray(tau),
    )


def _states():
    bra = jnp.asarray([[[0.7 + 0.1j, -0.2 + 0.3j, 0.4 - 0.1j, 9.0],
                        [0.1 - 0.2j, 0.5 + 0.0j, -0.3 + 0.2j, -7.0]]])
    ket = jnp.asarray([
        [[0.2 + 0.4j, 0.3 - 0.1j, -0.5 + 0.0j, 8.0],
         [0.6 + 0.0j, -0.1 + 0.2j, 0.2 + 0.3j, 6.0]],
        [[-0.4 + 0.1j, 0.2 + 0.2j, 0.1 - 0.3j, -5.0],
         [0.3 - 0.2j, 0.4 + 0.1j, -0.2 + 0.0j, 4.0]],
    ])
    return bra, ket


def _matrix_value(psi_bra, psi_ket, G, mask, k, setup):
    kdata = build_uniform_vnl_kdata(k, G, setup, g_mask=mask)
    mask_j = jnp.asarray(mask)[None, None, :]
    bra = psi_bra * mask_j
    ket = psi_ket * mask_j
    K = (G.astype(jnp.float64) + k[None, :]) @ jnp.asarray(setup.B)
    kinetic = jnp.einsum(
        "msG,G,nsG->mn", jnp.conj(bra), jnp.sum(K * K, axis=1), ket,
        optimize=True)
    B = jnp.asarray(setup.B)
    Z = vnl_ops._assemble_uniform_projectors(
        jnp.asarray(kdata.k_crys), jnp.asarray(kdata.G_int), B,
        jnp.asarray(setup.dq), setup.G_table, setup.Gp_table,
        setup.Gpp_table, jnp.asarray(setup.prefactor),
        setup.row_beta_idx, setup.row_l, setup.row_m, setup.row_tau,
        l_max=setup.l_max)
    c_bra = jnp.einsum("RG,msG->Rsm", jnp.conj(Z), bra)
    c_ket = jnp.einsum("RG,nsG->Rsn", jnp.conj(Z), ket)
    Ec = jnp.einsum("stRQ,Qtn->Rsn", setup.E_super, c_ket)
    nonlocal_value = jnp.einsum("Rsm,Rsn->mn", jnp.conj(c_bra), Ec)
    return kinetic + nonlocal_value


def test_exact_gamma_g0_curved_even_radial_hessian():
    # First gate the radial owner's exact moment rather than a one-sided FD.
    r = np.linspace(0.0, 4.0, 101)
    dr = r[1] - r[0]
    beta_over_r = np.exp(-r * r)[None, :]
    sp = SpeciesData(
        element="H", z_valence=1.0, z_atomic=1, r=r,
        rab=np.full_like(r, dr), vloc_r=np.zeros_like(r),
        rho_core_r=np.zeros_like(r), has_nlcc=False, n_proj=1,
        beta_r=beta_over_r, proj_l=np.asarray([0]),
        proj_j=np.asarray([0.5]), dij=np.ones((1, 1)), nspinor=1)
    got_table = projector_second_deriv_table(sp, 0, np.asarray([0.0, 0.1]))
    sw = np.ones_like(r)
    sw[1:-1:2] = 4.0 / 3.0
    sw[2:-1:2] = 2.0 / 3.0
    sw[[0, -1]] = 1.0 / 3.0
    beta = beta_over_r[0] * r
    expected_gpp0 = -np.sum(beta * r**3 * sw * dr) / 3.0
    np.testing.assert_allclose(got_table[0], expected_gpp0, rtol=0, atol=1e-15)

    # Then gate the full public contact at Gamma/G=0.
    setup = _setup(curved=True)
    psi = jnp.asarray([[[1.0], [0.0]]], dtype=jnp.complex128)
    got = static_gauge_vertices_matrix_k(
        psi, jnp.zeros((1, 3), dtype=jnp.int32), jnp.zeros(3), setup)
    z0 = setup.prefactor * 1.2 / np.sqrt(4.0 * np.pi)
    zpp = setup.prefactor * (-0.40) / np.sqrt(4.0 * np.pi)
    expected_nl = 2.0 * 0.81 * z0 * zpp
    np.testing.assert_allclose(got.gamma_cart, 0.0, rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        got.lambda_cart[:, :, 0, 0],
        np.eye(3) * (2.0 + expected_nl), rtol=2e-13, atol=2e-13)


def test_nonzero_k_cartesian_jvp_and_hessian_match_finite_difference():
    setup = _setup(curved=False)
    bra, ket = _states()
    G = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = np.asarray([1.0, 1.0, 1.0, 0.0])
    k = jnp.asarray([0.19, -0.17, 0.12])
    got = static_gauge_vertices_matrix_k(
        ket, G, k, setup, psi_bra_G=bra, g_mask=mask)
    h = 2.0e-5
    Binv = np.linalg.inv(setup.B)
    gamma_fd = np.zeros_like(np.asarray(got.gamma_cart))
    lambda_fd = np.zeros_like(np.asarray(got.lambda_cart))
    for a in range(3):
        step = np.zeros(3)
        step[a] = h
        kp = k + step @ Binv
        km = k - step @ Binv
        gamma_fd[a] = np.asarray(
            (_matrix_value(bra, ket, G, mask, kp, setup)
             - _matrix_value(bra, ket, G, mask, km, setup)) / (2.0 * h))
        gp = static_gauge_vertices_matrix_k(
            ket, G, kp, setup, psi_bra_G=bra, g_mask=mask).gamma_cart
        gm = static_gauge_vertices_matrix_k(
            ket, G, km, setup, psi_bra_G=bra, g_mask=mask).gamma_cart
        lambda_fd[:, a] = np.asarray((gp - gm) / (2.0 * h))
    np.testing.assert_allclose(got.gamma_cart, gamma_fd, rtol=4e-8, atol=4e-8)
    np.testing.assert_allclose(
        got.lambda_cart, lambda_fd, rtol=2e-7, atol=2e-7)


def test_distinct_left_right_band_blocks_match_full_square_slice():
    setup = _setup(curved=False)
    bra, ket = _states()
    G = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = np.asarray([1.0, 1.0, 1.0, 0.0])
    k = jnp.asarray([0.19, -0.17, 0.12])
    small_bra = 0.3j * bra
    small_ket = -0.2j * ket
    bra_bispinor = jnp.concatenate([bra, small_bra], axis=1)
    ket_bispinor = jnp.concatenate([ket, small_ket], axis=1)
    rectangular = static_gauge_vertices_matrix_k(
        ket_bispinor, G, k, setup, psi_bra_G=bra_bispinor, g_mask=mask)
    combined = jnp.concatenate([bra, ket], axis=0)
    square = static_gauge_vertices_matrix_k(
        combined, G, k, setup, g_mask=mask)
    np.testing.assert_allclose(
        rectangular.gamma_cart, square.gamma_cart[:, :1, 1:],
        rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        rectangular.lambda_cart, square.lambda_cart[:, :, :1, 1:],
        rtol=2e-13, atol=2e-13)
    assert rectangular.gamma_cart.shape == (3, 1, 2)

    # The exceptional G-space apply door must be the same operator as the
    # normal rectangular coefficient closure, without retaining d2Z.
    kdata = build_uniform_vnl_kdata(k, G, setup, g_mask=mask)
    left = vnl_projector_coefficients_k(
        kdata, bra_bispinor, state_block_identity="left")
    right = vnl_projector_coefficients_k(
        kdata, ket_bispinor, state_block_identity="right")
    closed = vnl_matrix_derivatives_from_projector_coefficients(left, right)
    applied = apply_uniform_vnl_derivatives_to_ket(kdata, ket_bispinor)
    bra_physical = bra * jnp.asarray(mask)[None, None, :]
    np.testing.assert_allclose(
        jnp.einsum(
            "msG,ansG->amn", jnp.conj(bra_physical),
            applied.gamma_cart_ket),
        closed.gamma_cart, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        jnp.einsum(
            "msG,abnsG->abmn", jnp.conj(bra_physical),
            applied.lambda_cart_ket),
        closed.lambda_cart, rtol=2e-13, atol=2e-13)
    with pytest.raises(ValueError, match="physical-spinor mismatch"):
        static_gauge_vertices_matrix_k(
            ket_bispinor[:, :3], G, k, setup, g_mask=mask)


def test_stale_coefficient_carrier_refuses():
    setup = _setup(curved=False)
    bra, ket = _states()
    G = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    first = build_uniform_vnl_kdata(jnp.asarray([0.2, -0.1, 0.1]), G, setup)
    second = build_uniform_vnl_kdata(jnp.asarray([0.2, -0.1, 0.1]), G, setup)
    left = vnl_projector_coefficients_k(
        first, bra, state_block_identity="left")
    right = vnl_projector_coefficients_k(
        second, ket, state_block_identity="right")
    with pytest.raises(ValueError, match="stale VNL coefficient carrier"):
        vnl_matrix_derivatives_from_projector_coefficients(left, right)


def test_finite_q_and_ordinary_kdata_bypass_both_refuse():
    setup = _setup(curved=False)
    _, ket = _states()
    G = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    k = jnp.asarray([0.2, -0.1, 0.1])
    with pytest.raises(NotImplementedError, match="EM-VERTEX-FINITE-Q-WILSON"):
        build_uniform_vnl_kdata(
            k, G, setup, q_cart_bohr_inv=(1.0e-6, 0.0, 0.0))

    ordinary = build_vnl_kdata_from_kvec(k, G, setup)
    # The old bypass shape was to reach a raw derivative/apply door without
    # the public q check.  Flipping the visible flag cannot forge the private
    # in-process context token.
    ordinary.uniform_gauge = True
    with pytest.raises(ValueError, match="build_uniform_vnl_kdata"):
        vnl_projector_coefficients_k(
            ordinary, ket, state_block_identity="bypass")
