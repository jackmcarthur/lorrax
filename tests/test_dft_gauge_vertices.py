"""Deterministic first-principles gates for uniform VNL gauge actions."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from common import mtxel_sweep
from common.bispinor_init import (
    ALPHA_FS,
    HALFALPHA,
    ISOMETRIC_KINETIC_BALANCE_LIFT,
    lift_to_4spinor,
)
from common.collectives import single_device_mesh
from common.gamma_matrices import gamma_apply, gamma_perm_phase
from common.wfn_transforms import gflat_to_rmu
from file_io.wfn_basis import WavefunctionBasisReceipt
from gw import w_isdf
from gw.wavefunction_bundle import (
    AuthenticatedWavefunctions, BandSlices, Wavefunctions)
from psp import vnl_ops
from psp import dft_operators, radial_tables
from psp.species import SpeciesData
from psp.vnl_ops import (
    ChannelMeta,
    VNLSetup,
    apply_icl_vnl_transfer_jet_to_ket,
    apply_uniform_vnl_derivatives_to_ket,
    build_vnl_kdata_from_kvec,
    compute_icl_vnl_finite_contact_to_ket,
    compute_icl_vnl_finite_transfer_to_ket,
)


_EXPECTED_SRC = Path(__file__).resolve().parents[1] / "src"
for _module in (mtxel_sweep, w_isdf, vnl_ops, dft_operators, radial_tables):
    if _EXPECTED_SRC not in Path(_module.__file__).resolve().parents:
        raise RuntimeError(
            f"split-source gate: {_module.__name__} imported from "
            f"{_module.__file__}, expected {_EXPECTED_SRC}")


def _setup(
    *, curved: bool, natoms: int = 1, third_derivatives: bool = False,
) -> VNLSetup:
    dq = 0.002
    q = dq * np.arange(2501, dtype=np.float64)
    if curved:
        radial = 1.2 - 0.20 * q * q
        radial_prime = -0.40 * q
        radial_second = np.full_like(q, -0.40)
        B = np.eye(3)
        tau = np.zeros((natoms, 3), dtype=np.float64)
    else:
        # Linear interpolation is exact for this table; nonzero K keeps its
        # intentionally non-even radial slope away from the origin.
        radial = 0.9 + 0.07 * q
        radial_prime = np.full_like(q, 0.07)
        radial_second = np.zeros_like(q)
        B = np.asarray([
            [1.10, 0.18, 0.02],
            [0.00, 0.93, 0.16],
            [0.07, 0.00, 1.04],
        ])
        base = np.asarray([0.13, 0.27, 0.09])
        step = np.asarray([0.11, 0.07, 0.05])
        tau = np.stack([base + ia * step for ia in range(natoms)])

    E = np.asarray([
        [[[0.81 + 0.0j]], [[0.12 + 0.05j]]],
        [[[0.12 - 0.05j]], [[-0.23 + 0.0j]]],
    ], dtype=np.complex128)
    channel = ChannelMeta(
        l=0, nbeta=1, msize=1, R=1, tau=tau, E=E,
        beta_table_start=0, natoms=natoms)
    E_super = np.zeros((2, 2, natoms, natoms), dtype=np.complex128)
    for ia in range(natoms):
        E_super[:, :, ia, ia] = E[:, :, 0, 0]
    return VNLSetup(
        channels=[channel], dq=dq, n_q=q.size, q_max=float(q[-1]),
        G_table=jnp.asarray(radial[None, :]),
        Gp_table=jnp.asarray(radial_prime[None, :]),
        Gpp_table=jnp.asarray(radial_second[None, :]),
        Gppp_table=(jnp.zeros((1, q.size), dtype=jnp.float64)
                    if third_derivatives else None),
        prefactor=0.67, B=B, cell_volume=1.0,
        total_R=natoms, nspinor=2, E_super=jnp.asarray(E_super), l_max=0,
        soc=True,
        row_beta_idx=jnp.zeros(natoms, dtype=jnp.int32),
        row_l=jnp.zeros(natoms, dtype=jnp.int32),
        row_m=jnp.zeros(natoms, dtype=jnp.int32),
        row_tau=jnp.asarray(tau),
        coupled_row_blocks=tuple((ia, ia + 1, 0) for ia in range(natoms)),
    )


def _basis_receipt(
    wfn, geom, r_mu, band_start, band_stop, *, bispinor_lift="raw",
):
    from runtime.padding import padded_mu_extent
    return WavefunctionBasisReceipt.from_source(
        wfn=wfn, role='transverse', bispinor=True,
        bispinor_lift=bispinor_lift,
        band_interval=(band_start, band_stop),
        fft_grid=geom.fft_grid, centroid_fft_idx=r_mu,
        n_rmu_logical=len(r_mu),
        n_rmu_padded=padded_mu_extent(len(r_mu), geom.mesh))


def _states():
    return jnp.asarray([
        [[0.2 + 0.4j, 0.3 - 0.1j, -0.5 + 0.0j, 8.0],
         [0.6 + 0.0j, -0.1 + 0.2j, 0.2 + 0.3j, 6.0]],
        [[-0.4 + 0.1j, 0.2 + 0.2j, 0.1 - 0.3j, -5.0],
         [0.3 - 0.2j, 0.4 + 0.1j, -0.2 + 0.0j, 4.0]],
    ], dtype=jnp.complex128)


def _ordinary_vnl_action(psi, G, mask, k, setup):
    """Independent value path used for Cartesian finite differences."""
    kdata = build_vnl_kdata_from_kvec(
        np.asarray(k, dtype=np.float64), np.asarray(G, dtype=np.int32), setup)
    mask_j = jnp.asarray(mask, dtype=psi.real.dtype)
    physical = psi * mask_j[None, None, :]
    value = vnl_ops.apply_vnl(physical, kdata.Z, kdata.E_super)
    return np.asarray(value * mask_j[None, None, :])


def _finite_setup(*, third_derivatives=False):
    return replace(
        _setup(curved=False, third_derivatives=third_derivatives),
        uniform_gauge_fingerprint="sha256:" + "5" * 64)


def _rectangular_vnl_action(
    psi, G_source, source_mask, G_target_unwrapped, target_mask, k, setup,
):
    """Independent rectangular ``Z_out E Z_in^dag`` value oracle."""
    Z_source = np.asarray(build_vnl_kdata_from_kvec(
        np.asarray(k), np.asarray(G_source), setup).Z)
    Z_target = np.asarray(build_vnl_kdata_from_kvec(
        np.asarray(k), np.asarray(G_target_unwrapped), setup).Z)
    source_mask = np.asarray(source_mask)
    target_mask = np.asarray(target_mask)
    physical = np.asarray(psi) * source_mask[None, None, :]
    coefficients = np.einsum(
        "RG,nsG->Rsn", np.conj(Z_source), physical, optimize=True)
    coupled = np.einsum(
        "stRQ,Qtn->Rsn", np.asarray(setup.E_super), coefficients,
        optimize=True)
    value = np.einsum(
        "RG,Rsn->nsG", Z_target, coupled, optimize=True)
    return value * target_mask[None, None, :]


def test_exact_origin_radial_moment_and_contact_action():
    r = np.linspace(0.0, 4.0, 101)
    dr = r[1] - r[0]
    beta_over_r = np.exp(-r * r)[None, :]
    species = SpeciesData(
        element="H", z_valence=1.0, z_atomic=1, r=r,
        rab=np.full_like(r, dr), vloc_r=np.zeros_like(r),
        rho_core_r=np.zeros_like(r), has_nlcc=False, n_proj=1,
        beta_r=beta_over_r, proj_l=np.asarray([0]),
        dij=np.ones((1, 1)))
    tables = radial_tables.build_all_tables(
        [species], q_max=0.1, n_q=3, second_derivatives=True)
    sw = np.ones_like(r)
    sw[1:-1:2] = 4.0 / 3.0
    sw[2:-1:2] = 2.0 / 3.0
    sw[[0, -1]] = 1.0 / 3.0
    beta = beta_over_r[0] * r
    expected_gpp0 = -np.sum(beta * r**3 * sw * dr) / 3.0
    np.testing.assert_allclose(
        tables["second_deriv_tables"][0][0, 0], expected_gpp0,
        rtol=0.0, atol=1.0e-15)

    setup = _setup(curved=True)
    psi = jnp.asarray([[[1.0], [0.0]]], dtype=jnp.complex128)
    got = apply_uniform_vnl_derivatives_to_ket(
        psi, jnp.zeros((1, 3), dtype=jnp.int32), jnp.zeros(3), setup,
        jnp.ones(1), projector_row_chunk=1, g_chunk=1)
    z0 = setup.prefactor * 1.2 / np.sqrt(4.0 * np.pi)
    zpp = setup.prefactor * (-0.40) / np.sqrt(4.0 * np.pi)
    expected = np.zeros((3, 3, 1, 2, 1), dtype=np.complex128)
    spin_column = 2.0 * z0 * zpp * np.asarray(
        setup.channels[0].E[:, 0, 0, 0])
    for a in range(3):
        expected[a, a, 0, :, 0] = spin_column
    np.testing.assert_allclose(got.gamma_cart_ket, 0.0, rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        got.lambda_cart_ket, expected, rtol=2e-13, atol=2e-13)
    kinetic = dft_operators.apply_kinetic_contact_to_ket(psi)
    np.testing.assert_allclose(
        kinetic[:, :, 0, :, 0],
        2.0 * np.eye(3)[:, :, None] * np.asarray(psi[0, :, 0]),
        rtol=0.0, atol=0.0)


def test_nonzero_k_action_matches_cartesian_value_finite_differences():
    setup = _setup(curved=False)
    psi = _states()
    G = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = np.asarray([1.0, 1.0, 1.0, 0.0])
    k = np.asarray([0.19, -0.17, 0.12])
    got = apply_uniform_vnl_derivatives_to_ket(
        psi, G, k, setup, mask, projector_row_chunk=1, g_chunk=2)
    Binv = np.linalg.inv(setup.B)
    h1 = 2.0e-5
    h2 = 4.0e-4
    value0 = _ordinary_vnl_action(psi, G, mask, k, setup)
    gamma_fd = np.zeros_like(np.asarray(got.gamma_cart_ket))
    lambda_fd = np.zeros_like(np.asarray(got.lambda_cart_ket))
    for a in range(3):
        da = np.zeros(3)
        da[a] = 1.0
        kp = k + h1 * da @ Binv
        km = k - h1 * da @ Binv
        gamma_fd[a] = (
            _ordinary_vnl_action(psi, G, mask, kp, setup)
            - _ordinary_vnl_action(psi, G, mask, km, setup)) / (2.0 * h1)
        kp2 = k + h2 * da @ Binv
        km2 = k - h2 * da @ Binv
        lambda_fd[a, a] = (
            _ordinary_vnl_action(psi, G, mask, kp2, setup)
            - 2.0 * value0
            + _ordinary_vnl_action(psi, G, mask, km2, setup)) / h2**2
        for b in range(a):
            db = np.zeros(3)
            db[b] = 1.0
            mixed = (
                _ordinary_vnl_action(
                    psi, G, mask, k + (h2 * da + h2 * db) @ Binv, setup)
                - _ordinary_vnl_action(
                    psi, G, mask, k + (h2 * da - h2 * db) @ Binv, setup)
                - _ordinary_vnl_action(
                    psi, G, mask, k + (-h2 * da + h2 * db) @ Binv, setup)
                + _ordinary_vnl_action(
                    psi, G, mask, k - (h2 * da + h2 * db) @ Binv, setup)
            ) / (4.0 * h2**2)
            lambda_fd[a, b] = mixed
            lambda_fd[b, a] = mixed
    np.testing.assert_allclose(
        got.gamma_cart_ket, gamma_fd, rtol=7e-8, atol=7e-8)
    np.testing.assert_allclose(
        got.lambda_cart_ket, lambda_fd, rtol=5e-6, atol=5e-6)


def test_fixed_row_and_g_scans_cover_each_coupled_block_once():
    setup = _setup(curved=False, natoms=3)
    assert vnl_ops._coupled_projector_row_blocks(setup, 1) == (
        (0, 1, 0), (1, 2, 0), (2, 3, 0))
    psi = _states()
    G = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = jnp.asarray([1.0, 1.0, 1.0, 0.0])
    k = jnp.asarray([0.19, -0.17, 0.12])
    one = apply_uniform_vnl_derivatives_to_ket(
        psi, G, k, setup, mask, projector_row_chunk=1, g_chunk=1)
    wide = apply_uniform_vnl_derivatives_to_ket(
        psi, G, k, setup, mask, projector_row_chunk=8, g_chunk=7)
    np.testing.assert_allclose(
        one.gamma_cart_ket, wide.gamma_cart_ket, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(
        one.lambda_cart_ket, wide.lambda_cart_ket, rtol=3e-13, atol=3e-13)
    np.testing.assert_array_equal(np.asarray(one.gamma_cart_ket[..., -1]), 0)
    np.testing.assert_array_equal(np.asarray(one.lambda_cart_ket[..., -1]), 0)

    duplicate = replace(
        setup, coupled_row_blocks=((0, 1, 0), (1, 2, 0), (1, 2, 0)))
    with pytest.raises(ValueError, match="ROW-COVERAGE"):
        vnl_ops._coupled_projector_row_blocks(duplicate, 1)


def test_strict_large_components_contact_setup_and_finite_q_refuse():
    setup = _setup(curved=False)
    psi = _states()
    G = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = jnp.asarray([1.0, 1.0, 1.0, 0.0])
    k = jnp.asarray([0.19, -0.17, 0.12])
    psi_4c = jnp.concatenate([psi, -0.2j * psi], axis=1)
    with pytest.raises(ValueError, match="EM-VERTEX-LARGE-COMPONENTS"):
        apply_uniform_vnl_derivatives_to_ket(psi_4c, G, k, setup, mask)
    with pytest.raises(NotImplementedError, match="EM-VERTEX-FINITE-Q-WILSON"):
        apply_uniform_vnl_derivatives_to_ket(
            psi, G, k, setup, mask,
            q_cart_bohr_inv=(1.0e-8, 0.0, 0.0))
    with pytest.raises(ValueError, match="EM-VERTEX-VNL-GPP-MISSING"):
        apply_uniform_vnl_derivatives_to_ket(
            psi, G, k, replace(setup, Gpp_table=None), mask)


def test_ordinary_and_gauge_projector_values_share_exact_origin_owner():
    setup = _setup(curved=True)
    k = jnp.zeros(3, dtype=jnp.float64)
    G = jnp.zeros((1, 3), dtype=jnp.int32)
    ordinary = build_vnl_kdata_from_kvec(k, G, setup).Z
    gauge = vnl_ops._assemble_uniform_projector_rows(
        k, G, setup, setup.row_beta_idx, setup.row_l, setup.row_m,
        setup.row_tau)
    expected = setup.prefactor * 1.2 / np.sqrt(4.0 * np.pi)
    np.testing.assert_allclose(ordinary, gauge, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(ordinary[0, 0], expected, rtol=2e-15, atol=0.0)


def test_icl_kminusq_jet_reuses_uniform_current_and_contact_exactly():
    setup = _setup(curved=False)
    psi = _states()
    G = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = jnp.asarray([1.0, 1.0, 1.0, 0.0])
    k = jnp.asarray([0.19, -0.17, 0.12])
    uniform = apply_uniform_vnl_derivatives_to_ket(
        psi, G, k, setup, mask, projector_row_chunk=1, g_chunk=2)
    icl = apply_icl_vnl_transfer_jet_to_ket(
        psi, G, k, setup, mask, projector_row_chunk=1, g_chunk=2)
    np.testing.assert_array_equal(
        np.asarray(icl.gamma0_cart_ket),
        np.asarray(uniform.gamma_cart_ket))
    np.testing.assert_array_equal(
        np.asarray(icl.lambda0_cart_ket),
        np.asarray(uniform.lambda_cart_ket))
    np.testing.assert_allclose(
        icl.dgamma_dq_cart_ket,
        -0.5 * uniform.lambda_cart_ket,
        rtol=0.0, atol=0.0)
    current_only = apply_icl_vnl_transfer_jet_to_ket(
        psi, G, k, setup, mask, projector_row_chunk=1, g_chunk=2,
        include_contact=False)
    np.testing.assert_array_equal(
        current_only.gamma0_cart_ket, uniform.gamma_cart_ket)
    assert current_only.lambda0_cart_ket is None
    assert current_only.dgamma_dq_cart_ket is None


def test_icl_kminusq_taylor_ward_and_transfer_hermiticity():
    """The q jet is the straight-line average of dV(k-lambda*q)."""
    setup = _setup(curved=False)
    psi = _states()
    G = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = np.asarray([1.0, 1.0, 1.0, 0.0])
    k = np.asarray([0.19, -0.17, 0.12])
    Binv = np.linalg.inv(setup.B)
    jet = apply_icl_vnl_transfer_jet_to_ket(
        psi, G, k, setup, mask, projector_row_chunk=1, g_chunk=2)
    gamma0 = np.asarray(jet.gamma0_cart_ket)
    dgamma = np.asarray(jet.dgamma_dq_cart_ket)
    direction = np.asarray([0.31, -0.47, 0.29])

    # Ward/Taylor identity for the positive Hamiltonian vertex:
    # q.Gamma = [V(k)-V(k-q)] through the retained O(q^2) terms.
    def ward_error(step):
        q = step * direction
        gamma_q = gamma0 + np.einsum("b,abnsg->ansg", q, dgamma)
        lhs = np.einsum("a,ansg->nsg", q, gamma_q)
        rhs = (
            _ordinary_vnl_action(psi, G, mask, k, setup)
            - _ordinary_vnl_action(
                psi, G, mask, k - q @ Binv, setup))
        return np.linalg.norm(lhs - rhs)

    err_big = ward_error(4.0e-3)
    err_small = ward_error(2.0e-3)
    assert err_small < 0.14 * err_big  # cubic remainder: ideal ratio 1/8

    # At one k, V_,a and V_,ab are Hermitian.  This is precisely the
    # O(q) content of Gamma(k,q)^dagger = Gamma(k-q,-q).
    bra = np.conj(np.asarray(psi) * mask[None, None, :])
    gamma_mtx = np.einsum("msg,ansg->amn", bra, gamma0)
    dgamma_mtx = np.einsum("msg,abnsg->abmn", bra, dgamma)
    np.testing.assert_allclose(
        gamma_mtx, np.swapaxes(np.conj(gamma_mtx), -1, -2),
        rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        dgamma_mtx, np.swapaxes(np.conj(dgamma_mtx), -1, -2),
        rtol=2e-12, atol=2e-12)


def test_icl_q2_physical_radial_third_derivative_and_operator_jet():
    # Independent analytic l=0 Gaussian Hankel transform:
    # int r^2 exp(-a r^2) j0(qr) dr = A exp(-q^2/(4a)).
    a = 0.8
    r = np.linspace(0.0, 8.0, 801)
    dr = r[1] - r[0]
    beta_over_r = np.exp(-a * r * r)[None, :]
    species = SpeciesData(
        element="H", z_valence=1.0, z_atomic=1, r=r,
        rab=np.full_like(r, dr), vloc_r=np.zeros_like(r),
        rho_core_r=np.zeros_like(r), has_nlcc=False, n_proj=1,
        beta_r=beta_over_r, proj_l=np.asarray([0]),
        dij=np.ones((1, 1)))
    tables = radial_tables.build_all_tables(
        [species], q_max=0.6, n_q=61,
        second_derivatives=True, third_derivatives=True)
    q = tables["q"]
    amplitude = np.sqrt(np.pi) / (4.0 * a ** 1.5)
    expected_gppp = amplitude * np.exp(-q * q / (4.0 * a)) * (
        3.0 * q / (4.0 * a * a) - q**3 / (8.0 * a**3))
    np.testing.assert_allclose(
        tables["third_deriv_tables"][0][0], expected_gppp,
        rtol=3e-12, atol=3e-13)
    assert tables["third_deriv_tables"][0][0, 0] == 0.0

    # Full separable VNL third derivative, through the same bounded
    # coefficient scan, against a centered derivative of the incumbent
    # analytic contact action.
    setup = _setup(curved=False, third_derivatives=True)
    psi = _states()
    G = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = np.asarray([1.0, 1.0, 1.0, 0.0])
    k = np.asarray([0.19, -0.17, 0.12])
    Binv = np.linalg.inv(setup.B)
    jet = apply_icl_vnl_transfer_jet_to_ket(
        psi, G, k, setup, mask, projector_row_chunk=1, g_chunk=2,
        include_q2=True)
    third = 3.0 * np.asarray(jet.d2gamma_dq2_cart_ket)
    h = 2.0e-4
    third_fd = np.zeros_like(third)
    for c in range(3):
        dc = np.zeros(3)
        dc[c] = h
        plus = apply_uniform_vnl_derivatives_to_ket(
            psi, G, k + dc @ Binv, setup, mask,
            projector_row_chunk=1, g_chunk=2).lambda_cart_ket
        minus = apply_uniform_vnl_derivatives_to_ket(
            psi, G, k - dc @ Binv, setup, mask,
            projector_row_chunk=1, g_chunk=2).lambda_cart_ket
        third_fd[:, :, c] = (np.asarray(plus) - np.asarray(minus)) / (2.0 * h)
    np.testing.assert_allclose(third, third_fd, rtol=3e-7, atol=3e-7)

    # The retained q2 ICL Taylor jet obeys q.Gamma=V(k)-V(k-q) with a
    # quartic remainder, and its third Hamiltonian derivative is Hermitian.
    gamma0 = np.asarray(jet.gamma0_cart_ket)
    dgamma = np.asarray(jet.dgamma_dq_cart_ket)
    d2gamma = np.asarray(jet.d2gamma_dq2_cart_ket)
    direction = np.asarray([0.31, -0.47, 0.29])

    def ward_error(step):
        qv = step * direction
        gamma_q = (
            gamma0
            + np.einsum("b,abnsg->ansg", qv, dgamma)
            + 0.5 * np.einsum("b,c,abcnsg->ansg", qv, qv, d2gamma))
        lhs = np.einsum("a,ansg->nsg", qv, gamma_q)
        rhs = (
            _ordinary_vnl_action(psi, G, mask, k, setup)
            - _ordinary_vnl_action(
                psi, G, mask, k - qv @ Binv, setup))
        return np.linalg.norm(lhs - rhs)

    err_big = ward_error(4.0e-3)
    err_small = ward_error(2.0e-3)
    assert err_small < 0.07 * err_big  # quartic remainder: ideal ratio 1/16

    bra = np.conj(np.asarray(psi) * mask[None, None, :])
    d2gamma_mtx = np.einsum("msg,abcnsg->abcmn", bra, d2gamma)
    np.testing.assert_allclose(
        d2gamma_mtx, np.swapaxes(np.conj(d2gamma_mtx), -1, -2),
        rtol=3e-11, atol=3e-11)


def test_fixed_large_component_photon_vertex_jet_taylor_ward_and_hermiticity():
    setup = _setup(curved=False, third_derivatives=True)
    G = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = jnp.asarray([1.0, 1.0, 1.0, 0.0])
    psi_L = _states() * mask[None, None, :]
    k = np.asarray([0.19, -0.17, 0.12])
    B = np.asarray(setup.B)
    Binv = np.linalg.inv(B)
    psi_4 = lift_to_4spinor(
        psi_L[None], G[None], jnp.asarray(k[None]), jnp.asarray(B))[0]

    # Exercise the incumbent transaction's apply-to-ket packing.  The
    # negative adjoint below is the production unpack rule that converts
    # <Psi|alpha_i dPsi_a> into the bra-(k-q) endpoint derivative.
    geom = SimpleNamespace(ns=4, ngkmax=int(G.shape[0]))
    operator = mtxel_sweep.uniform_gauge_operator(
        geom, bvec=B, blat=1.0, vnl_setup=setup,
        include_transfer_q2=True)
    packed_action = np.moveaxis(np.asarray(operator.apply(
        psi_4[None], G, mask, jnp.zeros((1, 1, 1, 1), jnp.int32),
        jnp.asarray(k))), -1, 0)[:, 0]
    packed_mtx = np.einsum(
        "msg,xnsg->xmn", np.conj(np.asarray(psi_4)), packed_action,
        optimize=True)
    gamma0 = packed_mtx[:3]
    contact = packed_mtx[3:12].reshape(3, 3, 2, 2)
    q1_source = packed_mtx[12:21].reshape(3, 3, 2, 2)
    q1 = -np.swapaxes(np.conj(q1_source), -1, -2)
    q2_source = packed_mtx[21:].reshape(3, 3, 3, 2, 2)
    q2 = np.swapaxes(np.conj(q2_source), -1, -2)

    uniform = apply_uniform_vnl_derivatives_to_ket(
        psi_L, G, k, setup, mask, projector_row_chunk=1, g_chunk=2)
    lambda_kin = dft_operators.apply_kinetic_contact_to_ket(psi_L)
    contact_action = np.zeros((3, 3, 2, 4, 4), np.complex128)
    contact_action[:, :, :, :2] = HALFALPHA * np.asarray(
        lambda_kin + uniform.lambda_cart_ket)
    contact_direct = np.einsum(
        "msg,abnsg->abmn", np.conj(np.asarray(psi_4)), contact_action,
        optimize=True)
    # The production sweep and the direct oracle associate the same complex
    # contraction in different orders.  Require roundoff-level parity rather
    # than bit identity across those two contraction trees.
    np.testing.assert_allclose(
        contact, contact_direct, rtol=5e-15, atol=2e-19)

    @jax.jit
    def vnl_gamma_at(k_crys):
        return apply_uniform_vnl_derivatives_to_ket(
            psi_L, G, k_crys, setup, mask,
            projector_row_chunk=1, g_chunk=2).gamma_cart_ket

    gauss_x, gauss_w = np.polynomial.legendre.leggauss(6)
    segment_s = 0.5 * (gauss_x + 1.0)
    segment_w = 0.5 * gauss_w
    alpha_vertices = tuple(gamma_perm_phase(i) for i in (1, 2, 3))

    def direct_vertex(base_k, q_cart):
        q_crys = q_cart @ Binv
        ket = lift_to_4spinor(
            psi_L[None], G[None], jnp.asarray(base_k[None]), jnp.asarray(B))[0]
        bra = lift_to_4spinor(
            psi_L[None], G[None],
            jnp.asarray((base_k - q_crys)[None]), jnp.asarray(B))[0]
        gamma_kin = np.stack([
            np.asarray(gamma_apply(ket, perm, phase, axis=1))
            for perm, phase in alpha_vertices
        ])
        gamma_vnl = sum(
            weight * np.asarray(vnl_gamma_at(base_k - s * q_crys))
            for s, weight in zip(segment_s, segment_w))
        action = gamma_kin.copy()
        action[:, :, :2] += HALFALPHA * gamma_vnl
        return np.einsum(
            "msg,insg->imn", np.conj(np.asarray(bra)), action,
            optimize=True)

    direction = np.asarray([0.31, -0.47, 0.29])

    def taylor_error(step):
        q_cart = step * direction
        direct = direct_vertex(k, q_cart)
        taylor = (
            gamma0
            + np.einsum("a,iamn->imn", q_cart, q1)
            + 0.5 * np.einsum("a,b,iabmn->imn", q_cart, q_cart, q2))
        return np.linalg.norm(direct - taylor)

    err_big = taylor_error(4.0e-3)
    err_small = taylor_error(2.0e-3)
    assert err_small < 0.14 * err_big  # cubic remainder: ideal ratio 1/8

    # Direct ICL Ward identity, including the kinetic spin term.  The
    # antisymmetric Pauli product cancels after contraction with q_i q_a.
    q_cart = 3.0e-3 * direction
    q_crys = q_cart @ Binv
    direct = direct_vertex(k, q_cart)
    lhs = (2.0 / ALPHA_FS) * np.einsum("i,imn->mn", q_cart, direct)
    K = (np.asarray(G) + k[None]) @ B
    Kmq = K - q_cart[None]
    kinetic_diff = np.asarray(psi_L) * (
        np.sum(K * K, axis=1) - np.sum(Kmq * Kmq, axis=1))[None, None]
    vnl_diff = (
        _ordinary_vnl_action(psi_L, G, mask, k, setup)
        - _ordinary_vnl_action(psi_L, G, mask, k - q_crys, setup))
    rhs = np.einsum(
        "msg,nsg->mn", np.conj(np.asarray(psi_L)),
        kinetic_diff + vnl_diff, optimize=True)
    np.testing.assert_allclose(lhs, rhs, rtol=2e-11, atol=2e-11)

    reverse = direct_vertex(k - q_crys, -q_cart)
    np.testing.assert_allclose(
        direct, np.swapaxes(np.conj(reverse), -1, -2),
        rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(
        q2, np.swapaxes(np.conj(q2), -1, -2),
        rtol=3e-11, atol=3e-11)
    # The historical raw q2 action was exposed directly.  The public carrier
    # now gives both q1 and q2 the same source orientation and performs their
    # (-dagger,+dagger) rules at one band-matrix boundary.  Hermiticity makes
    # the raw physical value invariant under that cleanup.
    np.testing.assert_allclose(
        q2, q2_source, rtol=3e-11, atol=3e-11)


def test_isometric_photon_vertex_jet_matches_the_direct_kminusq_endpoint():
    setup = _setup(curved=False, third_derivatives=True)
    G = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = jnp.asarray([1.0, 1.0, 1.0, 0.0])
    source = _states() * mask[None, None, :]
    k = np.asarray([0.19, -0.17, 0.12])
    B = np.asarray(setup.B)
    Binv = np.linalg.inv(B)
    ket = lift_to_4spinor(
        source[None], G[None], jnp.asarray(k[None]), jnp.asarray(B),
        representation=ISOMETRIC_KINETIC_BALANCE_LIFT)[0]

    geom = SimpleNamespace(ns=4, ngkmax=int(G.shape[0]))
    operator = mtxel_sweep.uniform_gauge_operator(
        geom, bvec=B, blat=1.0, vnl_setup=setup,
        include_transfer_q2=True,
        kinetic_balance_lift=ISOMETRIC_KINETIC_BALANCE_LIFT)
    packed_action = np.moveaxis(np.asarray(operator.apply(
        ket[None], G, mask, jnp.zeros((1, 1, 1, 1), jnp.int32),
        jnp.asarray(k))), -1, 0)[:, 0]
    packed_mtx = np.einsum(
        "msg,xnsg->xmn", np.conj(np.asarray(ket)), packed_action,
        optimize=True)
    gamma0 = packed_mtx[:3]
    q1_source = packed_mtx[12:21].reshape(3, 3, 2, 2)
    q2_source = packed_mtx[21:].reshape(3, 3, 3, 2, 2)
    q1 = -np.swapaxes(np.conj(q1_source), -1, -2)
    q2 = np.swapaxes(np.conj(q2_source), -1, -2)

    @jax.jit
    def vnl_gamma_at(k_crys):
        return apply_uniform_vnl_derivatives_to_ket(
            ket[:, :2], G, k_crys, setup, mask,
            projector_row_chunk=1, g_chunk=2).gamma_cart_ket

    gauss_x, gauss_w = np.polynomial.legendre.leggauss(8)
    segment_s = 0.5 * (gauss_x + 1.0)
    segment_w = 0.5 * gauss_w
    alpha_vertices = tuple(gamma_perm_phase(i) for i in (1, 2, 3))

    def direct_vertex(q_cart):
        q_crys = q_cart @ Binv
        bra = lift_to_4spinor(
            source[None], G[None],
            jnp.asarray((k - q_crys)[None]), jnp.asarray(B),
            representation=ISOMETRIC_KINETIC_BALANCE_LIFT)[0]
        gamma_kin = np.stack([
            np.asarray(gamma_apply(ket, perm, phase, axis=1))
            for perm, phase in alpha_vertices
        ])
        gamma_vnl = sum(
            weight * np.asarray(vnl_gamma_at(k - s * q_crys))
            for s, weight in zip(segment_s, segment_w))
        action = gamma_kin.copy()
        action[:, :, :2] += HALFALPHA * gamma_vnl
        return np.einsum(
            "msg,insg->imn", np.conj(np.asarray(bra)), action,
            optimize=True)

    # The differentiated endpoint is K_bra=k-q: q-first carries one minus,
    # q-second carries no sign.  These central differences independently pin
    # both signs before the directional Taylor discriminator below.
    step = 2.0e-4
    for a in range(3):
        direction = np.zeros(3)
        direction[a] = 1.0
        first_fd = (
            direct_vertex(step * direction)
            - direct_vertex(-step * direction)) / (2.0 * step)
        np.testing.assert_allclose(
            q1[:, a], first_fd, rtol=2.0e-7, atol=3.0e-9)

    direction = np.asarray([0.31, -0.47, 0.29])

    def taylor_error(scale):
        q_cart = scale * direction
        direct = direct_vertex(q_cart)
        taylor = (
            gamma0
            + np.einsum("a,iamn->imn", q_cart, q1)
            + 0.5 * np.einsum(
                "a,b,iabmn->imn", q_cart, q_cart, q2))
        return np.linalg.norm(direct - taylor)

    err_big = taylor_error(4.0e-3)
    err_small = taylor_error(2.0e-3)
    assert err_small < 0.14 * err_big  # cubic remainder: ideal ratio 1/8


def test_exact_icl_finite_transfer_matches_q0_and_landed_q2_jet():
    setup = _finite_setup(third_derivatives=True)
    psi = _states()
    G = np.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = np.asarray([1.0, 1.0, 1.0, 0.0])
    k = np.asarray([0.19, -0.17, 0.12])

    q0 = compute_icl_vnl_finite_transfer_to_ket(
        psi, G, G, k, k, np.zeros(3), np.zeros(3, dtype=np.int32),
        setup, mask, mask, path_order=8,
        projector_row_chunk=1, g_chunk=2)
    uniform = apply_uniform_vnl_derivatives_to_ket(
        psi, G, k, setup, mask, projector_row_chunk=1, g_chunk=2)
    assert bool(np.asarray(q0.certified))
    np.testing.assert_allclose(
        q0.gamma_cart_ket, uniform.gamma_cart_ket,
        rtol=3.0e-13, atol=3.0e-13)

    # The finite-transfer owner and the long-wave owner meet at the existing
    # q2 jet.  q is supplied in crystal coordinates; jet derivatives contract
    # the corresponding Cartesian q exactly once.
    q_cart = 1.5e-3 * np.asarray([0.31, -0.47, 0.29])
    q_crys = q_cart @ np.linalg.inv(setup.B)
    finite = compute_icl_vnl_finite_transfer_to_ket(
        psi, G, G, k, k - q_crys, q_crys,
        np.zeros(3, dtype=np.int32), setup, mask, mask,
        path_order=12, path_rtol=1.0e-11, path_atol=1.0e-13,
        projector_row_chunk=1, g_chunk=2)
    jet = apply_icl_vnl_transfer_jet_to_ket(
        psi, G, k, setup, mask, projector_row_chunk=1, g_chunk=2,
        include_q2=True)
    expected = (
        np.asarray(jet.gamma0_cart_ket)
        + np.einsum(
            "b,abnsg->ansg", q_cart,
            np.asarray(jet.dgamma_dq_cart_ket))
        + 0.5 * np.einsum(
            "b,c,abcnsg->ansg", q_cart, q_cart,
            np.asarray(jet.d2gamma_dq2_cart_ket)))
    assert bool(np.asarray(finite.certified))
    np.testing.assert_allclose(
        finite.gamma_cart_ket, expected, rtol=2.0e-9, atol=2.0e-10)


def test_exact_icl_finite_contact_q0_and_two_photon_ward():
    setup = _finite_setup()
    psi = _states()
    G = np.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = np.asarray([1.0, 1.0, 1.0, 0.0])
    k = np.asarray([0.19, -0.17, 0.12])

    q0 = compute_icl_vnl_finite_contact_to_ket(
        psi, G, k, np.zeros(3), setup, mask, path_order=8,
        projector_row_chunk=1, g_chunk=2)
    uniform = apply_uniform_vnl_derivatives_to_ket(
        psi, G, k, setup, mask, projector_row_chunk=1, g_chunk=2)
    assert bool(np.asarray(q0.certified))
    np.testing.assert_allclose(
        q0.lambda_cart_ket, uniform.lambda_cart_ket,
        rtol=3.0e-13, atol=3.0e-13)

    # Contracting the first photon leg removes it from the Wilson path.
    # The remaining two one-photon vertices fix the sign and unit prefactor:
    # q_a Lambda_ab(k;q,-q) = Gamma_b(k,-q)-Gamma_b(k-q,-q).
    q = np.asarray([0.017, -0.011, 0.009])
    contact = compute_icl_vnl_finite_contact_to_ket(
        psi, G, k, q, setup, mask, path_order=12,
        path_rtol=2.0e-10, path_atol=1.0e-12,
        projector_row_chunk=1, g_chunk=2)
    gamma_plus = compute_icl_vnl_finite_transfer_to_ket(
        psi, G, G, k, k + q, -q, np.zeros(3, dtype=np.int32),
        setup, mask, mask, path_order=12,
        path_rtol=2.0e-10, path_atol=1.0e-12,
        projector_row_chunk=1, g_chunk=2)
    gamma_minus = compute_icl_vnl_finite_transfer_to_ket(
        psi, G, G, k - q, k, -q, np.zeros(3, dtype=np.int32),
        setup, mask, mask, path_order=12,
        path_rtol=2.0e-10, path_atol=1.0e-12,
        projector_row_chunk=1, g_chunk=2)
    assert bool(np.asarray(contact.certified))
    assert bool(np.asarray(gamma_plus.certified))
    assert bool(np.asarray(gamma_minus.certified))
    q_cart = q @ setup.B
    lhs = np.einsum("a,abnsg->bnsg", q_cart, contact.lambda_cart_ket)
    rhs = gamma_plus.gamma_cart_ket - gamma_minus.gamma_cart_ket
    np.testing.assert_allclose(lhs, rhs, rtol=4.0e-10, atol=4.0e-11)
    assert (contact.vnl_path_operator_fingerprint
            != gamma_plus.vnl_path_operator_fingerprint)


def test_uniform_contact_and_paramagnetic_sum_have_fsum_prefactor():
    """Hellmann--Feynman curvature fixes contact/bubble sign and factor 2."""
    setup = _finite_setup()
    G = np.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1]])
    mask = np.ones(G.shape[0], dtype=np.float64)
    k = np.asarray([0.19, -0.17, 0.12])
    B = np.asarray(setup.B)
    Binv = np.linalg.inv(B)
    dim = 2 * G.shape[0]
    basis = np.eye(dim, dtype=np.complex128).reshape(dim, 2, G.shape[0])

    def operator_matrix(actions):
        return np.asarray(actions).reshape(dim, dim).T

    def hamiltonian(k_value):
        K = (G + k_value[None, :]) @ B
        kinetic = basis * np.sum(K * K, axis=1)[None, None, :]
        kdata = build_vnl_kdata_from_kvec(k_value, G, setup)
        nonlocal_action = vnl_ops.apply_vnl(
            basis, kdata.Z, kdata.E_super)
        result = operator_matrix(kinetic + np.asarray(nonlocal_action))
        np.testing.assert_allclose(
            result, np.conj(result.T), rtol=3.0e-13, atol=3.0e-13)
        return result

    K = (G + k[None, :]) @ B
    uniform = apply_uniform_vnl_derivatives_to_ket(
        basis, G, k, setup, mask, projector_row_chunk=1, g_chunk=2)
    q0_contact = compute_icl_vnl_finite_contact_to_ket(
        basis, G, k, np.zeros(3), setup, mask, path_order=8,
        projector_row_chunk=1, g_chunk=2)
    kinetic_gamma = np.stack([
        basis * (2.0 * K[:, a])[None, None, :] for a in range(3)
    ])
    kinetic_contact = np.zeros(
        (3, 3, dim, 2, G.shape[0]), dtype=np.complex128)
    for a in range(3):
        kinetic_contact[a, a] = 2.0 * basis
    gamma = kinetic_gamma + np.asarray(uniform.gamma_cart_ket)
    contact = kinetic_contact + np.asarray(q0_contact.lambda_cart_ket)

    direction = np.asarray([0.37, -0.51, 0.28])
    direction /= np.linalg.norm(direction)
    gamma_direction = operator_matrix(np.einsum(
        "a,anSG->nSG", direction, gamma))
    contact_direction = operator_matrix(np.einsum(
        "a,b,abnSG->nSG", direction, direction, contact))
    energies, states = np.linalg.eigh(hamiltonian(k))
    assert energies[1] - energies[0] > 1.0e-4
    gamma_eigen = np.conj(states.T) @ gamma_direction @ states
    contact_eigen = np.conj(states.T) @ contact_direction @ states
    response_curvature = (
        np.real(contact_eigen[0, 0])
        - 2.0 * np.sum(
            np.abs(gamma_eigen[1:, 0]) ** 2
            / (energies[1:] - energies[0])))

    step = 3.0e-4
    dk = step * direction @ Binv
    curvature_fd = (
        np.linalg.eigvalsh(hamiltonian(k + dk))[0]
        - 2.0 * energies[0]
        + np.linalg.eigvalsh(hamiltonian(k - dk))[0]) / step**2
    np.testing.assert_allclose(
        response_curvature, curvature_fd, rtol=2.0e-5, atol=2.0e-6)


def test_exact_icl_finite_contact_centered_q2_coefficient():
    setup = _finite_setup()
    psi = _states()
    G = np.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = np.asarray([1.0, 1.0, 1.0, 0.0])
    k = np.asarray([0.19, -0.17, 0.12])
    Binv = np.linalg.inv(setup.B)
    direction = np.asarray([0.37, -0.51, 0.28])
    direction /= np.linalg.norm(direction)
    step = 3.0e-3
    q_cart = step * direction
    q = q_cart @ Binv

    common = dict(
        setup=setup, g_mask=mask, path_order=10,
        projector_row_chunk=1, g_chunk=2)
    plus = compute_icl_vnl_finite_contact_to_ket(
        psi, G, k, q, **common)
    zero = compute_icl_vnl_finite_contact_to_ket(
        psi, G, k, np.zeros(3), **common)
    minus = compute_icl_vnl_finite_contact_to_ket(
        psi, G, k, -q, **common)
    np.testing.assert_allclose(
        plus.lambda_cart_ket, minus.lambda_cart_ket,
        rtol=3.0e-13, atol=3.0e-13)
    contact_curvature = (
        np.asarray(plus.lambda_cart_ket)
        - 2.0 * np.asarray(zero.lambda_cart_ket)
        + np.asarray(minus.lambda_cart_ket)) / step**2

    lambda_plus = apply_uniform_vnl_derivatives_to_ket(
        psi, G, k + q, setup, mask,
        projector_row_chunk=1, g_chunk=2).lambda_cart_ket
    lambda_minus = apply_uniform_vnl_derivatives_to_ket(
        psi, G, k - q, setup, mask,
        projector_row_chunk=1, g_chunk=2).lambda_cart_ket
    lambda_curvature = (
        np.asarray(lambda_plus)
        - 2.0 * np.asarray(zero.lambda_cart_ket)
        + np.asarray(lambda_minus)) / step**2
    # E[(s-t)^2] = 1/6 for two independent unit-interval path points.
    np.testing.assert_allclose(
        contact_curvature, lambda_curvature / 6.0,
        rtol=4.0e-4, atol=2.0e-5)


def test_exact_icl_finite_transfer_ward_wrap_and_hermiticity():
    setup = _finite_setup()
    G = np.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1]])
    mask = np.ones(G.shape[0], dtype=np.float64)
    k = np.asarray([0.45, -0.11, 0.08])
    q = np.asarray([-0.22, 0.03, -0.04])
    wrap = np.asarray([1, 0, 0], dtype=np.int32)
    k_target = k - q - wrap

    # Use a complete spin/G basis so the returned actions are the rectangular
    # operator matrices themselves, not a state-dependent Hermiticity probe.
    dim = 2 * G.shape[0]
    basis = np.eye(dim, dtype=np.complex128).reshape(dim, 2, G.shape[0])
    forward = compute_icl_vnl_finite_transfer_to_ket(
        basis, G, G, k, k_target, q, wrap, setup, mask, mask,
        path_order=12, path_rtol=2.0e-10, path_atol=1.0e-12,
        projector_row_chunk=1, g_chunk=2)
    assert bool(np.asarray(forward.certified))

    q_cart = q @ setup.B
    lhs = np.einsum("a,ansg->nsg", q_cart, forward.gamma_cart_ket)
    G_target_unwrapped = G - wrap[None, :]
    rhs = (
        _rectangular_vnl_action(
            basis, G, mask, G_target_unwrapped, mask, k, setup)
        - _rectangular_vnl_action(
            basis, G, mask, G_target_unwrapped, mask, k - q, setup))
    np.testing.assert_allclose(lhs, rhs, rtol=3.0e-10, atol=3.0e-11)

    reverse = compute_icl_vnl_finite_transfer_to_ket(
        basis, G, G, k_target, k, -q, -wrap, setup, mask, mask,
        path_order=12, path_rtol=2.0e-10, path_atol=1.0e-12,
        projector_row_chunk=1, g_chunk=2)
    assert bool(np.asarray(reverse.certified))

    def operator_matrices(result):
        values = np.asarray(result.gamma_cart_ket)
        return np.transpose(values, (0, 2, 3, 1)).reshape(3, dim, dim)

    forward_mtx = operator_matrices(forward)
    reverse_mtx = operator_matrices(reverse)
    np.testing.assert_allclose(
        forward_mtx, np.swapaxes(np.conj(reverse_mtx), -1, -2),
        rtol=4.0e-10, atol=4.0e-11)


def test_exact_icl_finite_transfer_rejects_noncanonical_wrap_and_stamps_rule():
    setup = _finite_setup()
    psi = _states()
    G = np.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0]])
    mask = np.asarray([1.0, 1.0, 1.0, 0.0])
    k = np.asarray([0.19, -0.17, 0.12])
    q = np.asarray([0.07, 0.02, -0.03])
    invalid = compute_icl_vnl_finite_transfer_to_ket(
        psi, G, G, k, k - q, q, np.asarray([1, 0, 0]),
        setup, mask, mask, path_order=8,
        projector_row_chunk=1, g_chunk=2)
    assert not bool(np.asarray(invalid.certified))

    first = vnl_ops.icl_vnl_finite_transfer_operator_fingerprint(
        setup, path_order=8, path_rtol=1.0e-9, path_atol=1.0e-12)
    second = vnl_ops.icl_vnl_finite_transfer_operator_fingerprint(
        setup, path_order=12, path_rtol=1.0e-9, path_atol=1.0e-12)
    third = vnl_ops.icl_vnl_finite_transfer_operator_fingerprint(
        setup, path_order=8, path_rtol=2.0e-9, path_atol=1.0e-12)
    assert first.startswith("sha256:") and len(first) == 71
    assert len({first, second, third}) == 3


