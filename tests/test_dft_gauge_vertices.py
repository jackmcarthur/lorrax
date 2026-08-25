"""Five deterministic first-principles gates for uniform VNL gauge actions."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import jax.numpy as jnp

from psp import vnl_ops
from psp import dft_operators, radial_tables
from psp.species import SpeciesData
from psp.vnl_ops import (
    ChannelMeta,
    VNLSetup,
    apply_uniform_vnl_derivatives_to_ket,
    build_vnl_kdata_from_kvec,
)


_EXPECTED_SRC = Path(__file__).resolve().parents[1] / "src"
for _module in (vnl_ops, dft_operators, radial_tables):
    if _EXPECTED_SRC not in Path(_module.__file__).resolve().parents:
        raise RuntimeError(
            f"split-source gate: {_module.__name__} imported from "
            f"{_module.__file__}, expected {_EXPECTED_SRC}")


def _setup(*, curved: bool, natoms: int = 1) -> VNLSetup:
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
        prefactor=0.67, B=B, cell_volume=1.0,
        total_R=natoms, nspinor=2, E_super=jnp.asarray(E_super), l_max=0,
        soc=True,
        row_beta_idx=jnp.zeros(natoms, dtype=jnp.int32),
        row_l=jnp.zeros(natoms, dtype=jnp.int32),
        row_m=jnp.zeros(natoms, dtype=jnp.int32),
        row_tau=jnp.asarray(tau),
        coupled_row_blocks=tuple((ia, ia + 1, 0) for ia in range(natoms)),
    )


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


def test_exact_origin_radial_moment_and_contact_action():
    r = np.linspace(0.0, 4.0, 101)
    dr = r[1] - r[0]
    beta_over_r = np.exp(-r * r)[None, :]
    species = SpeciesData(
        element="H", z_valence=1.0, z_atomic=1, r=r,
        rab=np.full_like(r, dr), vloc_r=np.zeros_like(r),
        rho_core_r=np.zeros_like(r), has_nlcc=False, n_proj=1,
        beta_r=beta_over_r, proj_l=np.asarray([0]),
        proj_j=np.asarray([0.5]), dij=np.ones((1, 1)), nspinor=1)
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
