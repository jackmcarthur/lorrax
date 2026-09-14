import numpy as np
import pytest

from gw.photon_sigma import (
    contract_instantaneous_tt_head_exchange,
    contract_ordered_tt_head_sigma,
)


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


def _explicit(gamma, matrices, weights, mask, nk_tot, volume, sign):
    nk, _, ne, nm = gamma.shape
    out = np.zeros((nk, ne), dtype=np.complex128)
    for k in range(nk):
        for n in range(ne):
            for m in range(nm):
                for a in range(3):
                    for b in range(3):
                        out[k, n] += sign * (
                            mask[k, m] * weights[k, m]
                            * gamma[k, a, n, m] * matrices[k, n, m, a, b]
                            * np.conj(gamma[k, b, n, m]) / (nk_tot * volume)
                        )
    return out


def test_ordered_tt_head_sigma_selects_green_pole_branch():
    rng = np.random.default_rng(2026091401)
    nk, ne, nm = 2, 2, 4
    gamma = rng.normal(size=(nk, 3, ne, nm)) + 1j * rng.normal(
        size=(nk, 3, ne, nm))
    plus = rng.normal(size=(nk, ne, nm, 3, 3)) + 1j * rng.normal(
        size=(nk, ne, nm, 3, 3))
    minus = rng.normal(size=(nk, ne, nm, 3, 3)) + 1j * rng.normal(
        size=(nk, ne, nm, 3, 3))
    occupations = np.asarray(((1.0, 0.25, 0.0, 1.0),
                              (0.0, 1.0, 0.75, 0.0)))
    mask = np.asarray(((1.0, 1.0, 0.0, 1.0),
                       (1.0, 0.0, 1.0, 1.0)))
    ordered = (occupations[:, None, :, None, None] * minus
               + (1.0 - occupations[:, None, :, None, None]) * plus)
    want = _explicit(
        gamma, ordered, np.ones_like(occupations), mask, nk_tot=nk,
        volume=11.0, sign=1.0)
    got = np.asarray(contract_ordered_tt_head_sigma(
        gamma, plus, minus, occupations, nk_tot=nk, cell_volume_bohr3=11.0,
        intermediate_mask_km=mask))
    np.testing.assert_allclose(got, want, rtol=2e-14, atol=2e-14)


def test_instantaneous_tt_head_exchange_is_occupied_only():
    rng = np.random.default_rng(2026091402)
    nk, ne, nm = 2, 3, 5
    gamma = rng.normal(size=(nk, 3, ne, nm)) + 1j * rng.normal(
        size=(nk, 3, ne, nm))
    interaction = rng.normal(size=(nk, 3, 3)) + 1j * rng.normal(
        size=(nk, 3, 3))
    occupations = np.asarray(((1, 1, 0, 0, 0), (1, 0, 1, 0, 0)), float)
    matrices = np.broadcast_to(
        interaction[:, None, None, :, :], (nk, ne, nm, 3, 3))
    want = _explicit(
        gamma, matrices, occupations, np.ones_like(occupations), nk_tot=7,
        volume=13.0, sign=-1.0)
    got = np.asarray(contract_instantaneous_tt_head_exchange(
        gamma, interaction, occupations, nk_tot=7,
        cell_volume_bohr3=13.0))
    np.testing.assert_allclose(got, want, rtol=2e-14, atol=2e-14)


def test_tt_head_sigma_refuses_shape_and_normalization_ambiguity():
    gamma = np.zeros((1, 3, 1, 2), dtype=np.complex128)
    ordered = np.zeros((1, 1, 2, 3, 3), dtype=np.complex128)
    occupations = np.ones((1, 2))
    with pytest.raises(ValueError, match="nk_tot must be positive"):
        contract_ordered_tt_head_sigma(
            gamma, ordered, ordered, occupations, nk_tot=0,
            cell_volume_bohr3=1.0)
    with pytest.raises(ValueError, match="ordered TT functions"):
        contract_ordered_tt_head_sigma(
            gamma, ordered[..., :2, :2], ordered, occupations, nk_tot=1,
            cell_volume_bohr3=1.0)


def test_instantaneous_contract_matches_production_gamma_on_green_axes():
    """Directly gate the gamma order used by _make_lorentz_convolution."""
    from common.gamma_matrices import gamma_apply, gamma_perm_phase

    rng = np.random.default_rng(2026091403)
    nb = 5
    psi = rng.normal(size=(nb, 4)) + 1j * rng.normal(size=(nb, 4))
    occupations = np.asarray((1.0, 1.0, 0.0, 0.0, 0.0))
    interaction = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))

    vertices = []
    for a in range(3):
        perm, phase = gamma_perm_phase(a + 1)
        gamma_psi = np.asarray(gamma_apply(psi, perm, phase, axis=1))
        vertices.append(np.einsum("ns,ms->nm", np.conj(psi), gamma_psi))
    gamma_ext_m = np.asarray(vertices)[None, :, :, :]

    # This is the q0 branch of the production convolution: build occupied
    # G, apply gamma_A on its left spin axis and conjugated gamma_B on its
    # right spin axis, then project through the unchanged endpoints.
    green = np.einsum("ms,m,mt->st", psi, occupations, np.conj(psi))
    sigma_spin = np.zeros((4, 4), dtype=np.complex128)
    for a in range(3):
        left = gamma_apply(green, *gamma_perm_phase(a + 1), axis=0)
        for b in range(3):
            perm, phase = gamma_perm_phase(b + 1)
            block = gamma_apply(left, perm, np.conj(phase), axis=1)
            sigma_spin += interaction[a, b] * np.asarray(block)
    projected = -np.einsum(
        "ns,st,nt->n", np.conj(psi), sigma_spin, psi) / (17.0 * 3.0)
    got = np.asarray(contract_instantaneous_tt_head_exchange(
        gamma_ext_m, interaction, occupations[None], nk_tot=3,
        cell_volume_bohr3=17.0))[0]
    np.testing.assert_allclose(got, projected, rtol=3e-14, atol=3e-14)


def test_ordered_scalar_reduction_has_iGW_correlation_sign_and_one_volume():
    gamma = np.zeros((1, 3, 1, 2), dtype=np.complex128)
    gamma[0, 0, 0] = (2.0, 3.0)
    plus = np.zeros((1, 1, 2, 3, 3), dtype=np.complex128)
    minus = np.zeros_like(plus)
    plus[0, 0, 1, 0, 0] = 5.0
    minus[0, 0, 0, 0, 0] = 7.0
    occupation = np.asarray(((1.0, 0.0),))
    got = np.asarray(contract_ordered_tt_head_sigma(
        gamma, plus, minus, occupation, nk_tot=3,
        cell_volume_bohr3=11.0))[0, 0]
    # Occupied m=0 selects F_minus, empty m=1 selects F_plus.  Positive
    # iGW correlation sign and exactly one Omega*Nk denominator.
    want = (4.0 * 7.0 + 9.0 * 5.0) / (11.0 * 3.0)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=2e-15)
