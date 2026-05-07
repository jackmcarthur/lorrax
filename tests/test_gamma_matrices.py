import numpy as np

from common.gamma_matrices import (
    gamma_left_perms,
    gamma_left_phases,
    gamma_right_perms,
    gamma_right_phases,
    gammas,
)


def test_gamma_left_permutation_matches_dense_matrix():
    psi = np.arange(20, dtype=np.float64).reshape(5, 4)
    psi = psi + 1j * (psi + 0.25)

    for mu, gamma in enumerate(gammas):
        dense = np.einsum("ab,xb->xa", np.asarray(gamma), psi)
        perm = np.asarray(gamma_left_perms[mu])
        phase = np.asarray(gamma_left_phases[mu])
        permuted = phase[None, :] * psi[:, perm]
        np.testing.assert_allclose(permuted, dense)


def test_gamma_right_permutation_matches_dense_matrix():
    G = np.arange(24, dtype=np.float64).reshape(6, 4)
    G = G + 1j * (0.5 - G)

    for mu, gamma in enumerate(gammas):
        dense = np.einsum("xg,gd->xd", G, np.asarray(gamma))
        perm = np.asarray(gamma_right_perms[mu])
        phase = np.asarray(gamma_right_phases[mu])
        permuted = phase[None, :] * G[:, perm]
        np.testing.assert_allclose(permuted, dense)


def test_two_sided_gamma_permutation_matches_lorentz_einsum():
    G = np.arange(2 * 4 * 3 * 4 * 5, dtype=np.float64).reshape(2, 4, 3, 4, 5)
    G = G + 1j * (G / 7.0 - 3.0)

    for mu, gamma_mu in enumerate(gammas):
        for nu, gamma_nu in enumerate(gammas):
            dense = np.einsum(
                "ab,kbxgy,gd->kaxdy",
                np.asarray(gamma_mu),
                G,
                np.asarray(gamma_nu),
            )
            left_perm = np.asarray(gamma_left_perms[mu])
            left_phase = np.asarray(gamma_left_phases[mu])
            right_perm = np.asarray(gamma_right_perms[nu])
            right_phase = np.asarray(gamma_right_phases[nu])

            permuted = np.take(G, left_perm, axis=1)
            permuted = permuted * left_phase[None, :, None, None, None]
            permuted = np.take(permuted, right_perm, axis=3)
            permuted = permuted * right_phase[None, None, None, :, None]
            np.testing.assert_allclose(permuted, dense)
