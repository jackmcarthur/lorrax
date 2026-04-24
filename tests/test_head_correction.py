import numpy as np

from gw.head_correction import (
    compute_ppm_head_sigma_kij,
    compute_static_head_terms,
    fit_head_gn,
    resolve_head_override,
    static_head_terms_to_kij,
)


def test_compute_static_head_terms_matches_cohsex_formulas():
    head = compute_static_head_terms(
        vc0=12.0 + 0.0j,
        wcoul0_static=5.0 + 0.0j,
        occ=np.array([True, True, False, False]),
        cell_volume=3.0,
        nk_tot=2,
        source="unit_test",
    )

    pref = 1.0 / (3.0 * 2.0)
    np.testing.assert_allclose(
        np.asarray(head.sigma_x_diag),
        np.array([-12.0 * pref, -12.0 * pref, 0.0, 0.0], dtype=np.complex128),
    )
    np.testing.assert_allclose(
        np.asarray(head.sigma_sx_diag),
        np.array([-5.0 * pref, -5.0 * pref, 0.0, 0.0], dtype=np.complex128),
    )
    np.testing.assert_allclose(
        np.asarray(head.sigma_sx_minus_x_diag),
        np.array([7.0 * pref, 7.0 * pref, 0.0, 0.0], dtype=np.complex128),
    )
    np.testing.assert_allclose(
        np.asarray(head.sigma_coh_diag),
        np.array([-3.5 * pref, -3.5 * pref, -3.5 * pref, -3.5 * pref], dtype=np.complex128),
    )


def test_static_head_terms_to_kij_broadcasts_diagonal_heads():
    head = compute_static_head_terms(
        vc0=8.0 + 0.0j,
        wcoul0_static=2.0 + 0.0j,
        occ=np.array([True, False, False]),
        cell_volume=4.0,
        nk_tot=3,
        source="unit_test",
    )

    sx_kij, coh_kij = static_head_terms_to_kij(head, nk_tot=3, do_screened=True)
    x_kij, _ = static_head_terms_to_kij(head, nk_tot=3, do_screened=False)

    expected_sx = np.diag(np.array([-2.0 / 12.0, 0.0, 0.0], dtype=np.complex128))
    expected_x = np.diag(np.array([-8.0 / 12.0, 0.0, 0.0], dtype=np.complex128))
    expected_coh = np.diag(np.array([-3.0 / 12.0, -3.0 / 12.0, -3.0 / 12.0], dtype=np.complex128))

    np.testing.assert_allclose(np.asarray(sx_kij), np.broadcast_to(expected_sx, (3, 3, 3)))
    np.testing.assert_allclose(np.asarray(x_kij), np.broadcast_to(expected_x, (3, 3, 3)))
    np.testing.assert_allclose(np.asarray(coh_kij), np.broadcast_to(expected_coh, (3, 3, 3)))


def test_compute_ppm_head_sigma_static_limit_matches_cohsex():
    # PPM Σ^c head reduces to the COHSEX static-head pieces (Σ^{SX-X} + Σ^COH)
    # in the limit ω = ε_n with a small η.  Use any pole frequency Ω_h > 0.
    # Physically W^c decays toward zero as |ω|→∞, so |W^c(iω_p)| < |W^c(0)|.
    # The two-sample GN fit needs that ordering for Ω_h² > 0.
    vc0 = 12.0
    wcoul0_static = 2.0      # → W^c(0) = -10
    wcoul0_imfreq = 7.0      # → W^c(iω_p) = -5
    omega_p_ry = 1.5

    head = fit_head_gn(
        vc0=vc0,
        wcoul0_static=wcoul0_static,
        wcoul0_imfreq=wcoul0_imfreq,
        omega_p_ry=omega_p_ry,
    )
    assert head.omega_h_sq > 0.0  # well-defined real pole

    cell_volume = 3.0
    nk_tot = 2
    nb = 4
    n_occ = 2
    enk = np.array([
        [-1.0, -0.5, +0.5, +1.5],
        [-0.9, -0.4, +0.6, +1.6],
    ], dtype=np.float64)
    efermi = 0.0

    # Evaluate exactly at each ε_n (one ω per (k, n) for the diagonal check)
    omega_grid = np.unique(enk - efermi)
    sigma_kij = compute_ppm_head_sigma_kij(
        head,
        omega_grid_ry=omega_grid,
        enk_ry=enk,
        efermi_ry=efermi,
        n_occ=n_occ,
        cell_volume=cell_volume,
        nk_tot=nk_tot,
        eta=1.0e-9,
    )
    # Off-diagonals are zero
    diag_mask = np.eye(nb, dtype=bool)
    np.testing.assert_allclose(
        sigma_kij[:, :, ~diag_mask],
        np.zeros_like(sigma_kij[:, :, ~diag_mask]),
        atol=0.0,
    )
    # COHSEX static-head limit per band, for the (Σ^{SX-X} + Σ^COH) head pieces:
    #   occupied: -0.5 W^c(0) / (V N_k);  empty: +0.5 W^c(0) / (V N_k).
    wc0 = wcoul0_static - vc0
    pref = 1.0 / (cell_volume * nk_tot)
    expected_occ = -0.5 * wc0 * pref
    expected_emp = +0.5 * wc0 * pref
    # Pick the diagonal entry where ω == ε_{k,n}.  For each (k,n), find the
    # corresponding ω-index and check the sigma_diag value.
    for ik in range(enk.shape[0]):
        for ib in range(nb):
            iw = int(np.argmin(np.abs(omega_grid - (enk[ik, ib] - efermi))))
            target = expected_occ if ib < n_occ else expected_emp
            np.testing.assert_allclose(
                sigma_kij[iw, ik, ib, ib].real, target, rtol=1.0e-4
            )


def test_compute_ppm_head_sigma_zero_when_pole_degenerate():
    # If the GN fit gives a degenerate (zero) residue, head Σ^c must vanish.
    head = fit_head_gn(vc0=10.0, wcoul0_static=10.0, wcoul0_imfreq=10.0, omega_p_ry=1.0)
    sigma_kij = compute_ppm_head_sigma_kij(
        head,
        omega_grid_ry=np.linspace(-1.0, 1.0, 5),
        enk_ry=np.zeros((2, 3), dtype=np.float64),
        efermi_ry=0.0,
        n_occ=1,
        cell_volume=2.0,
        nk_tot=4,
    )
    np.testing.assert_array_equal(sigma_kij, np.zeros_like(sigma_kij))


def test_resolve_head_override_uses_frequency_specific_whead():
    params = {
        "vhead": 100.0,
        "whead_0freq": 25.0,
        "whead_imfreq": 40.0,
    }

    static = resolve_head_override(params, 0.0 + 0.0j)
    imag = resolve_head_override(params, 1j * 2.0)

    assert static is not None
    assert imag is not None
    assert static.vc0 == complex(100.0)
    assert static.wcoul0 == complex(25.0)
    assert static.source == "override"
    assert imag.vc0 == complex(100.0)
    assert imag.wcoul0 == complex(40.0)
    assert "override(" in imag.source
