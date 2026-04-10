import numpy as np

from gw.head_correction import (
    compute_static_head_terms,
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
