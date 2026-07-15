"""GN-PPM mode-classification contract (Fix-3 census determinism).

Pins the four-class behavior of ``fit_gn_ppm_from_wc_pair``:

- valid       — resolvable dispersion, Ω² > 0; recovers the exact model pole.
- dead        — |Wc0| at roundoff; deterministically invalid (noise/noise Ω²
                must never enter the valid census).
- stiff       — |Wc0 − Wc_probe| ≤ 1e-8·|Wc0|; invalid.  Under the OLD
                absolute cut (|denom| > 1e-14) these fit garbage-huge Ω that
                polluted the window max-Ω statistic and flipped valid↔invalid
                with device-count reduction-order noise.
- Ω² ≤ 0      — one-pole ansatz failure (element grows toward the probe).

Plus the determinism contract itself: a ±1-ulp perturbation of the inputs
must not change any mode's classification (the whole point of the fix).
Runs on any platform, single device.
"""

import numpy as np

from gw.minimax_screening import fit_gn_ppm_from_wc_pair

PROBE = 2.0j          # standard GN probe, ω̄ = 2 Ry
FALLBACK = 2.0
N_PAD = 8             # padded extent
N_LOG = 6             # logical extent (rows/cols 6,7 are pad)

# Model element: Wc(iω) = Wc0·Ω²/(Ω²+ω̄²).  Wc0 = −1, Ω = 1, ω̄ = 2 → Wc_probe = −0.2.
W0_VALID, WP_VALID, OMEGA_TRUE = -1.0, -0.2, 1.0

SPECIALS = {
    # (i, j): (Wc0, Wc_probe, expected_valid)
    (0, 1): (-0.5, -0.5 * (1.0 - 1.0e-10), False),   # stiff: rel. dispersion 1e-10
    (1, 0): (-0.5, -0.5 + 1.0e-14, False),           # stiff AND straddles the old
                                                     #   absolute 1e-14 cut (the ulp-flipper)
    (2, 3): (1.0e-17, 3.0e-17, False),               # dead: roundoff element
    (4, 5): (-0.1, -0.2, False),                     # Ω² < 0: grows toward probe
}


def _build_pair():
    w0 = np.full((1, 1, 1, N_PAD, N_PAD), W0_VALID, dtype=np.complex128)
    wp = np.full((1, 1, 1, N_PAD, N_PAD), WP_VALID, dtype=np.complex128)
    for (i, j), (a, b, _) in SPECIALS.items():
        w0[..., i, j] = a
        wp[..., i, j] = b
    # Junk in the pad block — must be born dead regardless of value.
    w0[..., N_LOG:, :] = 5.0
    w0[..., :, N_LOG:] = 5.0
    wp[..., N_LOG:, :] = 4.0
    wp[..., :, N_LOG:] = 4.0
    return w0, wp


def _fit(w0, wp):
    omega, B, valid, unful = fit_gn_ppm_from_wc_pair(
        w0, wp, PROBE, fallback_omega=FALLBACK, n_mu_logical=N_LOG)
    return np.asarray(omega), np.asarray(B), np.asarray(valid), float(unful)


def test_four_class_contract():
    w0, wp = _build_pair()
    omega, B, valid, unful = _fit(w0, wp)

    # Everything finite — the stiff/dead lanes must not materialize inf/nan.
    assert np.all(np.isfinite(omega)) and np.all(np.isfinite(B))

    # Valid elements recover the exact model pole.
    plain = np.ones((N_PAD, N_PAD), dtype=bool)
    plain[N_LOG:, :] = plain[:, N_LOG:] = False
    for ij in SPECIALS:
        plain[ij] = False
    assert np.all(valid[0, 0, 0][plain])
    np.testing.assert_allclose(omega[0, 0, 0][plain], OMEGA_TRUE, rtol=1e-12)

    # Special elements are all invalid and carry the fallback pole.
    for ij, (_, _, expect) in SPECIALS.items():
        assert bool(valid[0, 0, 0][ij]) is expect, f"class wrong at {ij}"
        assert omega[0, 0, 0][ij] == FALLBACK

    # Pad modes born dead: Ω = B = 0, valid False, junk values notwithstanding.
    assert np.all(omega[0, 0, 0][N_LOG:, :] == 0.0)
    assert np.all(omega[0, 0, 0][:, N_LOG:] == 0.0)
    assert not np.any(valid[0, 0, 0][N_LOG:, :])
    assert np.all(B[0, 0, 0][N_LOG:, :] == 0.0)

    # unfulfilled counts logical modes only: 4 bad of 36.
    np.testing.assert_allclose(unful, 4.0 / 36.0, atol=1e-12)


def test_classification_is_ulp_stable():
    """±1-ulp input noise (the cross-P reduction-order signature) must not
    flip any mode's class — this is what the old absolute-denominator cut
    violated for dispersion-free elements."""
    w0, wp = _build_pair()
    _, _, valid_ref, _ = _fit(w0, wp)
    for direction in (np.inf, -np.inf):
        w0_p = (np.nextafter(w0.real, direction) + 1j * np.nextafter(w0.imag, direction))
        wp_p = (np.nextafter(wp.real, direction) + 1j * np.nextafter(wp.imag, direction))
        _, _, valid_pert, _ = _fit(w0_p, wp_p)
        np.testing.assert_array_equal(valid_pert, valid_ref)
