import types

import numpy as np
import pytest
from types import SimpleNamespace

from gw.head_correction import (
    HeadResolver,
    HeadResponseKind,
    HeadSample,
    _dipole_window_from_params,
    compute_static_head_terms,
    fit_head_ppm,
    fit_head_ppm_from_samples,
    resolve_head_override,
    static_head_terms_to_kij,
)
from gw.gw_config import HeadCorrection, coerce_head_correction


def _resolver(policy, monkeypatch, *, screened=True):
    head = SimpleNamespace(
        correction=policy, wcoul0_source="s_tensor", wcoul0_eta=0.0,
        vhead=None, whead_0freq=None, whead_imfreq=None,
        head_minibz_average=False, bgw_metal_q0_treatment="exact")
    config = SimpleNamespace(
        head=head, do_screened=screened, nval=4, ncond=4, nband=8,
        # HeadResolver resolves the four-current representation from the GW
        # run controls (head_correction.py, resolve_four_current_representation),
        # so a hand-built config must carry them.  Scalar defaults: this suite
        # is about the head POLICY, not the bispinor carrier.
        bispinor=False, bispinor_gw="bare_transverse")
    direct = HeadSample(
        vc0=100.0 + 0.0j, wcoul0=25.0 + 0.0j,
        source="unit direct", omega=0.0j,
        S_cart=np.eye(3),
        response_kind=HeadResponseKind.DIRECT_IRREDUCIBLE)
    monkeypatch.setattr(
        "gw.head_correction.resolve_head_sample",
        lambda *a, **k: direct)
    return HeadResolver(config, ".", object(), object(), object(), lambda *a: None)


def test_head_policy_enum_is_strict_and_has_one_physical_default_axis():
    for value in HeadCorrection:
        assert coerce_head_correction(value.value) is value
    with pytest.raises(ValueError, match="full, no_local_fields, off"):
        coerce_head_correction("local_fields_maybe")


def test_full_policy_refuses_an_unfolded_direct_epsilon_head(monkeypatch):
    resolver = _resolver(HeadCorrection.FULL, monkeypatch)
    with pytest.raises(RuntimeError, match="no finalized head"):
        resolver.at(0.0j)
    with pytest.raises(ValueError, match="Refusing the unfolded epsilon head"):
        resolver.install_samples([resolver.direct_at(0.0j)])


@pytest.mark.parametrize(
    "kind",
    [HeadResponseKind.FULL_LOCAL_FIELDS, HeadResponseKind.MICRO_REDUCIBLE],
)
def test_full_policy_accepts_only_once_reduced_head_kinds(monkeypatch, kind):
    resolver = _resolver(HeadCorrection.FULL, monkeypatch)
    final = HeadSample(
        vc0=100.0 + 0.0j, wcoul0=30.0 + 0.0j,
        source="unit final", omega=0.0j, S_cart=np.eye(3),
        response_kind=kind)
    resolver.install_samples([final])
    assert resolver.at(0.0j) is final


def test_diagnostic_and_off_policies_are_observable(monkeypatch):
    no_lf = _resolver(HeadCorrection.NO_LOCAL_FIELDS, monkeypatch)
    assert no_lf.at(0.0j).response_kind is HeadResponseKind.DIRECT_IRREDUCIBLE
    off = _resolver(HeadCorrection.OFF, monkeypatch)
    sample = off.at(0.0j)
    assert sample.response_kind is HeadResponseKind.OFF
    assert sample.vc0 == 0.0j and sample.wcoul0 == 0.0j


def test_head_resolver_forwards_the_gw_run_dipole_window():
    """The provenance message compares the reader's window, not defaults."""
    head = types.SimpleNamespace(
        wcoul0_source="s_tensor",
        wcoul0_eta=0.0,
        vhead=None,
        whead_0freq=None,
        whead_imfreq=None,
        head_minibz_average=False,
        bgw_metal_q0_treatment="off",
    )
    config = types.SimpleNamespace(
        head=head, nval=8, ncond=32, nband=40,
        bispinor=False, bispinor_gw="bare_transverse")
    wfn = types.SimpleNamespace(nbands=62, nelec=10)
    resolver = HeadResolver(
        config, ".", wfn, sym=None, meta=None, print_fn=lambda _msg: None)
    assert _dipole_window_from_params(resolver._params, wfn) == (8, 32, 40)


def test_an_absent_dipole_window_refuses_instead_of_defaulting_to_5_5():
    """No band window means NO reference — never a fabricated 5/5/nbands.

    The helper used to answer ``(5, 5, max(nbands, nelec+5))`` for a params
    dict with no window in it, which is exactly what ``HeadResolver`` handed
    it before the window was threaded: the provenance check then compared a
    correctly-stamped 26/26/600 file against an invented 5/5/610 run and
    accused it.  A reference the checker makes up cannot fail for the reason
    it claims, so the absent case is a refusal.
    """
    wfn = types.SimpleNamespace(nbands=610, nelec=26)
    with pytest.raises(ValueError, match="nval, ncond, nband"):
        _dipole_window_from_params({}, wfn)
    # The six head keys alone — the pre-fix ``HeadResolver._params`` — are
    # still not a band window, and must not silently become one.
    head_only = {"wcoul0_source": "s_tensor", "wcoul0_eta": 0.0,
                 "vhead": None, "whead_0freq": None, "whead_imfreq": None,
                 "head_minibz_average": False}
    with pytest.raises(ValueError, match="band window is missing"):
        _dipole_window_from_params(head_only, wfn)
    # A partial window is no better than none: 26/26 with nband absent must
    # not resolve nband from the WFN behind the deck's back.
    with pytest.raises(ValueError, match="nband"):
        _dipole_window_from_params({"nval": 26, "ncond": 26}, wfn)


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


# ---------------------------------------------------------------------------
# G3 (Σ_PPM tighten, WS0): head negative-branch (Ω_h² ≤ 0) regression.
#
# ``fit_head_ppm`` had ZERO coverage of the ``omega_h_sq <= 0`` branch
# (head_correction.py:320-338) — the branch Bug A lived in.  Bug A: the
# negative branch computed ``B_h = -w1 * omega_h_sq`` with the SIGNED
# (negative) omega_h_sq while ``omega_h = |omega_h_sq|**0.5`` is positive,
# so ``R_h = B_h / (2 omega_h)`` came out sign-FLIPPED relative to the
# positive branch — the entire q→0 head Σ_c (hundreds of meV) had the wrong
# sign whenever the GN head fit went imaginary.  The fix (2026-07-04, at
# :327) uses ``B_h = -w1 * |omega_h_sq|`` so that
#
#     R_h = -w1 * sqrt(|omega_h_sq|) / 2          (both branches)
#
# i.e. |R_h| is magnitude-continuous across Ω²=0 and sign(R_h) = -sign(w1)
# on BOTH sides.  These tests pin exactly that continuity.


def _analytic_head_omega_h_sq(vc0, wc0_static, wc0_probe, probe_omega):
    """The head fit's omega_h_sq formula, recomputed independently."""
    w1 = wc0_static - vc0
    w2 = wc0_probe - vc0
    omega_2_sq = (complex(probe_omega) ** 2).real
    return -w2 * omega_2_sq / (w1 - w2)


def test_fit_head_ppm_negative_branch_sign_matches_positive_limit():
    """Drive the Ω_h²<0 branch and assert R_h has the anti-Bug-A sign.

    GN probe (purely imaginary ω_p) with |W^c(iω_p)| > |W^c(0)| forces
    omega_h_sq < 0.  vc0=10, W^c(0)=w1=-5, W^c(iω_p)=w2=-8 (|w2|>|w1|):
    omega_h_sq = -w2·(iω_p)²/(w1-w2) = 8·(-4)/3 = -32/3 < 0.
    """
    vc0, wc0_static, wc0_probe = 10.0, 5.0, 2.0   # w1=-5, w2=-8
    probe_omega = 2.0j                             # GN: (iω_p)² = -4
    w1 = wc0_static - vc0                           # -5

    head = fit_head_ppm(vc0, wc0_static, wc0_probe, probe_omega)

    # The negative branch really was taken.
    assert head.omega_h_sq < 0.0
    s = _analytic_head_omega_h_sq(vc0, wc0_static, wc0_probe, probe_omega)
    np.testing.assert_allclose(head.omega_h_sq, s)
    np.testing.assert_allclose(head.omega_h, abs(s) ** 0.5)

    # The property Bug A violated: R_h = -w1·sqrt(|Ω²|)/2, so sign(R_h) is
    # the SAME sign the positive branch would give (= -sign(w1)), NOT flipped.
    expected_R_h = -w1 * abs(s) ** 0.5 / 2.0
    np.testing.assert_allclose(head.R_h, expected_R_h)
    assert np.sign(head.R_h) == np.sign(-w1)          # would FAIL under Bug A
    # B_h likewise uses |Ω²| (the fix), not the signed value.
    np.testing.assert_allclose(head.B_h, -w1 * abs(s))

    # The resolved-sample entry point must agree bit-for-bit.
    head_s = fit_head_ppm_from_samples(
        HeadSample(vc0=complex(vc0), wcoul0=complex(wc0_static),
                   source="unit", omega=0.0 + 0.0j),
        HeadSample(vc0=complex(vc0), wcoul0=complex(wc0_probe),
                   source="unit", omega=probe_omega),
        probe_omega=probe_omega,
    )
    np.testing.assert_allclose(head_s.R_h, head.R_h)
    np.testing.assert_allclose(head_s.B_h, head.B_h)


def test_fit_head_ppm_R_h_continuous_across_omega_sq_zero():
    """R_h is sign- and magnitude-continuous across the Ω_h²=0 crossing.

    Hold w1 = W^c(0)-vc0 fixed and sweep W^c(iω_p) through vc0 (w2→0),
    where omega_h_sq changes sign.  Just below (positive branch) and just
    above (negative branch) the crossing, R_h must have the same sign and
    nearly equal magnitude.  Under Bug A the negative side flipped sign,
    making |ΔR_h| ≈ 2·|R_h| instead of →0.
    """
    vc0, wc0_static = 10.0, 5.0        # w1 = -5
    probe_omega = 2.0j
    w1 = wc0_static - vc0
    delta = 0.01

    # w2 = -delta  → omega_h_sq > 0 (positive branch, just below crossing)
    pos = fit_head_ppm(vc0, wc0_static, vc0 - delta, probe_omega)
    # w2 = +delta  → omega_h_sq < 0 (negative branch, just above crossing)
    neg = fit_head_ppm(vc0, wc0_static, vc0 + delta, probe_omega)

    assert pos.omega_h_sq > 0.0 and neg.omega_h_sq < 0.0   # bracket the crossing
    # Same sign on both sides (the anti-Bug-A property) ...
    assert np.sign(pos.R_h) == np.sign(neg.R_h) == np.sign(-w1)
    # ... and continuous in magnitude (→0 as delta→0): |ΔR_h| ≪ |R_h|.
    # Under Bug A the two sides are opposite-signed, so |ΔR_h| ≈ 2|R_h|.
    assert abs(pos.R_h - neg.R_h) < 0.02 * abs(pos.R_h)
