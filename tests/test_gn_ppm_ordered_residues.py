"""First-principles oracle for the TR-broken two-point GN pole algebra."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gw.gw_config import ComputeMode, ScreeningDiagrams
from gw.minimax_screening import (
    fit_gn_ppm_from_wc_pair,
)
from gw.ppm_sigma import _residue_for_space
from gw.qsgw_head import head_samples_from_s
from gw.screening import (
    ScreeningRequest,
    _assert_imaginary_probe_supported,
    compute_screening_model,
    _gate_w,
)


def _ordered_samples(R_positive, R_negative, omega, z):
    omega = np.asarray(omega, dtype=np.float64)
    return (
        R_positive / (z - omega)
        - R_negative / (z + omega)
    )


def test_ordered_gn_fit_recovers_hall_residues_and_both_samples():
    R_positive = np.array(
        [[2.0, 0.4 + 0.7j], [0.4 - 0.7j, 1.2]],
        dtype=np.complex128,
    )[None, ...]
    R_negative = np.swapaxes(R_positive, -1, -2)
    omega = np.full((1, 2, 2), 1.7, dtype=np.float64)
    z = 1.3j
    W0 = _ordered_samples(R_positive, R_negative, omega, 0.0j)
    W_plus = _ordered_samples(R_positive, R_negative, omega, z)
    W_minus = _ordered_samples(R_positive, R_negative, omega, -z)

    # The q=0 Hall probe is real but not symmetric/Hermitian.
    np.testing.assert_allclose(W_plus.imag, 0.0, atol=2.0e-16)
    assert not np.allclose(W_plus, np.swapaxes(W_plus, -1, -2))

    fit = fit_gn_ppm_from_wc_pair(
        W0,
        W_plus,
        z,
        fallback_omega=2.0,
        n_mu_logical=2,
        include_frequency_odd_response=True,
        q_neg_index=np.asarray([0]),
    )

    got_positive = np.asarray(fit.B_qmunu)
    got_negative = np.asarray(fit.B_negative_qmunu)
    got_omega = np.asarray(fit.omega_qmunu)
    np.testing.assert_allclose(got_omega, omega, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(
        got_positive, R_positive, rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(
        got_negative, R_negative, rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(
        _ordered_samples(got_positive, got_negative, got_omega, z),
        W_plus,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        _ordered_samples(got_positive, got_negative, got_omega, -z),
        W_minus,
        rtol=3.0e-14,
        atol=3.0e-14,
    )


def test_reciprocal_gn_fit_keeps_one_residue_and_branch_selector_is_exact(
    monkeypatch,
):
    from gw import minimax_screening as fit_module

    one_q = np.array(
        [[2.0, 0.3 - 0.2j], [0.3 + 0.2j, 1.0]],
        dtype=np.complex128,
    )
    B = np.stack([one_q, 1.25 * one_q, 0.75 * one_q])
    omega = np.stack([
        np.full((2, 2), 1.4, dtype=np.float64),
        np.full((2, 2), 1.6, dtype=np.float64),
        np.full((2, 2), 1.2, dtype=np.float64),
    ])
    z = 0.9j
    W0 = _ordered_samples(B, B, omega, 0.0j)
    W_probe = _ordered_samples(B, B, omega, z)

    per_q_bytes = int(W0[0].size * W0.dtype.itemsize)
    assert fit_module._gn_ppm_fit_q_block(3, per_q_bytes) == 3
    single_shot = fit_gn_ppm_from_wc_pair(
        W0,
        W_probe,
        z,
        fallback_omega=2.0,
        n_mu_logical=2,
    )
    monkeypatch.setattr(
        fit_module, "_GN_PPM_FIT_ARENA_BUDGET_BYTES", 1)
    assert fit_module._gn_ppm_fit_q_block(3, per_q_bytes) == 1
    fit = fit_gn_ppm_from_wc_pair(
        W0,
        W_probe,
        z,
        fallback_omega=2.0,
        n_mu_logical=2,
    )
    explicit_false = fit_gn_ppm_from_wc_pair(
        W0,
        W_probe,
        z,
        fallback_omega=2.0,
        n_mu_logical=2,
        include_frequency_odd_response=False,
    )
    np.testing.assert_allclose(np.asarray(fit.B_qmunu), B)
    assert fit.B_negative_qmunu is None
    assert explicit_false.B_negative_qmunu is None
    for field in ("omega_qmunu", "B_qmunu", "valid_qmunu"):
        values = [
            np.asarray(getattr(result, field))
            for result in (single_shot, fit, explicit_false)
        ]
        assert np.array_equal(values[0], values[1]), (
            f"legacy single-shot and donated q-chunk differ in {field}")
        assert np.array_equal(values[1], values[2]), (
            f"legacy default and explicit False differ in {field}")
    for field in (
        "unfulfilled_fraction", "n_valid", "omega_min_raw",
        "omega_max_raw", "pair_relative_separation_min",
        "n_tail_low", "n_tail_high", "omega_min_after",
        "omega_max_after", "tail_anchor_omega",
    ):
        values = [
            getattr(result, field)
            for result in (single_shot, fit, explicit_false)
        ]
        for left, right in zip(values, values[1:]):
            assert left == right or (np.isnan(left) and np.isnan(right)), (
                f"legacy single-shot/chunk/default/False differ in {field}")

    positive = object()
    negative = object()
    assert _residue_for_space("cond", positive, negative) is positive
    assert _residue_for_space("val", positive, negative) is negative
    assert _residue_for_space("val", positive, None) is positive


def test_broken_tr_probe_gate_keeps_q_reciprocity_not_fixed_q_hermiticity(
    monkeypatch,
):
    from common import sanity

    calls = []
    monkeypatch.setattr(
        sanity, "check_finite", lambda *args, **kwargs: calls.append("finite"))
    monkeypatch.setattr(
        sanity, "check_hermitian",
        lambda *args, **kwargs: calls.append("hermitian"))
    monkeypatch.setattr(
        sanity, "check_q_conjugate_reciprocity",
        lambda *args, **kwargs: calls.append("q_reciprocity"))

    W = np.zeros((1, 2, 2), dtype=np.complex128)
    _gate_w(
        W,
        ScreeningRequest(1.0j, "probe"),
        kgrid=(1, 1, 1),
        frequency_odd_response=True,
    )
    assert calls == ["finite", "q_reciprocity"]

    calls.clear()
    _gate_w(
        W,
        ScreeningRequest(0.0j, "static"),
        kgrid=(1, 1, 1),
        frequency_odd_response=True,
    )
    assert calls == ["finite", "hermitian", "q_reciprocity"]


def test_dynamic_one_residue_models_refuse_measured_broken_tr():
    config = SimpleNamespace(
        screening=SimpleNamespace(diagrams=ScreeningDiagrams.W_RPA))
    common = dict(
        quad=None,
        e_ref=0.0,
        sym=SimpleNamespace(trs_allowed=False),
        centroid_indices=None,
        config=config,
        meta=None,
        mesh_xy=None,
        run_dir="",
        label="test",
    )

    for mode, gate in (
        (ComputeMode.HL_PPM, "hl_ppm_broken_tr_response"),
        (ComputeMode.MPA, "mpa_broken_tr_response"),
    ):
        with np.testing.assert_raises_regex(RuntimeError, gate):
            compute_screening_model(mode, None, None, **common)

    # Static screening has no positive/negative-frequency residue pair.
    assert compute_screening_model(
        ComputeMode.MPA, None, None, static_only=True, **common) == {}


def test_w_bse_refuses_broken_tr_before_the_ladder():
    config = SimpleNamespace(
        screening=SimpleNamespace(diagrams=ScreeningDiagrams.W_BSE))
    with np.testing.assert_raises_regex(
            RuntimeError, "w_bse_requires_measured_trs"):
        compute_screening_model(
            ComputeMode.GN_PPM,
            None,
            None,
            quad=None,
            e_ref=0.0,
            sym=SimpleNamespace(trs_allowed=False),
            centroid_indices=None,
            config=config,
            meta=None,
            mesh_xy=None,
            run_dir="",
            label="test",
        )


def test_resolvent_probe_gate_obeys_the_same_ordered_response_policy(
    monkeypatch,
):
    from common import sanity
    from gw.screening_bse import _gate_w_or_refuse

    calls = []
    monkeypatch.setattr(
        sanity, "check_finite", lambda *args, **kwargs: calls.append("finite"))
    monkeypatch.setattr(
        sanity, "check_hermitian",
        lambda *args, **kwargs: calls.append("hermitian"))
    monkeypatch.setattr(
        sanity, "check_q_conjugate_reciprocity",
        lambda *args, **kwargs: calls.append("q_reciprocity"))

    _gate_w_or_refuse(
        np.zeros((1, 2, 2), dtype=np.complex128),
        ScreeningRequest(1.0j, "probe"),
        stage="ordered resolvent probe",
        kgrid=(1, 1, 1),
        frequency_odd_response=True,
    )
    assert calls == ["finite", "q_reciprocity"]


def test_broken_tr_route_refuses_an_even_probe_plan():
    with np.testing.assert_raises_regex(
            RuntimeError, "gn_ppm_ordered_probe_producer"):
        _assert_imaginary_probe_supported(trs_allowed=False)
    _assert_imaginary_probe_supported(trs_allowed=True)


def test_broken_tr_gn_head_refuses_unpaired_average():
    config = SimpleNamespace(
        compute_mode=ComputeMode.GN_PPM,
        head=SimpleNamespace(
            vhead=None,
            whead_0freq=None,
            whead_imfreq=None,
            head_minibz_average=False,
        ),
    )

    for sys_dim in (2, 3):
        for trs_holds in (False, None):
            with np.testing.assert_raises_regex(
                    RuntimeError, "gn_ppm_unpaired_head_average"):
                head_samples_from_s(
                    np.eye(3, dtype=np.complex128)[None],
                    (0.41j,),
                    wfn=SimpleNamespace(trs_holds=trs_holds),
                    meta=SimpleNamespace(sys_dim=sys_dim),
                    config=config,
                )
