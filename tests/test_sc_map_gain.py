"""Synthetic two-map checks for the output-only SC map-gain diagnostic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gw import sc_iteration
from gw.sc_iteration import measure_sc_map_gain


def test_two_map_gain_uses_only_the_non_scissored_set():
    """Map 2 reports adjacent L-infinity changes; excluded tails do not win."""
    e_map1 = np.array([
        [-2.0, 0.0, 1.0, 3.0],
        [-1.5, 0.5, 2.0, 4.0],
    ])
    sigma_map1 = np.array([
        [0.2, 0.4, 0.6, 0.8],
        [0.3, 0.5, 0.7, 0.9],
    ], dtype=np.complex128)
    e_map2 = e_map1.copy()
    sigma_map2 = sigma_map1.copy()

    # The numerator and denominator both attain their included maxima at
    # k=1, active column 2.  The much larger changes in column 3 are outside
    # the non-scissored set and must not affect either maximum.
    e_map2[1, 2] += 0.040
    sigma_map2[1, 2] += 0.010
    e_map2[0, 3] += 9.0
    sigma_map2[0, 3] += 7.0

    result = measure_sc_map_gain(
        e_map2, sigma_map2, e_map1, sigma_map1,
        np.array([True, True, True, False]),
        band_offset=4,
    )

    assert result.gain == pytest.approx(0.25)
    assert result.max_dsigma_mev == pytest.approx(10.0)
    assert result.max_de_mev == pytest.approx(40.0)
    assert result.worst_k == 1
    assert result.worst_band == 7  # absolute, 1-based: offset 4 + column 2 + 1
    assert result.worst_de_mev == pytest.approx(40.0)
    assert result.summary() == (
        "SC map gain: max |dSigma_on-shell| / max |dE_in| = 0.25 "
        "(worst k=1 band=7: dSigma=10.000000 meV / dE=40.000000 meV)"
    )
    stamp = result.stamp()
    for field in (
        "sc_map_gain=2.500000000000e-01",
        "max_abs_dSigma_on_shell_mev=1.000000000000e+01",
        "max_abs_dE_in_mev=4.000000000000e+01",
        "worst_k=1",
        "worst_band_1based=7",
        "worst_state_abs_dE_in_mev=4.000000000000e+01",
    ):
        assert field in stamp


def test_zero_input_motion_is_reported_as_undefined_gain():
    e = np.zeros((1, 2))
    result = measure_sc_map_gain(
        e, np.array([[0.0, 0.01]]), e, np.zeros((1, 2)),
        np.array([True, True]),
    )
    assert np.isnan(result.gain)
    assert result.max_dsigma_mev == pytest.approx(10.0)
    assert result.max_de_mev == 0.0


def test_map_two_is_the_first_driver_diagnostic():
    """Map 1 seeds history even when SigmaResult.e_eval_ev is its None sentinel."""
    mask = np.array([True, True])
    partition = SimpleNamespace(
        protected_mask=mask, in_range_mask=mask)
    inputs = SimpleNamespace(
        partition=partition,
        band_slices=SimpleNamespace(sigma=slice(4, 6)),
    )

    def state(sigma, iteration):
        sigma_result = SimpleNamespace(
            sigma_c_at_dft_diag_ev=np.asarray(sigma, dtype=np.complex128),
            # assemble_eqp uses None as the first map's E_eval == E_DFT
            # sentinel; the map-gain history must not depend on that field.
            e_eval_ev=None,
        )
        return sc_iteration.SCState(
            H_qp_dft=np.zeros((1, 2, 2)), iteration=iteration,
            partition=partition,
            outputs=SimpleNamespace(sigma_result=sigma_result),
        )

    diagnostic1, history = sc_iteration._sc_map_gain_for_call(
        inputs, state([[0.0, 0.1]], 1), np.array([[1.0, 2.0]]), None)
    assert diagnostic1 is None
    assert history is not None

    diagnostic2, history = sc_iteration._sc_map_gain_for_call(
        inputs, state([[0.0, 0.12]], 2), np.array([[1.0, 2.04]]), history)
    assert diagnostic2 is not None
    assert diagnostic2.gain == pytest.approx(0.5)
    assert diagnostic2.worst_band == 6
    assert history is not None
