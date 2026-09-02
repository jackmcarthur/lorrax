"""Algebra and fail-closed gates of the packed static photon q=0 completion.

Covers the pure kernels of ``gw.head_correction`` (Hall tensor, coupled 4x4
moment chunk, numerical certificates, response residuals) and the deck /
runtime envelope of ``bispinor_gw = full_static_cohsex``: the completion runs
by default, ``off`` is an announced DEBUG skip, ``no_local_fields`` and the
retired ``charge_hall_cubature`` spelling refuse, and no caller-supplied
response can reach the completion.
"""

import inspect
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from gw.gw_config import (
    HeadCorrection,
    LorraxConfig,
    uses_coupled_photon_head,
    uses_static_photon_response,
)
from gw.head_correction import (
    _reduce_static_photon_order_diagnostics,
    _require_static_photon_numerical_certificate,
    _static_photon_mixed_error_ratio,
    complete_static_slab_photon_q0,
    static_gauge_tensor_residuals,
    static_hall_linear_response,
    static_slab_photon_head_moment_chunk,
)
from gw.static_gauge_response import StaticPhotonHeadResponse
from gw.w_isdf import compute_static_photon_response


def _mesh():
    devices = jax.devices()
    side = 2 if len(devices) >= 4 else 1
    return Mesh(
        np.asarray(devices[:side * side]).reshape(side, side),
        axis_names=("x", "y"),
    )


def _ward_closed_S():
    """One nonzero canonical S whose spatial response is transverse."""
    S = np.zeros((2, 2, 4, 4), dtype=np.complex128)
    # An unconstrained charge-charge dielectric tensor.
    S[0, 0, 0, 0] = 0.4
    S[0, 1, 0, 0] = S[1, 0, 0, 0] = 0.1
    S[1, 1, 0, 0] = 0.7

    # Pi_ij = beta * (q^2 delta_ij - q_i q_j), i,j in (x,y).
    beta = 1.3
    S[1, 1, 1, 1] = beta
    S[0, 0, 2, 2] = beta
    for a, b in ((0, 1), (1, 0)):
        S[a, b, 1, 2] = -0.5 * beta
        S[a, b, 2, 1] = -0.5 * beta
    return S


def test_hall_builder_has_only_transverse_ct_tc_and_fixed_sign():
    sigma = np.asarray((0.0, 0.0, 0.37))
    H = np.asarray(static_hall_linear_response(sigma))

    np.testing.assert_array_equal(H[:, 0, 0], 0.0)
    np.testing.assert_array_equal(H[:, 1:, 1:], 0.0)
    np.testing.assert_allclose(H, np.conj(np.swapaxes(H, 1, 2)))
    np.testing.assert_allclose(H[0, 0, 2], 1j * sigma[2])
    np.testing.assert_allclose(H[1, 0, 1], -1j * sigma[2])

    q = np.asarray((0.23, -0.41))
    Pi = np.einsum("a,aij->ij", q, H)
    np.testing.assert_allclose(q @ Pi[1:3, :], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(Pi[:, 1:3] @ q, 0.0, atol=1.0e-15)


def test_hall_builder_with_zero_sigma_is_exactly_zero():
    H = np.asarray(static_hall_linear_response(np.zeros(3)))
    np.testing.assert_array_equal(H, 0.0)


def test_static_head_moment_matches_direct_coupled_dyson_algebra():
    q = np.asarray(
        ((0.17, -0.09, 0.0), (-0.12, 0.21, 0.0), (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    D = np.zeros((3, 4, 4), dtype=np.complex128)
    D[0] = np.diag((2.0, -0.7, -0.9, -0.4))
    D[1] = np.diag((1.6, -0.8, -0.6, -0.3))
    sigma = np.asarray((0.0, 0.0, 0.11), dtype=np.float64)
    S = _ward_closed_S()

    weight = np.asarray((1.0, 1.0, 0.0), dtype=np.float64)
    (moments, D_sum, count, residual, sigma_min,
     condition_max,
     conditioned_backward) = static_slab_photon_head_moment_chunk(
        q, D, sigma, S, 2, weight)

    H = np.asarray(static_hall_linear_response(sigma))
    R = (
        np.einsum("sa,aij->sij", q[:2, :2], H)
        + np.einsum("sa,sb,abij->sij", q[:2, :2], q[:2, :2], S)
    )
    W = np.linalg.solve(np.eye(4)[None] - D[:2] @ R, D[:2])
    basis = np.column_stack((np.ones(2), q[:2, :2]))
    expected = np.einsum("su,sij,sv->uvij", basis, W, basis)

    np.testing.assert_allclose(np.asarray(moments), expected, rtol=2e-14)
    np.testing.assert_allclose(np.asarray(D_sum), D[:2].sum(axis=0))
    assert int(np.asarray(count)) == 2
    assert float(np.asarray(residual)) < 1.0e-14
    assert float(np.asarray(sigma_min)) > 0.0
    assert float(np.asarray(condition_max)) >= 1.0
    assert (float(np.asarray(conditioned_backward))
            >= np.finfo(np.float64).eps)


def test_static_head_moment_applies_deterministic_cubature_weights():
    q = np.asarray(
        ((0.17, -0.09, 0.0), (-0.12, 0.21, 0.0), (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    D = np.zeros((3, 4, 4), dtype=np.complex128)
    D[0] = np.diag((2.0, -0.7, -0.9, -0.4))
    D[1] = np.diag((1.6, -0.8, -0.6, -0.3))
    sigma = np.asarray((0.0, 0.0, 0.11), dtype=np.float64)
    S = _ward_closed_S()
    weight = np.asarray((0.23, 0.77, 0.0), dtype=np.float64)

    (moments, D_sum, count, residual, sigma_min,
     condition_max,
     conditioned_backward) = static_slab_photon_head_moment_chunk(
        q, D, sigma, S, 2, weight)
    H = np.asarray(static_hall_linear_response(sigma))
    R = (
        np.einsum("sa,aij->sij", q[:2, :2], H)
        + np.einsum("sa,sb,abij->sij", q[:2, :2], q[:2, :2], S)
    )
    lhs = np.eye(4)[None] - D[:2] @ R
    W = np.linalg.solve(lhs, D[:2])
    basis = np.column_stack((np.ones(2), q[:2, :2]))
    expected = np.einsum(
        "s,su,sij,sv->uvij", weight[:2], basis, W, basis)
    np.testing.assert_allclose(np.asarray(moments), expected, rtol=2e-14)
    np.testing.assert_allclose(
        np.asarray(D_sum), np.einsum("s,sij->ij", weight[:2], D[:2]))
    assert int(np.asarray(count)) == 2
    assert float(np.asarray(residual)) < 1.0e-14
    singular_values = np.linalg.svd(lhs, compute_uv=False)
    condition_fro = (
        np.linalg.norm(lhs, axis=(-2, -1))
        * np.linalg.norm(np.linalg.inv(lhs), axis=(-2, -1)))
    np.testing.assert_allclose(
        np.asarray(sigma_min), singular_values[:, -1].min(), rtol=2e-14)
    np.testing.assert_allclose(
        np.asarray(condition_max), np.max(condition_fro), rtol=2e-14)
    residual_matrix = lhs @ W - D[:2]
    backward = np.linalg.norm(residual_matrix, axis=(-2, -1)) / (
        np.linalg.norm(lhs, axis=(-2, -1))
        * np.linalg.norm(W, axis=(-2, -1))
        + np.linalg.norm(D[:2], axis=(-2, -1)))
    expected_theta = np.max(
        condition_fro * np.maximum(backward, np.finfo(np.float64).eps))
    np.testing.assert_allclose(
        np.asarray(conditioned_backward), expected_theta, rtol=2e-14)


def test_static_head_numerical_certificate_gates_transformed_forward_bound():
    finite = np.zeros((4, 4), dtype=np.complex128)
    theta = 0.75e-9
    with pytest.raises(ValueError, match=r"2\*theta/\(1-theta\)"):
        _require_static_photon_numerical_certificate(
            finite, finite,
            max_backward=2.0e-12,
            min_sigma=1.0,
            max_condition=1.0,
            # Raw theta is below 1e-9; the rigorous bound is above it.
            max_conditioned_backward=theta,
            mixed_error_ratios=(0.2, 0.3))
    passing_theta = 0.25e-9
    bound = _require_static_photon_numerical_certificate(
        finite, finite,
        max_backward=2.0e-12,
        min_sigma=1.0,
        max_condition=1.0,
        max_conditioned_backward=passing_theta,
        mixed_error_ratios=(0.2, 0.3))
    assert bound == pytest.approx(
        2.0 * passing_theta / (1.0 - passing_theta))


def test_static_head_numerical_certificate_refuses_bound_denominator():
    finite = np.zeros((4, 4), dtype=np.complex128)
    with pytest.raises(
            ValueError, match="forward_bound_denominator.*theta < 1"):
        _require_static_photon_numerical_certificate(
            finite, finite,
            max_backward=1.0e-16,
            min_sigma=1.0,
            max_condition=1.0,
            max_conditioned_backward=1.0,
            mixed_error_ratios=(0.2, 0.3))


def test_static_head_numerical_certificate_refuses_nonfinite_theta():
    finite = np.zeros((4, 4), dtype=np.complex128)
    with pytest.raises(ValueError, match="static_photon_dyson_nonfinite"):
        _require_static_photon_numerical_certificate(
            finite, finite,
            max_backward=1.0e-16,
            min_sigma=1.0,
            max_condition=1.0,
            max_conditioned_backward=np.nan,
            mixed_error_ratios=(0.2, 0.3))


def test_static_head_numerical_certificate_refuses_nonfinite_convergence():
    finite = np.zeros((4, 4), dtype=np.complex128)
    with pytest.raises(ValueError, match="static_photon_polygon_nonfinite"):
        _require_static_photon_numerical_certificate(
            finite, finite,
            max_backward=1.0e-16,
            min_sigma=1.0,
            max_condition=1.0,
            max_conditioned_backward=np.finfo(np.float64).eps,
            mixed_error_ratios=(0.2, np.nan))


def test_static_head_mixed_error_refuses_nonfinite_nonfirst_block():
    finite = np.eye(4, dtype=np.complex128)
    nonfinite = finite.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="static_photon_polygon_nonfinite"):
        _static_photon_mixed_error_ratio(
            (finite, finite), (finite, nonfinite))


@pytest.mark.parametrize("diagnostic_index", range(4))
def test_static_head_order_reduction_refuses_later_nonfinite_diagnostic(
        diagnostic_index):
    diagnostics = [
        [1.0e-16, 2.0e-16, 3.0e-16],
        [1.0, 0.9, 0.8],
        [2.0, 3.0, 4.0],
        [4.0e-16, 6.0e-16, 8.0e-16],
    ]
    diagnostics[diagnostic_index][1] = np.nan
    with pytest.raises(ValueError, match="static_photon_dyson_nonfinite"):
        _reduce_static_photon_order_diagnostics(*diagnostics)


def test_static_head_completion_has_no_independent_cell_volume_seam():
    assert "cell_volume" not in inspect.signature(
        complete_static_slab_photon_q0).parameters


def test_static_gauge_tensor_residuals_accept_a_nonzero_ward_closed_tensor():
    ward, hermiticity = static_gauge_tensor_residuals(_ward_closed_S())
    assert ward < 1.0e-15
    assert hermiticity == 0.0


def test_static_gauge_tensor_residuals_flag_a_ward_breaking_tensor():
    S = _ward_closed_S()
    S[0, 0, 1, 1] += 0.2
    ward, _ = static_gauge_tensor_residuals(S)
    assert ward > 1.0e-2


def test_static_gauge_tensor_residuals_flag_a_nonhermitian_tensor():
    S = _ward_closed_S()
    S[0, 0, 0, 3] = 0.2j
    _, hermiticity = static_gauge_tensor_residuals(S)
    assert hermiticity > 1.0e-2


def test_static_photon_head_response_is_sealed_to_its_producer():
    with pytest.raises(TypeError, match="issued only by"):
        StaticPhotonHeadResponse(
            layout=None, S_direct=None, sigma_H=None, hall_source="",
            Y_x=None, Z_y=None, ward_residual=0.0,
            hermiticity_residual=0.0, wing_reciprocity_residual=0.0,
            _producer_token=object())


def test_packed_runtime_has_no_caller_supplied_head_response_seam():
    parameters = inspect.signature(compute_static_photon_response).parameters
    assert "gauge_head_response" not in parameters


def test_packed_runtime_refuses_a_config_less_call_before_opening_a_body():
    mesh = _mesh()
    with pytest.raises(ValueError, match="requires the run config"):
        compute_static_photon_response(
            None, None, None, None, None, mesh, config=None)


def test_packed_runtime_refuses_no_local_fields_before_opening_a_body():
    mesh = _mesh()
    config = SimpleNamespace(
        head=SimpleNamespace(correction=HeadCorrection.NO_LOCAL_FIELDS),
        bispinor_gw="full_static_cohsex",
    )
    with pytest.raises(ValueError, match="head_correction=full"):
        compute_static_photon_response(
            None, None, None, None, None, mesh, config=config)


def _packed_deck(*, sys_dim=2, extra=""):
    return (
        "[cohsex]\n"
        "nval = 2\n"
        "ncond = 2\n"
        "number_bands = 8\n"
        "memory_per_device_gb = 4.0\n"
        f"sys_dim = {sys_dim}\n"
        "bispinor = true\n"
        "bispinor_gw = full_static_cohsex\n"
        "compute_mode = cohsex\n"
        "low_mem_bands = true\n"
        "w_dyson_solver = distributed\n"
        "restart = false\n"
        + extra
    )


def _parse(tmp_path, deck_text):
    deck = tmp_path / "packed.in"
    deck.write_text(deck_text)
    return LorraxConfig.from_input_file(str(deck), print_fn=lambda *_: None)


def test_packed_deck_default_head_runs_the_completion(tmp_path):
    config = _parse(tmp_path, _packed_deck())
    assert config.head.correction is HeadCorrection.FULL
    assert uses_static_photon_response(config)
    assert uses_coupled_photon_head(config)


def test_packed_deck_off_head_is_an_announced_debug_skip(tmp_path):
    config = _parse(tmp_path, _packed_deck(extra="head_correction = off\n"))
    assert uses_static_photon_response(config)
    assert not uses_coupled_photon_head(config)


def test_packed_deck_refuses_no_local_fields(tmp_path):
    with pytest.raises(ValueError, match="static_bispinor_photon_envelope"):
        _parse(tmp_path,
               _packed_deck(extra="head_correction = no_local_fields\n"))


def test_packed_deck_completion_is_slab_only_and_says_why(tmp_path):
    with pytest.raises(
            ValueError,
            match="(?s)static_bispinor_photon_head_slab_only.*no derived "
                  "integrator") as info:
        _parse(tmp_path, _packed_deck(sys_dim=3))
    assert "sys_dim = 2" in str(info.value)


def test_packed_deck_off_head_keeps_the_bulk_body_reachable(tmp_path):
    config = _parse(
        tmp_path, _packed_deck(sys_dim=3, extra="head_correction = off\n"))
    assert not uses_coupled_photon_head(config)


def test_retired_charge_hall_cubature_spelling_names_the_new_mode(tmp_path):
    deck = _packed_deck().replace(
        "bispinor_gw = full_static_cohsex",
        "bispinor_gw = charge_hall_cubature")
    with pytest.raises(
            ValueError,
            match="(?s)bispinor_gw_charge_hall_cubature_retired.*"
                  "full_static_cohsex"):
        _parse(tmp_path, deck)
