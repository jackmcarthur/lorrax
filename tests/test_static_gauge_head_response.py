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
    complete_static_photon_q0,
    static_gauge_tensor_residuals,
    static_hall_linear_response,
    static_photon_head_moment_chunk,
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


def test_q0_factor_update_rank_is_derived_not_truncated():
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw.photon_layout import (
        PhotonBasisLayout, add_photon_q0_low_rank)

    mesh = _mesh()
    layout = PhotonBasisLayout.from_centroid_extents(2, 2, mesh)
    packed_sharding = NamedSharding(mesh, P(None, "x", "y"))
    left_sharding = NamedSharding(mesh, P(None, "x"))
    right_sharding = NamedSharding(mesh, P(None, "y"))
    rank = 5
    left = np.arange(
        rank * layout.packed_extent, dtype=np.float64).reshape(
            rank, layout.packed_extent).astype(np.complex128)
    right = (left + 0.5j).astype(np.complex128)
    packed = jax.device_put(
        np.zeros((1, layout.packed_extent, layout.packed_extent),
                 dtype=np.complex128), packed_sharding)
    got = add_photon_q0_low_rank(
        packed, layout, mesh,
        left_rows_X=jax.device_put(left, left_sharding),
        right_rows_Y=jax.device_put(right, right_sharding))
    np.testing.assert_array_equal(
        np.asarray(got[0]), np.einsum("ai,aj->ij", left, right))
    assert layout.q0_basis_size(2) == 3
    assert layout.q0_basis_size(3) == 4


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
    H = np.asarray(static_hall_linear_response(sigma, dimension=2))

    np.testing.assert_array_equal(H[:, 0, 0], 0.0)
    np.testing.assert_array_equal(H[:, 1:, 1:], 0.0)
    np.testing.assert_allclose(H, np.conj(np.swapaxes(H, 1, 2)))
    np.testing.assert_allclose(H[0, 0, 2], -1j * sigma[2])
    np.testing.assert_allclose(H[1, 0, 1], 1j * sigma[2])

    q = np.asarray((0.23, -0.41))
    Pi = np.einsum("a,aij->ij", q, H)
    np.testing.assert_allclose(q @ Pi[1:3, :], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(Pi[:, 1:3] @ q, 0.0, atol=1.0e-15)


def test_hall_builder_with_zero_sigma_is_exactly_zero():
    H = np.asarray(static_hall_linear_response(
        np.zeros(3), dimension=2))
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
     conditioned_backward) = static_photon_head_moment_chunk(
        q, D, sigma, S, 2, weight)

    H = np.asarray(static_hall_linear_response(sigma, dimension=2))
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
     conditioned_backward) = static_photon_head_moment_chunk(
        q, D, sigma, S, 2, weight)
    H = np.asarray(static_hall_linear_response(sigma, dimension=2))
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


# A hexagonal slab shaped like the MoS2 3x3 gate deck (a = 5.9 bohr,
# c = 23.3 bohr, k = 3x3x1, cell volume 702 bohr^3), so the cells below
# exercise the rule on the integrand production actually integrates rather
# than on an invented one.  ``_S_ISOTROPIC`` is lane J's fitted in-plane
# scalar for that deck (claim 0586: s = -0.317143 bohr^2).
_SLAB_A_BOHR = 5.9
_SLAB_C_BOHR = 23.3
_S_ISOTROPIC = -0.317143


def _hex_slab_geometry():
    """One hexagonal slab cell and k-grid, shared by the owner cells below."""
    import vcoul
    a = _SLAB_A_BOHR
    bvec = np.asarray((
        (2.0 * np.pi / a, 2.0 * np.pi / (a * np.sqrt(3.0)), 0.0),
        (0.0, 4.0 * np.pi / (a * np.sqrt(3.0)), 0.0),
        (0.0, 0.0, 2.0 * np.pi / _SLAB_C_BOHR),
    ), dtype=np.float64)
    celvol = float(a * a * np.sqrt(3.0) / 2.0 * _SLAB_C_BOHR)
    return vcoul.CoulombGeometry(bvec=bvec, cell_volume=celvol), (3, 3, 1)


def test_scalar_head_owner_and_packed_completion_share_one_q0_quadrature():
    """ONE q->0 cell-average owner: same nodes, same weights, same numbers.

    The scalar charge head (``vcoul.Slab2D.q0_average``) and the packed
    completion's bare/screened Gamma average must be the SAME reduction of
    the SAME provider receipt, not two estimators that happen to agree.
    Measured on both companions:

    * bare       ``<v>``                 == ``D_mean[0, 0]``
    * screened   ``<v/(1 - v qSq)>``     == ``moments[0, 0][0, 0] / measure``
      with the charge-only ``S_quadratic`` block, which reduces the coupled
      4x4 Dyson solve to exactly the scalar denominator.

    RED TWIN: the same comparison against the superseded ``sobol_debug``
    rule, which must MISS by ~0.1 % -- the +5.72 meV/state error lane J
    measured (claim 0586).  Without it a broken owner and a coincidence
    look the same.
    """
    import vcoul
    from vcoul import Q0_RULE_EXACT, Q0_RULE_SOBOL_DEBUG

    geometry, kgrid = _hex_slab_geometry()
    kernel = vcoul.get_kernel(2)
    S_cart = np.diag(
        (_S_ISOTROPIC, _S_ISOTROPIC, 0.0)).astype(np.complex128)

    receipt = vcoul.slab_minibz_photon_cubature(kernel, geometry, kgrid)
    chunk = receipt.chunks[-1]
    n_valid = int(chunk.physical_count)
    measure = float(np.sum(chunk.sample_weight[:n_valid]))

    S_quadratic = np.zeros((2, 2, 4, 4), dtype=np.complex128)
    S_quadratic[:, :, 0, 0] = S_cart[:2, :2]
    moments, D_sum, count, *_ = static_photon_head_moment_chunk(
        np.asarray(chunk.q_cart), np.asarray(chunk.D_raw), np.zeros(3),
        S_quadratic, n_valid, np.asarray(chunk.sample_weight))
    assert int(np.asarray(count)) == n_valid
    packed_bare = complex(np.asarray(D_sum)[0, 0] / measure)
    packed_screened = complex(np.asarray(moments)[0, 0, 0, 0] / measure)

    scalar_bare, _ = kernel.q0_average(
        geometry, kgrid, S_cart=np.zeros((3, 3)), rule=Q0_RULE_EXACT)
    _, scalar_screened = kernel.q0_average(
        geometry, kgrid, S_cart=S_cart, rule=Q0_RULE_EXACT)

    assert abs(complex(scalar_bare) - packed_bare) <= 1.0e-10
    assert abs(complex(scalar_screened) - packed_screened) <= 1.0e-10

    sobol_bare, sobol_screened = kernel.q0_average(
        geometry, kgrid, S_cart=S_cart, rule=Q0_RULE_SOBOL_DEBUG)
    assert abs(complex(sobol_bare) - packed_bare) > 1.0e-6 * abs(packed_bare)
    assert (abs(complex(sobol_screened) - packed_screened)
            > 1.0e-6 * abs(packed_screened))


def test_slab_q0_ladder_certificate_can_refuse():
    """NEGATIVE CONTROL: the ladder gate is reachable in the FAIL direction.

    A check that cannot fail is not evidence (TASTE.md).  A screened head
    whose denominator varies far more sharply across the cell than any
    measured deck (a much thicker slab at the same fitted ``S``) leaves the
    24->32 pair above the mixed budget, and the production rule REFUSES
    instead of returning the order-32 value.  The gate is the same
    absolute+relative budget the packed completion applies to its own
    polygon ladder.
    """
    import vcoul

    a = _SLAB_A_BOHR
    thick_c = 4.0 * _SLAB_C_BOHR
    bvec = np.asarray((
        (2.0 * np.pi / a, 2.0 * np.pi / (a * np.sqrt(3.0)), 0.0),
        (0.0, 4.0 * np.pi / (a * np.sqrt(3.0)), 0.0),
        (0.0, 0.0, 2.0 * np.pi / thick_c),
    ), dtype=np.float64)
    geometry = vcoul.CoulombGeometry(
        bvec=bvec, cell_volume=float(a * a * np.sqrt(3.0) / 2.0 * thick_c))
    S_cart = np.diag(
        (_S_ISOTROPIC, _S_ISOTROPIC, 0.0)).astype(np.complex128)
    with pytest.raises(ValueError, match="slab_q0_polygon_not_converged"):
        vcoul.get_kernel(2).q0_average(geometry, (3, 3, 1), S_cart=S_cart)


def test_slab_q0_rule_is_a_named_selection_that_refuses_its_alternatives():
    """No silent rule: an unknown name and the 3D sphere key both refuse."""
    import vcoul

    geometry, kgrid = _hex_slab_geometry()
    kernel = vcoul.get_kernel(2)
    zero = np.zeros((3, 3))
    with pytest.raises(ValueError, match="slab_q0_rule_unknown"):
        kernel.q0_average(geometry, kgrid, S_cart=zero, rule="sobol")
    with pytest.raises(
            ValueError, match="slab_q0_analytic_sphere_unavailable"):
        kernel.q0_average(geometry, kgrid, S_cart=zero, analytic_sphere=True)
    # The bulk head keeps its incumbent Sobol + Baldereschi rule and takes
    # no ``rule`` selection: the polygon construction is two-dimensional.
    with pytest.raises(TypeError):
        vcoul.get_kernel(3).q0_average(
            geometry, kgrid, S_cart=zero, rule=vcoul.Q0_RULE_EXACT)


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
    with pytest.raises(ValueError, match="static_photon_cell_nonfinite"):
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
    with pytest.raises(ValueError, match="static_photon_cell_nonfinite"):
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
        complete_static_photon_q0).parameters


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
            layout=None, dimension=2, S_direct=None, sigma_H=None, hall_source="",
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
            None, None, None, None, None, mesh, screen_current=True,
            config=None)


def test_packed_runtime_refuses_no_local_fields_before_opening_a_body():
    mesh = _mesh()
    config = SimpleNamespace(
        head=SimpleNamespace(correction=HeadCorrection.NO_LOCAL_FIELDS),
        bispinor_gw="full_static_cohsex",
    )
    with pytest.raises(ValueError, match="head_correction=full"):
        compute_static_photon_response(
            None, None, None, None, None, mesh, screen_current=True,
            config=config)


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
    """Refused for EVERY bispinor deck now, not only the packed envelope.

    The gate moved up (2026-09-01): ``no_local_fields`` is a scalar
    diagnostic, and on the bare-transverse route it also silently moved a
    deck off the packed path -- a head dial choosing a route.
    """
    with pytest.raises(
            ValueError,
            match="bispinor_head_correction_no_local_fields_unavailable"):
        _parse(tmp_path,
               _packed_deck(extra="head_correction = no_local_fields\n"))


def test_packed_deck_bulk_completion_is_reachable(tmp_path):
    config = _parse(tmp_path, _packed_deck(sys_dim=3))
    assert uses_static_photon_response(config)
    assert uses_coupled_photon_head(config)


def test_packed_deck_off_head_keeps_the_bulk_body_reachable(tmp_path):
    config = _parse(
        tmp_path, _packed_deck(sys_dim=3, extra="head_correction = off\n"))
    assert not uses_coupled_photon_head(config)
    assert uses_static_photon_response(config)


# ---------------------------------------------------------------------------
# ONE envelope table, labelled, with the mode-required settings DERIVED
# (lane J architecture review section 3)
# ---------------------------------------------------------------------------

def test_the_route_and_the_refusal_read_the_same_envelope_table(tmp_path):
    """The envelope had two owners; a condition written twice will differ.

    ``packed_bare_transverse_route``'s conditions and
    ``refuse_unsupported_bispinor_gw``'s requirements restated five
    conjuncts with separately formatted got/want strings.  Both now walk
    ``packed_static_envelope``.
    """
    import inspect
    from gw import gw_config

    for fn in (gw_config.packed_bare_transverse_route,
               gw_config.refuse_unsupported_bispinor_gw):
        assert "packed_static_envelope(" in inspect.getsource(fn), fn.__name__
    shared = [row[2] for row in gw_config.packed_static_envelope(
        _parse(tmp_path, _packed_deck()), screened=False)]
    both = [row[2] for row in gw_config.packed_static_envelope(
        _parse(tmp_path, _packed_deck()), screened=True)]
    assert both[:len(shared)] == shared
    assert len(both) == len(shared) + 2
    assert "sys_dim in {2, 3}" in shared
    assert "w_dyson_solver = distributed" in shared


def test_material_class_is_owned_by_wfn_validation_not_the_envelope(tmp_path):
    """The removed material-class deck key must not survive as a shadow row."""
    from gw import gw_config

    config = _parse(tmp_path, _packed_deck())
    rows = list(gw_config.packed_static_envelope(config, screened=True))
    assert all("material_class" not in row[1] + row[2] for row in rows)


def test_every_envelope_row_says_physics_or_implementation_limit(tmp_path):
    """A refusal that does not say WHICH cannot be acted on.

    "the envelope refuses this" tells a reader nothing about whether the
    deck is wrong or the tree is incomplete.
    """
    from gw import gw_config
    config = _parse(tmp_path, _packed_deck())
    for row in gw_config.packed_static_envelope(config, screened=True):
        assert row[3] in (gw_config._ENV_PHYSICS, gw_config._ENV_IMPL), row


def test_the_eight_scalar_head_overrides_are_one_conjunct(tmp_path):
    """Eight hand-written got/want rows became one predicate that names
    only what the deck actually set."""
    from gw.gw_config import scalar_head_overrides_named
    assert scalar_head_overrides_named(_parse(tmp_path, _packed_deck())) == ()
    with pytest.raises(ValueError) as exc:
        _parse(tmp_path, _packed_deck(
            extra="wcoul0_eta = 0.05\nuse_bgw_vcoul = true\n"))
    message = str(exc.value)
    assert "static_bispinor_photon_envelope" in message
    assert "use_bgw_vcoul = true" in message
    assert "wcoul0_eta = 0.05" in message
    assert "no scalar q->0 head override named" in message
    # and it names ONLY what was set
    assert "vhead" not in message and "mc_average_placement" not in message


def test_mode_required_settings_are_derived_from_the_envelope_table(tmp_path):
    """``low_mem_bands`` / ``w_dyson_solver`` are the only layout and the
    only Dyson plan the packed screened mode has, so the deck must not
    have to write them."""
    lines = []
    deck = (_packed_deck()
            .replace("low_mem_bands = true\n", "")
            .replace("w_dyson_solver = distributed\n", ""))
    path = tmp_path / "derived.in"
    path.write_text(deck)
    config = LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert config.memory.low_mem_bands is True
    assert str(config.backend.w_dyson_solver) == "distributed"
    assert uses_static_photon_response(config)
    assert any("low_mem_bands was not named" in ln for ln in lines), lines
    assert any("w_dyson_solver was not named" in ln for ln in lines), lines


def test_an_explicit_conflicting_value_is_still_refused_not_overridden(
        tmp_path):
    """Rule 13: derive what the deck left unsaid, refuse what it said."""
    deck = _packed_deck().replace(
        "low_mem_bands = true", "low_mem_bands = false")
    with pytest.raises(ValueError) as exc:
        _parse(tmp_path, deck)
    message = str(exc.value)
    assert "static_bispinor_photon_envelope" in message
    assert "low_mem_bands = false" in message


def test_a_deck_outside_the_envelope_still_sees_its_own_reason(tmp_path):
    """The promotion must not fire for a deck that is outside the envelope
    for some OTHER reason, or a bad deck would be told about a key it
    never wrote."""
    deck = (_packed_deck()
            .replace("low_mem_bands = true\n", "")
            .replace("compute_mode = cohsex", "compute_mode = mpa"))
    # ``mpa`` is the one dynamic mode OUTSIDE PACKED_PHOTON_COMPUTE_MODES
    # (the plasmon-pole pair joined it with the dynamic packed route, lane N).
    with pytest.raises(ValueError) as exc:
        _parse(tmp_path, deck)
    message = str(exc.value)
    assert "compute_mode = mpa" in message
    assert "low_mem_bands" not in message


# ---------------------------------------------------------------------------
# Heads are always on: the incumbent route must say what it did, and
# `restart` must not swap the head mechanism behind the deck's back
# (owner ruling 2026-09-01 / TASTE.md row 20; lane J section 3)
# ---------------------------------------------------------------------------

_INCUMBENT_DECK = (
    "[cohsex]\n"
    "nval = 2\n"
    "ncond = 2\n"
    "number_bands = 8\n"
    "memory_per_device_gb = 4.0\n"
    "sys_dim = 3\n"
    "bispinor = true\n"
    "bispinor_gw = bare_transverse\n"
    "compute_mode = cohsex\n"
    "restart = false\n")


def test_bulk_head_off_takes_the_packed_debug_route(
        tmp_path):
    config = _parse(tmp_path,
                    _INCUMBENT_DECK + "head_correction = off\n")
    assert uses_static_photon_response(config)
    assert not uses_coupled_photon_head(config)


def test_bulk_head_full_takes_the_packed_completion(tmp_path):
    config = _parse(tmp_path, _INCUMBENT_DECK)
    assert uses_static_photon_response(config)
    assert uses_coupled_photon_head(config)


def test_the_driver_prints_the_incumbent_head_record(tmp_path):
    """The one owner is gw_config; gw_jax must not grow a second copy."""
    import inspect
    from gw import gw_jax
    src = inspect.getsource(gw_jax.main)
    assert "incumbent_bispinor_head_record(config)" in src
    assert "Photon head    : " in src


def test_the_driver_replays_config_provenance_into_the_production_report():
    """Unnamed physics defaults must survive the production chatter filter."""
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src/gw/gw_jax.py").read_text()
    assert 'if "[config provenance]" in text:' in src
    assert 'report.heading("Configuration provenance")' in src
    assert "report.emit(line.strip())" in src


def test_restart_may_not_swap_the_head_mechanism_on_a_slab_cohsex_deck(
        tmp_path):
    """restart = true used to move a slab bispinor COHSEX deck from the
    packed Gamma-cell completion to the incumbent scalar head, silently,
    for 5.7 meV plus the whole transverse head."""
    deck = (_INCUMBENT_DECK
            .replace("sys_dim = 3", "sys_dim = 2")
            .replace("restart = false", "restart = true"))
    with pytest.raises(ValueError) as exc:
        _parse(tmp_path, deck)
    message = str(exc.value)
    assert ("bispinor_slab_cohsex_restart_changes_the_head_mechanism"
            in message)
    # it must NAME BOTH mechanisms, not just say "unsupported"
    assert "Wigner-Seitz" in message and "StaticHeadTerms" in message
    assert "5.72 meV" in message
    assert "IMPLEMENTATION LIMIT" in message


def test_an_unnamed_restart_uses_the_fresh_physics_default(tmp_path):
    """The global default is fresh physics and its provenance is visible."""
    lines = []
    deck = (_INCUMBENT_DECK
            .replace("sys_dim = 3", "sys_dim = 2")
            .replace("restart = false\n", ""))
    path = tmp_path / "unnamed_restart.in"
    path.write_text(deck)
    config = LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert config.restart is False
    assert uses_static_photon_response(config)
    assert any("restart was not named" in ln for ln in lines), lines
    assert any("restart = false" in ln for ln in lines), lines
    assert any("authenticated restart loader" in ln for ln in lines), lines


def test_a_scalar_deck_gets_the_same_fresh_restart_default(tmp_path):
    deck = (_INCUMBENT_DECK
            .replace("sys_dim = 3", "sys_dim = 2")
            .replace("bispinor = true\n", "")
            .replace("bispinor_gw = bare_transverse\n", "")
            .replace("restart = false\n", ""))
    path = tmp_path / "scalar_restart.in"
    path.write_text(deck)
    lines = []
    config = LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert config.restart is False
    assert any("[config provenance] restart was not named" in line
               for line in lines)


def test_a_bulk_bispinor_restart_deck_is_untouched(tmp_path):
    """The gate is aimed at the deck class the packed route would have
    served, not at every bispinor restart."""
    config = _parse(tmp_path,
                    _INCUMBENT_DECK.replace("restart = false",
                                            "restart = true"))
    assert config.restart is True
    assert not uses_static_photon_response(config)


def test_retired_charge_hall_cubature_spelling_names_the_new_mode(tmp_path):
    deck = _packed_deck().replace(
        "bispinor_gw = full_static_cohsex",
        "bispinor_gw = charge_hall_cubature")
    with pytest.raises(
            ValueError,
            match="(?s)bispinor_gw_charge_hall_cubature_retired.*"
                  "full_static_cohsex"):
        _parse(tmp_path, deck)
