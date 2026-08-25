"""Focused algebra and fail-closed gates for the static photon q=0 schema."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.gw_config import (
    HeadCorrection,
    LorraxConfig,
)
from gw.head_correction import (
    StaticGaugeHeadResponse,
    require_static_gauge_head_response,
    static_gauge_tensor_residuals,
    static_hall_linear_response,
    static_slab_photon_head_moment_chunk,
)
from gw.photon_layout import PhotonBasisLayout
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


def _response(mesh, *, S=None, **changes):
    layout = PhotonBasisLayout.from_centroid_extents(1, 1, mesh)
    n_body = layout.packed_extent
    base = StaticGaugeHeadResponse(
        layout=layout,
        S_direct=jax.device_put(
            _ward_closed_S() if S is None else S,
            NamedSharding(mesh, P()),
        ),
        sigma_H=np.asarray((0.0, 0.0, 0.37), dtype=np.float64),
        Y_x=jax.device_put(
            np.zeros((2, 4, n_body), dtype=np.complex128),
            NamedSharding(mesh, P(None, None, "x")),
        ),
        Z_y=jax.device_put(
            np.zeros((2, n_body, 4), dtype=np.complex128),
            NamedSharding(mesh, P(None, "y", None)),
        ),
        hamiltonian_config_operator_fingerprint="sha256:" + "a" * 64,
        operator_current_equivalent=True,
        contact_is_exact=True,
        ward_residual=0.0,
        hermiticity_residual=0.0,
    )
    return replace(base, **changes)


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

    moments, D_sum, count, residual = static_slab_photon_head_moment_chunk(
        q, D, sigma, S, 2,
    )

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


def test_static_gauge_response_accepts_a_nonzero_ward_closed_tensor():
    mesh = _mesh()
    response = _response(mesh)
    ward, hermiticity = static_gauge_tensor_residuals(response.S_direct)
    assert ward < 1.0e-15
    assert hermiticity == 0.0
    assert require_static_gauge_head_response(response, mesh) is response


def test_static_gauge_response_refuses_ward_breaking_tensor():
    mesh = _mesh()
    S = _ward_closed_S()
    S[0, 0, 1, 1] += 0.2
    with pytest.raises(ValueError, match="static_gauge_head_ward"):
        require_static_gauge_head_response(_response(mesh, S=S), mesh)


def test_static_gauge_response_refuses_nonhermitian_tensor():
    mesh = _mesh()
    S = _ward_closed_S()
    S[0, 0, 0, 3] = 0.2j
    with pytest.raises(ValueError, match="static_gauge_head_hermiticity"):
        require_static_gauge_head_response(_response(mesh, S=S), mesh)


def test_static_gauge_response_refuses_a_wrong_wing_axis():
    mesh = _mesh()
    if int(mesh.shape["x"]) == 1:
        pytest.skip("x/y shardings are equivalent on a 1x1 mesh")
    response = _response(mesh)
    wrong_Y = jax.device_put(
        np.zeros(response.Y_x.shape, dtype=np.complex128),
        NamedSharding(mesh, P(None, None, "y")),
    )
    with pytest.raises(ValueError, match="Y_x must arrive"):
        require_static_gauge_head_response(
            replace(response, Y_x=wrong_Y), mesh)


@pytest.mark.parametrize(
    ("changes", "gate"),
    (
        ({"operator_current_equivalent": False}, "static_gauge_head_operator"),
        ({"contact_is_exact": False}, "static_gauge_head_contact"),
        ({"hamiltonian_config_operator_fingerprint": ""},
         "static_gauge_head_fingerprint"),
    ),
)
def test_static_gauge_response_refuses_missing_physics_provenance(changes, gate):
    mesh = _mesh()
    with pytest.raises(ValueError, match=gate):
        require_static_gauge_head_response(
            _response(mesh, **changes), mesh)


@pytest.mark.parametrize("caller_response", (None, "fabricated"))
def test_full_screened_runtime_refuses_before_opening_a_body(caller_response):
    mesh = _mesh()
    config = SimpleNamespace(
        head=SimpleNamespace(correction=HeadCorrection.FULL),
    )
    response = None if caller_response is None else _response(mesh)
    with pytest.raises(
            ValueError, match="static_gauge_head_producer_unavailable"):
        compute_static_photon_response(
            None, None, None, None, None, mesh, config=config,
            gauge_head_response=response,
        )


def test_full_screened_deck_refuses_on_the_real_parse_path(tmp_path):
    deck = tmp_path / "full_static_bispinor.in"
    deck.write_text(
        "[cohsex]\n"
        "nval = 2\n"
        "ncond = 2\n"
        "number_bands = 8\n"
        "memory_per_device_gb = 4.0\n"
        "bispinor = true\n"
        "bispinor_gw = full_static_cohsex\n"
        "head_correction = full\n"
    )
    with pytest.raises(
            ValueError, match="full_static_bispinor_gauge_head_unavailable"):
        LorraxConfig.from_input_file(str(deck), print_fn=lambda *_: None)
