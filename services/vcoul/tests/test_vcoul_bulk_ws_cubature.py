"""Analytic checks for the arbitrary-3D exact mini-BZ provider.

The skew and hexagonal cases deliberately expose row/column mistakes; the
cube separately exercises the six-face Wigner--Seitz topology.
"""
import numpy as np
import pytest

import vcoul


SKEW_BVEC = np.asarray([
    [2.10, 0.35, 0.20],
    [-0.25, 1.70, 0.45],
    [0.15, -0.30, 1.25],
], dtype=np.float64)
SKEW_KGRID = (2, 3, 4)
FCC_BVEC = 0.5 * np.asarray([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])
BCC_BVEC = 0.5 * np.asarray([
    [-1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, -1.0],
])


def _receipt(bvec, kgrid):
    return vcoul.bulk_minibz_photon_cubature(
        vcoul.get_kernel(3),
        vcoul.CoulombGeometry(bvec=bvec, cell_volume=7.0),
        kgrid,
    )


def _order_moments(receipt, *, check_halfspaces=False):
    final_sites = None
    if check_halfspaces:
        axes = [range(-bound, bound + 1)
                for bound in receipt.lattice_index_bounds]
        indices = np.stack(
            np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        indices = indices[np.any(indices != 0, axis=1)]
        final_sites = vcoul.minibz_frac_to_cart(
            indices, np.asarray(receipt.mini_lattice_rows))
    accumulators = {
        order: [0.0, np.zeros(3), np.zeros((3, 3)), [], 0]
        for order in receipt.orders
    }
    for chunk in vcoul.iter_bulk_minibz_photon_cubature(receipt):
        total_weight, centroid, second_moment, sample, chunks = (
            accumulators[chunk.order])
        chunks += 1
        physical = chunk.physical_count
        q = np.asarray(chunk.q_cart[:physical])
        weight = np.asarray(chunk.sample_weight[:physical])
        assert chunk.q_cart.shape == (receipt.padded_chunk_count, 3)
        assert chunk.D_raw.shape == (receipt.padded_chunk_count, 4, 4)
        assert chunk.sample_weight.shape == (receipt.padded_chunk_count,)
        assert physical == 2 * chunk.order ** 3
        assert np.all(weight > 0.0)
        assert np.array_equal(q[1::2], -q[::2])
        assert np.array_equal(weight[1::2], weight[::2])
        assert np.array_equal(
            chunk.D_raw[1:physical:2], chunk.D_raw[:physical:2])
        q2 = np.einsum("ni,ni->n", q, q)
        cc = chunk.D_raw[:physical, 0, 0]
        tt = chunk.D_raw[:physical, 1:, 1:]
        projector = (np.eye(3)[None, :, :]
                     - np.einsum("ni,nj->nij", q, q) / q2[:, None, None])
        np.testing.assert_allclose(
            q2 * cc, 8.0 * np.pi, rtol=2.0e-15, atol=2.0e-14)
        normalized_tt = tt / cc[:, None, None]
        np.testing.assert_allclose(
            normalized_tt, -projector, rtol=0.0, atol=5.0e-15)
        qhat = q / np.sqrt(q2)[:, None]
        np.testing.assert_allclose(
            np.einsum("nij,nj->ni", normalized_tt, qhat), 0.0,
            rtol=0.0, atol=5.0e-15)
        if final_sites is not None:
            site_norm2 = np.einsum("ni,ni->n", final_sites, final_sites)
            violations = q @ final_sites.T - 0.5 * site_norm2[None, :]
            scale = np.max(np.linalg.norm(
                np.asarray(receipt.mini_lattice_rows), axis=1))
            assert np.max(violations) <= (
                32768.0 * np.finfo(np.float64).eps * scale ** 2)
        assert np.all(chunk.q_cart[physical:] == 0.0)
        assert np.all(chunk.D_raw[physical:] == 0.0)
        assert np.all(chunk.sample_weight[physical:] == 0.0)
        total_weight += float(np.sum(weight))
        centroid += np.einsum("n,ni->i", weight, q)
        second_moment += np.einsum("n,ni,nj->ij", weight, q, q)
        sample.append(q[:4].copy())
        accumulators[chunk.order] = [
            total_weight, centroid, second_moment, sample, chunks]
    result = {}
    for order, values in accumulators.items():
        total_weight, centroid, second_moment, sample, chunks = values
        assert chunks == receipt.chunks_per_order
        result[order] = (
            total_weight, centroid, second_moment, np.concatenate(sample))
    return result


def test_skew_ws_polyhedron_has_the_lattice_volume_and_wraps_to_itself():
    receipt = _receipt(SKEW_BVEC, SKEW_KGRID)
    mini_lattice = vcoul.minibz_frac_to_cart(
        np.diag(1.0 / np.asarray(SKEW_KGRID)), SKEW_BVEC)
    expected_volume = abs(float(np.linalg.det(mini_lattice)))
    assert receipt.orders == (8, 12, 16)
    assert receipt.polyhedron_volume == expected_volume
    assert 6 <= receipt.face_count <= 14
    assert receipt.face_count % 2 == 0
    assert receipt.chunks_per_order > 0
    assert receipt.physical_counts == tuple(
        2 * receipt.chunks_per_order * order ** 3
        for order in receipt.orders)
    assert receipt.quadrature_provenance == (
        vcoul.GAUSS_LEGENDRE_INTERVAL_PROVENANCE)
    assert len(receipt.quadrature_digest) == 64
    assert vcoul.validate_bulk_minibz_photon_receipt(receipt) is receipt
    moments = _order_moments(receipt, check_halfspaces=True)
    for total_weight, centroid, _, _ in moments.values():
        np.testing.assert_allclose(
            total_weight, 1.0, rtol=0.0, atol=2.0e-13)
        np.testing.assert_allclose(
            centroid, 0.0, rtol=0.0, atol=2.0e-15)

    # This is an independent geometry check through the pre-existing
    # canonical WS wrapping owner.  Interior cubature nodes must choose the
    # zero lattice shift.
    wrapped = np.asarray(vcoul.wrap_points_to_voronoi(
        moments[receipt.orders[0]][3], mini_lattice,
        nmax=max(receipt.lattice_index_bounds)))
    np.testing.assert_allclose(
        wrapped, moments[receipt.orders[0]][3], rtol=0.0, atol=2.0e-13)


def test_hexagonal_prism_rule_reproduces_analytic_second_moment():
    in_plane = 1.8
    out_of_plane = 1.25
    bvec = np.asarray([
        [in_plane, 0.0, 0.0],
        [0.5 * in_plane, 0.5 * np.sqrt(3.0) * in_plane, 0.0],
        [0.0, 0.0, out_of_plane],
    ])
    kgrid = (2, 2, 3)
    receipt = _receipt(bvec, kgrid)
    moments = _order_moments(receipt)

    # The WS cell is a regular hexagonal prism.  A regular hexagon whose
    # nearest-neighbour lattice spacing is a has
    # <x^2>=<y^2>=5 a^2/72; a centered interval of length c has <z^2>=c^2/12.
    a = in_plane / kgrid[0]
    c = out_of_plane / kgrid[2]
    expected = np.diag((5.0 * a * a / 72.0,
                        5.0 * a * a / 72.0,
                        c * c / 12.0))
    for total_weight, centroid, second_moment, _ in moments.values():
        np.testing.assert_allclose(
            total_weight, 1.0, rtol=0.0, atol=2.0e-13)
        np.testing.assert_allclose(
            centroid, 0.0, rtol=0.0, atol=2.0e-15)
        np.testing.assert_allclose(
            second_moment, expected, rtol=2.0e-12, atol=2.0e-14)


def test_cubic_rule_reproduces_analytic_second_moment():
    lattice_spacing = 1.5
    kgrid = (3, 3, 3)
    receipt = _receipt(np.eye(3) * lattice_spacing, kgrid)
    moments = _order_moments(receipt)
    mini_spacing = lattice_spacing / kgrid[0]
    expected = np.eye(3) * mini_spacing ** 2 / 12.0

    assert receipt.face_count == 6
    assert receipt.chunks_per_order == 6
    for total_weight, centroid, second_moment, _ in moments.values():
        np.testing.assert_allclose(
            total_weight, 1.0, rtol=0.0, atol=2.0e-13)
        np.testing.assert_allclose(
            centroid, 0.0, rtol=0.0, atol=2.0e-15)
        np.testing.assert_allclose(
            second_moment, expected, rtol=2.0e-12, atol=2.0e-14)


def test_bulk_ws_provider_refuses_a_nonbulk_kernel():
    with pytest.raises(TypeError, match="exact Bulk3D kernel"):
        vcoul.bulk_minibz_photon_cubature(
            vcoul.get_kernel(2),
            vcoul.CoulombGeometry(bvec=SKEW_BVEC, cell_volume=7.0),
            SKEW_KGRID,
        )


@pytest.mark.parametrize(
    ("bvec", "face_count", "tetrahedron_pairs"),
    ((FCC_BVEC, 12, 12), (BCC_BVEC, 14, 22)),
    ids=("fcc-rhombic-dodecahedron", "bcc-truncated-octahedron"),
)
def test_fcc_bcc_ws_topology_twins_are_canonical(
        bvec, face_count, tetrahedron_pairs):
    first = _receipt(bvec, (1, 1, 1))
    second = _receipt(bvec, (1, 1, 1))
    assert first.face_count == face_count
    assert first.chunks_per_order == tetrahedron_pairs
    assert first.positive_tetrahedra == second.positive_tetrahedra
    assert first._provider_digest == second._provider_digest


def test_bulk_receipt_authentication_and_geometry_budgets_fail_closed():
    import vcoul.minibz as minibz

    receipt = _receipt(np.eye(3), (2, 2, 2))
    object.__setattr__(receipt, "quadrature_digest", "0" * 64)
    with pytest.raises(ValueError, match="quadrature provenance"):
        vcoul.validate_bulk_minibz_photon_receipt(receipt)
    with pytest.raises(ValueError, match="site budget"):
        minibz._bulk_minibz_site_count((100, 100, 100))
    ill_conditioned = np.diag((1.0, 1.0, 1.0e-11))
    with pytest.raises(ValueError, match="condition-number"):
        _receipt(ill_conditioned, (1, 1, 1))
