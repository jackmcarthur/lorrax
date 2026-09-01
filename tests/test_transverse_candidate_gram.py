"""Synthetic gates for the transverse pivoted-Cholesky candidate metric.

The fit CCT for one transverse Lorentz component is Hermitian indefinite;
it is not a legal pivoted-Cholesky target.  Candidate pruning instead uses
the Gram of the stacked current transition densities,
``G_perp = sum_i Z_i Z_i^H``.  These tests compare the pair-density
factorisation against those explicit features and pin the two consequences
that matter for selection: Cartesian rotation covariance and additive
leverage from a point supported in only one component.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh


def _one_device_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _synthetic_pair_problem(seed: int = 20260830):
    """Return physical pair densities and their explicit transverse Z_i."""
    from common.gamma_matrices import gamma1, gamma2, gamma3
    from isdf import pair_density

    rng = np.random.default_rng(seed)
    nk, nl, nr, ns, npoint = 2, 3, 4, 4, 6
    left = (rng.standard_normal((nk, nl, ns, npoint))
            + 1j * rng.standard_normal((nk, nl, ns, npoint)))
    right = (rng.standard_normal((nk, nr, ns, npoint))
             + 1j * rng.standard_normal((nk, nr, ns, npoint)))
    weights = np.asarray([0.375, 0.625], dtype=np.float64)
    mesh = _one_device_mesh()

    # Canonical pair_density inputs: X is already conjugated and has
    # (k, point, band, spin); Y has (k, band, spin, point).
    P_l = pair_density(
        jnp.asarray(np.conj(left).transpose(0, 3, 1, 2)),
        jnp.asarray(left), mesh)
    P_r = pair_density(
        jnp.asarray(np.conj(right).transpose(0, 3, 1, 2)),
        jnp.asarray(right), mesh)

    gamma = np.asarray(jax.device_get(jnp.stack((gamma1, gamma2, gamma3))))
    # A[i,k,a,n,m] = left[k,n,:,a]^H gamma[i] right[k,m,:,a].
    # The production feature convention is Z=conj(A), so Z Z^H is exactly
    # conj(A[a]) A[b], the contraction returned below.
    features = np.einsum(
        "knsa,ist,kmta->ikanm", np.conj(left), gamma, right,
        optimize=True)
    return mesh, P_l, P_r, weights, features


def _explicit_stacked_gram(features, weights):
    return np.einsum(
        "k,ikanm,ikbnm->ab", weights, np.conj(features), features,
        optimize=True)


def test_transverse_pair_fold_is_the_hermitian_psd_feature_gram():
    """The production fold equals explicit stacked transition features."""
    from centroid.pivoted_cholesky import candidate_gram_q0_from_pair

    mesh, P_l, P_r, weights, features = _synthetic_pair_problem()
    got = np.asarray(candidate_gram_q0_from_pair(
        P_l, P_r, jnp.asarray(weights), mesh_xy=mesh,
        gamma_mode="transverse"))
    expected = _explicit_stacked_gram(features, weights)

    np.testing.assert_allclose(got, expected, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(got, got.conj().T, rtol=0.0, atol=0.0)
    eig = np.linalg.eigvalsh(got)
    assert eig[0] >= -3e-13 * max(1.0, float(eig[-1])), eig


def test_summed_transverse_gram_is_cartesian_rotation_invariant():
    """Equal unnormalised component weights make the sum frame-covariant."""
    mesh, P_l, P_r, weights, features = _synthetic_pair_problem(seed=19)
    del mesh, P_l, P_r
    rng = np.random.default_rng(23)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    rotated = np.einsum("ij,jkanm->ikanm", Q, features, optimize=True)

    G = _explicit_stacked_gram(features, weights)
    G_rot = _explicit_stacked_gram(rotated, weights)
    np.testing.assert_allclose(G_rot, G, rtol=8e-15, atol=8e-13)

    # Negative control: component-wise normalisation is coordinate-frame
    # dependent and must not be introduced as an innocuous rescaling.
    norm = np.linalg.norm(features.reshape(3, -1), axis=1)
    norm_rot = np.linalg.norm(rotated.reshape(3, -1), axis=1)
    G_bad = _explicit_stacked_gram(features / norm[:, None, None, None, None],
                                   weights)
    G_bad_rot = _explicit_stacked_gram(
        rotated / norm_rot[:, None, None, None, None], weights)
    assert not np.allclose(G_bad_rot, G_bad, rtol=1e-10, atol=1e-10)


def test_point_supported_in_only_one_component_keeps_its_leverage():
    """A large one-component feature can be the first selected point."""
    from common.pivoted_cholesky import pivoted_cholesky_select

    # Three components, three candidates, two feature columns. Candidate 2
    # exists ONLY in component 3; summing component Grams gives it the
    # largest residual diagonal and it must survive a one-point prune.
    features = np.zeros((3, 3, 2), dtype=np.complex128)
    features[0, 0, 0] = 1.0
    features[1, 1, 1] = 2.0
    features[2, 2, 0] = 7.0
    G = np.einsum("iaf,ibf->ab", features, np.conj(features),
                  optimize=True)
    piv, *_ = pivoted_cholesky_select(jnp.asarray(G), 1)
    assert int(np.asarray(piv)[0]) == 2
    assert np.real(G[2, 2]) == 49.0


def test_charge_candidate_fold_is_bit_identical_to_the_scalar_owner():
    """The new mode dispatch leaves the historical scalar path untouched."""
    from centroid.pivoted_cholesky import candidate_gram_q0_from_pair
    from isdf import gram_q0_from_pair

    mesh, P_l, P_r, weights, _ = _synthetic_pair_problem(seed=41)
    kw = jnp.asarray(weights)
    direct = gram_q0_from_pair(P_l, P_r, kw, mesh_xy=mesh)
    dispatched = candidate_gram_q0_from_pair(
        P_l, P_r, kw, mesh_xy=mesh, gamma_mode="charge")
    assert np.array_equal(np.asarray(dispatched), np.asarray(direct))


def test_transverse_orbit_block_deflates_every_emitted_current_point():
    """One representative per orbit does not span a generic current Gram."""
    from centroid.pivoted_cholesky import candidate_gram_q0_from_pair
    from common.pivoted_cholesky import (
        group_block_pivoted_cholesky_select,
        pivoted_cholesky_select,
    )

    mesh, P_l, P_r, weights, _ = _synthetic_pair_problem(seed=260831)
    G = np.asarray(candidate_gram_q0_from_pair(
        P_l, P_r, jnp.asarray(weights), mesh_xy=mesh,
        gamma_mode="transverse"))
    group_id = np.repeat(np.arange(3, dtype=np.int32), 2)
    assert np.linalg.matrix_rank(G, tol=1.0e-11 * np.linalg.norm(G, 2)) == 6

    # Historical behavior: three representatives authorize all six emitted
    # points, but only three current-feature directions enter the recurrence.
    representative = pivoted_cholesky_select(
        jnp.asarray(G), 3, jnp.asarray(group_id))
    assert int(representative[2]) == 3
    rep_L = np.asarray(representative[1])
    rep_residual = G - rep_L @ rep_L.conj().T

    # A point budget cannot authorize a partial two-point orbit.
    bounded = group_block_pivoted_cholesky_select(
        jnp.asarray(G), 5, jnp.asarray(group_id), n_groups=3)
    bounded_piv = np.asarray(bounded[0])
    assert np.count_nonzero(bounded_piv >= 0) == 4
    assert bounded_piv[-1] == -1
    bounded_counts = np.bincount(
        group_id[bounded_piv[bounded_piv >= 0]], minlength=3)
    assert set(bounded_counts.tolist()) <= {0, 2}

    # Production behavior: the same six-point delivery pivots both members
    # of every complete orbit before another orbit is scored.
    block = group_block_pivoted_cholesky_select(
        jnp.asarray(G), 6, jnp.asarray(group_id), n_groups=3)
    piv = np.asarray(block[0])
    assert np.all(piv >= 0)
    assert int(block[2]) == 6
    np.testing.assert_array_equal(
        np.bincount(group_id[piv], minlength=3), np.full(3, 2))
    block_L = np.asarray(block[1])
    block_residual = G - block_L @ block_L.conj().T

    scale = max(1.0, float(np.linalg.norm(G)))
    assert np.linalg.norm(rep_residual) > 1.0e-4 * scale
    assert np.linalg.norm(block_residual) < 2.0e-12 * scale
