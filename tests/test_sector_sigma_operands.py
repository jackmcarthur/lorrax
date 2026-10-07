"""Sector Σ operand forms against their per-node originals (CPU 2x2 mesh).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
* The axis route's W(τ) with its right operand formed once per Σ call
  (``shared_pole_right_operand``, ``right_formed=True``) equals the per-node form,
  the mixed sector's partner (by conjugate weights) included.
* One reduce-scatter of every band bracket's partial (``BandProjector.finish(stacked=True)``)
  equals one per bracket.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


@pytest.mark.parametrize("partner", [False, True])
def test_formed_right_operand_matches_the_per_node_form(partner):
    from distrib_la import gemm_plan
    from gw.mpa.sigma import _shared_pole_contract, shared_pole_right_operand
    mesh = _mesh()
    rng = np.random.default_rng(7)
    q, m, n, nc, nt, K = 3, 8, 12, 1, 3, 10
    c = lambda *s: jnp.asarray(rng.normal(size=s) + 1j * rng.normal(size=s))
    b_x = jax.device_put(c(q, m, nc, K), NamedSharding(mesh, P(None, 'x', None, None)))
    b_y = jax.device_put(c(q, n, nt, K), NamedSharding(mesh, P(None, 'y', None, None)))
    weights = jax.device_put(c(q, K), NamedSharding(mesh, P()))
    intervals = jnp.asarray([[0, K], [2, 7], [1, 9]], jnp.int32)
    gemm = gemm_plan(mesh, m=m * nc, k=K, n=n * nt, nq=q, dtype=jnp.complex128, layout='axis',
                     enable_active_range=True, warmup=False)
    before = _shared_pole_contract(b_x, b_y, weights, gemm=gemm, layout='axis', intervals=intervals,
                                   partner=partner)
    right = shared_pole_right_operand(b_y)
    after = _shared_pole_contract(b_x, right, weights, gemm=gemm, layout='axis', intervals=intervals,
                                  partner=partner, right_formed=True)
    # The W(τ) owner takes the formed operand too (its shape checks included).
    from gw.mpa.sigma import synthesize_shared_pole_parents
    poles = jnp.asarray(rng.uniform(0.1, 2.0, (q, K)))
    owner = lambda y, formed: synthesize_shared_pole_parents(
        b_x, y, poles, intervals, 0.0, 1.0 + 0.5j, mesh_xy=mesh, gemm=gemm, layout='axis',
        active_range=True, same_factor=not partner, right_formed=formed)
    for a, b in zip(jax.tree.leaves(owner(b_y, False)), jax.tree.leaves(owner(right, True))):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=0, atol=1e-12)
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=0, atol=1e-12)
    if partner:
        # The partner is conj(b_x) d b_yᵀ, the conjugate-weight form of W = b_x d b_y†.
        x = np.asarray(b_x).reshape(q, m * nc, K)
        y = np.asarray(b_y).reshape(q, n * nt, K)
        d = np.asarray(weights) * (np.arange(K)[None] >= np.asarray(intervals)[:, :1]) \
            * (np.arange(K)[None] < np.asarray(intervals)[:, 1:])
        want = np.einsum('qmk,qk,qnk->qmn', np.conj(x), d, y)
        np.testing.assert_allclose(np.asarray(after[1]), want, rtol=0, atol=1e-12)


def test_stacked_finish_matches_one_reduce_scatter_per_bracket():
    from common.contract_bands import contract_bands_block_reshard
    mesh = _mesh()
    nk, nb, mu, ns = 2, 4, 8, 1
    project = contract_bands_block_reshard(mesh, channels="none", layout="axis",
                                           face_shape=(nk, nb, mu, ns), right_face_shape=(nk, nb, mu, ns),
                                           face_band_extent=nb)
    rng = np.random.default_rng(1)
    spec = NamedSharding(mesh, P(None, ('x', 'y')))
    accs = [jax.device_put(jnp.asarray(rng.normal(size=(1, 4 * nk, nb, nb))), spec) for _ in range(3)]
    one = project.finish(jnp.concatenate(accs, axis=0), stacked=True)
    each = jnp.stack([project.finish(a) for a in accs])
    assert np.array_equal(np.asarray(one), np.asarray(each))
