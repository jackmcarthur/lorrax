"""Sector Σ: one reduce-scatter of every band bracket's partial against one per bracket (CPU 2x2 mesh).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
``BandProjector.finish(stacked=True)`` equals one ``finish`` per bracket, bitwise.
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
