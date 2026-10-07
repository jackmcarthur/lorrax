"""The photon bank's resident Dyson operands against the moving ones (CPU 2x2 mesh).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
The photon pair with V, the contact and W_inf - V laid out once in the batch layout
(``distrib_la.batch_layout`` / ``batch_broadcast``) gives the same bits as the pair that
moves them face -> batch on every call; ``batch_broadcast`` equals the broadcast copies.
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


def test_batch_broadcast_is_the_broadcast_copies():
    from distrib_la import batch_broadcast, batch_layout
    mesh = _mesh()
    face = NamedSharding(mesh, P(None, "x", "y"))
    rng = np.random.default_rng(2)
    one = jax.device_put(jnp.asarray(rng.normal(size=(1, 8, 8)) + 1j * rng.normal(size=(1, 8, 8))), face)
    got = batch_broadcast(one, mesh, 7)
    want = batch_layout(jax.device_put(jnp.broadcast_to(one, (7, 8, 8)), face), mesh)
    assert got.shape == want.shape == (8, 8, 8)
    assert np.array_equal(np.asarray(got)[:7], np.asarray(want)[:7])


def test_resident_photon_pair_matches_the_moving_pair_bitwise():
    from distrib_la import batch_broadcast, batch_layout
    from gw.response_bank import _response_programs
    mesh = _mesh()
    face = NamedSharding(mesh, P(None, "x", "y"))
    rng = np.random.default_rng(5)
    nq, n = 5, 8
    c = lambda *s: rng.normal(size=s) + 1j * rng.normal(size=s)
    put = lambda a: jax.device_put(jnp.asarray(a), face)
    v = put(0.1 * c(nq, n, n) + np.eye(n))
    chi, dchi, constant = put(0.05 * c(nq, n, n)), put(0.05 * c(nq, n, n)), put(0.02 * c(nq, n, n))
    contact = put(0.01 * c(1, n, n))
    dyson = _response_programs(mesh, n, "off", "batch_reshard", 0.5, True, 2.0)[0]
    moving = dyson.pair("face")(v, chi, dchi, contact, constant)
    resident = dyson.pair("resident")(batch_layout(v, mesh), chi, dchi, batch_broadcast(contact, mesh, nq),
                                      batch_layout(constant, mesh))
    for a, b in zip(moving, resident):
        assert np.array_equal(np.asarray(a), np.asarray(b))
