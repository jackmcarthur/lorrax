"""Cached distributed programs must not keep caller matrices or tracers alive."""
import gc
import weakref

import runtime
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from distrib_la import matmul


def _case(mesh, nb, dtype, transa, transb, accumulate):
    side = int(mesh.shape['x'])
    m, k, n = 2*side, 3*side, 4*side
    rng = np.random.default_rng(913 + nb)
    def values(shape):
        a = rng.normal(size=shape)
        if np.issubdtype(dtype, np.complexfloating):
            a = a + 1j*rng.normal(size=shape)
        return a.astype(dtype)
    a = values((nb, m, k) if transa == 'N' else (nb, k, m))
    b = values((nb, k, n) if transb == 'N' else (nb, n, k))
    c = values((nb, m, n)) if accumulate else None
    alpha, beta = 1.315313151, .217 if accumulate else 0.
    def op(a, trans):
        a = a if trans == 'N' else a.swapaxes(-1, -2)
        return a.conj() if trans == 'C' else a
    expected = alpha*(op(a, transa) @ op(b, transb))
    if c is not None:
        expected += beta*c
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    inputs = [device_put_process_local(v, face) if v is not None else None
              for v in (a, b, c)]
    refs = [weakref.ref(v) for v in inputs if v is not None]
    got = matmul(*inputs, mesh=mesh, alpha=alpha, beta=beta,
                 transa=transa, transb=transb, backend='off',
                 batched_route='batch_reshard')
    jax.block_until_ready(got)
    for shard in got.addressable_shards:
        np.testing.assert_allclose(np.asarray(shard.data), expected[shard.index],
                                   rtol=3e-13, atol=3e-13)
    return refs


def cache_ownership_controls(mesh):
    """Literal ragged/transpose/accumulate identities and input lifetime, also MPI-safe."""
    records = []
    for nb, dtype, ta, tb, accumulate in (
            (4, np.float64, 'N', 'N', False),
            (3, np.complex128, 'C', 'T', True),
            (1, np.float64, 'T', 'N', True),
            (5, np.complex128, 'N', 'C', False)):
        refs = _case(mesh, nb, dtype, ta, tb, accumulate)
        gc.collect()
        assert all(ref() is None for ref in refs), 'cached GEMM retained a caller matrix'
        records.append(dict(batch=nb, dtype=np.dtype(dtype).name,
                            transa=ta, transb=tb, accumulate=accumulate))
    # Trace the factory at a fresh geometry: an escaped input tracer is a
    # separate failure even when only its dtype was read by the closure.
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    width = 2*int(mesh.shape['x'])
    a = device_put_process_local(np.ones((7, width, width)), face)
    @jax.jit
    def traced(a):
        return matmul(a, a, mesh=mesh, backend='off', batched_route='batch_reshard')
    with jax.checking_leaks():
        jax.block_until_ready(traced(a))
    return records


def test_cached_matmul_does_not_retain_matrices_or_outer_tracers():
    devices = jax.devices('cpu')
    if len(devices) < 4 or jax.process_count() != 1:
        pytest.skip('requires four emulated CPU devices')
    jax.config.update('jax_enable_x64', True)
    mesh = Mesh(np.asarray(devices[:4]).reshape(2, 2), ('x', 'y'))
    cache_ownership_controls(mesh)
