"""Executable reuse must preserve data, donation, dtype and device placement."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

M = importlib.import_module('distrib_la.matmul')


def test_zero_and_transpose_cache_separation():
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices.reshape(2, 2), ('x', 'y'))
    equivalent = Mesh(devices.copy().reshape(2, 2), ('x', 'y'))
    other = Mesh(devices[::-1].reshape(2, 2), ('x', 'y'))
    tile = NamedSharding(mesh, P(None, 'x', 'y'))
    same = NamedSharding(equivalent, P(None, 'x', 'y'))
    changed = NamedSharding(other, P(None, 'x', 'y'))
    M._zeros_kernel.cache_clear()
    M._transpose_kernel.cache_clear()
    zero = M._zeros_kernel((1, 4, 6), np.dtype('complex128'), tile)
    assert zero is M._zeros_kernel((1, 4, 6), np.dtype('complex128'), same)
    for shape, dtype, sharding in [((1, 6, 4), 'complex128', tile),
                                   ((1, 4, 6), 'float64', tile),
                                   ((1, 4, 6), 'complex128', changed)]:
        assert zero is not M._zeros_kernel(shape, np.dtype(dtype), sharding)
    x = np.arange(24).reshape(1, 4, 6) * (1 + 2j)
    for op in ('T', 'C'):
        fn = M._transpose_kernel(op, tile)
        assert fn is M._transpose_kernel(op, same)
        assert fn is not M._transpose_kernel(op, changed)
        for scale in (1, 2):
            got = fn(jax.device_put(scale*x, tile))
            want = (scale*x).swapaxes(-1, -2)
            if op == 'C':
                want = want.conj()
            np.testing.assert_array_equal(got, want)
            assert got.sharding.is_equivalent_to(tile, 3)
        assert fn._cache_size() == 1
    assert M._transpose_kernel('T', tile) is not M._transpose_kernel('C', tile)
    # The zero factory caches code, so donating one allocation cannot poison
    # a later allocation from that same factory.
    consume = jax.jit(lambda a: a + 1, donate_argnums=(0,))
    np.testing.assert_array_equal(consume(zero()), np.ones((1, 4, 6)))
    np.testing.assert_array_equal(zero(), np.zeros((1, 4, 6)))
    assert zero._cache_size() == 1


def test_contract_faces_reuses_code_with_changing_data():
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    put = lambda a, spec: jax.device_put(a, NamedSharding(mesh, spec))
    x = np.arange(24).reshape(1, 4, 6).astype(complex) * (1 + 1j)
    y = x + 2j
    M._contract_faces_kernel.cache_clear()
    for paired in (False, True):
        fn = M._contract_faces_kernel(mesh, False, paired)
        for stop in (3, 5):
            weights = np.arange(6)[None, :].astype(complex) * (1 - 2j)
            got = M.contract_faces(put(x, P(None, 'x', None)),
                put(y, P(None, 'y', None)), put(weights, P()),
                put(np.array([1]), P()), put(np.array([stop]), P()),
                mesh=mesh, return_transpose=paired)
            d = weights * ((np.arange(6) >= 1) & (np.arange(6) < stop))
            want = (x*d[:, None, :]) @ y.conj().swapaxes(-1, -2)
            np.testing.assert_allclose(got[0] if paired else got, want, atol=1e-12)
            if paired:
                np.testing.assert_allclose(got[1], (x.conj()*d[:, None, :]) @
                                           y.swapaxes(-1, -2), atol=1e-12)
        assert fn._cache_size() == 1
    assert M._contract_faces_kernel(mesh, False, False) is not M._contract_faces_kernel(mesh, True, False)


def test_gemm_builder_separates_provider_signature(monkeypatch):
    """Native warming happens once; changed coefficients/context cannot alias."""
    G = importlib.import_module('distrib_la.matmul_plan')
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    builds = []

    def build(mesh, **kw):
        builds.append(kw)
        if kw['with_c']:
            return lambda a, b, c: kw['alpha']*(a @ b) + kw['beta']*c
        return lambda a, b: kw['alpha']*(a @ b)

    monkeypatch.setattr(G, '_build_kernel', build)
    G._warmed_gemm_kernels.cache_clear()
    key = (mesh, 1, 4, 4, 4, np.dtype('complex128'), 1., 0., 17)
    pair = G._warmed_gemm_kernels(*key)
    assert pair is G._warmed_gemm_kernels(*key)
    assert len(builds) == 2
    for index, value in ((1, 2), (2, 6), (5, np.dtype('float64')),
                         (6, 2.), (7, 1.), (8, 18)):
        changed = list(key)
        changed[index] = value
        assert pair is not G._warmed_gemm_kernels(*changed)
    # Do not retain mocked provider kernels for another test in this process.
    G._warmed_gemm_kernels.cache_clear()
