"""Executable reuse must preserve data, donation, dtype and device placement."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

M = importlib.import_module('distrib_la.matmul')


def test_transpose_cache_separation():
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices.reshape(2, 2), ('x', 'y'))
    equivalent = Mesh(devices.copy().reshape(2, 2), ('x', 'y'))
    other = Mesh(devices[::-1].reshape(2, 2), ('x', 'y'))
    tile = NamedSharding(mesh, P(None, 'x', 'y'))
    same = NamedSharding(equivalent, P(None, 'x', 'y'))
    changed = NamedSharding(other, P(None, 'x', 'y'))
    M._transpose_kernel.cache_clear()
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
