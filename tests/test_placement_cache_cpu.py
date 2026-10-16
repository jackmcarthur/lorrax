"""lxkit.device_put_process_local_cached: bit-identical, placed once per distinct value, bounded."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from jax.sharding import NamedSharding, PartitionSpec as P

import lxkit
from lxkit import placement


def _mesh():
    devs = np.array(jax.devices()[:4]).reshape(2, 2) if len(jax.devices()) >= 4 else np.array(jax.devices()[:1]).reshape(1, 1)
    return jax.sharding.Mesh(devs, ("x", "y"))


def test_cached_is_bit_identical_and_reused():
    rep = NamedSharding(_mesh(), P())
    before = lxkit.placement_cache_stats()
    v = np.array([3, -1, 7], np.int64)
    a = lxkit.device_put_process_local_cached(v, rep)
    b = lxkit.device_put_process_local_cached(v.copy(), rep)
    ref = lxkit.device_put_process_local(v, rep)
    assert a is b
    assert a.sharding == ref.sharding and a.dtype == ref.dtype and a.shape == ref.shape
    np.testing.assert_array_equal(np.asarray(a), np.asarray(ref))
    after = lxkit.placement_cache_stats()
    assert after["misses"] - before["misses"] == 1 and after["hits"] - before["hits"] == 1


def test_key_separates_dtype_shape_and_value():
    rep = NamedSharding(_mesh(), P())
    a = lxkit.device_put_process_local_cached(np.int32(5), rep)
    b = lxkit.device_put_process_local_cached(np.int64(5), rep)
    c = lxkit.device_put_process_local_cached(np.array([5], np.int32), rep)
    d = lxkit.device_put_process_local_cached(np.int32(6), rep)
    assert len({id(a), id(b), id(c), id(d)}) == 4
    assert a.shape == () and a.dtype == np.int32 and int(d) == 6


def test_large_values_bypass_and_cache_is_bounded():
    rep = NamedSharding(_mesh(), P())
    big = np.zeros(placement._CACHE_ENTRY_BYTES // 8 + 1, np.float64)
    s0 = lxkit.placement_cache_stats()
    x = lxkit.device_put_process_local_cached(big, rep)
    y = lxkit.device_put_process_local_cached(big, rep)
    assert x is not y and lxkit.placement_cache_stats()["bypass"] - s0["bypass"] == 2
    old = placement._CACHE_TOTAL_BYTES
    try:
        placement._CACHE_TOTAL_BYTES = 1024
        for i in range(400):
            lxkit.device_put_process_local_cached(np.arange(i, i + 4, dtype=np.int64), rep)
        assert lxkit.placement_cache_stats()["bytes"] <= 1024
    finally:
        placement._CACHE_TOTAL_BYTES = old


def test_device_arrays_pass_through():
    rep = NamedSharding(_mesh(), P())
    d = jax.device_put(np.arange(3.0), rep)
    out = lxkit.device_put_process_local_cached(d, rep)
    np.testing.assert_array_equal(np.asarray(out), np.arange(3.0))


def test_parent_rows_source_cached_and_identical():
    import jax.numpy as jnp
    from gw import centroid_k_unfold as cku
    irr = np.array([0, 1, 1, 2, 0, 2], np.int64)
    vals = np.arange(6 * 3, dtype=np.float64).reshape(6, 3)
    out1 = cku.parent_rows(irr, 3, vals)
    out2 = cku.parent_rows(irr, 3, jnp.asarray(vals))
    src = np.full(3, -1, np.int32)
    for full, parent in enumerate(irr):
        src[parent] = full
    np.testing.assert_array_equal(np.asarray(out1), vals[src])
    np.testing.assert_array_equal(np.asarray(out2), vals[src])
    assert cku._parent_source(irr, 3) is cku._parent_source(irr.copy(), 3)
