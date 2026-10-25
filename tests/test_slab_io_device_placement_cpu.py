"""Accepted local/replicated SlabIO device inputs never bounce through hosts."""
import numpy as np
import pytest

import runtime  # noqa: F401
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def test_foreign_or_single_device_write_placement_stays_on_device(monkeypatch):
    from file_io import _slab_io_ffi as io
    devices = jax.devices('cpu')
    if len(devices) < 4:
        pytest.skip('requires four virtual CPU devices')
    source = Mesh(np.asarray(devices[:4]).reshape(2, 2), ('a', 'b'))
    target = Mesh(np.asarray(devices[:4]).reshape(2, 2), ('x', 'y'))
    # Complex64 also checks that placement never silently canonicalizes to
    # double precision; this is transport, not a numerical conversion.
    want = (np.arange(32).reshape(4, 8)*(1+.25j)).astype(np.complex64)
    arrays = (
        jax.device_put(want, devices[0]),
        jax.device_put(want, NamedSharding(source, P())),
        jax.device_put(want, NamedSharding(source, P('a', 'b'))),
    )
    original = np.asarray

    def host_copy(*args, **kwargs):
        raise AssertionError('SlabIO device operand copied to host')

    def no_device_array(value, *args, **kwargs):
        if isinstance(value, jax.Array):
            host_copy()
        return original(value, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(io, 'gather_to_host', host_copy)
        guard.setattr(np, 'asarray', no_device_array)
        placed = [io._onto_handle_mesh(a, target, 'A') for a in arrays]
        jax.block_until_ready(placed)
    for before, after in zip(arrays, placed):
        assert after.sharding.is_equivalent_to(NamedSharding(target, P()), 2)
        assert after.dtype == want.dtype
        np.testing.assert_array_equal(np.asarray(after), want)
        np.testing.assert_array_equal(np.asarray(before), want)
