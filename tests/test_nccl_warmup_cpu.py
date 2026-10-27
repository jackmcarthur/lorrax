"""Startup communication must execute, finish, and keep slab-sized local fields."""
import math
import re
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh


@pytest.mark.parametrize('side,abstract', [(2, False), (12, True)])
def test_warmup_real_collectives_and_owned_abstract_bounds(side, abstract, monkeypatch):
    if len(jax.devices()) < side*side:
        pytest.skip(f'needs {side*side} virtual CPU devices')
    from runtime import nccl_warmup
    mesh = Mesh(np.asarray(jax.devices()[:side*side]).reshape(side, side), ('x', 'y'))
    original_jit, original_ones = jax.jit, jax.numpy.ones
    # Exercise the actual GPU warmup owner on virtual CPU collectives; only
    # bypass its process-count early return. No transport operation is mocked.
    monkeypatch.setattr(jax, 'process_count', lambda: 2)
    if abstract:
        monkeypatch.setattr(jax.numpy, 'ones', lambda shape, dtype, device:
            jax.ShapeDtypeStruct(shape, dtype, sharding=device))
    observed = []

    def observe(fn, *a, **kw):
        program = original_jit(fn, *a, **kw)
        def run(x):
            exe = program.lower(x).compile()
            hlo = exe.as_text()
            alltoall = len(re.findall(r'\ball-to-all\(', hlo))
            sums = len(re.findall(r'\ball-reduce\(', hlo))
            assert (alltoall, sums) in ((1, 0), (0, 1)), (alltoall, sums)
            local = math.prod(x.shape) // side**(2 if x.sharding.spec[0] == ('x', 'y') else 1)
            assert local*8 <= 8 << 20
            for dims in re.findall(r'f64\[([0-9,]+)\]', hlo):
                assert math.prod(map(int, dims.split(','))) <= local
            ready = [False]
            value = None if abstract else exe(x)
            def finish():
                if not abstract:
                    value.block_until_ready()
                    group = side**(2 if x.sharding.spec[0] == ('x', 'y') else 1)
                    for shard in value.addressable_shards:
                        np.testing.assert_array_equal(np.asarray(shard.data),
                            np.full(shard.data.shape, group if sums else 1., dtype=np.float64))
                ready[0] = True
            observed.append((alltoall, sums, local, ready))
            return SimpleNamespace(block_until_ready=finish)
        return run

    monkeypatch.setattr(jax, 'jit', observe)
    nccl_warmup(mesh)
    assert len(observed) == 12
    assert [(a, b) for a, b, _, _ in observed] == [(1, 0), (0, 1)] * 6
    assert all(ready[0] for _, _, _, ready in observed)
