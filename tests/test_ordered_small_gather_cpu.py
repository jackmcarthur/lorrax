"""Bounded ordered all-reduce: literal rank arithmetic, VMA and executed HLO."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=144')


def _mesh(world):
    import jax
    import numpy as np
    import pytest
    from jax.sharding import Mesh
    if len(jax.devices()) < world:
        pytest.skip(f'needs {world} virtual host devices')
    side = int(world ** .5)
    return Mesh(np.array(jax.devices()[:world]).reshape(side, side), ('x', 'y'))


def _fold(rows):
    import numpy as np
    out = np.array(rows[0], copy=True)
    for row in rows[1:]:
        out = out + row
    return out


def test_bounded_gather_is_literal_rank_order_and_has_no_all_to_all():
    import jax
    import numpy as np
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    from lxkit import ordered_psum
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(71)
    for world in (4, 9, 16, 144):
        mesh = _mesh(world)
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            for length in (2, 500):  # includes L>P: q batches also use the bounded gather
                rows = rng.standard_normal((world, length)).astype(dtype)
                if np.dtype(dtype).kind == 'c':
                    rows += 1j * rng.standard_normal(rows.shape).astype(dtype)
                large = 2. ** (24 if np.dtype(dtype).itemsize in (4, 8) and dtype != np.float64 else 53)
                rows[0, 0], rows[1, 0], rows[2, 0] = large, 1, -large
                call = jax.jit(shard_map(lambda x: ordered_psum(x[0], ('x', 'y')),
                    mesh=mesh, in_specs=P(('x', 'y'), None), out_specs=P(), check_vma=True))
                exe = call.lower(rows).compile()
                got = np.asarray(exe(rows))
                assert got.tobytes() == _fold(rows).tobytes(), (world, dtype, length)
                text = exe.as_text()
                assert 'all-to-all(' not in text and 'all-to-all-start' not in text
                assert 'all-gather(' in text or 'all-gather-start' in text


def test_gather_cap_boundary_partial_axes_and_pytree():
    import jax
    import numpy as np
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    from lxkit import ordered_psum
    import lxkit.collectives as C
    jax.config.update('jax_enable_x64', True)
    mesh = _mesh(4)
    rows = np.arange(12, dtype=np.float64).reshape(4, 3) / 7
    needed = rows.nbytes
    for slab in (needed - 1, needed):
        call = jax.jit(shard_map(lambda x: ordered_psum(x[0], ('x', 'y'), slab_bytes=slab),
            mesh=mesh, in_specs=P(('x', 'y'), None), out_specs=P(), check_vma=True))
        exe = call.lower(rows).compile()
        assert np.asarray(exe(rows)).tobytes() == _fold(rows).tobytes()
        text = exe.as_text()
        assert ('all-to-all(' in text or 'all-to-all-start' in text) == (slab < needed)
    call = jax.jit(shard_map(lambda x: ordered_psum({'a': x, 'b': 2*x}, 'y'),
        mesh=mesh, in_specs=P(('x', 'y'), None), out_specs={'a': P('x', None), 'b': P('x', None)}, check_vma=True))
    result = call(rows)
    want = np.stack([_fold(rows[0:2]), _fold(rows[2:4])])
    assert np.asarray(result['a']).tobytes() == want.tobytes()
    assert np.asarray(result['b']).tobytes() == (2*want).tobytes()
    # The declared gather+two-partial footprint stays in the existing two slabs.
    old_mode = C._GUARD['mode']
    try:
        C._GUARD['mode'] = 'log'
        C._TRACE.clear()
        call = jax.jit(shard_map(lambda x: ordered_psum(x[0], ('x', 'y'), slab_bytes=needed),
            mesh=mesh, in_specs=P(('x', 'y'), None), out_specs=P(), check_vma=True))
        call.lower(rows).compile()
        assert C._TRACE and C._TRACE[-1]['extra_bytes'] <= 2*needed
    finally:
        C._GUARD['mode'] = old_mode
        C._TRACE.clear()
