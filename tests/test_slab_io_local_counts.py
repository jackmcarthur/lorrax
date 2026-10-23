"""Check posted union-read counts against JAX's independent shard ownership."""


def main():
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from file_io._slab_io_ffi import (
        _derive_window_counts, _rank_block_offset, _sharding_to_axis_info,
        _shard_divisors,
    )

    mesh = stack.mesh
    mesh_shape = tuple(int(mesh.shape[axis]) for axis in mesh.axis_names)
    layouts = (P(('x', 'y'), None, None), P(None, ('y', 'x'), None),
               P('x', None, 'y'), P(None, 'y', 'x'),
               P(None, None, ('x', 'y')))
    rng = np.random.default_rng(917)
    cases = 0
    for spec in layouts:
        sharding = NamedSharding(mesh, spec)
        axis_counts, axis_flat = _sharding_to_axis_info(sharding, 3)
        divisors = _shard_divisors(
            axis_count_per_dim=axis_counts, axis_flat=axis_flat,
            mesh_shape=mesh_shape, ndim=3)
        shape = tuple(2*int(divisor)+5 for divisor in divisors)
        shape = tuple(-(-extent//int(divisor))*int(divisor)
                      for extent, divisor in zip(shape, divisors))
        local_shape = tuple(int(v) for v in sharding.shard_shape(shape))
        valid = np.asarray([[0, 0, 0], [1, 1, 1], list(shape),
                            [max(0, n-1) for n in shape],
                            *[rng.integers(0, np.asarray(shape)+1).tolist()
                              for _ in range(23)]], dtype=np.int64)
        ownership = sharding.devices_indices_map(shape)
        for rank, device in enumerate(mesh.devices.flat):
            indices = ownership[device]
            offset = _rank_block_offset(
                rank=rank, per_rank_shape=local_shape,
                axis_count_per_dim=axis_counts, axis_flat=axis_flat,
                mesh_shape=mesh_shape)
            np.testing.assert_array_equal(offset, [s.start or 0 for s in indices])
            actual = _derive_window_counts(
                per_rank_shape=local_shape, rank_offsets=offset[None, :],
                valid_shapes=valid)
            expected = np.asarray([
                [np.count_nonzero(np.arange(shape[d])[indices[d]] < extent[d])
                 for d in range(3)] for extent in valid], dtype=np.int64)
            np.testing.assert_array_equal(actual, expected)
            assert actual.shape == (len(valid), len(shape))
            cases += len(valid)
    if jax.process_index() == 0:
        print(f'LOCAL_UNION_COUNTS PASS mesh={mesh_shape} cases={cases}', flush=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip('run directly under the configured MPI launcher')


if __name__ == '__main__':
    main()
