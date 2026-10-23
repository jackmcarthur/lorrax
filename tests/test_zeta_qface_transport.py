"""P4+ regression for the measured q-owner Z-tile replication.

Run as a compute-node script after the machine launch prelude. This checks
transport and its compiled scratch, not a replacement zeta solver.
"""


def main():
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import all_gather_processes
    from isdf.zeta_mubatch import _face_kernels

    mesh = stack.mesh
    world = int(mesh.size)
    if world < 4:
        raise ValueError('q-face transport regression requires at least a P4 square mesh')
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    m, g = -(-63 // px) * px, -(-79 // py) * py
    batch = NamedSharding(mesh, P(('x', 'y'), None, None))
    face = NamedSharding(mesh, P(None, 'x', 'y'))

    cases = 0
    for q in (1, world + 1):
        qpad = -(-q // world) * world
        shape = (qpad, m, g)

        def values(index, extent=shape):
            a, b, c = [np.arange(*sl.indices(n), dtype=np.int64)
                       for sl, n in zip(index, extent)]
            real = ((a[:, None, None] * m + b[None, :, None]) * g + c[None, None, :])
            return real.astype(np.complex128) + 1j * (real % 37 + 1)

        # Pad parents and M/G carrier cells are deliberately nonzero, so
        # a wrong crop or permutation cannot pass by reading exact zeros.
        z = jax.make_array_from_callback(shape, batch, values)
        for with_v, inverse in ((True, False), (False, False), (False, True)):
            move = _face_kernels(mesh, q, g, with_v, inverse=inverse)[0]
            executable = move.lower(z).compile()
            stats = executable.memory_analysis()
            assert stats is not None
            # The observed regression allocates exactly this full padded
            # stack as scratch. Explicit transport needs only local tiles.
            full_replica = 16 * qpad * m * g
            assert stats.temp_size_in_bytes < full_replica, (
                q, with_v, inverse, stats.temp_size_in_bytes, full_replica)
            out = executable(z)
            out.block_until_ready()
            assert out.shape == (q, m, g)
            assert out.sharding.is_equivalent_to(face, 3)
            for shard in out.addressable_shards:
                np.testing.assert_array_equal(np.asarray(shard.data),
                                              values(shard.index, (q, m, g)))
            del out
            cases += 1
        del z
    passed = np.asarray(all_gather_processes(np.array(cases, np.int32)))
    assert np.all(passed == 6), passed
    if jax.process_index() == 0:
        print('ZETA_QFACE_TRANSPORT PASS 6 cases; exact values, q padding, charge/inverse/file routes, bounded scratch', flush=True)
    return 0


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
