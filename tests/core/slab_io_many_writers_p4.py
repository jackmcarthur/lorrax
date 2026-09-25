"""P4: three writable SlabIO handles with queued writes on all of them.

Each handle's ``write_slab`` is an asynchronous collective ``H5Dwrite``.  With a
writer thread per handle, the three files' collectives reached MPI-IO at the
same time and each rank matched them in its own order: a CrI3 run with
``zeta_q_mu1..3.h5`` open together hung for 26 min at 0 B/s.  SlabIO now sends
every handle's collectives through one process lane in program order
(``file_io._slab_io_ffi._CollectiveLane``).

The cell interleaves writes to three files (a strided 2-D tile layout, so every
write is a real two-phase collective), closes them, and reads every tile back.
Each rank exits 0 only if every byte matches.  Run it under ``timeout``: on a
tree without the lane it hangs instead of failing.
"""
import sys
from pathlib import Path


def main(argv=None):
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from file_io.slab_io import SlabIO

    rank = jax.process_index()
    assert jax.process_count() == 4, jax.process_count()
    out = Path((argv or sys.argv[1:])[0])
    out.mkdir(parents=True, exist_ok=True)
    mesh = RUNTIME.mesh
    sharding = NamedSharding(mesh, P("x", "y"))
    n, tiles, files = 2048, 6, 3            # 32 MiB per tile, 8 MiB per rank

    def tile(f, t):
        seed = 1000 * f + t
        return jax.make_array_from_callback(
            (n, n), sharding,
            lambda index: (np.arange(n * n, dtype=np.float64).reshape(n, n)
                           + seed)[index])

    paths = [out / f"many_writers_{f}.h5" for f in range(files)]
    handles = [SlabIO(str(p), mode="w", mesh=mesh) for p in paths]
    for io in handles:
        io.create_dataset("z", shape=(tiles * n, n), dtype=np.float64)
    for t in range(tiles):
        for f, io in enumerate(handles):
            io.write_slab("z", tile(f, t), offset=(t * n, 0))
    for io in handles:
        io.close()
    bad = []
    for f, p in enumerate(paths):
        with SlabIO(str(p), mode="r", mesh=mesh) as io:
            for t in range(tiles):
                got = io.read_slab("z", shape=(n, n), offset=(t * n, 0),
                                   partition_spec=P("x", "y"))
                want = tile(f, t)
                if not bool(jax.jit(lambda a, b: (a == b).all())(got, want)):
                    bad.append((f, t))
    print(f"rank {rank} mismatched tiles: {bad}", flush=True)
    assert not bad, bad
    print(f"done: {files} files x {tiles} interleaved collective writes "
          f"read back bit-exact", flush=True)
    return 0


if __name__ == "__main__":
    from runtime import initialize_communicator_stack, run_main_and_finalize
    RUNTIME = initialize_communicator_stack(platform="gpu")
    run_main_and_finalize(main)
