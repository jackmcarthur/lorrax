"""Parallel-HDF5 FFI subpackage.

Public API:

    from ffi.phdf5 import open_file, close_file, \\
                          write_sharded_slab, read_sharded_slab

    fh = open_file("/path/to/out.h5", mesh=mesh, mode="w")
    write_sharded_slab(fh, ds_name="eigenvectors", A=Q,
                       global_shape=(n, n), mesh=mesh)
    close_file(fh)

    fh = open_file("/path/to/out.h5", mesh=mesh, mode="r")
    Q = read_sharded_slab(fh, ds_name="eigenvectors",
                          global_shape=(n, n),
                          dtype=jnp.complex128, mesh=mesh)
    close_file(fh)

Each process reads/writes its local shard directly to a hyperslab of
the shared HDF5 file via MPI-IO — no gather through rank 0.  See
``src/ffi/AGENTS.md`` for required environment and
``src/ffi/PORTING.md`` for per-cluster setup.
"""
from .context import open_file, close_file
from .write import write_sharded_slab
from .read import read_sharded_slab

__all__ = [
    "open_file", "close_file",
    "write_sharded_slab", "read_sharded_slab",
]
