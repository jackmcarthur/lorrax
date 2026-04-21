"""SlabIO FFI per-rank write kernel.

Models one invocation of the shard_map'd write at
``file_io._slab_io_ffi._FfiBackend.write_slab``.  The FFI itself is
opaque (a C++ handle to MPI-IO collective write), but the shard_map's
Python-facing memory is one local slab per rank plus an int64 offset.

AOT measures only the slab layout — the *asynchronous queue* holding
`(K+2) × slab` across writes is a Python-level effect not visible to
``memory_analysis()``.  See ``MEMORY_MODEL.md`` orchestrator section
for the meta-formula that consumes this kernel's β to compute total
I/O overhead.

Input shapes (modeled for a zeta (nq, μ, B_r) write; pass other layouts
via the preset):
    A  (nq, μ, B_r)  P(None, 'x', 'y')   -- the local slab
    off_arr  (3,) int64  P()

There is no output — the write returns only a token.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..core import AotKernel, Knobs, MeshSpec, SysDims, register_kernel

_B = 16.0


def _T_slab(sys, knobs, mesh):
    """Per-rank local slab for a zeta (nq, μ, B_r) write sharded
    ``P(None, 'x', 'y')``. Per-device bytes = 16·nq·μ·B_r/(p_x·p_y)."""
    Br = knobs.get("chunk_r", sys.n_r or 1)
    return _B * sys.n_k * sys.n_rmu * Br / (mesh.p_x * mesh.p_y)


@register_kernel
class SlabWriteKernel(AotKernel):
    """shard_map'd local write — one rank's slab goes to HDF5.

    In production this is queued via a Python thread; AOT measures
    only the per-call footprint.  Total live memory =
    (queue_cap + 2) × β[slab] × T_slab.
    """
    name = "slab_write"
    SYSTEM_DIMS = ("n_k", "n_rmu", "n_r", "kgrid")
    KNOBS = ("chunk_r",)
    PRIMITIVES = {"slab": _T_slab}

    def build_specs(self, sys, knobs, mesh):
        nq, mu = sys.n_k, sys.n_rmu
        Br = knobs.get("chunk_r", sys.n_r)
        sh_A = NamedSharding(mesh, P(None, "x", "y"))
        sh_off = NamedSharding(mesh, P())
        return (
            jax.ShapeDtypeStruct((nq, mu, Br), jnp.complex128, sharding=sh_A),
            jax.ShapeDtypeStruct((3,), jnp.int64, sharding=sh_off),
        )

    def build_callable(self, sys, knobs, mesh):
        mesh_xy = mesh

        # We model the shard_map "identity-return + drop" that
        # production wraps around the FFI call.  The FFI call itself
        # cannot be AOT-lowered (it's a side-effecting C extension);
        # but the memory it captures is proportional to A_local so the
        # identity model is faithful to what memory_analysis sees.
        def _per_rank_identity(A_local, offset_local):
            # Keep the buffer alive long enough that XLA counts it in
            # the peak.  A trivial op (add 0) prevents it from being
            # elided as dead.
            return A_local + jnp.zeros_like(A_local)

        sm = shard_map(
            _per_rank_identity, mesh=mesh_xy,
            in_specs=(P(None, "x", "y"), P()),
            out_specs=P(None, "x", "y"),
            check_rep=False,
        )
        return jax.jit(sm, donate_argnums=(0,))
