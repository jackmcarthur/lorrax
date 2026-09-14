# Shared orthogonalization

`distrib_la.plan_orthogonalization` returns a callable for removing a block's
components along an existing orthonormal basis. It uses two classical
Gram–Schmidt passes (CGS2). It does not normalize the result, detect rank, or
orthogonalize vectors within the new block; use the shared subspace QR or
normalization operation for those separate steps.

```python
from distrib_la import plan_orthogonalization

orthogonalize = plan_orthogonalization(
    capacity=512, max_block_size=32, vector_sharding=vector_sharding)
# Inspect orthogonalize.workspace_specs before allocating operands.
# basis: (512, *vector_shape), block: (32, *vector_shape), complex128.
# start and count can change inside a compiled loop.
result = orthogonalize(basis, block, count, start=start)
```

The basis rows in `[start, start + count)` must be orthonormal under the
ordinary complex Euclidean inner product. For each of two passes the service
computes `C = basis.conj() @ block.T`, then `block -= C.T @ basis`, with
vector dimensions flattened and only the selected basis interval present in
the products. Empty intervals preserve the block; inactive basis rows may
contain NaNs. Two passes control projection roundoff; they cannot repair an
incorrect basis or a different physical metric.

The row axis is unsharded. Vector axes must cover every nontrivial mesh axis.
Both operands retain that sharding, and vectors never gather to fewer ranks.
`start` and `count` must agree across ranks; the optional native-collective
path explicitly checks this agreement at runtime.
Plan before tracing. Collective invocations must occur in the same order on
all ranks. With `native_collectives=True`, planning also initializes the shared
native communicator collectively; do not submit independent calls on it from
concurrent host threads. Solver iterations order calls through their vector
data dependencies. Invocation is JIT-compatible. Changing
capacity, vector shape, or block width changes compiled geometry; changing
only start/count does not.

## Providers and memory

`orthogonalize.backend` identifies the selected implementation:

- `cuda`: local cuBLAS CGS2 with a declared block input/output alias.
- `cuda_jax_collectives`: the default for distributed CUDA. A local Gram
  operation, an in-place subtraction combined with the next Gram operation,
  and a final in-place subtraction surround two JAX coefficient reductions.
  GEMMs use only the selected basis rows. Reductions retain the fixed
  `capacity * block_width` coefficient array. This path creates no native
  NCCL context and avoids reconstructed vector temporaries.
- `cuda_nccl`: explicit `native_collectives=True` on one process per GPU
  over the complete X/Y mesh. One native handler updates the local block
  in place and allreduces only `count * block_width` complex coefficients
  per pass. A fixed-size exchange checks interval agreement before mutation.
  This option can reduce communication and host waits, but initializing its
  shared native context has a substantial separate memory and startup cost.
- `jax_collectives`: distributed CPU callbacks operate on local vector tiles,
  with capacity-sized JAX coefficient reductions.
- `cpu_callback`: both active NumPy passes in one host callback.

CUDA scratch is bounded by `capacity * max_block_size` complex128
coefficients, 4 MiB of BLAS workspace, and, for the native distributed
operation, two int32 range entries per rank. These are XLA-owned arrays.
`native_collectives` affects distributed CUDA only. There is no eigensolver workspace query in this standalone planner. An
undonated input must remain unchanged: XLA may copy the correction block to
honor that contract. A surrounding JIT can donate the block when its caller
no longer needs it; inspect optimized HLO and `memory_analysis()` for the
actual buffer schedule.

The optional native distributed handler reuses the linear-algebra service's existing NCCL
context, stream, and pooled CUDA events. Initializing that context has a
separate time and device-memory cost. Native library/context allocations and
CPU callback temporaries are not included in `workspace_specs`. Native
runtime dimensions still require host reads and stream waits: three for the
default distributed CUDA path, one for the optional native collective path. This is
not a host-free CUDA graph.

The default distributed CUDA path requires `ActiveSubspaceSubtractFfi` and
`ActiveSubspaceSubtractGramFfi` alongside the existing Gram target. The optional
native path requires `ActiveSubspaceDistributedOrthoFfi`. Missing required
targets refuse at planning. Rebuild through the [FFI owner](../architecture/ffi_layout.md).
No private driver binding or additional shared library is introduced.

## Existing solvers and scope

`plan_subspace(...).orthogonalize` uses the same implementation. Planned
Davidson and ordinary scalar/block/thick-restart Lanczos therefore share the
optimization without duplicate solver math. Symplectic BSE orthogonalization
has a different metric and is not replaced by this Euclidean operation.

Tests in `tests/test_subspace_orthogonalize.py` cover active windows,
poisoned inactive rows, nearly dependent inputs, input preservation, and
real distributed execution. The Run398 sandbox investigation owns measured
performance and memory results; source structure alone is not a speedup
certificate.
