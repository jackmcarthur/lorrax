# Shared orthogonalization

`distrib_la.plan_orthogonalization(*, capacity, max_block_size,
vector_sharding=None, native_collectives=False)` returns a callable that
removes a block's components along an existing orthonormal basis by two
classical Gram–Schmidt passes (CGS2). It does not normalize the result,
detect rank, or orthogonalize vectors within the block; the subspace QR and
normalization operations of [`plan_subspace`](davidson.md#shared-subspace-service)
do that.

```python
from distrib_la import plan_orthogonalization

orthogonalize = plan_orthogonalization(
    capacity=512, max_block_size=32, vector_sharding=vector_sharding)
# Inspect orthogonalize.workspace_specs before allocating operands.
# basis: (512, *vector_shape), block: (b <= 32, *vector_shape), complex128.
result = orthogonalize(basis, block, count, start=start)   # start/count may be traced
```

For each pass, with vector axes flattened and only basis rows
`[start, start + count)` present,

$$C = \bar{Q}\,B^{\mathsf T}, \qquad B \leftarrow B - C^{\mathsf T} Q .$$

The selected rows must be orthonormal in the complex Euclidean inner product.
An empty interval returns the block unchanged, and inactive basis rows may
hold NaNs. Two passes control projection roundoff; they cannot repair a wrong
basis or a different metric (the symplectic BSE orthogonalization is a
separate operation).

**Contract.** Operands are complex128 with `basis.shape[0] == capacity`,
matching trailing shapes and `1 ≤ block.shape[0] ≤ max_block_size`; anything
else raises. The row axis is unsharded, vector axes must cover every
nontrivial mesh axis, both operands keep that sharding, and no vector is
gathered to fewer ranks. `start` and `count` must agree across ranks. Plan
before tracing; the callable is `jit`-compatible. Capacity, vector shape or
block width change the compiled geometry; `start`/`count` do not. Collective
invocations must be issued in the same order on every rank; with
`native_collectives=True` planning initializes the shared native communicator
collectively, and calls on it must not be issued from concurrent host threads.

## Providers and memory

`orthogonalize.backend` names the resolved implementation:

| backend | when | mechanism |
|---|---|---|
| `cuda` | CUDA, one device | local cuBLAS CGS2 with a declared block input/output alias |
| `cuda_jax_collectives` | CUDA, distributed (default) | local Gram, an in-place subtraction fused with the next Gram, and a final in-place subtraction around two JAX coefficient reductions of the fixed `capacity × block_width` array; GEMMs use only selected rows; no native NCCL context |
| `cuda_nccl` | CUDA, distributed, `native_collectives=True`, one GPU per process over the whole X/Y mesh | one native handler updates the block in place and all-reduces only `count × block_width` coefficients per pass; a fixed-size exchange checks interval agreement across ranks before any mutation |
| `jax_collectives` | CPU, distributed | host callbacks on local vector tiles with capacity-sized JAX coefficient reductions |
| `cpu_callback` | CPU, one device | both passes in one NumPy host callback |

`workspace_specs` lists the XLA-owned scratch: `capacity × max_block_size`
complex128 coefficients, 4 MiB of BLAS workspace and, for `cuda_nccl`, two
int32 range entries per rank. Native context allocations and CPU callback
temporaries are not included, and there is no eigensolver workspace. An
undonated block must stay unchanged, so XLA may copy it; a surrounding `jit`
can donate it when the caller no longer needs it. Read the actual buffer
schedule from optimized HLO and `memory_analysis()`.

`cuda_nccl` reuses the service's existing NCCL context, stream and pooled CUDA
events; initializing that context has its own time and device-memory cost.
Every CUDA route reads runtime dimensions to the host with a stream
synchronization, so none is a host-free CUDA graph.

Required handlers are probed at planning and refuse by name when missing:
`ActiveSubspaceOrthoFfi` on CUDA; `ActiveSubspaceGramFfi`,
`ActiveSubspaceSubtractFfi` and `ActiveSubspaceSubtractGramFfi` for
`cuda_jax_collectives`; `ActiveSubspaceDistributedOrthoFfi` for `cuda_nccl`.
Rebuild through the [FFI owner](../architecture/ffi_layout.md).

## Consumers

`plan_subspace(...).orthogonalize` uses the same implementation, so planned
Davidson and scalar, block and thick-restart Lanczos share it.
`tests/test_subspace_orthogonalize.py` covers active windows, poisoned inactive
rows, nearly dependent inputs, input preservation and real distributed
execution.
