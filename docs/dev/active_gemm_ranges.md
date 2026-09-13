# Active ranges in planned distributed GEMM

`gemm_plan(..., enable_active_range=True)` adds
`plan.active_range(A, B, lo, hi, C=None, *, out=None)` to the existing
cuBLASMp face plan. The full allocation shapes remain `(nq,m,k)` and
`(nq,k,n)`, at `P(None,x,y)`. The operation is
`alpha * A[...,lo:hi] @ B[...,lo:hi,:] + beta*C`; output remains at
`P(None,x,y)`. Bounds are replicated integer scalars or arrays of shape `(nq,)`, with scalar/vector broadcasting supported. No active-range
fallback to full-K arithmetic is provided for unsupported backends/layouts.

The provider reads the packed bounds to the host once per invocation
because vendor GEMM dimensions are host scalars. It intersects the interval
with each original contraction owner's slab. For each nonempty intersection it
creates descriptor views: A's single contraction block belongs to that owner's
process column, B's single contraction block to its process row. View pointers advance
within the existing allocations; original leading dimensions and batch strides
remain unchanged. The contraction K is the exact intersection length.
All processor tiles participate in updating the distributed output; subsequent
intersections use beta=1. No wavefunction panel packing or redistribution is
needed. Empty intervals apply the declared beta semantics without native GEMM. Zero beta uses an output memset; unit beta preserves aliased C; other values use a context-owned local cuBLAS handle with bounded integer chunks. No extra output-sized buffer is needed. Identical batch intervals share one descriptor setup, and a full interval executes the original dense GEMM unchanged.

This uses legal descriptor views rather than arbitrary submatrix origins.
cuBLASMp requires submatrix origins to align with descriptor block sizes;
with the original one-slab-per-owner descriptors, arbitrary contraction cuts do not
meet that condition. Source ownership and physical leading dimensions are
explicit descriptor parameters. See NVIDIA's
[cuBLASMp C API](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html).

The second canonical FFI target is `lorrax_cublasmp_active_range_gemm`; the
original target and default plan ABI are unchanged. Both paths share the same
native GEMM implementation. An older provider refuses the opt-in capability by
name; rebuild the canonical native library. No private GW backend is used.

Initial scope is scalar or per-batch bounds, float64/complex128,
N,N multiplication on a square face mesh. Arbitrary holes inside an interval
remain live arithmetic; the API is not a sparse selector. Cross-invocation descriptor/workspace-plan caching is a subsequent extension. As with existing
GEMM, native private workspace is distinct from XLA's memory report. The
initial provider builds descriptors per invocation; bounds synchronization
and repeated owner GEMMs must be measured alongside arithmetic savings.

Focused P4 tests are `services/distrib_la/tests/test_active_gemm_range.py`.
They compare against independent dense results for exact owner intersections,
nontrivial alpha/beta, empty intervals, and inactive inputs poisoned with NaNs.
Run391 in the sandbox owns runtime and memory evidence; availability of this
API alone is not a performance or production-GW certification.
