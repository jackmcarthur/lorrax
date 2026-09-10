# Dense workspace query

`distrib_la.workspace_bytes_per_rank(plan, op, shapes, dtype) -> int`
returns device workspace bytes per rank for a resolved CUDA `Plan` or
`GemmPlan`. Query collectively on the actual mesh before allocating operands.
Supported dtypes are float64 and complex128; unsupported providers refuse.

For `eigh`, pass `((n,n),)` or `((batch,n,n),)`. For `gemm`, pass
`((m,k),(k,n))` or matching batched shapes, after any transpose/adjoint
staging. A common `Plan` supplies the local/distributed policy for GEMM.
A distributed single-matrix eigensolve remains distributed even when its
batched route is `batch_reshard`.

The native scalar bindings call cusolverMp SyevdBufferSize and cublasMp
MatmulBufferSize with the execution descriptors. An existing context info
pointer supplies a non-null address token required by the sizing API; it is
not read as an operand. No matrix, result, or queried workspace is allocated.
Context/communicator/handle initialization and compiler metadata can allocate
resources. Query results are cached by context, operation, sizes and dtype;
no factors or dense operands are cached.

For distributed eigh, the execution handler's dynamic XLA ScratchAllocator
allocation is **exactly `native_device_bytes`**: vendor workspace rounded up
to 256-byte alignment plus one local operand tile, `n/Px * n/Py * itemsize`.
The tile preserves the non-donating input contract despite destructive vendor
tridiagonalisation. Execution and query share the native byte-layout helper;
do not add the tile or workspace twice. `vendor_device_bytes` is not separately
reported for this combined native allocation. The
separate host allocation is `vendor_host_bytes`. The internal receipt helper
records both. For distributed GEMM, one vendor workspace is retained by the
context and reused across batch slices. Thus a safe workspace envelope is
`max(planned GEMM workspace) + max(concurrent eigh scratch)`, rather than the
maximum of the two. Concurrent independent contexts need separate budgets.

Local eigh queries cuSolverDn divide-and-conquer workspace, with a conservative
Jacobi maximum for small matrices, plus a four-byte info integer. Staged serial
calls use one workspace; native whole-batch calls conservatively reserve one
per batch member. Local GEMM returns compiled local matmul temporary bytes
from shape-only compilation, without execution; this is a workspace upper
bound and may include compiler copies/epilogues. Do not count these same
compiler temporaries again elsewhere. Local host zero means the API requests
no caller-provided host workspace, not that all host memory use is zero.

This entry does not account for operand/output carriers, transpose staging,
communication buffers, persistent context resources, or other operations.
The caller still owns the aggregate memory gate and must query all actual
planned shapes when the SC state changes. See the campaign workspace receipt
for measured P4/P16 complex128 sizes at n=896, 2n=1792 and R=2576. These are
query measurements, not dense-solve or end-to-end memory measurements.
