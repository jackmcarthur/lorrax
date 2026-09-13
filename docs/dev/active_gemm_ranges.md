# Active ranges in planned GEMM

`gemm_plan(..., enable_active_range=True)` enables
`plan.active_range(A, B, lo, hi, C=None, *, out=None, weights=None)`.
The product is `alpha * A[q,:,lo:hi] @ B[q,lo:hi,:] + beta*C[q]`.
Optional `weights` has shape `(nq,K)` and scales A's contraction columns
before multiplication; complex weights require a complex-valued plan. Bounds are replicated integer scalars or arrays of
shape `(nq,)`. Allocation shapes stay `(nq,m,K)`, `(nq,K,n)` and `(nq,m,n)`;
output is always `P(None,x,y)` for the active-range API.

`low_mem_bands=true` selects distributed face operands; `false` selects
local axis products with replicated bands. Both production tau paths derive
exact per-parent support intervals after applying energy windows and selector
weights. There is no numerical threshold. Interior holes still incur arithmetic
within their enclosing interval. The separate `linalg` profile controls
upstream operations, not this Sigma carrier selection.

| Route | Input placement | Active kernel |
|---|---|---|
| CUDA face | Both operands `P(None,x,y)` | cuBLASMp original-owner descriptor views |
| CUDA axis | A `P(None,x,None)`, B `P(None,None,y)` | Local cuBLAS pointer views, no collectives |
| CPU axis | Same as CUDA axis | JAX exact selected panels |
| CPU face | Distributed bands | Unsupported planned ScaLAPACK GEMM; explicit refusal |

The public service page is [distrib_la](../services/distrib_la.md); vendor
routing belongs to [FFI layout](../architecture/ffi_layout.md). An older
CUDA provider refuses a missing active target by name; rebuild the canonical
library. No driver chooses an external backend, and there is no full-Green
gather fallback.

## Distributed CUDA kernel

Target `lorrax_cublasmp_active_range_gemm` reads only small bound metadata to
the host, then intersects each interval with original contraction-owner slabs.
For every nonempty intersection it creates legal descriptor views: A's block
belongs to that owner's process column, and B's to its process row. Base
pointers advance inside the original storage; leading dimensions and batch
strides remain unchanged. Each GEMM contracts the exact intersection length,
accumulating into the same all-processor output. No selected wavefunction panel
is packed. Identical parent intervals reuse batched descriptor setup.

Arbitrary submatrix origins on the original descriptors are not equivalent:
cuBLASMp requires origins aligned to descriptor blocks. Source ownership and
physical leading dimensions make the views legal; see the
[NVIDIA cuBLASMp C API](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html).
Full intervals call the original dense target unchanged. Empty intervals apply
beta semantics without reading A/B. Native context workspace is separate from
XLA memory accounting; the context grows it as needed. Range-descriptor caching
across invocations is not implemented.

## Local CUDA kernel

`distrib_la._active_local_cuda.active_local_cuda` invokes
`lorrax_cublas_local_active_range_gemm`. It views row-major local operands as
column-major transposes, computing `C^T = B^T A^T` through classic cuBLAS.
The interval changes pointers and K, retaining the original leading dimensions
and batch strides. Consecutive parents with equal bounds share a strided-batched
call. No selected panels or distributed context are created.

The handler has a dedicated checked thread-local cuBLAS handle. Each invocation
binds the supplied CUDA stream and a **4 MiB XLA-owned workspace**, returned as
an explicit scratch output. Workspace is reset after binding the stream, so the
handle never relies on a previous invocation's scratch allocation. C aliases
the result inside the FFI. Empty intervals honor beta semantics without GEMM.

Weighting still forms the same full weighted-A tile required by the original
GPU implementation. The memory improvement removes additional slice panels
and accumulation buffers; it does not claim to eliminate this baseline tile.
When all parents request the full interval, a JAX conditional retains the
original weighted dense dot and its evaluation order.

## Local CPU kernel

`distrib_la._active_local.active_local_matmul` uses existing JAX dot lowering
without a native context or callback. A `lax.while_loop` traverses an interval
using widths selected by `lax.switch`: powers of two capped at256 columns.
Only wholly active, disjoint slices enter each dot; the last piece never pads
inactive columns. Weights are applied inside the selected slices, avoiding a
full weighted-A temporary in partial branches. Compiled width variants are
bounded independently of K above256; changing bounds does not recompile them.

Equal parent intervals retain batched products; differing intervals scan
parents into the original output shape. Full intervals use the original dense
dot. Slice scratch and dot launches remain real costs and must be measured.
This kernel also ran on GPUs as a development control, but production CUDA
axis plans select the local cuBLAS implementation.

## API boundaries and verification

Supported dtypes are float64 and complex128, with N,N operands. Local active
plans require replicated K and two-axis output; centroid-reduction and
single-axis-output plans refuse the opt-in. Dense plan behavior is retained
for those other scopes. C/out retain their existing public plan semantics;
a local beta-zero `out` remains live, while beta-nonzero C is donated.

Require `0 <= lo <= hi <= K`. Invalid Python integer bounds raise eagerly.
Invalid dynamic bounds raise in the distributed provider; local kernels return
an all-NaN output without a callback, rather than silently clamping slices.
Plan construction executes a genuine partial interval when K>1; warming only
full intervals would bypass the active native handlers.

- `services/distrib_la/tests/test_active_gemm_range.py`: distributed CUDA
  scalar/per-parent intervals, owner crossings, alpha/beta, full/empty bounds,
  poisoned inactive operands and bitwise full-range parity.
- `services/distrib_la/tests/test_local_active_gemm_range.py`: local CPU/CUDA
  intervals, non-power-of-two capacities, complex weights, poisoned inactive
  A/B/weights, empty parents, one compiled scan and invalid-bound behavior.
- `tests/multi_device/active_band_sigma_gate.py`: typed scalar/spinor tau
  projections, time reversal, selectors, energy windows and bracket additivity.

Run391 and Run392 in the sandbox own numerical, HLO and timing evidence.
CPU emulation verifies local JAX semantics; it is not real CPU/MPI or Frontera
certification. The deployed MPI adapter must be available and attested before
claiming the complete CPU driver route.
