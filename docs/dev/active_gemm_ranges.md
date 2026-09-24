# Active ranges in planned GEMM

`gemm_plan(..., enable_active_range=True)` enables
`plan.active_range(A, B, lo, hi, C=None, *, out=None, weights=None)`.
The product is `alpha * A[q,:,lo:hi] @ B[q,lo:hi,:] + beta*C[q]`.
Optional `weights` has shape `(nq,K)` and scales A's contraction columns
before multiplication; complex weights require a complex-valued plan. Bounds are replicated integer scalars or arrays of
shape `(nq,)`. Allocation shapes stay `(nq,m,K)`, `(nq,K,n)` and `(nq,m,n)`;
output is always `P(None,x,y)` for the active-range API.

When the interval is already known on the host, call
`prepared = plan.prepare_active_range(lo, hi)` outside a compiled loop and
then use `prepared(A, B, C=None, *, out=None, weights=None)`. Preparation makes
an immutable copy of the scalar or per-parent bounds. The returned callable
has no bounds array among its runtime operands, but retains the same
alpha/beta, C/out, weighting, shape and sharding contract as `active_range`.

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
routing belongs to [FFI layout](../architecture/ffi_layout.md). A CUDA library
without the active target refuses by name; the fix is to rebuild the canonical
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

The prepared target `lorrax_cublasmp_prepared_active_range_gemm` receives the
validated intervals as immutable FFI metadata. It enters the same descriptor
view implementation as the dynamic target, without copying bounds from the
device or synchronizing the CUDA stream to read them. Requesting a prepared
callable probes this target eagerly. Its first operand call still compiles the
bound-specific executable.

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

Weighting forms one full weighted-A tile; the kernel adds no slice panels or
accumulation buffers beyond it. When all parents request the full interval, a
JAX conditional takes the dense weighted dot and its evaluation order.

The prepared target `lorrax_cublas_local_prepared_active_range_gemm` uses the
same pointer and leading-dimension implementation with immutable FFI metadata:
no per-call device-to-host bounds copy or stream synchronization. The weighted-A
tile and the 4 MiB scratch output remain.

## Local CPU kernel

`distrib_la._active_local.active_local_matmul` uses existing JAX dot lowering
without a native context or callback. A `lax.while_loop` traverses an interval
using widths selected by `lax.switch`: powers of two capped at 256 columns.
Only wholly active, disjoint slices enter each dot; the last piece never pads
inactive columns. Weights are applied inside the selected slices, avoiding a
full weighted-A temporary in partial branches. At most nine width variants are
compiled, independent of K; changing bounds does not recompile them.

Equal parent intervals keep batched products; differing intervals scan
parents into the original output shape. Full intervals use the dense dot.
Slice scratch and dot launches are real costs. CUDA axis plans select the
local cuBLAS kernel instead.

Prepared CPU calls close over the validated bounds and use this same panel
kernel. They remain callback-free and accept no runtime bounds operand.

## Prepared callable lifetime and compilation

The caller owns every callable returned by `prepare_active_range`. There is no
process-global cache of bound vectors and no mutable native handle carrying
the interval. Retaining a callable retains only its plan and immutable host
metadata until JAX compiles it on first use. Each distinct captured interval
may create another executable, so this interface is intended for intervals
that recur across many calls. Dynamic or frequently changing intervals should
continue to use `active_range`.

Preparation itself does not execute a GEMM or allocate full-size dummy
operands. A full-range prepared call selects the original dense operation so
its established numerical order is unchanged. Explicit preparation still
performs the CUDA capability probe immediately, even for full bounds, so a
requested unavailable provider cannot be hidden by that dense fast path.

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
  poisoned inactive operands, bitwise full-range parity, and immutable
  prepared metadata after mutation of the caller's original bounds arrays.
- `services/distrib_la/tests/test_local_active_gemm_range.py`: local CPU/CUDA
  intervals, non-power-of-two capacities, complex weights, poisoned inactive
  A/B/weights, empty parents, one compiled scan, prepared calls without
  runtime bounds operands, and invalid-bound behavior.
- `tests/multi_device/active_band_sigma_gate.py`: typed scalar/spinor tau
  projections, time reversal, selectors, energy windows and bracket additivity.

CPU emulation verifies local JAX semantics only; the complete CPU driver
route additionally needs the attested MPI adapter
([MPI collectives](mpi_collectives.md)).
