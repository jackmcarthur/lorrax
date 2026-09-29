# Fixed-shape dynamic Sigma execution

`gw.ppm_tau_kernel` owns the shared GN/HL-PPM and MPA tau contraction.
The band-extrapolation path scans bracket endpoints while retaining the
same Green builder, distributed GEMM plans, FFT service, and band projector.
Only the projected Sigma outputs acquire a leading bracket axis. The Green
and FFT temporaries remain within the loop body; projected output sharding
is explicitly `P(None, None, 'x', 'y')`.

Both face and axis Green plans enable the shared [active-range GEMM service](active_gemm_ranges.md).
After forming the exact phases and selector weights, `build_G_tau` finds
nonzero support bounds for each parent k. There is no numerical threshold:
only exact zero columns outside that interval are omitted. Explicit bracket
bounds further restrict the interval. Interior holes retain zero weights;
this is an interval contraction rather than arbitrary sparse compaction.
The same range is used for the conjugated endpoint needed by antiunitary
transport. Both bracketed and unbracketed Sigma paths use this owner. Face products
use native distributed descriptor views; axis products use local CPU/GPU
JAX interval decomposition. Neither changes the wavefunction carrier layout.

The wavefunction allocation shapes and distributed Green tiles stay fixed.
Only native contraction dimensions change. Do not replace the sequential
bracket scan with a stack of Green functions.

A bracket whose masked selector has no nonzero band on any parent builds an
exactly zero Green, so `_bracketed_face` skips it: a `lax.cond` on the
replicated selector returns zeros without the Green build, the k-convolution
or the Σ_mn projection, and every rank takes the same branch. The result is
bitwise; on CrI3 (N_μ 3088, GN-PPM, extrapolation on) the Σ τ sweep went from
129.8 to 55.7 s (sandbox claim 2944, addendum).

`gw.mpa.sigma._batch_rows` returns fixed-width pole-index, bound, and phase
arrays plus an int32 active-prefix count. Production passes that count as a
replicated dynamic operand to the shared tau callable. The W synthesis loop
runs only over the occupied prefix; its carry is one sharded `(q, mu, nu)`
array. The count is in `[0, selector_capacity]`, and rows before the count
are in the intended summation order. Standalone `build_shared_w_tau` calls
may omit the count to evaluate all supplied selector rows. Empty production
windows are skipped before dispatch.

SC band-extrapolation postprocessing caches JIT callables by output sharding.
Bracket indices and affine weights are operands, not captured constants.
The spectral estimator forms one symmetric per-state weight matrix at a time
and sums in bracket order. Unsharded callers run the eager operation sequence.

One compiled executable accepts changed active counts and skips poisoned
inactive pole rows. One bracket is bitwise equal to the unbracketed kernel,
and the production bracket partition keeps the explicit P4 output sharding.

For changes to the loop or layouts, inspect optimized P4 HLO and compiled
memory statistics as well as numerical outputs. The distributed GEMM and
FFT custom calls ([kernel catalog](../architecture/ffi_layout.md#kernel-catalog))
must still consume rank-local tiles. HLO collective counts
do not describe communication or private workspace inside native providers.
