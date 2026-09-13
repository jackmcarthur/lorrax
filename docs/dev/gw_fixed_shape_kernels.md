# Fixed-shape dynamic Sigma execution

`gw.ppm_tau_kernel` owns the shared GN/HL-PPM and MPA tau contraction.
The band-extrapolation path scans bracket endpoints while retaining the
same Green builder, distributed GEMM plans, FFT service, and band projector.
Only the projected Sigma outputs acquire a leading bracket axis. The Green
and FFT temporaries remain within the loop body; projected output sharding
is explicitly `P(None, None, 'x', 'y')`.

This removes repeated graph bodies, not the masked bands in the Green GEMM.
The contraction still uses its configured full band extent. Changing that
requires a separate tiled implementation through the common Green owner.
Do not pad a stack of Green functions to replace the sequential loop.

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
The spectral estimator still forms one symmetric per-state weight matrix at
a time and sums in bracket order. Unsharded callers retain the existing eager
operation sequence.

Validation is deliberately split:

- `tests/test_gw_fixed_shape.py` verifies one compiled executable accepts
  changed active counts and skips poisoned inactive pole rows, and that
  postprocessing accepts changed indices and weights.
- `tests/test_sigma_box_plan.py` checks selector capacity/count packing.
- `tests/multi_device/band_bracket_partition_p4.py` checks the production
  bracket partition, explicit P4 output sharding, and bitwise equality
  between one bracket and the unbracketed kernel.

For changes to the loop or layouts, inspect optimized P4 HLO and compiled
memory statistics as well as numerical outputs. The distributed GEMM and
FFT custom calls must still consume rank-local tiles. HLO collective counts
do not describe communication or private workspace inside native providers.
