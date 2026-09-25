# The vendor-BLAS batched-GEMM handler (`ffi.gemm`)

The body of one contraction: the large right GEMM of
`common.contract_bands.contract_bands_block_reshard`
([band-projection primitive](staged_reshard_primitive.md)) on a CPU mesh.
Nothing else routes through it, and nothing else should without its own
measurement. Python: `src/ffi/gemm.py` (`ffi.mklblas` is a re-export shim);
handler: `src/ffi/cpp/cblas/gemm_batch_ffi.cc`; gate: `ffi.gate.Gate`
([gate contract](ffi_gate_contract.md)).

## API

```python
from ffi.gemm import gemm_ffi_enabled, gemm_ffi_mode, require_gemm_ffi, gemm_batch

C = gemm_batch(A, B)      # A (BA, M, K), B (BB, K, N) -> C (BA, M, N)
```

    C[i] = A[i] @ B[i % BB]      row-major NN, BA % BB == 0

| | |
|---|---|
| dtypes | f64, f32, c128, c64; both operands the same dtype, dispatched inside the `.so` onto `cblas_{d,s,z,c}gemm[_batch]` |
| platform | host only: `lorrax_mklblas_gemm_batch` is in `ffi_loader`'s host table, not the CUDA one |
| sharding | none: a rank-local handler with no communicator, called inside the caller's own `shard_map` |
| aliasing | no `input_output_aliases`; a `(BA, M, N)` output cannot alias a `(BA, M, K)` or `(BB, K, N)` operand |

B cycles with period `BB`, which is why one handler serves both the plain
per-k batch (`BA = BB = nk`) and the extra-stacked batch (`BA = E·nk` against
the k-only ψ, stack axis outermost). The handler builds per-batch pointer
arrays, since A's batch walks `e·nk` while B's walks `k`.

## The dial

`LORRAX_BANDS_GEMM_FFI`, modes `on`/`off`, default `on`: the FFI layer is
required ([decisions](../architecture/decisions.md)). On a CPU mesh a missing
handler refuses at startup (`Gate.enforce` in
`runtime.initialize_communicator_stack`) and at the factory, naming
`liblorrax_ffi_host.so` / `LORRAX_FFI_HOST_SO`. `=0` is an announced,
uncertified debug opt-out onto the XLA einsum arm, which is retained because
`extra="minor"` cannot ride a batched GEMM. On a non-CPU mesh the dial does
not exist: the gate's `silent_platform_demote` keeps XLA's dot, which already
calls cuBLAS. `gemm_ffi_enabled()` is read at factory time and is safe before
`jax.distributed.initialize`; consumers key kernel caches on it
(`gw.ppm_tau_kernel`'s pipeline key).

Refusals, by phase:

| phase | refusal |
|---|---|
| factory | non-CPU mesh under `require_gemm_ffi`; host `.so` without the handler (quotes `probe_target`'s reason) |
| trace | dtype outside f64/f32/c128/c64, or a mismatched pair (a de-promotion bug upstream, distinguished in the message) |

`extra="minor"` keeps the XLA plan under every mode; that is
`contract_bands`' structural fact, not a refusal of this service.

## Why a handler

XLA:CPU lowers the right contraction through Eigen dots that run 1.6–1.9×
below vendor BLAS at full threads (the in-module rate is lower still), and
the client thread pool scales near-linearly, so the gap is not fixable on the
XLA side. The small left dots of the same primitive are 1.6e-3 of the right's
flops and stay on XLA. Measured at P = 64 on Frontera (MKL 2020.1, 28 threads
per rank): the staged projection 29.4 → 19.6 s, Σ execution 58.3 → 49.2 s.

## Batched versus plain entry: decided at run time, per precision

`cblas_?gemm_batch` is an MKL extension (OpenBLAS has it; Cray LibSci does
not). Each precision whose batched entry `dlsym` resolves gets one batched
call per invocation; each that does not falls back, for that precision only,
to a sequential loop of plain `cblas_?gemm` calls. There is no build-time
probe: a `check_symbol_exists` try-compile links an executable, which must
resolve the whole shared-library closure (`--no-allow-shlib-undefined`) and so
false-negatives against an MKL whose `MPI_*`/`fi_*` references resolve only at
run time.

Whole plain-loop invocations are serialised by a process mutex: XLA can
dispatch the four `split_reim` custom calls concurrently, and four OpenMP
teams inside Cray LibSci 25.09 corrupted results (18 % relative error) and
crashed. Batched entries take no mutex. Tested providers: MKL 2020.1
(batched) and Cray LibSci 25.09 (plain; correct, not a performance claim).

The receipt is unconditional (not behind `LORRAX_DEBUG_PRINT`), one line per
precision at first use, on the launcher's rank 0
(`SLURM_PROCID`/`PMI_RANK`/`OMPI_COMM_WORLD_RANK`, the same order as
`ffi.gate.rank_id`):

```
[mklblas] GEMM entry (c128): cblas_zgemm_batch (batched) — ...
[mklblas] gemm_batch first call: dtype=c128 BA=12 BB=4 M=16 N=8 K=16 threads=28 via cblas_?gemm_batch (batched entry)
```

## Threading

`LORRAX_MKLBLAS_THREADS`: `auto` (ambient `omp_get_max_threads()`), `off`
(= 1) or an integer; strict full-string grammar, and an unrecognised value
announces and falls back to `auto`. Applied through the shared
`cpp/common/mkl_thread_pin.h` (a thread-local `mkl_set_num_threads_local`
resolved by `dlsym`; a no-op on non-MKL BLAS, where `OMP_NUM_THREADS`
governs). Unlike the ScaLAPACK handlers, which cap their team because they are
collective ([linalg_ffi](linalg_ffi.md#inside-the-scalapack-handlers)), this
is a rank-local call that wants the full team.

## Gating a change

1. Parity is value-level at 1e-12, never bit-exact: a different BLAS
   reassociates.
2. HLO pins on the 4-emulated-device mesh: `lorrax_mklblas_gemm_batch`
   custom-call counts, reduce-scatter payloads identical on and off, zero
   rank ≥ 2 `convert(f64)→c128`. `tests/test_contract_bands.py` is the
   reference.
3. HLO and collective-table gates only from a cache-cold compile.
4. Gate the dial itself: grammar, announce strings, refusal texts, and that an
   off dial loads no library, each with a deliberately broken twin that must
   fail.
