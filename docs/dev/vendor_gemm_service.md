# The vendor-BLAS batched-GEMM service (`ffi.mklblas`)

*Mechanism documentation for `LORRAX_BANDS_GEMM_FFI` and the host handler
behind it, written to the shape of `staged_reshard_primitive.md` (which
documents a primitive with physics in it; this documents a mechanism, so it
is short).  Sources: `src/ffi/mklblas/gemm.py`,
`src/ffi/mklblas/cpp/gemm_batch_ffi.cc`, `src/ffi/common/gate.py`.
Measurements: `wk_REL/RESHARD_OVERHEAD_MEMO.md` Sec. 4.4/4.5,
`wk_REL/contract_bands_notes.md`, `wk_REL/gemm_portability_bse_notes.md`.*

## 1. API contract

```python
from ffi.mklblas import (
    bands_gemm_ffi_enabled,   # the dial (factory-time, cache-key safe)
    bands_gemm_ffi_mode,      # raw grammar: "on" | "off" | "auto"
    require_bands_gemm_ffi,   # announce-or-REFUSE on a mesh
    gemm_batch,               # the ffi_call
)

C = gemm_batch(A, B)          # A (BA, M, K), B (BB, K, N) -> C (BA, M, N)
```

    C[i] = A[i] @ B[i % BB]      row-major, NN (no transposes), BA % BB == 0

| | |
|---|---|
| dtypes | f64, f32, c128, c64 — **both operands the same dtype** |
| dispatch | inside the `.so`, from the buffer element type, onto `cblas_{d,s,z,c}gemm[_batch]` |
| platform | **host only** (`ffi_loader.py:146`; not in the CUDA table) |
| `shard_map` | **none here** — rank-local handler, called from inside the caller's own `shard_map` (which cannot nest) |
| `input_output_aliases` | **none, and none is legal** — a `(BA, M, N)` output never matches a `(BA, M, K)`/`(BB, K, N)` operand buffer.  Contrast the mklfft handlers (`{0:0}`), which are shape-preserving.  Do not cargo-cult aliases onto shape-changing ops |

**The B-cycling broadcast is the whole reason one handler suffices**: it
serves the plain per-k batch (`BA == BB == nk`) and the extra-stacked batch
(`BA == E·nk` against the k-only ψ, stack axis outermost) with the same
call.  Expressed with per-batch pointer arrays, not a constant stride,
because A's batch walks `e·nk` while B's walks `k` only
(`gemm_batch_ffi.cc:407-420`).

Refusals, split by phase (the two-phase contract —
`docs/dev/ffi_gate_contract.md` §1.5):

**Factory/resolve time**, under an EXPLICIT `=1`:

1. non-CPU mesh — the dial is CPU-only by design;
2. host `.so` without the handler — quotes `probe_target`'s reason verbatim;
3. `extra="minor"` — raised by `contract_bands`, not by this service: it is
   a fact about the primitive's operand layout, not about BLAS.

**Trace time** (dtype is not known earlier and cannot be):

4. an operand dtype outside f64/f32/c128/c64, or a MISMATCHED pair.  The
   message distinguishes them: a mismatched pair means the de-promotion
   policy failed to split a mixed real/complex contraction upstream (a bug,
   report it); half/extended precision is genuinely unserved.

Under the AUTO default none of these refuse — auto quietly keeps the native
lowering wherever the dial cannot apply.

## 2. Why this is NOT a general GEMM service

It is the body of **one** contraction: the large right contraction of
`common.contract_bands.contract_bands_block_reshard`
(`contract_bands.py` `_right_none` / `_right_leading`).  Nothing else in
the tree routes through it, and nothing else should without its own
measurement.  In particular the small LEFT dots of the same primitive are
deliberately untouched — they are a measured 1.6e-3 of the right's flops,
so routing them would add call overhead to buy nothing.

The reason the right contraction is worth a handler at all is a measured
gap, not a preference: XLA:CPU lowers it through Eigen dots that saturate
**1.6–1.9× below vendor BLAS at full threads** (bare-dot probe, job
7879008; the in-module production rate is a further ~2× below the bare
dot), i.e. 295 GF/s promoted zgemm / ~172 GF/s split dgemm against 1263
GF/s MKL (memo Sec. 4.4/4.5).  Job 7879008 also showed the client
thread-pool scaling is near-linear, so this is **not** a pool-wiring defect
that could be fixed on the XLA side.

## 3. The `auto` contract

Default since the 2026-07-29 owner order: capability detection, not policy
(env-vars doctrine #8).  `auto`/unset turns the body ON iff **the platform
is CPU AND the handler resolves in the host `.so`**, announced once on rank
0:

```
[bands_gemm] AUTO-ON: CPU platform and FFI target 'lorrax_mklblas_gemm_batch'
resolves in the host .so -> ...
```

Platform comes from `JAX_PLATFORMS` **lexically**
(`ffi_loader.platform_from_env`, `ffi_loader.py:191-199`) and this is
load-bearing, not incidental: the verdict is consumed as a KERNEL-CACHE KEY
(`gw/ppm_tau_kernel.py:232`, `:405`) at a point that may precede
`jax.distributed.initialize`, and `ffi_loader.py:178-180` records why
`get_lib(None)` — and by the same argument `jax.process_index()`, which
also goes through `get_backend()` — is forbidden there.  Since 2026-07-30
the rank-0 test used by the announcement reads the launcher's rank instead
(`ffi.common.gate.rank_id`), which closes a hole where the announce itself
initialized the backend two lines after a docstring promising it would not.

The lexical read has exactly **two known miss cases**, both resolving in
the SAFE direction (auto-OFF: XLA lowering, no error, no speedup), and both
fixed identically — `export JAX_PLATFORMS=cpu`, or set the dial `=1`:

* a bare driver that leaves `JAX_PLATFORMS` unset (`platform_from_env`
  defaults CUDA-first);
* a driver that reaches a CPU mesh while `JAX_PLATFORMS` still reads
  `cuda,cpu`.

Production CPU runs are covered: the harnesses export it, and
`runtime.bootstrap()`'s GPU-less downgrade (`fallback_to_cpu_if_no_gpu_backend`)
*forces* it to `cpu`.

On a non-CPU platform `auto` is OFF **silently**, and the silence is
declared with its reason in the gate itself
(`Gate.silent_platform_demote`): the target is not in `ffi_loader`'s CUDA
table at all, so the dial does not merely resolve off — it does not exist.

## 4. Vendor portability: the batched entry is looked up at run time, per precision

`cblas_?gemm_batch` is an MKL extension of CBLAS (OpenBLAS has it; Cray
LibSci does not).  **There is no build-time feature probe and no
`HAVE_BATCH` macro** (`gemm_batch_ffi.cc:7-40`, `:188-202`;
`host/CMakeLists.txt:333-345`).  Each of `cblas_{s,d,c,z}gemm_batch` that
`dlsym` resolves gets one batched call per invocation; each that does not
falls back **for that precision** to a loop of plain `cblas_?gemm` calls
(standard CBLAS, threaded internally by the vendor, loop sequential by
design so it does not fight the BLAS's own team).  One binary serves either
vendor.  **Tested with Intel only so far** (Frontera MKL 2020.1).

Why the probe was deleted rather than fixed (owner order 2026-07-29, after
it cost a gate cycle): `check_symbol_exists` links a try_compile
EXECUTABLE, where `ld` defaults to `--no-allow-shlib-undefined` and demands
the *whole* shared-library closure resolve — which a `.so` target never has
to.  On Frontera that produced a FALSE NEGATIVE twice, jobs **7879278** and
**7879281**, both compiling the slow plain loop against an MKL that has the
batched entry: first from BLACS's open `MPI_*` references, then — after
adding `libmpi` — from `libmpi`'s own `fi_*@FABRIC_*` references (libfabric
is on `LD_LIBRARY_PATH` at run time, not on the link search path).  The
general rule: a build-time question whose wrong answer is invisible and
costs 1.6–1.9× does not belong in the build.

What replaces it as the receipt: `announce_entry_once` prints one line per
precision at that precision's first use, **unconditionally** — not behind
`LORRAX_MKLBLAS_LOG` — so a silent downgrade is impossible by construction:

```
[mklblas] GEMM entry (c128): cblas_zgemm_batch (batched) — this BLAS provides the batched entry.
[mklblas] gemm_batch first call: dtype=c128 BA=12 BB=4 M=16 N=8 K=16 threads=28 via cblas_?gemm_batch (batched entry)
```

Rank-guarded via `announce_here()` (`gemm_batch_ffi.cc:229-235`), which
reads `SLURM_PROCID`/`PMI_RANK`/`OMPI_COMM_WORLD_RANK` — the same list, in
the same order, as `ffi.common.gate.rank_id` on the Python side.

## 5. Threading

`LORRAX_MKLBLAS_THREADS` — `auto` (ambient `omp_get_max_threads()`, i.e.
the harness's `OMP_NUM_THREADS` under `taskset`; the production 28/rank) |
`off` (= 1) | an integer.  Strict full-string grammar; unrecognized values
announce on stderr and fall back to `auto` (`gemm_batch_ffi.cc:318-336`).
Applied through `MklLocalPin` (`:285-298`, `:407`), a thread-local
`MKL_Set_Num_Threads_Local` resolved by `dlsym` — a no-op on a non-MKL
BLAS, where `OMP_NUM_THREADS` governs.

**The workstream-AW cliff does not apply here** and the handler says so in
place (`:76-80`): capping MKL threads inside a *ScaLAPACK* handler is
required because that handler is collective; this is a rank-LOCAL BLAS
call, the same class as the plan-A local eigh that NEEDS the full thread
count.

The pin is currently the third copy of the same ~60-line block
(`scalapack/cpp/blacs_grid.h:157-233`, `mklfft/cpp/fft_flat_k_ffi.cc:96-159`,
here at `:277-338`), kept separate so each TU stays MPI-free
(`blacs_grid.h:40` includes `<mpi.h>`).  The copies have already DRIFTED:
this one resolves the pin with `resolve_sym` (`RTLD_DEFAULT` →
`RTLD_NEXT`), the mklfft copy with bare `dlsym(RTLD_DEFAULT, ...)`, so
under a local-scope `dlopen` the GEMM handler still pins MKL and the FFT
handler silently does not.  `TEMPLATE.md:194-195`'s own threshold ("extract
when a THIRD library would copy it a third time") is met — the fix is a new
MPI-free `ffi/common/cpp/mkl_thread_pin.h`, **not** including
`blacs_grid.h`.  Not done in this wave (C++ was out of scope); recorded so
it is not rediscovered.

## 6. Platform reach — what a GPU user sees

Nothing, by design, and no action is needed.  `lorrax_mklblas_gemm_batch`
is in `ffi_loader`'s host table only, so a forced CUDA probe reports
*unknown target*; `auto` resolves OFF from `JAX_PLATFORMS` before any
probe; the primitive re-checks `mesh.devices.flat[0].platform`; and an
explicit `=1` refuses.  XLA:GPU's own `dot` lowering already dispatches
cuBLAS, which is optimal there.

## 7. How to gate a change

1. **Parity class: value-level, 1e-12 — never bit-exact.**  A different
   BLAS reassociates.  (The 2026-07-29 A/B did come out h5-bit-identical at
   P=64, job 7879296 — record that as an observation, not as the contract.)
2. **HLO pins** on the 4-emulated-device mesh: `lorrax_mklblas_gemm_batch`
   custom-call counts, reduce-scatter payloads byte-equal off-vs-on, zero
   rank≥2 `convert(f64)→c128`.  `tests/test_contract_bands.py` is the
   reference implementation (9 cells, 13 `LORRAX_BANDS_GEMM_FFI` mutations).
3. **Cache-cold rule (AY.2)**: collective-table and HLO gates are valid ONLY
   from a cache-cold compile.
4. **Gate the dial itself**, not only the math: `wk_REL/gatecheck.py` pins
   the grammar against the pre-consolidation parser over a 23-spelling
   matrix, the announce strings byte-for-byte, the refusal texts, and that
   an OFF dial loads no library at all.  Every cell runs a deliberately
   broken twin that must FAIL; a cell whose twin passes is reported VOID.

Reference runs: 7879008 (unit) / 7879010 (P=64 A/B: nb=128 project_rs
29.407→19.622 s, sigma.exec 58.313→49.224; nb=256 20.565→14.162 /
35.234→29.979; .dat exact-0, h5 ≤2.5e-14 eV), 7879296 (AUTO A/B),
**7882084**/**7882116** (the microservice refactor gate: 9/9 gate cells with their
RED twins, `tests/test_contract_bands.py` 9/9, `tests/test_projection_lgemm.py`
2/2, owned-manifest identical, zero bytecode written).

## 8. What is NOT claimed

* No GPU measurement of this dial exists and none is possible — host table
  only.  `wk_REL/audit_gpu_gemm.log:3` ran on a GPU node but every row is
  `XLA:CPU dot_general` / bare MKL.
* Measured domain of every number above: **one Frontera CLX node, 28 cores
  per rank under `taskset`, Intel MKL 2020.1, SHM/`impl=mpi` collectives**.
  No other vendor BLAS has been run.
* The batched-vs-plain entry choice has only ever been exercised with the
  batched entry present.  The plain-loop arm is compiled and shares the
  broadcast rule and the pin with the batched arm (`:419`), but has not
  been measured.
