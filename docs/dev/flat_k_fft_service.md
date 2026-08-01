# The flat-k FFT service (`ffi.mklfft` + `ffi.cufft`)

*Mechanism documentation for `LORRAX_FFT_FFI` / `LORRAX_FFT_FFI_FUSED` and
the two platform handlers behind them, written to the shape of
`staged_reshard_primitive.md`.  Sources: `src/ffi/fft.py`,
`src/ffi/cpp/mklfft/fft_flat_k_ffi.cc`,
`src/ffi/cpp/cufft/fft_flat_k_cuda_ffi.cc`, `src/ffi/gate.py`.
Measurements: `wk_REL/sigma_perf_results.md`, `wk_REL/audit_gpu_fft.log`,
`wk_REL/cufft_unit.log`, `wk_REL/audit_gpu_hlo.log` (job 7879378).*

> **Adoption state, 2026-07-30 — SUPERSEDED 2026-07-31.**  The 07-30 text
> below is kept as history; the delegation has landed.  `ffi.mklfft`
> (`src/ffi/fft.py`) IS the single implementation: `common/fft_helpers.py`
> imports the gate and both wrapper bodies from it (`fft_helpers.py:304`)
> and carries no copy of its own.  The A2/E/E2 pins now guard the
> re-export seam, not a second copy.
>
> *(historical, 2026-07-30)* `ffi.mklfft` is the intended single
> implementation and is byte-parity-gated against the live path, but
> `src/common/fft_helpers.py` still carries its own copy of the gate and the
> two wrapper bodies (that file belonged to another workstream this wave).
> Until it delegates, `fft_helpers` is what production calls.
> `wk_REL/gatecheck.py` cells A2/E/E2 pin the two copies against each other
> — decision-for-decision on 23 env spellings, and BIT-EXACT on the arrays —
> so the interim cannot drift silently.

## 1. API contract

```python
from ffi.mklfft import (
    fft_ffi_enabled,          # LORRAX_FFT_FFI        (factory-time read)
    fused_fft_ffi_enabled,    # LORRAX_FFT_FFI_FUSED  (factory-time read)
    require_fft_ffi,          # announce-or-REFUSE on a mesh
    make_flat_k_fft_ffi,      # the two plain transforms
    make_gw_conv_ffi,         # the fused convolution
)

fft = make_flat_k_fft_ffi(mesh, kgrid, spec, kind='ifftn'|'fftn',
                          norm='ortho'|'forward'|'backward'|None,
                          out_spec=None)
y = fft(x)                       # (nk, *trail) -> (nk, *trail), c128

conv = make_gw_conv_ffi(mesh, kgrid, g_spec, v_spec, norm='ortho', mult=1.0)
sigma = conv(G_flat, W_flat)     # G (nk,a,mx,b,my), W (nk,mx,my)
```

| | |
|---|---|
| `spec` | the `PartitionSpec` on the **3-D form** `(nkx,nky,nkz,*trail)`; the three leading k axes MUST be `None` (replicated) |
| dtype | **complex128 only**, both platforms |
| layout | k-major end to end.  The operand is never reshaped to the k-minor 3-D form — that is the entire point |
| norms | computed in Python (`ffi_fft_scale`) to match `jnp.fft` exactly, shipped to the handler as a plain scale.  The handlers implement no norm convention of their own |
| `input_output_aliases` | `{0: 0}` — shape-preserving, so aliasing is legal and is the terminal form of donation (zero extra big tiles when the operand is dead) |
| `mult` (conv only) | folded into the forward-transform scale, e.g. Σ's `-1/√N_k` |

Refusals, split by phase (`docs/dev/ffi_gate_contract.md` §1.5):

**Factory time** — 1. mesh platform outside {cpu, CUDA}; 2. the platform's
library lacks the target (quotes `probe_target`'s reason); 3. `out_spec !=
spec` (no post-FFT reshard is implemented); 4. FFT axes not replicated in
`spec`.

**Trace time** — 5. dtype != c128; 6. rank mismatch against the 3-D-form
spec; 7. leading extent != `nkx·nky·nkz`; 8. (conv) G/W shard-shape
disagreement.  These are trace-time FACTS and cannot fire earlier.

Both flags are read at FACTORY time and **must be part of every consumer's
kernel cache key** — `gw/ppm_tau_kernel.py:231-233` and `:404-406` do this
for all three FFI dials plus the stage-timing flag.

## 2. Raison d'être: the layout anchor, stated structurally

XLA's `fft` custom-call requires the transformed axes **minor-most**.  The Σ
τ kernel alternates `dot` (k-major flat) with `fft` (k-minor 3-D), so every
helper boundary pays a full transpose copy of the ~398 MB/rank μ² tile.
Measured at 60–65% of `sigma.exec` at nb=128/P=64, and CLOSED as structural
for any XLA-side arrangement (`wk_REL/sigma_perf_results.md`): the cost can
be moved, not removed, as long as the FFT is XLA's.

Stride descriptors read the dot-layout tile where it lies.  MKL's DFTI
descriptor API and `cufftPlanMany64`'s advanced data layout both express
"batch of N-D transforms at stride T, distance 1" directly, so the boundary
stops anchoring layouts and the transposes disappear instead of moving.
Measured on CPU: `sigma.exec` 272 → 71.9 s (3.8×), τ HLO transposes 6 → 0,
h5 parity 2.5e-14 eV.

"DFTI" names Intel's *descriptor interface* and nothing else — this is a
genuine O(N log N) FFT at any k-count, **not** a DFT-as-matmul (owner veto,
restated at `flat_k.py`'s module docstring and `fft_flat_k_ffi.cc`).

## 3. The two entry LAYERS, and why the gate reaches only one

This is the most over-read thing about the service, so it is stated as
structure, not as a TODO.

| layer | what it is | FFI-gated? |
|---|---|---|
| `fft_helpers.make_flat_k_*` | wraps its **own** `shard_map` | **yes** |
| `fft_helpers.local_fftn3` / `local_ifftn3` | bare `jnp.fft` aliases, for code ALREADY inside a `shard_map` | **no route at all** |

`shard_map` cannot nest, so a call site that is already inside one cannot
use the gated layer, and the gated layer cannot be pushed inside it without
a shard_map-free FFI entry that does not exist.  Consequences, by grep:

* `src/isdf/core.py:35-39` imports `make_flat_k_ifftn`/`make_flat_k_fftn`
  and **never calls them**; it makes six raw `jnp.fft` calls (`:355`,
  `:360`, `:373`, `:845`, `:850`, `:861`).  The reason is stated at
  `:262-280` and is legitimate: the pair-density pipeline is one monolithic
  `shard_map`, and a decomposed chain costs 21 GiB vs 13 GiB of
  BufferAssignment slots.
* `src/common/wfn_transforms.py` likewise (`:536`, `:709`, `:975`, `:1266`).
* BSE has **no** `make_flat_k_*` call site, so neither flag affects it on
  either platform (`env_vars.md:129` records the symptom; this is the
  cause).  BSE's fp32-GMRES FFTs are complex64 and these handlers are
  c128-only, so routing them at this layer would have to refuse them.

Adding a `local_*` FFI entry is the real gap, and it is a **measurement**
(is a stride-descriptor FFT even right at those layouts?), not a refactor.
Do not stage it as a pure move.

## 4. Platform mirror: same targets, different symbols

| target | host symbol | CUDA symbol |
|---|---|---|
| `lorrax_mklfft_flat_k` | `MklFftFlatKHostFfi` (`ffi_loader.py:142`) | `CufftFlatKCudaFfi` (`:105`) |
| `lorrax_mklfft_gw_conv` | `MklFftGwConvHostFfi` (`:143`) | `CufftGwConvCudaFfi` (`:106`) |

One platform-agnostic `ffi_call` per site therefore resolves the right
handler from the LOWERING platform — the same split jaxlib uses for cpu
(lapack) vs CUDA (cusolver), and the same one `ffi/phdf5` uses in-tree.
**This is why `src/ffi/cufft/` holds C++ only**; its `__init__.py` is the
design note plus a `__getattr__` that redirects the obvious wrong import to
`ffi.mklfft`.  The target *strings* keep the "mklfft" name because the CPU
prototype coined them: the name is historical, the dispatch is not.

Platform-specific, deliberately not mirrored:

* **CUDA**: `cufftPlanMany64` with `inembed`/`istride=T`/`idist=1`; cuFFT
  auto-allocation **off** (`fft_flat_k_cuda_ffi.cc:435`) with two grow-only
  `cudaMalloc` arenas outside the XLA allocator (`:401-425`), a
  `cudaDeviceSynchronize()` before every growth (`:412`), and plan cache /
  arena / enqueue serialized under one process mutex with the
  single-compute-stream assumption stated honestly (`:72-78`, `:519`).  The
  fused multiply is a plain CUDA kernel compiled by **NVRTC** (not a cuFFT
  callback, and no nvcc at build time — `:46-55`).
* **host**: an OpenMP chunk loop with MKL pinned to 1 thread inside
  (`fft_flat_k_ffi.cc:332-337`) and a process-global `Arena` under a mutex
  (`:418-426`).  The compact-chunk L2 staging (`:299-315`, measured 2.8×) is
  **explicitly not mirrored on CUDA** (`fft_flat_k_cuda_ffi.cc:20-23`: "the
  host engine's per-thread L2 buffer was a CLX cache artifact, not part of
  the contract").  That is the right call and the right way to say it.

Three structurally different scratch/concurrency models, each measured, is
also why there is no shared C++ handler base (`TEMPLATE.md:188-195`).

## 5. Memory contract — three classes, and what the query covers

1. **XLA-visible** buffers: operand, result, temps.  What
   `compiled.memory_analysis()` reports.
2. **cuFFT plan workspace**: taken by jaxlib's `FftThunk` from a runtime
   scratch allocator, so it is **not in XLA's buffer assignment at all**
   (`src/runtime/aot_memory.py`).
3. **LORRAX arenas** — *only under `LORRAX_FFT_FFI` on CUDA*: the handler
   disables cuFFT auto-allocation (`fft_flat_k_cuda_ffi.cc:435`) and
   `cudaMalloc`s two grow-only arenas of its own (`:401-425`), invisible to
   both 1 and 2.  The `V_R` arena is **measured at 99.7 MB** at production
   shape (`audit_gpu_fft.log:64`).

**Status, 2026-07-30 — this was the service's worst defect and it is
FIXED**, by the `fft_helpers` owner in the same wave as this document.
`query_fft_peak_bytes` used to promise "input buffer + output buffer +
cuFFT scratch" in its docstring while summing only class 1, fall back
silently to a 3×data guess on any exception under a comment claiming it was
"logged so the caller notices" (there was no log call), and be swallowed
again by an `except Exception: pass` in `gflat_memory_model`.  It now
compiles **the factory production actually uses** (`make_sharded_fftn_3d`,
one rank-3 cuFFT plan per rank — not the per-axis form, which sizes three
rank-1 plans no production path builds) and returns
`aot_kernel_peak_bytes(...).total` = class 1 **+** class 2, with every
weaker path announced once per process: a failed compile, and a compiled
FFT whose optimized HLO exposes no `fft` op (which on a CUDA mesh would
zero the cuFFT term by omission rather than by measurement).
`gflat_memory_model._fft_box_bytes`'s analytic fallback announces too.

**What is still NOT covered: class 3.**  Under the FFI gate on CUDA the
handler's own arenas are outside both accountings, so the number is a lower
bound by ~100 MB/rank at production shape.  Note also that this handler's
*own* plan workspace is measured **0.0 MB** for these strided Z2Z plans
(`audit_gpu_fft.log:12,29,44,72,83`) — that is a fact about the FFI plan,
and says nothing about XLA's `FftThunk` plan, which uses a different
layout and is what class 2 measures.

## 6. Threading, chunking, logging

| var | where | effect |
|---|---|---|
| `LORRAX_FFT_FFI_THREADS` | `fft_flat_k_ffi.cc::team_threads` | OpenMP team size for the chunk loop |
| `LORRAX_FFT_FFI_CHUNK` | `fft_flat_k_ffi.cc::chunk_elems` | compact-chunk staging size (the L2 lever, 2.8×) |
| `LORRAX_FFT_FFI_LOG` | `fft_flat_k_ffi.cc::log_enabled`, `fft_flat_k_cuda_ffi.cc::log_enabled` | descriptor/plan/arena/first-call diagnostics, ONE spelling on both platforms |

`LORRAX_MKLFFT_{THREADS,CHUNK,LOG}` and `LORRAX_CUFFT_LOG` are deprecated
aliases, honored with a one-time announcement (`mklpin::knob_value`).

> **SUPERSEDED 2026-07-31** — two claims that stood here are closed:
> *(a)* "none of these is documented in `docs/dev/env_vars.md`" — rows now
> exist there (§2b for THREADS/CHUNK, §3b for LOG), under the current
> `LORRAX_FFT_FFI_*` spellings with the alias policy stated;
> *(b)* "the C++ log output is NOT rank-guarded" — both handlers are
> rank-scoped since the 2026-07-30 audit (rank 0 by default, `=all` for
> every rank), via the shared MPI-free header `cpp/common/mkl_thread_pin.h`
> (`announce_here` / `log_value_here`) that `vendor_gemm_service.md` §5
> proposed and that now exists.  Pre-audit, `LORRAX_MKLFFT_LOG` printed
> once per thread per descriptor geometry per rank (kept as the reason the
> default is rank 0).

## 7. How to gate a change

1. **Parity class: value-level ~1e-15, never bit-exact** against `jnp.fft` —
   it is a different FFT engine.  Gate the Σ path at 1e-12.  (Between two
   *callers of the same handler* the result IS bit-exact, which is what
   `gatecheck.py` cells E/E2 assert.)
2. **HLO pins**: transpose/copy counts at the flat-k boundary, and the
   custom-call count.  On GPU read the **temp-bytes** column, not the
   census's verdict line — see §8.
3. **Cache-cold rule (AY.2)** for any HLO or collective-table gate.
4. **Gate the dials**, not only the math: `wk_REL/gatecheck.py` pins the
   grammar of both flags against the pre-consolidation parsers over 23
   spellings, the announce strings byte-for-byte, and both wrapper bodies
   bit-exactly.  Every cell runs a broken twin that must FAIL.
5. **Run it on BOTH platforms.**  The whole point of §4 is that one Python
   module reaches two different C++ handlers, so a CPU-only gate leaves the
   cuFFT half unverified.  `wk_REL/ffisvc_gate.sbatch` (host MKL) and
   `wk_REL/ffisvc_gpu.sbatch` (rtx-dev, cuFFT) run the identical cells.
5. **There is still no `tests/` unit gate for this service** — `grep -rn
   "fft_helpers" tests` returns nothing.  The 15-case correctness cell
   exists only in `wk_REL/`, which does not ship.  For a public release that
   is a gap; numbered request in the wave report.

## 8. Evidence index, with measured domains

| claim | status |
|---|---|
| CPU: `sigma.exec` 272 → 71.9 s (3.8×), τ transposes 6 → 0, h5 2.5e-14 eV | measured, Frontera CLX, P=64, nb=128 |
| cuFFT handler is CORRECT on GPU | **measured** — 15 cells, max rel err 0–3.7e-16, incl. fused-vs-decomposed (2.87e-16), both donation modes, all four norms on an odd (3,2,1)×(5,7,3) grid; NVRTC produced a real sm_75 cubin (`cufft_unit.log:10-33`) |
| cuFFT handler is ~2× the XLA path at production shapes | **measured**, 1 device, standalone bench: S2 flat-k 22.28 → 9.18 ms, S4 21.85 → 10.53, fused conv 33.89 → 15.62 (`audit_gpu_fft.log:104-105`, `:117-120`) |
| this handler's plan workspace is 0 for these strided Z2Z plans | measured (`audit_gpu_fft.log:12,29,44,72,83`); says nothing about XLA's `FftThunk` |
| the `V_R` arena is ~100 MB and invisible to `memory_analysis()` | measured (`audit_gpu_fft.log:64`) |
| the RELOCATED `ffi.mklfft` wrappers lower identically through cuFFT | **measured on real GPU**, job **7882123** (rtx-dev, `[CudaDevice(id=0)]`): `make_flat_k_fft_ffi` and `make_gw_conv_ffi` are **bit-exact** against the live `fft_helpers` path for both directions × three norms, and all 9 gate cells pass with their RED twins |
| multi-GPU / sharded meshes | **UNMEASURED** — every GPU log is `[CudaDevice(id=0)]` |
| the service inside the production Σ driver on GPU | **UNMEASURED** — all GPU runs are standalone benches |
| an `auto` mode for `LORRAX_FFT_FFI` | **not decided.**  The GPU margin (2.1–2.4×) is larger than the one that justified the GEMM `auto`.  The defect the assessment recorded is the absence of a written REASON for the current default, not the value.  Owner call |

Job 7879378, 2026-07-29, node c196-012, 1× Quadro RTX 5000 sm_75,
`src@0dd94a8` (`audit_gpu.7879378.out:1-13`).

**One claim in the C++ is not supported by the project's own census, and is
flagged rather than repeated here.**  `fft_flat_k_cuda_ffi.cc:7-11` says
XLA "transposes the tile before/after EVERY fft" on GPU, by the same
mechanism as on host.  `audit_gpu_hlo.log:9-55` reports `transpose ops: 0 /
copy ops: 0` and prints "NO LARGE LAYOUT TRANSPOSE" at every flat-k site.
But the census is a **defective instrument for that question**: the same
rows report `fusion ops: 1` and `XLA temp bytes: 398.7 MB` for the flat-k
site (`:50-53`) against `fusion ops: 0` / `0.0 MB` for the minor-most twin
at identical element count (`:66-71`).  A 398.7 MB temp produced by a fusion
*is* the layout materialization; the census only counts `opcode ==
transpose|copy`, so its verdict line is wrong by construction.  The
defensible statement: **on GPU the layout cost is real (398.7 MB temp, plus
the measured 22.28 vs 8.90 ms flat-k-vs-minor gap, `audit_gpu_fft.log:104`,
`:108`) but it is emitted as a fusion, not as a `transpose`.**  The header
sentence and the census's verdict line both need correcting — neither file
was owned this wave.
