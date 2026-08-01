# FFI layout: the compact tree

Status: **wave 1 executed 2026-07-31** (one C++ tree, one CMakeLists, first
facade modules + shims).  **Wave-2 module merges executed 2026-08-01**:
`ffi/io.py` (phdf5 python), `ffi/linalg/_slate.py`, `ffi/linalg/_scalapack.py`
— all three former packages remain as re-export shims (§6).  Consumer
migration and dead-leg deletion are still open.

Owner criterion: `src/ffi` must read as ONE FFI seam with vendor legs, not a
sprawling vendor monorepo.  Before this restructure the tree held **9 separate
`cpp/` directories, 3 `scripts/` directories, 3 CMakeLists, and 13 build/stage
shell scripts** spread across 10 python packages.

## 1. Inventory (before, 2026-07-31, branch fix/zq-band-gather-device-invariance)

Excludes `__pycache__` and `profile*`/`bench*` drivers (owned by the P4B
bench-relocation wave).

| package        | .py files / LOC | C++ files / LOC | build files / LOC | notes |
|----------------|-----------------|------------------|-------------------|-------|
| common         | 5 / 1216        | 4 / 536          | 7 / 1689          | loader, gate, dispatch, broadcast + 2 CMakeLists + 5 shell scripts |
| linalg         | 4 / 1008        | —                | —                 | the resolve/plan/dispatch facade (pure python) |
| mklfft         | 2 / 386         | 1 / 619          | —                 | flat-k FFT (DFTI); python serves BOTH platforms |
| mklblas        | 2 / 174         | 1 / 528          | —                 | vendor-CBLAS batched GEMM |
| cufft          | 1 / 79          | 1 / 794          | —                 | C++ only by design; python lives in mklfft |
| scalapack      | 3 / 321         | 3 / 1144         | —                 | distributed eigh + fused LU (host) |
| slate          | 6 / 963         | 8 / 2087         | 3 / 400 (scripts) | + README.md |
| cusolvermp     | 4 / 712         | 7 / 1856         | 3 / 237 (scripts) | **deletion candidate — §5** |
| cublasmp       | 2 / 342         | 4 / 1110         | —                 | **deletion candidate — §5** |
| phdf5          | 4 / 772         | 7 / 2736         | 2 / 167 (scripts) | + ARCHITECTURE.md |
| loose          | 1 / 37 (`__init__`) | —            | —                 | + AGENTS.md, PORTING.md, TEMPLATE.md |

Totals: 90 files; 10 `__init__.py` totalling 421 lines (mostly re-export
boilerplate); build entry points = 2 CMakeLists (CUDA + host) + `build.sh`,
`host/build_host.sh`, `run_shifter.sh`, `in_container.sh`, `select_gpu.sh`
+ 8 vendor stage scripts + `config/frontera/build_ffi{,_host}.sh` (outside
the tree, they stay).

## 2. Hard invariants (checked, not aspirational)

1. **Registered FFI custom-call target names do not change.**  The full set
   lives in `src/ffi/common/ffi_loader.py` (`_CUDA_TARGET_SYMBOLS`,
   `_HOST_TARGET_SYMBOLS`); the restructure moves files, never edits a target
   string or a C++ handler symbol.  Verified post-move: `nm -D` of the host
   lib is identical to the pre-move baseline.
2. **Env knob spellings do not change** (`LORRAX_FFT_FFI`,
   `LORRAX_FFT_FFI_FUSED`, `LORRAX_BANDS_GEMM_FFI`, `LORRAX_FFI_SO`,
   `LORRAX_FFI_HOST_SO`, `LORRAX_MKL_ROOT`, `LORRAX_SCALAPACK_LIBRARIES`,
   `LORRAX_SLATE_HOST_INSTALL_DIR`, …).  Aliases only; no new env vars were
   introduced by this wave (`LORRAX_FFI_PLATFORM` is a CMake `-D` cache
   variable, not an environment variable).
3. **Built `.so` names and consumed paths stay stable or their consumers are
   updated in the same commit.**  `liblorrax_ffi.so` / `liblorrax_ffi_host.so`
   names unchanged; stage-dir outputs (`$LORRAX_FFI_STAGE/build*`,
   `$LORRAX_FFI_STAGE_WTA/build_host`) unchanged.  The two IN-TREE fallback
   build dirs moved (`common/cpp/build` → `cpp/build`,
   `common/cpp/host/build` → `cpp/build_host`) and their only consumers —
   `ffi_loader._PLATFORMS[*]["build_subdir"]` and the build scripts — were
   updated in the same commit.
4. **Existing imports keep working this wave** via thin re-export shims
   (`ffi.mklfft`, `ffi.mklblas`, `ffi.common.gate`).  Consumer churn is wave 2.

## 3. Target layout

### Executed this wave

```
src/ffi/
├── __init__.py                  ffi_dial_key (unchanged)
├── gate.py                      ← common/gate.py   (announce-or-refuse gate grammar)
├── fft.py                       ← mklfft/flat_k.py (ONE module, MKL-DFTI + cuFFT dispatch)
├── gemm.py                      ← mklblas/gemm.py  (vendor-CBLAS batched GEMM)
├── linalg/                      resolve / plan / dispatch facade (unchanged)
├── common/                      ffi_loader, dispatch, broadcast (+ gate.py shim)
├── phdf5/  slate/  scalapack/   pure-python service packages (C++ gone from under them)
├── cusolvermp/  cublasmp/       pure-python; deletion candidates (§5)
├── mklfft/  mklblas/  cufft/    re-export shims / doc modules (wave-2 removals)
└── cpp/                         ★ THE ONE C++ TREE ★
    ├── CMakeLists.txt           ONE entry point; -DLORRAX_FFI_PLATFORM=cuda|host
    │                            selects the leg, REFUSES when unset (no silent default)
    ├── build.sh                 CUDA leg, Perlmutter/Shifter
    ├── build_host.sh            host leg, Perlmutter
    ├── run_shifter.sh  in_container.sh  select_gpu.sh
    ├── stage/                   8 vendor stage/build scripts, vendor-prefixed
    │                            (cusolvermp_stage_nvhpc.sh, phdf5_stage_cray.sh,
    │                             slate_build_perlmutter.sh, …)
    ├── common/                  api.cc  ffi_helpers.h  mkl_thread_pin.h  scalapack_descriptor.h
    ├── mklfft/   fft_flat_k_ffi.cc
    ├── cufft/    fft_flat_k_cuda_ffi.cc
    ├── mklblas/  gemm_batch_ffi.cc
    ├── scalapack/ blacs_grid.h  eigh_ffi.cc  solve_lu_ffi.cc
    ├── slate/    ctx.h context.cc host_ffi.cc eigh/potrf/trsm/batched_*.cc
    ├── cusolvermp/ ctx.h context.cc eigh/batched_*.cc cusolvermp_interface.h
    ├── cublasmp/ batched_gemm_ffi.cc batched_w_solve_ffi.cc w_solve_kernels.cu
    └── phdf5/    ctx.h context.cc api.cc read_ffi.cc write_ffi.cc platform_seam.h
```

Vendor subdirectory names inside `cpp/` are kept ON PURPOSE this wave: every
filename stays unique (`slate/ctx.h` vs `phdf5/ctx.h` vs `cusolvermp/ctx.h`),
every historical reference (`docs/`, `SPEEDUP_SCORECARD.md`, commit messages)
maps mechanically (`src/ffi/<v>/cpp/X` → `src/ffi/cpp/<v>/X`), and the move is
a pure `git mv` + include-path fix whose host build is byte-checkable against
the pre-move baseline.

### The ONE CMakeLists

`src/ffi/cpp/CMakeLists.txt` contains both legs verbatim behind an explicit
selector:

* `-DLORRAX_FFI_PLATFORM=host` → `liblorrax_ffi_host.so` (CUDA-free by
  construction; feature options `LORRAX_FFI_HAVE_PHDF5`,
  `LORRAX_HOST_HAVE_SCALAPACK` (MKL / explicit-link-line ScaLAPACK leg),
  `LORRAX_HOST_HAVE_SLATE`; MKL supplies the DFTI FFT + CBLAS GEMM legs).
* `-DLORRAX_FFI_PLATFORM=cuda` → `liblorrax_ffi.so` (feature options
  `LORRAX_FFI_HAVE_CAL`, `LORRAX_FFI_HAVE_CUBLASMP`, `LORRAX_FFI_HAVE_CUFFT`,
  `LORRAX_FFI_HAVE_PHDF5`, SLATE probe).
* unset → `FATAL_ERROR` naming both legs (announce-or-refuse; the old
  behaviour of "which directory you pointed cmake at" was an implicit dial).

Callers updated in the same commit: `config/frontera/build_ffi.sh`,
`config/frontera/build_ffi_host.sh`, `src/ffi/cpp/build.sh`,
`src/ffi/cpp/build_host.sh`.

### Wave-2 end state (module merges EXECUTED 2026-08-01)

* `ffi/io.py` — merge of `phdf5/{context,read,write}.py` (~740 LOC, one
  module) — DONE; `phdf5/` is now a re-export shim package (kept until the
  `file_io/` consumers move — deletion stays gated on
  `grep -rn "ffi\.phdf5"` empty; `ARCHITECTURE.md` stays with the shim).
  `ffi/linalg` absorbed `slate/*.py` → `linalg/_slate.py` (~900 LOC; the
  three per-module `_FFI_TARGET` constants were disambiguated to
  `_POTRF/_TRSM/_EIGH_FFI_TARGET` — target STRINGS unchanged) and
  `scalapack/*.py` → `linalg/_scalapack.py` (shares the SlateCtx lifecycle
  via `from ._slate import ...`).  `slate/` and `scalapack/` are re-export
  shims, submodule paths preserved (`ffi.slate.context`, `.batched` for
  `_mesh_key`, `ffi.phdf5.read/.write`); vendor dispatch stays inside
  `linalg/resolve.py` (`backend_module` still hands out the shim packages).
* shim packages `mklfft/`, `mklblas/`, `cufft/` deleted after consumers move
  to `ffi.fft` / `ffi.gemm` (§4 lists the call sites).
* optional cpp compaction to per-service dirs (`cpp/fft/`, `cpp/gemm/`,
  `cpp/linalg/`, `cpp/io/`) with vendor-prefixed filenames — only worth doing
  together with the single-TU-per-service merge; requires a GPU-leg build
  validation on rtx/Perlmutter, so it is NOT a login-node wave.

## 4. Consumer-migration plan (wave 2)

Current import counts outside `src/ffi` (src + tests, 2026-07-31):
`ffi.linalg` 36, `ffi.common` 23, `ffi.cusolvermp` 13, `ffi.slate` 10,
`ffi.mklblas` 7, `ffi.phdf5` 7, `ffi.mklfft` 6, `ffi.cublasmp` 4,
`ffi.scalapack` 3, `ffi.cufft` 0.

Mechanical rewrites, one consumer commit each:
`from ffi.mklfft import X` → `from ffi.fft import X`;
`from ffi.mklblas import X` → `from ffi.gemm import X`;
`from ffi.common.gate import Gate` → `from ffi.gate import Gate`;
`from ffi.phdf5 import X` → `from ffi.io import X` (after the io.py merge).
The shims make each rewrite independently landable; deleting a shim is the
gate that the rewrite is complete (`grep -rn "ffi\.mklfft"` must be empty).

## 5. Dead vendor legs — deletion proposal (NOT this wave)

* **cusolvermp** (4 py / 7 C++ / 3 stage scripts, ~2800 LOC): its deck keys
  are being removed by the P4B wave.  Reference count after this wave: 13
  import sites outside `src/ffi` (chiefly `linalg/resolve.py` backend table,
  `isdf/core.py`, `w_isdf.py` distributed-Dyson plan, plus tests).  The
  distributed CPU story is ScaLAPACK; the GPU story is SLATE.  Deletion
  removes the `auto|cusolvermp` spelling from the linalg backend grammar —
  that is an input-deck surface, so it needs the deprecation-window doctrine
  and an rtx-partition run proving the SLATE GPU path covers the eigh/LU
  tiers cusolvermp served.
* **cublasmp** (2 py / 4 C++, ~1450 LOC): 4 import sites
  (`bse/vq_interp.py`, `bandstructure/htransform.py`, tests).  Requires an
  rtx gate (the fused W-solve path has no measured replacement); until that
  run exists this leg stays, explicitly marked candidate.  Its
  `LORRAX_FFI_HAVE_CUBLASMP` CMake option already defaults OFF on Frontera
  (`config/frontera/build_ffi.sh`).

Neither leg was deleted or altered this wave beyond the mechanical path move.

## 6. What landed this wave vs deferred

Landed: the `src/ffi/cpp` tree (pure `git mv` + include-path updates), the
merged single CMakeLists, updated build scripts/loader/modulefile/doc paths,
`ffi/gate.py` + `ffi/fft.py` + `ffi/gemm.py` with re-export shims.
Validation: host-leg `cmake` configure + full build (`ninja`, ≤4 jobs) on the
Frontera login node with the `config/frontera/build_ffi_host.sh` stack, all
groups ON (phdf5 + ScaLAPACK + SLATE + FFT + GEMM), exported-symbol set
byte-identical to the pre-move baseline.

Deferred (with reasons): CUDA-leg build validation (needs rtx/Perlmutter —
the leg is a verbatim text relocation, path-existence-checked only);
shim deletion (consumer churn); cusolvermp/cublasmp deletion (§5 gates).
The `io.py`/`linalg` merges landed 2026-08-01 (see "Wave-2 end state"
above) with the former packages kept as shims; the same commit removed the
blanket `-Wno-unused-*` compile flags (third-party headers went SYSTEM,
the two real warnings were fixed) — host lib rebuilt, LORRAX
handler/entry-point symbol set identical.

## 7. FFT engine portability — the Occam target (owner-directed, 2026-07-31)

The CPU flat-k TU is today source-locked to MKL's DFTI descriptor API.  The
layout requirement it encodes — "howmany transforms, element stride =
batch count, distance 1" — is expressible verbatim in the FFTW *advanced*
interface: `fftw_plan_many_dft(istride=howmany, idist=1)`.  Advanced, not
guru, is the adjudicated target: the flat-k layout is one uniform batch
(single stride, single distance), exactly what `fftw_plan_many_dft`
expresses, and the guru interface's arbitrary nested loops buy nothing here
(the `CMakeLists.txt` RESOLVE-3/4 comment, which used to name
`fftw_plan_guru_dft`, was updated 2026-07-31 to match).  Two facts make
the advanced interface the portable spelling:

1. **MKL natively exports the FFTW3 C interface from its core libraries**
   (no wrapper build), so one source against `fftw_plan_many_dft` links MKL
   on Frontera — same engine, same threading — and `cray-fftw` or stock
   FFTW3 on a Cray/AMD machine.
2. The GEMM service already proved the resolution pattern: **runtime dlsym
   per symbol with announced refusal** (`gemm_batch_ffi.cc`), falling back
   to the ungated XLA path.  No engine present means slower-and-loud, never
   broken — the FFI stays an accelerator, not a dependency.

**Symbol resolution on a non-MKL site — the design, fixed here.**  The
FFTW entry points (`fftw_plan_many_dft`, `fftw_execute_dft`,
`fftw_destroy_plan`, …) are resolved at RUN time by `dlsym`
(`RTLD_DEFAULT` → `RTLD_NEXT`, the `mkl_thread_pin.h` resolver), exactly
the pattern the GEMM service proved: on Frontera they resolve out of
`libmkl_intel_lp64` already on the link line (MKL exports the FFTW3 C
interface natively — no new link dependency); on a Cray/AMD site the host
leg gains a CMake feature option (`LORRAX_HOST_HAVE_FFTW3`, detected from
`fftw3.h` + `libfftw3` like the other RESOLVE legs) that links the
system FFTW, and the same `dlsym` then finds it.  No new environment
variable is introduced in either case — the engine is named by what the
`.so` links, not by a knob — which is what invariant 2 (§2) requires;
`LORRAX_FFT_FFI` still announces-or-refuses when no engine resolves, so
absence stays slower-and-loud, never wrong.

**Parity gate for the engine swap, stated once with its class:**
value-level, **relative 1e-12** (the Σ-path class,
`flat_k_fft_service.md` §7).  Not bit-exactness — `wk_REL/gatecheck.py`
cells E/E2 assert `np.array_equal` only between two *callers of the same
handler*, and swapping DFTI for FFTW changes the engine, where bit
equality is not promised.  And not the 1e-16 figures — those are
*measured* unit residuals (max 3.7e-16) sitting at the c128 ULP, where a
threshold tests nothing (`ffi_gate_contract.md` §3.5: the first run of the
gate found two cells perturbing below the ULP and testing nothing).

The cuFFT mirror (`cufftPlanMany64` advanced layout) already covers the GPU
leg of any site, Cray included.  Remaining items, in order: (i) rework the
plan-creation section of `src/ffi/cpp/mklfft/fft_flat_k_ffi.cc` from DFTI
to the FFTW-many API + dlsym table, then rerun the automated parity gates
at the 1e-12 class above (the h5 end-to-end receipt stays ≤2.5e-14 eV);
(ii) verify MKL's FFTW3 interface maps interleaved c2c
many-plans cleanly (documented wrapper limitations are r2c-stride and
wisdom no-ops — one-job check); (iii) threading: `fftw_init_threads` /
`plan_with_nthreads` on non-MKL engines under the same
`LORRAX_FFT_FFI_THREADS` grammar; (iv) the `fftwf_` twin table if BSE
adoption wants c64; (v) only then gate the shard_map-interior
`local_*fftn3` entry so the FFI can become the permanent backend of
`make_sharded_ifftn_3d` — per `flat_k_fft_service.md` that flip is a
measurement, not a move.
