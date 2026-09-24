# The FFI layer

How LORRAX reaches vendor libraries: what the layers are, what each machine
provides, which knobs decide correctness rather than speed, and how to tell
the failure modes apart.

> **Verification scope.** Every `file:line` and every default below was read
> from source on **Perlmutter**, on **2026-08-06**. The first pass read
> `886139f`; the page was then rebased onto `8789131` and the statements in
> §§3–8 that a commit in between could have moved were **re-read at
> `8789131`** the same day — the stripe defaults, the CAL option default, the
> `run_shifter.sh` library path, the DFTI/FFTW3 counts, the `phdf5` context
> defaults and the three shim call-site counts. Statements marked *(Frontera,
> unverified 2026-08-06)* were not checked on Frontera and must be re-read
> there before being relied on. Line numbers drift: they are provided so you
> can find the code, not so you can quote them. **Read the file.**
>
> This page owns the *native boundary*. It does not own owner rulings
> ([`decisions.md`](decisions.md)), the SlabIO contract and its measurements
> ([`slab_io.md`](slab_io.md)), or knob spellings and defaults
> ([`../dev/env_vars.md`](../dev/env_vars.md)). See the
> [register](../index.md#register).

---

## 1. The layers

Five, outermost first. Each one can refuse; none silently substitutes.

| # | Layer | Lives in | Job |
|---|---|---|---|
| 1 | **Consumer** | `src/gw/`, `src/file_io/`, `src/bse/`, … | states *logical* intent — shapes, not strides |
| 2 | **Service facade** (Python) | `src/ffi/io.py`, `fft.py`, `gemm.py`, `linalg/` | owns the gate grammar, builds the descriptor, picks a backend |
| 3 | **Gate** | `src/ffi/gate.py` | announce-or-refuse; an explicit request that cannot be honoured **refuses**, never downgrades |
| 4 | **XLA FFI custom call** | `src/ffi/common/ffi_loader.py` | resolves handler symbols out of the built `.so` |
| 5 | **C++ handler + vendor library** | `src/ffi/cpp/<vendor>/` | the actual MPI-IO / BLAS / FFT / solver call |

The rule that makes this tractable: **a vendor dependency enters only
through a facade with runtime resolution and an announced refusal.** Since
the 2026-08-01 ruling (`decisions.md`) the FFI is *required*, so a missing
library is a startup refusal naming the `.so` — not a demotion to a slower
Python path. Vendor-portability fallbacks *inside* a handler (FFTW-vs-MKL
symbol resolution, plain-loop CBLAS) are a different thing and stay: they
are how the required layer remains buildable everywhere.

### Python-side module map

`src/ffi/io.py` (the parallel-HDF5 service), `fft.py` (flat-k FFT and the
k-convolution router),
`gemm.py` (batched vendor GEMM), `gate.py`, and the `linalg/` facade are the
real modules. `phdf5/`, `slate/`, `scalapack/`, `mklfft/`, `mklblas/`,
`cufft/` survive as **re-export shims** — `src/ffi/phdf5/` is 40 lines
across four files.

Deleting a shim is gated on its consumers moving, and **one has now moved
in part**: outside `src/ffi/` there are **5** `ffi.phdf5` references left
(down from 10), all in `file_io/_slab_io_ffi.py` — `file_io/wfn_loader.py`
had 3 and has none since the wave-1 wfn_loader extraction promoted the
union read into `SlabIO.read_slabs` (2026-08-07,
[docs/services/wfn_loader.md](../services/wfn_loader.md)). Also 6 `ffi.mklblas`
(`common/contract_bands.py`), and 4 `ffi.mklfft` (`common/fft_helpers.py`,
`gw/ppm_tau_kernel.py`) — counted at `886139f` and **re-counted unchanged at
`8789131`, 2026-08-06**. The gate is
`grep -rn "ffi\.mklfft" src/ tests/ | grep -v '^src/ffi/'` returning empty.
Run the grep; do not trust a count written down here.

---

## 2. The one C++ tree

`src/ffi/cpp/` — one directory, one `CMakeLists.txt`, both platform legs
behind an explicit selector.

```
src/ffi/cpp/
├── CMakeLists.txt      -DLORRAX_FFI_PLATFORM=cuda|host; FATAL_ERROR when unset
├── build.sh            CUDA leg (inside Shifter)
├── build_host.sh       host leg
├── run_shifter.sh  in_container.sh  select_gpu.sh  gate_one_mpi.sh
├── stage/              vendor stage scripts (phdf5_stage_cray.sh, …)
├── common/  mklfft/  cufft/  mklblas/  scalapack/  slate/  cusolvermp/  cublasmp/
└── phdf5/   api.cc  context.cc  ctx.h  read_ffi.cc  write_ffi.cc
            phdf5_interface.h  platform_seam.h  shard_index.h
```

Two legs, two libraries:

* `-DLORRAX_FFI_PLATFORM=host` → `liblorrax_ffi_host.so`. CUDA-free by
  construction. Feature options `LORRAX_FFI_HAVE_PHDF5`,
  `LORRAX_HOST_HAVE_SCALAPACK`, `LORRAX_HOST_HAVE_SLATE`,
  `LORRAX_HOST_HAVE_FFTW3`.
* `-DLORRAX_FFI_PLATFORM=cuda` → `liblorrax_ffi.so`. Feature options
  `LORRAX_FFI_HAVE_CAL`, `LORRAX_FFI_HAVE_CUBLASMP`, `LORRAX_FFI_HAVE_CUFFT`,
  `LORRAX_FFI_HAVE_PHDF5`, plus a SLATE probe.
* unset → `FATAL_ERROR` naming both legs (`CMakeLists.txt:1399`). "Which
  directory you pointed cmake at" was an implicit dial; it is gone.

`LORRAX_FFI_PLATFORM` is a **CMake cache variable, not an environment
variable** — it is never read from the environment by any Python module.

Vendor subdirectory names are kept deliberately: every filename stays unique
(`slate/ctx.h` vs `phdf5/ctx.h` vs `cusolvermp/ctx.h`), and every historical
reference maps mechanically `src/ffi/<v>/cpp/X` → `src/ffi/cpp/<v>/X`. Docs
and commit messages written before 2026-07-31 use the old spelling.

### 2a. The deployable unit is one sealed pair

The two independently built legs become one production provider only through
`src/ffi/cpp/stage/seal_bundle.py`.  It publishes a new, non-overwriting
directory containing both canonical shared objects and one
`lorrax_ffi_bundle.json`.  That manifest binds the pair, the handler ABI, exact
file sizes and SHA-256 values, each ELF SONAME/`DT_NEEDED` record, and the
dependency-first private redistributable closure.  Both libraries put literal
`$ORIGIN` first in their existing RPATHs; the common loader also preloads every
manifested private provider by exact path, so an ordinary run script owns no
library search path.

The private closure is deliberately narrow: cuSolverMp/cuBLASMp/CAL,
SLATE/BLAS++/LAPACK++, and NVSHMEM may be sealed.  MPI, site HDF5, the system
and compiler runtimes, CUDA runtime/driver libraries, and NCCL remain owned by
the selected machine runtime and are never copied or manifest-owned.  At load,
`lxkit.native_provider` rehashes both legs and the private closure, checks the
live ABI-symbol origin with `dladdr`, and refuses any mapped engine-private
provider not named by the manifest.  The core loader and standalone
`distrib_la` loader call this same policy; neither carries a private copy.

`source.revision` records which clean source build produced the sealed pair.
It is provenance, not a demand that the active Python checkout have the same
Git SHA: a stable FFI ABI intentionally permits a certified installed wheel or
backported provider to serve another source commit.  Compatibility is the
exported `LORRAX_FFI_ABI_VERSION`, live feature/target probes, and the build
acceptance contract.  A changed ABI is always refused.  An unsealed build-tree
artifact remains a developer migration path and prints `LEGACY-UNSEALED` with
its actual hash; it is not production attestation.

---

## 3. What each machine provides

| Service | Perlmutter (GPU leg + host leg) | Frontera |
|---|---|---|
| FFT (CPU) | `cray-fftw/3.3.10.11` | MKL's native FFTW3 export *(verified 2026-08-06, below)* |
| FFT (GPU) | **nvidia-mathdx** (cuFFTDx, NVRTC) for every k-axis transform and convolution; the in-`shard_map` spatial 3-D FFTs are XLA's (cuFFT in jaxlib) | n/a on the CPU leg |
| GEMM | Cray LibSci CBLAS | MKL CBLAS |
| Dense solvers | SLATE (GPU + host), cuSOLVERMp | ScaLAPACK (MKL), SLATE |
| Parallel HDF5 | `cray-hdf5-parallel` + Cray MPICH | HDF5 + Intel MPI |
| Container | Shifter (`run_shifter.sh`) | apptainer |

### 3a. The dependency matrix — one row per routine we call out for

**This table is the register's answer to "how LORRAX reaches a vendor
library", at routine granularity.** The service table above says which
*vendors* are on each machine; this one says, for each numerical or I/O
routine LORRAX does not implement itself, who serves it, what else could,
**how you would know it built right**, and whether that check is passing.

The last two columns are the point. A row whose "how you know" is *(none)*
is a routine we are trusting without evidence.

| Routine | Perlmutter | Frontera | Reachable alternatives | How you know it built right | Passing? |
|---|---|---|---|---|---|
| **3-D FFT** (in-`shard_map`) | XLA:GPU `fft` → **cuFFT** in jaxlib | XLA:CPU `fft` → **DUCC/Eigen** in XLA | none — `fft_helpers.local_fftn3`/`local_ifftn3` are bare `jnp.fft` aliases with **no FFI route**; `LORRAX_FFT_FFI` structurally cannot reach them | **(none)** | — |
| **k-axis convolutions + k-minor transforms** (ζ fit, Σ, COHSEX, BSE) | CUDA: **nvidia-mathdx** cuFFTDx thread FFTs in one fused pass, NVRTC-built per k-grid from the wheel's headers, disk-cached. Host: the FFTW3-ABI plan handlers below | host plan handlers | none on CUDA by ruling (decisions.md 2026-09-24); a missing wheel refuses at startup (`GATE mathdx-headers`) | `tests/multi_device/kconv_router_p4.py` (every mode vs `np.fft` or dense sums, red twins) | **PASS**, Perlmutter P4, 2026-09-24 |
| **flat-k FFT** (batched 3-D) | CUDA leg: **nvidia-mathdx** k-leading transform (router mode 3). Host leg: FFTW3 ABI by `dlopen` → **cray-fftw** bare-metal; **nothing in-container** | **MKL**'s FFTW3 export, bound at `resolve_sym` stage 1 (MKL is already loaded via the ScaLAPACK link line, so the ladder never runs) | the whole `fftw3_candidates()` ladder: `$LORRAX_FFTW3_SO` → build-time `LORRAX_FFTW3_SO_HINT` → `libfftw3.so.3` → `libfftw3.so.mpi31.3` → `libmkl_rt.so` → `libfftw3.so`. On Frontera `libfftw3.so.3` **is** reachable (`/usr/lib64`, FFTW 3.3.2) and would win over `libmkl_rt.so` if MKL were not already resident | **GATE 5b** — zero `fftw` in `DT_NEEDED` (`build_ffi_host.sh`). Covers *load* time only | **PASS**, Perlmutter, measured 2026-08-06 |
| ↳ *which engine actually answered* | — | — | — | **GATE 8** (`gate_one_fftw.sh`) — **NOT ON THIS BRANCH**, see §3b | **no check** |
| **Host band-block contraction** | **Cray LibSci** CBLAS | **MKL** CBLAS | LibSci exports no `cblas_?gemm_batch`, so the run-time `dlsym` picks the plain-`cblas_?gemm` loop; MKL has the batched entry. Also netlib/AOCL/OpenBLAS/BLIS/ATLAS are accepted as CBLAS providers | **GATE 2** — one LibSci flavour, no sequential/threaded mix. Which *entry* was chosen is **announced at run time, not gated** | **PASS**, Perlmutter (`seq=0 mp=2`), measured 2026-08-06 |
| **Planned axis GEMM** | Full range: XLA:GPU `dot` → **cuBLAS**. Active range: local classic-**cuBLAS** FFI over pointer views, with 4 MiB XLA-owned workspace | Full range: XLA:CPU `dot`; active range: bounded JAX dot panels | CUDA active plans require `CublasLocalActiveRangeGemmFfi`; CPU needs no provider. Both keep K local and shard only output centroid axes | `services/distrib_la/tests/test_local_active_gemm_range.py` on a real P4 CUDA mesh and emulated P4 CPU mesh | branch evidence in `docs/dev/active_gemm_ranges.md`; not on main |
| **Distributed face GEMM** | GPU leg: **cuBLASMp** planned full-range and active-range N,N GEMM | No planned PBLAS/SLATE GEMM handler; CPU face plans refuse by name | The axis carrier above keeps the contraction axis local. `batch_reshard` is available through the one-shot `matmul` surface only when complete matrices fit each device; the planned face surface never selects it silently | Full planned GEMM: `test_distrib_la_multiproc.py` P4 provider cell. Active ranges: `services/distrib_la/tests/test_active_gemm_range.py`, production integration gate `tests/multi_device/active_band_sigma_gate.py` | **PASS on branch `perf/jittable-davidson-2026-09-13`**, P4 jobs 58278132.11–13 and P16 job 58278132.17; not on main |
| **`eigh`** | default **native** `jnp.linalg.eigh` → cuSOLVER via jaxlib | default **native** → LAPACK via jaxlib | `eigh_backend` deck key: `scalapack` \| `cusolvermp` \| `slate`. SLATE `heev` on a **host** mesh is refused outright (bug L-2, deterministic SIGSEGV) | the 11-symbol ScaLAPACK/BLACS **pre-flight** in `build_ffi_host.sh`; then `pytest -m distrib_la` (`services/distrib_la/tests/test_distrib_la_contract.py`) at run time | pre-flight **PASS**, Perlmutter. Contract tests: 2 skips are the SLATE `heev` pair |
| **`eigh`, distributed** | **cuSOLVERMp** `syevd` (GPU); **ScaLAPACK** `pzheevd`/`pdsyevd` from Cray LibSci (CPU) | **ScaLAPACK** from MKL (`libmkl_scalapack_lp64` + `libmkl_blacs_intelmpi_lp64`) | SLATE `heev` (CUDA only) | as above — **no gate asserts which vendor answered**; `CMakeLists.txt` states nothing at run time can observe it | see above |
| **Cholesky** (`potrf`) | **cuSOLVERMp** batched `potrf`/`potrs`; **SLATE** `potrf`/`trsm` | **SLATE** host `potrf` against MKL (opt-in; unset stage ⇒ phdf5-only lib) | **there is no ScaLAPACK `potrf` handler anywhere in the tree** | **(none at build time)** — `test_distrib_la_contract.py` only | contract tests only |
| **LU** (`getrf`/`getrs`) | **cuSOLVERMp** batched `solve_lu` (GPU); **ScaLAPACK** `pXgetrf`/`pXgetrs` from LibSci (CPU) | **ScaLAPACK** from MKL | fused `solve_lu` **and** the split `getrf`+`getrs` pair; resolve refuses if any of the three targets is missing | **(none at build time)** — contract tests only. `LORRAX_LU_NO_PIVOT` can disable pivoting at run time with no gate | contract tests only |
| **Distributed transport** | **NCCL** (cuSOLVERMp ≥0.7.2) or **CAL** (≤0.6.x); **NVSHMEM** transitively via cuBLASMp; `MPI_COMM_WORLD` dup for SLATE/ScaLAPACK | Intel MPI; NCCL for the pip cuSOLVERMp on the rtx leg | the stage choice *is* the transport choice — one string, `LORRAX_NVHPC_SUBPATH`; build refuses if unstated | build-time refusal on an unstated stage (`build.sh`) | — |
| **Parallel HDF5** | **`cray-hdf5-parallel/1.14.3.7`** (`libhdf5_parallel_gnu.so.310`) over **Cray MPICH** `libmpi_gnu_123.so.12` | **phdf5 1.14.6** (`libhdf5.so.310`) over **Intel MPI 2020.4** | **none — the alternatives were deleted 2026-08-06.** There is one transport; a deployment that cannot serve it refuses at open naming the probe that declined. `bse_loading`'s serial-h5py tile readers are a loud, memory-correct fallback at ANY process count (~17x slower, CLAIMS 76 vs 69) and are deliberately not a tier | **GATE 1** (`gate_one_mpi.sh`, one cray-mpich object) and **GATE 7** (`gate_one_hdf5.sh`, one HDF5 SOVERSION + the stage provides it) | **both PASS**, Perlmutter, measured 2026-08-06 |
| **OpenMP runtime** | `libgomp.so.1` | `libiomp5` (Intel) | `libgomp` \| `libiomp5` \| `libomp` | **GATE 6** — the OpenMP runtime really is OpenMP | **PASS**, Perlmutter — but see §3b, it passes on an empty set too |

**Scope of every "PASS" above:** Perlmutter, login node, bare metal, against
`lorrax_hdf5/src/ffi/cpp/build_host/liblorrax_ffi_host.so` (2026-08-06) —
the newest host artifact, built from `fix/host-ffi-hdf5-closure-2026-08-06`,
which is an ancestor of this branch. **No host `.so` has been built from
`integration/2026-08-06` itself**, and **no artifact exists on Frontera at
all** (`find $WORK $SCRATCH -name 'liblorrax_ffi*.so'` returns only
`lorrax_ffi_wtA` / `lorrax_ffi_unified` build dirs from earlier campaigns,
none from this branch). Every Frontera cell above is a claim about what the
build *would* select, verified at the library level — `nm -D` on
`libmkl_rt.so` exports all three FFTW3 advanced-ABI entry points — not a
claim about a built artifact.

### 3b. The routines with no check, and the gates that cannot fail

Four gaps, worst first.

1. **Nothing verifies which FFT engine actually answered.** GATE 5b proves
   nothing *binds* at load time; after that the engine arrives by `dlopen`
   and no static tool can see it. The gate written for this — **GATE 8**,
   `src/ffi/cpp/gate_one_fftw.sh`, which drives one real flat-k FFT and
   reads `/proc/self/maps` — **exists, is certified, and is not on this
   branch.** It is four commits (707 insertions) on
   `fix/host-ffi-fftw-container-stage-2026-08-06`, unmerged:

   ```
   c973968 docs(env_vars): register the five deployment variables the FFTW3 stage adds
   85f346a shifter: mount the FFTW3 stage at /lorrax_fftw, beside phdf5/slate/nvhpc
   7e48d66 ffi build: GATE 8 -- one FFTW3 engine, MAPPED, and it is the staged one
   a3fafdc ffi stage: the container ships no FFTW3, so stage the one the ladder needs
   ```

   The hazard is concrete: the Shifter image ships **`libcufftw.so.11`**,
   which exports `fftw_plan_many_dft` / `fftw_execute_dft` /
   `fftw_destroy_plan` — all three names the ladder binds. Point
   `LORRAX_FFTW3_SO` at it and every FFT cell goes green while the **host**
   handler transforms on the GPU. Merging that branch closes both this and
   the in-container "no FFTW3 engine in this process" failure below.

2. **GATE 5a reads 0 by construction, and still prints in the PASS banner.**
   `nm -D --undefined-only … | grep -c fftw_` counts *undefined symbol
   references*, which `dlsym`ing every entry point drives to zero whatever
   else is true. Measured on both artifacts, same day:

   | | GATE 5a (`nm -D`) | GATE 5b (`DT_NEEDED`) |
   |---|---|---|
   | post-fix `.so` (`lorrax_hdf5`) | 0 → pass | 0 → pass |
   | **pre-fix `.so` (`lorrax_P`, the broken one)** | **0 → pass** | **3 → fail** |

   Same library, same day: the check that was used for certification passes
   on the build that could not load. 5b is the load-bearing half.

3. **Two gates announce PASS having scanned nothing.**
   * **GATE 6** — with zero `lib*omp*.so` entries in `DT_NEEDED`,
     `omp_needed` is empty, `bad_omp` strips to empty, and the gate prints
     `GATE 6 … PASSED`. Verified by running the gate's own expression on an
     empty input.
   * **`gate_one_hdf5.sh`** guards `ldd` (`GATE FAILED (7d): ldd is not
     available`) but never guards `readelf`. Handed a non-ELF file it takes
     the `GATE 7 N/A: none of the 1 artifact(s) link HDF5 at all` branch and
     exits **0**. Verified directly. This is the failure mode
     `gate_one_mpi.sh:30-36` was rewritten to remove — *"A GATE THAT CANNOT
     RUN IS NOT A GATE THAT PASSED"* — left in place one tool over.

4. **Whole routine classes have no build-time check at all.** Cholesky and
   LU are asserted only by `services/distrib_la/tests/test_distrib_la_contract.py`
   at run time;
   nothing at build time says which vendor supplies them, and
   `CMakeLists.txt` notes that nothing at run time can observe it either.
   The 3-D FFT row has no check of any kind, on either machine.
   `LORRAX_LU_NO_PIVOT` disables pivoting from the environment with no gate
   and, until 2026-08-06, no registry row.

### 3c. The shape of this surface: an arch.mk expressed as environment

Census of `integration/2026-08-06`: **355 distinct `LORRAX_*` names appear
in the tree; 234 are actually read.** Of those 234, **120 are build-time
shell variables** — read only by `config/**/*.sh`, `src/ffi/cpp/**/build*.sh`
and CMake, never by the running Python. 111 are run-time, 3 are both, and
121 more names survive only in prose and post-mortem comments.

That majority is the finding. **120 build-time knobs is an `arch.mk`
expressed as environment.** BerkeleyGW answers the same questions this
layer answers — which FFT, which BLAS/LAPACK, which ScaLAPACK, which HDF5,
CPU or GPU — in a single ~70-line file per machine
(`config/<machine>.<compiler>.<target>.<site>.mk`, symlinked to `arch.mk`),
where the answers are **compile-time cpp macros** in one variable:

```make
MATHFLAG = -DUSESCALAPACK -DUNPACKED -DUSEFFTW3 -DHDF5   # Frontera
MATHFLAG = -DUSESCALAPACK -DUNPACKED -DUSEFFTW3 -DHDF5 -DOMP_TARGET -DOPENACC
FFTWLIB  = $(FFTW_DIR)/libfftw3.so ...                   # a PATH, not a soname
```

Its entire run-time environment is one commented `module load` line. Note
what that buys on the exact hazard in §3b: BerkeleyGW's Perlmutter build
links `libfftw3.so` **and** `-lcufft` into one binary and still cannot
suffer the `libcufftw.so.11` substitution, because it names a **file path**
at link time rather than resolving a **soname** at run time. There is no
moment at which the question "which engine answered?" is open, so there is
nothing for a GATE 8 to check.

**A build-time fact should be a build-time fact.** The knobs worth moving
are the ones that defer a decision the build already made — `LORRAX_FFTW3_SO`
is the clearest (the build already recorded its engine as the compile-time
`LORRAX_FFTW3_SO_HINT`; the env var exists to override it at run time).
This is a direction, not a scheduled refactor: a census of read sites
cannot tell you which knobs a deployment actually sets, and every knob
tested during this audit turned out **reachable** — see the deletion
finding below.

**No `LORRAX_*` name was found safely deletable.** Every candidate a
read-site census flagged as dead proved reachable through indirection the
census could not see: `LORRAX_FFT_FFI_{CHUNK,LOG,THREADS}` are read through
a C++ helper taking a name plus a deprecated-alias list;
`LORRAX_KIN_ION_LOOKAHEAD` is bound to `collectives.SWEEP_LOOKAHEAD_ENV`
and read through that constant; `LORRAX_RUN_DIR` is read in a `.sbatch`
template. Treat "nothing reads it" as a hypothesis requiring a hand grep,
not a result.

**The Frontera leg shares no numbered gate with Perlmutter.**
`config/frontera/build_ffi_host.sh` calls neither `gate_one_mpi.sh` nor
`gate_one_hdf5.sh`; it has its own CUDA-free grep, an exported-handler list,
and a `readelf -d | grep -E 'scalapack|blacs|libsci'` non-emptiness test.
`config/frontera/build_ffi.sh` asserts nothing beyond the file existing.
So the machine with no built artifact is also the machine with the weakest
gates — the two facts compound.

**One FFT source serves both.** The CPU flat-k translation unit was
source-locked to MKL's DFTI descriptor API until 2026-08-05; it is not any
more. `src/ffi/cpp/mklfft/fft_flat_k_ffi.cc` now contains **zero**
`DftiCreateDescriptor` calls and four `fftw_plan_many_dft` calls. Entry
points are resolved at *run* time by `dlsym` (`RTLD_DEFAULT` → `RTLD_NEXT`),
so the same object links MKL's native FFTW3 export on Frontera and
`cray-fftw` on Perlmutter. No environment variable names the engine — **the
engine is named by what the `.so` links.**

That design produces a signature worth recognising, because it looks like a
defect and is not: `nm -D --undefined-only | grep -c fftw_` → **0** while
`libfftw3.so.mpi31.3` sits in `DT_NEEDED`. The `DT_NEEDED` entry exists so
the library is *loaded* for `dlsym` to resolve against; nothing binds at
link time. That is the `-Wl,--no-as-needed` idiom, not a dangling
dependency.

> **That count is not a gate, and it was used as one.** The host-FFI leg was
> certified partly on `nm -D --undefined-only | grep -c fftw_` → 0. The check
> is **necessary but not sufficient**: once entry points are resolved by
> `dlsym`, the count is driven to 0 *by construction* — it would read 0 for a
> build with no FFTW anywhere near it. It never inspects `DT_NEEDED`, which
> is the part that actually decides whether the library loads, and the next
> paragraph is what happens when only the count is checked. A check that
> cannot fail is not evidence. The sufficient form is `readelf -d` for the
> `DT_NEEDED` entries plus `ldd` **inside the container on a compute node**
> (§7c on why the login node cannot answer this).

**`DT_NEEDED` is load-bearing at load time**, and until 2026-08-06 that was
a live failure. Measured that day, in-container on compute node nid001644:
the host `.so` carried the right directory in its **RPATH**
(`/opt/cray/pe/fftw/3.3.10.11/x86_milan/lib`), so on the bare host it
resolved -- but inside the Shifter image `/opt/cray/pe` **does not exist at
all**, so all three FFTW entries reported `not found` and the entire
`liblorrax_ffi_host.so` failed to load. Tier-1 on the CUDA leg was
consequently 33 passed / **19 skipped**, every skip reading
`liblorrax_ffi_host.so unavailable: libfftw3.so.mpi31.3: cannot open
shared object file`.

No `LD_LIBRARY_PATH` value repaired that -- the files are not in the
container's mount namespace. **The containerized host leg had never been
green**; the "35/35 on both vendors" certification was a bare-host run.
A skip is not a pass.

> **Fixed, and IN this tree.** `411e257` ("the run-time-resolved FFT engine
> stops being a load-time dependency", from
> `fix/host-ffi-fftw-dt-needed-2026-08-06`) removes the link-time dependency
> so the engine is `dlopen`'d rather than `DT_NEEDED`, and adds the
> sufficient invariant -- **zero `fftw` in `DT_NEEDED`** -- as GATE 5 in
> `config/perlmutter/build_ffi_host.sh`.
>
> **Verified 2026-08-06 on `integration/2026-08-06`.** `411e257` is an
> ancestor here (it arrives via `fix/host-ffi-hdf5-closure-2026-08-06`).
> `readelf -d` on the rebuilt host `.so` shows **0** `fftw` `NEEDED` entries,
> against **3** on the pre-fix library still staged in `lorrax_P`. In-container
> Tier-1 host then reads **49 passed / 2 skipped / 1 failed** against the
> 33/19/0 above; the 2 skips are the known SLATE `heev` L-2 pair and the 1
> failure is `test_compute_wfns_fi_scalapack_matches_native_cpu`, the FFT cell
> honestly reporting `mklfft: no FFTW3 engine in this process` because the
> Shifter image ships no FFTW3. The library itself loads: ScaLAPACK, SLATE and
> GEMM all pass.
>
> **Read the launch geometry before comparing numbers.** Those counts require
> **exactly one visible GPU**. With four GPUs visible (`lx run -G 4`), eight
> SLATE/ScaLAPACK cells fail with `blas::get_device_count()=4 but JAX
> one-process-per-GPU model requires exactly 1`, giving 41/2/9 -- a launch
> artifact, not a regression. Measured both ways on this branch and on
> `fix/host-ffi-hdf5-closure-2026-08-06`: identical in both trees.
>
> The closure hole that used to survive this fix is also closed:
> `libhdf5_parallel_gnu.so.310` needs the **1.14.3.7** phdf5 stage
> (`bbfa026`), not the 1.12 stage. `config/perlmutter/site_config.sh` now
> defaults `LORRAX_FFI_PHDF5_DIR_DEFAULT` to
> `$HOME/software/lorrax_phdf5_cray_1.14.3.7/stage`. **An installed modulefile
> generated before 2026-08-06 still mounts the 1.12 stage**, and against that
> stage the repaired `.so` fails to load with
> `libhdf5_parallel_gnu.so.310: cannot open shared object file`. Re-run the
> installer, or pass `LORRAX_FFI_PHDF5_DIR` explicitly.
>
> The earlier repair suggestion here -- bind-mount `/opt/cray/pe` -- is
> withdrawn. It treats a load-time dependency that should not exist as a
> mount problem, and `411e257` is the better shape.

---

## 4. Picking an nvhpc stage picks a communication path

**This is the section that has caused real skew. Read it before touching
the CUDA leg.**

The staged NVIDIA HPC SDK trees under `/lorrax_nvhpc` are *not* the same
library at different version numbers. They differ in how cuSOLVERMp
communicates:

| Stage | cuSOLVERMp | Ships `cal.h` / `libcal`? | Comm path | Build flag |
|---|---|---|---|---|
| `25.5_cuda12.9` | 0.6.0 | **yes** | CAL | `-DLORRAX_FFI_HAVE_CAL=ON` (the CMake default, `CMakeLists.txt:51`) |
| `0.7.2_cuda12.9` | 0.7.2 | **no** | NCCL-native | `-DLORRAX_FFI_HAVE_CAL=OFF` — required |

So a default here does not merely guess a version, it **silently picks a
communication path**. That is why `build.sh` refuses rather than choosing:
`src/ffi/cpp/build.sh:54-91` exits 2 when neither `LORRAX_NVHPC_ROOT` nor
`LORRAX_NVHPC_SUBPATH` is set, and enumerates the stages actually present,
probing each for `cal.h` and printing which flag it needs
(`build.sh:80-85`).

Three further facts, each of which has bitten:

1. **Every stage exports the same SONAME**, `libcusolverMp.so.0`. Building
   against one and running against another links cleanly and warns about
   nothing.
2. **`25.5_cuda12.9` (0.6.0) is racy on any mesh with `Px>1` *and*
   `Py>1`.** MEASURED 2026-08-06: the failure signature is
   **nondeterminism, not a stable wrong answer** — at 2×2 it trips a
   rerun-bit-determinism assert before the residual is ever compared,
   consistent with `config/perlmutter/site_config.sh` crediting 0.7.2 with
   "the race fix". `0.7.2_cuda12.9` carries both the CAL→NCCL ABI fix and
   the race fix, and is what `site_config.sh` selects; `run_shifter.sh:171`
   defaults `LORRAX_NVHPC_SUBPATH` to
   `0.7.2_cuda12.9/math_libs/12.9/lib64`.
   *Note for whoever edits the source:* the comments at
   `run_shifter.sh:165-166` and `build.sh:27-29,65-66` describe this as
   "returns WRONG getrf/getrs answers", which overstates a race as a
   deterministic result. A rerun that agrees proves nothing here.
3. **Both stages are on `LD_LIBRARY_PATH` at runtime.** `run_shifter.sh:202`
   places the *selected* stage first and `25.5_cuda12.9` after it, on
   purpose: only that tree ships `libcal.so.0`, which a CAL-built `.so`
   carries in `DT_NEEDED`. It means the ordering, not the mount, is what
   decides which `libcusolverMp` you get.
   *Still true at `8789131`, re-read 2026-08-06.* Commit `b2df35f`
   ("the 25.5 libcal fallback is vestigial after the 0.7.2 rebuild") is
   **comment-only** — it adds sixteen lines above an unchanged `LDLIB=`
   and says so: "The entry is left in place — an older CAL-linked .so still
   needs it … No behaviour change." Do not read that commit subject as a
   removal.

The single source of truth is `LORRAX_NVHPC_SUBPATH`. `run_shifter.sh`
exports both it and a `LORRAX_NVHPC_ROOT` derived from its first component
(`run_shifter.sh:240-241`), so a build launched through `run_shifter.sh`
agrees with the run it is built for **by construction** rather than by two
people remembering the same string. Launch builds that way.

**Verified end to end on the machine, 2026-08-06** (Perlmutter, compute
node, in-container build):

* `NVHPC_ROOT` resolved to `/lorrax_nvhpc/0.7.2_cuda12.9` from the single
  `LORRAX_NVHPC_SUBPATH` string.
* The rebuilt `.so` links cuSOLVERMp **0.7.2 with no CAL**: `libcal` absent
  from `DT_NEEDED`, zero hits in the build log, `nm -D | grep cal_` → 0
  symbols. The previous `.so` has `U cal_comm_create`. The runtime banner
  reads `library 0.7.2, comm path: NCCL`.

So the stage genuinely selects the comm path — this is measured, not
inferred from the CMake option.

> **Open skew, not yet resolved in code.** The CMake default is
> `LORRAX_FFI_HAVE_CAL=ON` (`CMakeLists.txt:51`) while the runtime default
> stage is `0.7.2_cuda12.9`, which ships no `libcal` and needs `OFF`. The
> two defaults disagree. `build.sh` refusing an unstated stage is what
> currently prevents the mismatch from being silent; nothing else does.

---

## 5. Parallel HDF5 — what the FFI side of it is

**[`slab_io.md`](slab_io.md) owns this subsystem**: the tile contract and
what a call site may assume of it, the striping campaign, the launcher
requirements, the multi-node certification, the one-owner-per-file rule,
and the measured failure signatures. Only the FFI-side facts belong here.

**There is ONE transport, and no router.** The three tiers (`PHDF5_FFI`,
`PHDF5_HOST`, `H5PY_ALLGATHER`), the `SlabIOBackend` enum, the `slab_io`
deck key, the `use_ffi_io` boolean and the `auto` router were **deleted
2026-08-06**, along with the seven separate refusals that had been guarding
the allgather tier. `file_io/slab_io.py` today takes a path, a mode and a
mesh, and a deployment that cannot serve the tile path refuses at open
naming the probe that declined. Anything on any page that describes
choosing between tiers, or a gap in a refusal that guards one, describes a
tree that no longer exists —
[history](slab_io.md#tiers-history) records why, because the shape of the
mistake recurs.

**The C++ handler is one source serving both legs.** The same `phdf5/`
sources compile into `liblorrax_ffi.so` and into the CUDA-free
`liblorrax_ffi_host.so`, where the D2H staging into a pinned buffer degrades
to an in-place read of the XLA host buffer. That degradation is why the
control-operand stream race ([`slab_io.md`](slab_io.md#stream-race)) is a
CUDA-leg-only defect: on the host leg `copy_index_to_host` is a `memcpy`
and there is no stream to race.

**Since `fix/ffi-odr-2026-08-08` the two legs' C entry points are NOT
interchangeable.** The host leg's carry a `_host` suffix
(`cpp/common/c_abi.h`) and each leg's internal definitions are localised by
`exports_{cuda,host}.map`, precisely so that one `PhdfCtx` type name with
two struct layouts can no longer alias across the two `.so`s under
`RTLD_GLOBAL`. A library built before that fix still exports the plain
names and still collides —
[`slab_io.md#odr-host-so`](slab_io.md#odr-host-so) has the current
measurement and the acceptance test, and `tests/KNOWN_FAILURES.md` L1 owns
the defect.

**One boolean grammar spans every reader of these knobs**, so
`LORRAX_PHDF5_COLLECTIVE_WRITES=0` means "independent" wherever it is
read — §6.

---

## 6. phdf5 defaults — read them here, then read the file

**The struct initialisers in `ctx.h:155-160` are not the effective
defaults.** Every field is reassigned from the environment at `open_file`
time in `context.cc:352-373`. Two of the six differ between the two places.
Read `context.cc`, not the header — quoting a declaration and stopping is
exactly how the stale `use_collective_write=false` claim survived for ten
days.

| Field | `ctx.h` decl | **Effective** | Env override | Notes |
|---|---|---|---|---|
| `use_collective_read` | `true` | `true` | `LORRAX_PHDF5_INDEPENDENT=1` → independent **reads** | `context.cc:352,355` |
| `use_collective_write` | `true` | **`true`** | `LORRAX_PHDF5_COLLECTIVE_WRITES=0` | flipped `false`→`true` on **2026-07-27**, commit `d40e7fd` |
| `coll_metadata` | `false` | `false` | `LORRAX_PHDF5_COLL_META=1` | non-collective metadata lets `H5Dcreate`/extend bypass the collective driver |
| `dedup_replicas` | `true` | `true` | `LORRAX_PHDF5_DEDUP_REPLICAS=0` | drops all-but-one writer of a replica group's identical hyperslab |
| `align_threshold` | 1 MiB | **4 MiB** | `LORRAX_PHDF5_ALIGN_MB` (default `4`) | `context.cc:367,372` |
| `align_length` | 1 MiB | **4 MiB** | same knob — set together | `context.cc:373` |

Alignment is deliberately *not* tied to the striping unit, and is measured
non-load-bearing on this filesystem: at 16 × 1 MiB striping, `ALIGN_MB` of
4 / 1 / 0 gave 0.830 / 0.809 / 0.813 GiB/s at 1 node and 2.975 / 2.883 /
2.915 at 4 nodes — all inside ±1.5 % repeat noise (job 56389339). It stays
at 4 rather than becoming a knob that must be kept in sync with a value it
does not depend on.

**Boolean grammar.** All the flags parse through `env_flag`
(`context.cc`), which mirrors Python's `file_io/_slab_io_ffi._env_flag`
exactly (it mirrored `_slab_io_mpi_host._env_flag` until that module was
deleted with the host tier on 2026-08-06): unset or exactly-empty →
the default; otherwise trimmed, lowercased, and **true only for
`1` / `true` / `yes` / `on`**. Everything else is false — including `off`,
`no`, and any typo. There is no "unrecognised value" diagnostic, so
`LORRAX_PHDF5_COLLECTIVE_WRITES=ture` silently disables collective writes.
One grammar, every writer.

Two of these are correctness, not tuning:

* **`dedup_replicas`** is *required* under collective writes. Overlapping
  hyperslab selections are undefined in HDF5. Under independent writes the
  same flag is pure waste-removal.
* **`use_collective_write`** decides whether a PMI-mismatched launch fails
  loudly or corrupts silently — see §7.

History of the write default, since prose elsewhere in the tree still
carries the old value: introduced `3a7f2e5` (2026-04-17); set `false` in
`d37c47a` (2026-04-20, "independent writes by default; Cray MPICH now
works"); set `true` in `d40e7fd` (2026-07-27).

> **Known stale comments in source (reported, not edited — those files are
> owned elsewhere).** `src/ffi/cpp/stage/phdf5_stage_cray.sh:13-19` and
> `src/ffi/cpp/run_shifter.sh:100-104` both still assert that the phdf5
> default is *independent* writes with non-collective metadata. That has
> been false since 2026-07-27. The first of the two is actively harmful —
> see §7.

---

## 7. Failure modes, and how to tell them apart

### 7a. The PMI-flavour mismatch — the one that gives wrong answers

**Measured**, job 56389339, 4 nodes / 16 ranks, launched `srun --mpi=pmi2`
(the wrong PMI flavour for Cray MPICH; the right one is `cray_shasta`):

```
MPI_Comm_size(MPI_COMM_WORLD) == 1   on every rank
jax.process_count()           == 16
→ 8 hostile geometries written and read back BIT-EXACT, rc=0,
  file 16-striped and fully populated, no warning anywhere.
```

Nothing in the stack noticed, and the reason is worth internalising:
`ffi.io.open_file` checks `p*q == jax.process_count()`, and
`shard_index.h::validate_shard_encoding` checks
`prod(mesh_shape) == ctx->world_size` — but `ctx->world_size` *is*
`jax.process_count()`, passed down from Python. **Both checks compare JAX to
JAX and agree.** The MPI communicator `H5Dwrite` actually collects on was
never consulted.

It "worked" only because the hyperslabs happened to be disjoint, so there
was no collective handshake left to fail. Change the geometry so two ranks
touch one HDF5 chunk, or let one rank's metadata update race another's, and
it is silent corruption with rc=0.

The guard is in `file_io/_slab_io_ffi.py`: ask MPI once, at the first
collective open. The verdict is rank-invariant by construction, so it
refuses everywhere or nowhere — the only kind of refusal a collective
tolerates.

### 7b. The ROMIO OOM — and the documented remedy that makes it worse

With collective writes ON (the current default) that same mismatched launch
does **not** survive. But it does not diagnose either. It dies as:

```
Out of memory in .../ad_cray/ad_cray_write_coll.c, line 669
… MPI_Abort … "HDF5: infinite loop closing library"
```

That is the *same* line `stage/phdf5_stage_cray.sh:13-19` documents as a
known Cray-MPICH `≥1 GB/rank` collective-buffer OOM — whose documented
remedy is `LORRAX_PHDF5_COLLECTIVE_WRITES=0` / `LORRAX_PHDF5_INDEPENDENT=1`.

**That documented remedy is wrong in both halves.**

* `LORRAX_PHDF5_COLLECTIVE_WRITES=0` does exactly what it says — and
  converts the loud crash into the silent-wrong-answer regime of §7a.
* `LORRAX_PHDF5_INDEPENDENT=1` forces independent **reads**
  (`context.cc:352,355`). It does nothing to the write path at all, so
  against a write-side OOM it is simply inert.

The misdiagnosis is not hypothetical — the tree points straight at it, and
the half of the advice that *does* something is the half that hides the
bug.

How to tell the two apart before reaching for the knob:

| | genuine Cray collective-buffer OOM | PMI-flavour mismatch |
|---|---|---|
| `MPI_Comm_size(MPI_COMM_WORLD)` | == `jax.process_count()` | **1**, on every rank |
| per-rank aggregate | ≳ 1 GB | any size |
| independent writes | genuinely fixes it | **hides it** |

Check the world size **first**. `LORRAX_PHDF5_REQUIRE_MPI_WORLD=1` makes an
unprobeable world a refusal rather than a warning;
`LORRAX_PHDF5_SKIP_MPI_WORLD_CHECK` disables the guard entirely and should
be treated as a debugging-only escape hatch, never a remedy.

Not reproduced at 512 MiB/rank in the 2026-08-05 Perlmutter campaign, and
the collective default was revalidated there (job 56389339): keep it at `1`.

### 7c. SONAME aliases that look like two ABIs and are not

`libmpi_gnu_123.so.12` is **a deliberate symlink, not a second MPI**.
`src/ffi/cpp/stage/phdf5_stage_cray.sh:128-130` creates one shim per
cray-pe compiler-specific SONAME — `libmpi_gnu_{91,110,123}.so.12` — all
pointing at the container's generic MPICH-ABI library,
`/opt/udiImage/modules/mpich/libmpi.so.12` (`SHIM_TARGET`, line 92). The
loader follows the symlink at container startup; every variant is MPICH 4.x
`libmpi.so.12` underneath. **One object, not two.**

A 2026-08-05 report of a two-MPI-ABI defect on the CUDA leg was **retracted
on 2026-08-06** for this reason. The `ldd` behind it was run on a **login
node**, where the closure is incomplete and four dependencies show
`not found`.

> **Method note, because this cost real time.** `ldd` on a login node does
> not describe what a container run loads. Run link-closure checks inside
> the container, on a compute node, through `lx run`. A `not found` in a
> login-node `ldd` is evidence of nothing.

### 7d. Bounds-check asymmetry — the hang with no traceback

Bounds are tested once, on the *logical* slab `offset + valid_shape`, which
is a replicated quantity, so every rank reaches the same verdict. Testing a
rank-local advanced offset splits the ranks into those that refuse and those
that enter the collective, stranding the communicator with no HDF5 error and
no traceback (**measured**: 306 s hang at P=4; silent 420 s timeout on the
read path). No rank may skip a collective because of its own error: record
it, participate in the teardown, then raise. See `decisions.md` 2026-08-04.

---

## 8. Hard invariants

Checked at `886139f` and re-checked at `8789131` on 2026-08-06; not aspirational.

1. **Registered FFI custom-call target names do not change.** The full set
   is in `src/ffi/common/ffi_loader.py` (`_CUDA_TARGET_SYMBOLS`,
   `_HOST_TARGET_SYMBOLS`). Refactors move files; they never edit a target
   string or a C++ handler symbol.
2. **Env knob spellings do not change.** Aliases only.
3. **Built `.so` names and consumed paths stay stable, or their consumers
   are updated in the same commit.** `liblorrax_ffi.so` /
   `liblorrax_ffi_host.so`.
4. **A stage script refuses an unstated environment fact rather than
   guessing it.** `phdf5_stage_cray.sh` refuses an unset `HDF5_DIR`
   (lines 57-70) and an unset `MPICH_DIR` (73-84); `build.sh` refuses an
   unset nvhpc stage (54-91). Each of those was a hardcoded guess until
   2026-08-05/06, and each guess had gone stale: the HDF5 fallback named
   1.12.2.9 while the host build uses `cray-hdf5-parallel/1.14.3.7`. What
   is staged is what every later build *links against*, and a wrong guess
   does not fail at link time — it fails much later, as a wrong answer or a
   hang, with nothing on disk recording the substitution.

Invariant 4 is the generalisation of §4 and §7b: **in this layer, a
substituted default is a wrong answer with a long fuse.**

---

## 9. Deletion candidates and open work

* **cusolvermp** (~2800 LOC): 11 import sites outside `src/ffi` at
  `886139f`. The distributed CPU story is ScaLAPACK; the GPU story is
  SLATE. Deletion removes the `auto|cusolvermp` spelling from the linalg
  backend grammar — an input-deck surface, so it needs the deprecation
  window plus a GPU run proving SLATE covers the eigh/LU tiers cusolvermp
  served.
* **cublasmp** (~1450 LOC): 4 import sites (`bse/vq_interp.py`,
  `bandstructure/htransform.py`, tests). The fused W-solve path has no
  measured replacement, so this leg stays until a GPU gate exists.
* **Shim deletion**: blocked on consumer migration (§1).
* **FFT, remaining items**: `fftw_init_threads` / `plan_with_nthreads` on
  non-MKL engines under the existing `LORRAX_FFT_FFI_THREADS` grammar; the
  `fftwf_` twin table if BSE adoption wants c64; and only then gating the
  shard_map-interior `local_*fftn3` entry so the FFI can back
  `make_sharded_ifftn_3d`. That last flip is a **measurement**, not a move,
  and until it happens that layer stays XLA by ruling.

Parity gate for any engine swap, stated once with its class: value-level,
**relative 1e-12** (the Σ-path class, `flat_k_fft_service.md` §7). Not
bit-exactness — swapping engines changes the arithmetic ordering, where bit
equality is not promised. And not the 1e-16 figures: those are *measured*
unit residuals sitting at the c128 ULP, where a threshold tests nothing.

### k-convolution router and the mathdx family

Every k-axis convolution and every k-axis transform in the physics is requested
through one factory in `ffi/fft.py` (re-exported by `common.fft_helpers`). The
factory picks the backend from the mesh platform and nothing else; no
environment variable or deck key selects a route (ruling:
[`decisions.md`](decisions.md), 2026-09-24).

| platform | backend |
|---|---|
| CUDA | nvidia-mathdx: cuFFTDx thread FFTs inside one fused shared-memory pass per k-row, compiled by NVRTC per k-grid |
| cpu | the FFTW3-ABI host plan handlers, composed with XLA elementwise work |
| other | refusal, `GATE kconv-platform` |

Both legs return the same callable contract, so a consumer never branches on
the backend.

**Why this way.** The pair convolution `U_q = Σ_k conj(A_k)·B_{k+q}` over
`N_k` points costs `O(rows·N_k²)` as a direct sum and `O(rows·N_k log N_k)` as
`U = s·FFT_k[conj(IFFT_k A)·IFFT_k B]`. The CUDA kernel keeps each k-row in
shared memory from the inverse transforms through the product (or spin
contraction) to the forward transform, so one convolution reads each operand
from HBM once and writes the result once. The line FFTs are the library's,
specialised per grid when NVRTC compiles the kernel.

| Layer | What |
|---|---|
| 1 consumer | ζ fit: `isdf.core.c_q_downfold` (pair); `isdf.core.c_q_from_psi_sm`, `_z_q_face_parent`, `parent_projector_kconv` and `isdf.zeta_mubatch.make_route_g_kernel` (parent). Σ: `gw.ppm_tau_kernel.get_sigma_spatial_kernel`, `gw.cohsex_sigma._make_static_convolution`. BSE: `bse_stack_matvec._conv_decode`, `bse_ring_comm._make_ring_rung`, and the W_R transforms in `bse_densify.make_w_densifier`, `bse_lanczos`, `davidson_absorption`, `absorption_haydock`, `bse_nontda`, `exciton_bands`. Flat-k transform: `common.fft_helpers.make_flat_k_fft` and its `make_flat_k_ifftn` / `make_flat_k_fftn` / `make_local_flat_k_fftn` wrappers (`gw.w_isdf`, `gw.qsgw_head`, `gw.cohsex_sigma`, `gw.wavefunction_bundle`, `bandstructure.htransform`, `bandstructure.orbital`) |
| 2 router | `ffi/fft.py`: `make_fused_conv_kpair`, `make_fused_conv_kparent`, `make_kconv_klead` (→ `KConvStored(prep, apply)`), `make_kconv_kminor` / `make_local_kconv_kminor`, `make_kfft_klead` / `make_local_kfft_klead`, `make_kfft_kminor` / `make_local_kfft_kminor`. `common.fft_helpers.get_donated_kfft_kminor` is `make_kfft_kminor` jitted with its input donated, memoised per `(mesh, kgrid, spec, kind, norm)`; the caller drops its own reference after the call |
| 3 gate | `require_kconv`, called by `runtime.initialize_communicator_stack` after the FFT and GEMM gates. CUDA: the wheel's headers and all six `lorrax_mathdx_*` targets. cpu: `lorrax_mklfft_flat_k`. Each factory re-probes its own target; operand shapes and dtypes are checked at trace time |
| 4 target | CUDA, in `liblorrax_ffi.so`: the six `lorrax_mathdx_*` names. cpu, in `liblorrax_ffi_host.so`: `lorrax_mklfft_flat_k`, `lorrax_mklfft_gw_conv` |
| 5 handler | CUDA: `cpp/cufft/kconv_mathdx_cuda_ffi.cc`, one embedded cuFFTDx source compiled per (CUDA context, mode, `nkx`, `nky`, `nkz`, `ns`, precision) into an in-process cache, backed by the disk cubin cache below. cpu: `cpp/mklfft/fft_flat_k_ffi.cc` (`MklFftFlatKHostFfi`, `MklFftGwConvHostFfi`) |

**Doors.** Pick the door whose k position matches the tile you hold. A caller
never transposes to reach another door.

| door | k axis of the operand | CUDA mode | cpu leg |
|---|---|---|---|
| `make_fused_conv_kpair` | 3-D leading `(nkx, nky, nkz, …)` | 0 | two host flat-k inverse transforms, the spin contraction in XLA, one host forward transform |
| `make_fused_conv_kparent` | parent tables | 1 | the typed parent load in XLA, which materialises `(N_k, ns, μ, ν, ns)` per side, then the pair tail |
| `make_kconv_klead` | flat leading `(N_k, …)` | `prep` 3, `apply` 2 | `prep` is the identity; `apply` is `lorrax_mklfft_gw_conv`, which transforms W itself and holds the R-space T tile only in per-thread compact chunks |
| `make_kfft_klead` | flat leading `(N_k, …)` | 3 | `lorrax_mklfft_flat_k` |
| `make_kconv_kminor` | flat trailing `(…, N_k)` | 4 | XLA moves k to the front, then host inverse transform, product, host forward transform, and k moves back |
| `make_kfft_kminor` | 3-D trailing `(…, nkx, nky, nkz)` | 5 | the same transpose around one host transform |

- **Sharding.** The pair, parent and `make_local_*` doors are rank-local
  callables for use inside the caller's `shard_map`; the others wrap their own
  `shard_map`. The k axes are replicated. Specs of the k-leading doors are
  given in the 3-D form, with the three leading axes `None`. For
  `make_kconv_kminor`, `K_R`'s `(d1, d2)` must sit on the same mesh axes as
  X's.
- **Scale.** Every handler takes one total scale `s`, computed in Python from
  `jnp.fft`'s norm conventions (`ffi_fft_scale`, `conv_kpair_scale`). The
  handlers implement no norm of their own. The parent door fixes
  `norm="forward"`, so `s = 1/N_k`.
- **`KConvStored`.** `prep(W)` does everything that depends on W alone, once
  per W. `apply(T, W_prep)` does the rest, once per T. `W_prep` is in the
  backend's own form (R space on CUDA, W unchanged on cpu), so pass it only to
  the `apply` of the same pair.
- **Vertex attributes (modes 0 and 1).** `perm_l`, `perm_r` are permutations
  of `range(ns)` and `phase_l`, `phase_r` are exact monomials in
  `{+1, +i, −1, −i}`; the CUDA leg refuses anything else at factory time.
  `ns ≤ 4`.

The modes of the one kernel source:

| mode | operation | layout | resident banks per k-row |
|---|---|---|---|
| 0 pair | `U = s·FFT_k Σ_ab φ_l[a]φ_r[b]·conj(IFFT_k A[…,a,…,b])·IFFT_k B[…,π_l a,…,π_r b]` | `A`, `B` `(nkx,nky,nkz, ns, col, μ, ns)` → `U` `(nkx,nky,nkz, col, μ)` | 3 |
| 1 parent | mode 0 on the typed parent load ([below](#parent-load-isdf-pair-convolution-mode-1)) | `D_l`, `D_r` `(n_parent, ns, μ, ns, ν)` and ten tables → `U` `(N_k, μ, ν)` | 3 |
| 2 klead conv | `U = s·FFT_k(IFFT_k T · V_R[:, None, :, None, :])`, `V_R` already in R space (mode 3 made it) | `T`, `U` `(N_k, a, m_x, b, m_y)`; `V_R` `(N_k, m_x, m_y)` | 1 |
| 3 klead fft | `Y = s·FFT^±_k X` | `(N_k, rows)` | 1 |
| 4 kminor conv | `U = s·FFT_k(IFFT_k X · K_R[None, :, :, None, None, :])`, `K_R` already in R space (the caller made it with mode 5) | `X` `(d0, d1, d2, d3, d4, N_k)`, `K_R` `(d1, d2, N_k)` → `U` in X's layout (`out_layout=0`) or `(d0, N_k, d3, d1, d4, d2)` (`out_layout=1`) | 1 |
| 5 kminor fft | `Y = s·FFT^±_k X` | `(rows, N_k)` | 1 |

- **Flat k** is C order, with `kz` fastest.
- **Dtype.** Modes 0 and 1 are complex128 only. Modes 2–5 take all-complex128
  or all-complex64 operands (the complex64 image serves the fp32-GMRES BSE
  arm) and never cast. The cpu host handlers are complex128 only, so a
  complex64 operand refuses at trace time on a cpu mesh.
- **In place.** Modes 2, 3 and 5, and mode 4 with `out_layout=0`, alias
  operand 0 to the result (`input_output_aliases={0: 0}`). This is safe
  because each block reads all `N_k` values of its rows before it stores any
  of them.
- **Launch geometry.** One 256-thread block per `rb` rows; a row needs
  `banks·16·(N_k|1)` bytes of shared memory (8 per element for complex64).
  Modes 0 and 1 take `rb ≤ 16` within 100 KiB, modes 2–5 `rb ≤ 64` within
  48 KiB (three blocks per A100 SM); a row larger than the budget gets what
  the device's opt-in maximum holds.
- **Cost.** Each transform is `O(rows·N_k log N_k)` flops. HBM traffic is one
  read of each operand and one write of the result. The CUDA kernels allocate
  no device workspace beyond dynamic shared memory.
- **Host workspace (cpu leg).** `gw_conv` stages `V_R = IFFT_k W` once per
  call in a reused host arena of `N_k·m_x·m_y·16` bytes, invisible to XLA.

**Refusals.**

| refusal | raised at | condition | fix |
|---|---|---|---|
| `GATE kconv-platform` | startup, factory | the mesh platform is neither CUDA nor cpu | run on a CUDA or cpu mesh |
| `GATE mathdx-headers` | startup (`mathdx_root`); kernel build | no importable `nvidia.mathdx` with `include/cufftdx.hpp` | `pip install nvidia-mathdx`; the `cuda12`/`cuda13` extras of `pyproject.toml` pin it (`==25.6.0`) |
| `GATE kconv-target` | startup, factory | the loaded library lacks the target the router selects | rebuild the library and point `LORRAX_FFI_SO` (CUDA) or `LORRAX_FFI_HOST_SO` (cpu) at it |
| `GATE mathdx-kconv-axis` | factory; handler | a k-grid axis outside `[1, 40]` (`KCONV_AXIS_MAX`, the fp64 cuFFTDx thread-FFT limit). The klead, kminor and kfft factories check it on cpu meshes too; the pair and parent factories check it on CUDA only | a smaller k-grid |
| `GATE mathdx-kconv-residency` | first call (kernel build) | one resident row, `banks·16·(N_k|1)` bytes (8 per element for complex64), exceeds the device's opt-in shared memory per block. On an A100 (166 912 B) that is `N_k > 3477` for modes 0/1 and `N_k > 10431` for modes 2–5 in complex128 | a smaller k-grid; the family has no out-of-core arm |
| `LORRAX_FFT_FFI=0` | factory | the cpu leg refuses, and `make_flat_k_fft` refuses on both platforms | unset `LORRAX_FFT_FFI` |

A kernel-build failure (NVRTC compile, missing toolkit headers, module load)
is sticky: the handler caches it per in-process key and returns it on every
later call, naming the stage (`kconv_mathdx (fused cuFFTDx k-convolution):
<stage> failed -- …`).

**Headers and build.** The router passes the wheel's `nvidia/mathdx` directory
(from the `nvidia.mathdx` package spec) to every handler as the string
attribute `mathdx_root`. NVRTC includes `include/` and
`external/cutlass/include` beneath it, plus the CUDA toolkit's `include/` and
`include/cccl`, found beside the loaded libnvrtc. No environment variable names
either path. Building `liblorrax_ffi.so` needs no mathdx: the translation unit
links libnvrtc and resolves the driver API by `dlsym`. CMake compiles it only
when its probe (option `LORRAX_FFI_HAVE_CUFFT`, default on) finds `cufft.h`,
`libcufft`, `nvrtc.h` and `libnvrtc`; otherwise the six targets are absent and
startup refuses with `GATE kconv-target`. The leg still links libcufft, which
nothing calls.

**Disk cubin cache.** The images live in `ffi.fft.cubin_cache_dir()`:
`$SCRATCH/.cache/lorrax/kconv_mathdx`, or `~/.cache/lorrax/kconv_mathdx` where
the site defines no `SCRATCH`. The cache is always on, has no knob, and is not
the XLA compile cache (`ISDF_JAX_CACHE_DIR`). One directory serves every world
size, because an image depends on the device and the wheel, not on P.

- **Key.** FNV-1a over the embedded source; the NVRTC options that decide the
  image (C++ standard, architecture, mode, grid, `ns`, rows per block,
  precision, SM); the wheel's `cufftdx_version.hpp` and
  `commondx_version.hpp`; and the NVRTC version. Include paths are not in the
  key, so two installs of one wheel version share images. Editing the kernel
  source invalidates every image.
- **File.** `kconv_m<mode>_<nkx>x<nky>x<nkz>_ns<ns>[_c64]_sm<XY>_<key>.cubin`,
  with a `LRXKCONV1` header that carries the key and a hash of the payload.
- **Writes and reads.** A write goes to a unique temporary and is `rename`d
  into place, which is atomic on one filesystem, so concurrent ranks each
  publish a whole file. A read re-hashes the payload; a torn or foreign file
  is recompiled and replaced.
- **Cost.** A cold NVRTC build takes 5–7 s per image per process; a disk hit
  takes 5–15 ms (sandbox CLAIMS 2673). Without the cache, a CrI3-class
  run pays about 18 s of NVRTC per process.
- **Receipts.** Under `LORRAX_DEBUG_PRINT=1` the startup `[kconv]` line names
  the backend, the wheel root and the cache directory with its image count
  and size. Every kernel build prints `[kconv_mathdx] disk-cache hit` or
  `NVRTC built …` on rank 0, with the grid, rows per block, shared memory and
  whether the cubin was stored.

**Test-only cpu arm.** `tests/conftest.py` sets `LORRAX_KFFT_CPU_TEST_XLA=1`.
In-process pytest cpu meshes on Perlmutter have no host library, so under it
the cpu leg announces itself and uses `jnp.fft` for its k-axis transforms. It
is never read on CUDA and is never a production route.

**The gate.** `tests/multi_device/kconv_router_p4.py`, run as
`lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/kconv_router_p4.py`,
covers every mode at P4, including an odd grid, 8×8×8 and the complex64
k-minor image. The tolerance is 1e-13 against the cpu composition and 1e-12
against dense sums or `np.fft`; every check has a red twin that must miss by
more than 1e-3.

A new mode is added in four steps:

1. Add a kernel entry to the embedded source under its `LRX_MODE` value.
2. Add a handler and `XLA_FFI_DEFINE_HANDLER_SYMBOL` in the same translation
   unit. Register its target in `ffi_loader._CUDA_TARGET_SYMBOLS`,
   `ffi.fft.KCONV_TARGETS` (which `require_kconv` checks at startup) and
   `ffi.cufft.CUDA_TARGETS`/`CUDA_SYMBOLS`.
3. Add a router factory in `ffi/fft.py` that returns the mathdx call on CUDA
   and the plan-route composition on cpu.
4. Add a case, with a red twin, to `tests/multi_device/kconv_router_p4.py`.

#### Parent-load ISDF pair convolution (mode 1)

Mode 1 is the pair convolution with its operands unfolded from the raw parent
k-points inside the load, so no full-k open-spin array is written to HBM. With
`p = irr[k]`, `o = sym[k]`, `m = left[o, μ]`, `n = right[o, ν]`, and `𝒯_k`
complex conjugation when `trs[k] ≠ 0`, the load builds

```text
P_{k,ab}(μ, ν) = conj( Σ_{c,e} coef[k, a·ns+b, c·ns+e] ·
                       𝒯_k( e^{2πi q_p·L_{o,μ}} · D_{p,c,e}(m, n) · e^{−2πi q_p·R_{o,ν}} ) )
```

from `D_l` with `coef_l` on the left and from `D_r` with `coef_r` on the
right, and the kernel then runs mode 0 on `P^L`, `P^R`.

| positional operand | shape | dtype |
|---|---|---|
| `D_l`, `D_r` | `(n_parent, ns, μ_local, ns, ν_local)` logical | complex128 |
| `irr`, `sym` | `(N_k,)` | int32 |
| `left`, `right` owner-local source maps | `(n_ops, μ_local)`, `(n_ops, ν_local)` | int32 |
| `L`, `R` lattice wraps | `(n_ops, μ_local, 3)`, `(n_ops, ν_local, 3)` | float64 |
| `q` parent fractional k | `(n_parent, 3)` | float64 |
| `trs` antiunitary mask | `(N_k,)` | int32 |
| `coef_l`, `coef_r` open-spin coefficients | `(N_k, ns², ns²)` | complex128 |
| result `U` | `(N_k, μ_local, ν_local)` | complex128 |

- **Tables.** They are built by `isdf.core._parent_conv_tables_local` from the
  typed unfold plan and must be authenticated owner-local plan tables. The
  handler checks their shapes and dtypes, not the device-side map values.
- **Layout.** The static attribute `centroid_major` states the physical layout
  of `D`. It is 1 for the CCT build (`c_q_from_psi_sm`): major-to-minor
  `(parent, ν, spin_r, μ, spin_l)`, requested through the `ffi_call` input
  layout `(0, 4, 3, 2, 1)`, so the GEMM's output feeds the kernel with no
  transpose. It is 0 for the ZCT tails and route G: row-major
  `(parent, spin_l, μ, spin_r, ν)`. Only the load's address arithmetic
  differs.
- **Vertex.** Production folds the post-unfold Lorentz vertex into `coef_r`
  (`isdf.core._parent_conv_vertices`, conjugating the phase because the load
  returns a conjugate). The kernel's `perm`/`phase` attributes therefore stay
  the identity, and every channel of one shape reuses one executable.
