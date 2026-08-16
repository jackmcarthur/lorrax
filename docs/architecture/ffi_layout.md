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

`src/ffi/io.py` (the parallel-HDF5 service), `fft.py` (flat-k FFT),
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

---

## 3. What each machine provides

| Service | Perlmutter (GPU leg + host leg) | Frontera |
|---|---|---|
| FFT (CPU) | `cray-fftw/3.3.10.11` | MKL's native FFTW3 export *(verified 2026-08-06, below)* |
| FFT (GPU) | cuFFT (`cufftPlanMany64`) | n/a on the CPU leg |
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
| **flat-k FFT** (batched 3-D) | CUDA leg: **cuFFT** `cufftPlanMany64` + NVRTC. Host leg: FFTW3 ABI by `dlopen` → **cray-fftw** bare-metal; **nothing in-container** | **MKL**'s FFTW3 export, bound at `resolve_sym` stage 1 (MKL is already loaded via the ScaLAPACK link line, so the ladder never runs) | the whole `fftw3_candidates()` ladder: `$LORRAX_FFTW3_SO` → build-time `LORRAX_FFTW3_SO_HINT` → `libfftw3.so.3` → `libfftw3.so.mpi31.3` → `libmkl_rt.so` → `libfftw3.so`. On Frontera `libfftw3.so.3` **is** reachable (`/usr/lib64`, FFTW 3.3.2) and would win over `libmkl_rt.so` if MKL were not already resident | **GATE 5b** — zero `fftw` in `DT_NEEDED` (`build_ffi_host.sh`). Covers *load* time only | **PASS**, Perlmutter, measured 2026-08-06 |
| ↳ *which engine actually answered* | — | — | — | **GATE 8** (`gate_one_fftw.sh`) — **NOT ON THIS BRANCH**, see §3b | **no check** |
| **GEMM / batched GEMM** | Host leg: **Cray LibSci** CBLAS. GPU: no FFI — XLA:GPU `dot` → **cuBLAS** | **MKL** CBLAS | LibSci exports no `cblas_?gemm_batch`, so the run-time `dlsym` picks the plain-`cblas_?gemm` loop; MKL has the batched entry. Also netlib/AOCL/OpenBLAS/BLIS/ATLAS are accepted as CBLAS providers | **GATE 2** — one LibSci flavour, no sequential/threaded mix. Which *entry* was chosen is **announced at run time, not gated** | **PASS**, Perlmutter (`seq=0 mp=2`), measured 2026-08-06 |
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

The GPU flat-k mirror (`cufftPlanMany64`, advanced layout) was **not**
re-certified in the 2026-08-05 Perlmutter campaign. Treat a CUDA-leg FFT
number as carried forward, not measured.

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
* **CUDA-leg re-certification**: outstanding. See §3.

Parity gate for any engine swap, stated once with its class: value-level,
**relative 1e-12** (the Σ-path class, `flat_k_fft_service.md` §7). Not
bit-exactness — swapping engines changes the arithmetic ordering, where bit
equality is not promised. And not the 1e-16 figures: those are *measured*
unit residuals sitting at the c128 ULP, where a threshold tests nothing.
