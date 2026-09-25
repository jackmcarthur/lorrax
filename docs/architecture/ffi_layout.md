# The FFI layer

How LORRAX reaches vendor libraries: the layers, the two build legs and their
acceptance gates, what each machine provides, which cuSOLVERMp selects which
communication path, which FFT engine the host library binds, the C++ phdf5
defaults, and how to tell the native-layer failure modes apart.

This page owns the *native boundary*. Owner rulings are in
[`decisions.md`](decisions.md), the SlabIO contract in
[`slab_io.md`](slab_io.md), knob spellings and defaults in
[`../dev/env_vars.md`](../dev/env_vars.md). See the
[register](../index.md#register).

---

## 1. The layers

Five, outermost first. Each one can refuse; none silently substitutes.

| # | Layer | Lives in | Job |
|---|---|---|---|
| 1 | **Consumer** | `src/gw/`, `src/file_io/`, `src/bse/`, … | states *logical* intent: shapes, not strides |
| 2 | **Service facade** (Python) | `src/ffi/io.py`, `fft.py`, `gemm.py`; `services/distrib_la/` | owns the call grammar, builds the descriptor, picks the backend |
| 3 | **Gate** | `src/ffi/gate.py` (env dials); `distrib_la.resolve` (deck choices); `ffi.fft.require_kconv` (router) | announce-or-refuse: an explicit request that cannot be honoured refuses, never downgrades |
| 4 | **XLA FFI custom call** | `src/ffi/common/ffi_loader.py`, `distrib_la.loader`, both over `lxkit.native_provider` | locates and attests the `.so`, registers its handler symbols |
| 5 | **C++ handler + vendor library** | `src/ffi/cpp/<vendor>/` | the MPI-IO / BLAS / FFT / solver call |

**A vendor dependency enters only through a facade with run-time resolution
and an announced refusal.** The FFI is required
([`decisions.md`](decisions.md), 2026-08-01): a missing library is a startup
refusal naming the `.so`, never a demotion to a Python path. Portability
fallbacks *inside* a handler (the FFTW3 `dlsym` ladder, batched versus plain
CBLAS) are how the required layer stays buildable everywhere; each announces
which entry it bound.

### Python-side module map

* **Real modules:** `ffi/io.py` (parallel HDF5), `ffi/fft.py` (the host flat-k
  FFT and the [k-convolution router](#k-convolution-router-and-the-mathdx-family)),
  `ffi/gemm.py` (host batched GEMM), `ffi/gate.py`,
  `ffi/common/ffi_loader.py`, and `ffi/cublasmp/` (cuBLASMp GEMM and W-solve,
  reached by the bench drivers). Distributed dense linear algebra is
  `services/distrib_la`, which opens the same two libraries through its own
  `distrib_la.loader`.
* **Target table only:** `ffi/cufft/` lists the six CUDA `lorrax_mathdx_*`
  targets and their C++ symbols. It has no callable.
* **Re-export shims:** `ffi.phdf5` → `ffi.io`, `ffi.mklfft` → `ffi.fft`,
  `ffi.mklblas` → `ffi.gemm`, `ffi.cusolvermp` → `distrib_la._cusolvermp`.
  New code imports the real module. A shim is deleted when
  `grep -rnE "ffi\.<shim>" src/ services/ tests/ | grep -v '^src/ffi/'` is
  empty.

---

## 2. The one C++ tree

`src/ffi/cpp/`: one `CMakeLists.txt`, both platform legs behind an explicit
selector.

```
src/ffi/cpp/
├── CMakeLists.txt           -DLORRAX_FFI_PLATFORM=cuda|host; FATAL_ERROR when unset
├── build.sh  build_host.sh  leg build scripts (container CUDA leg; generic host leg)
├── exports_cuda.map  exports_host.map     version scripts: LORRAX internals are local
├── gate_one_mpi.sh  gate_one_hdf5.sh  gate_one_fftw.sh  gate_one_odr.py
├── run_shifter.sh  in_container.sh  select_gpu.sh    container composition
├── stage/                   vendor stage scripts, seal_bundle.py, stamp_provenance.sh
├── common/                  C ABI, ABI and build-config stamps, contour accumulator
├── phdf5/  slate/                                    both legs
├── mklfft/  mklblas/  scalapack/                     host leg
└── cusolvermp/  cublasmp/  cublas/  cufft/  active_subspace/  symmetry/   CUDA leg
```

Vendor directory names are historical: `mklfft/` holds the FFTW3-ABI handler
and `cufft/` the nvidia-mathdx family. Every filename stays unique across
directories (`slate/ctx.h`, `phdf5/ctx.h`, `cusolvermp/ctx.h`).

| leg | CMake | library | feature options (all default `ON`) |
|---|---|---|---|
| host | `-DLORRAX_FFI_PLATFORM=host` | `liblorrax_ffi_host.so` | `LORRAX_FFI_HAVE_PHDF5`, `LORRAX_HOST_HAVE_SCALAPACK`, `LORRAX_HOST_HAVE_SLATE`, `LORRAX_HOST_HAVE_FFTW3` |
| CUDA | `-DLORRAX_FFI_PLATFORM=cuda` | `liblorrax_ffi.so` | `LORRAX_FFI_HAVE_CAL`, `LORRAX_FFI_HAVE_PHDF5`, `LORRAX_FFI_HAVE_CUBLASMP`, `LORRAX_FFI_HAVE_CUFFT`; SLATE is probed at `LORRAX_SLATE_INSTALL_DIR` |
| unset | — | `FATAL_ERROR` naming both legs | — |

* **Host leg.** CUDA-free by construction: its sources compile with
  `LORRAX_FFI_NO_CUDA`, and GATE 3 refuses a CUDA-stack `DT_NEEDED`. A feature
  group whose dependency is absent is skipped with a `STATUS` line. The FFT
  handler is always built (it declares the FFTW3 ABI itself);
  `LORRAX_HOST_HAVE_FFTW3` only records a found FFTW3 as the run-time
  `dlopen` hint `LORRAX_FFTW3_SO_HINT` and never links it. The GEMM handler is
  built when a CBLAS header and provider are found. The link uses
  `-Wl,--no-undefined`.
* **CUDA leg.** Always links cuSOLVERMp, cuBLASMp, NCCL, cudart, cuSOLVER and
  cuBLAS, and builds the cuSOLVERMp, local-cuBLAS, active-subspace, contour
  and spin-rotation handlers. `LORRAX_FFI_HAVE_CUBLASMP` adds the fused
  cuBLASMp GEMM and W-solve handlers and is the switch that enables the CUDA
  language (`enable_language(CUDA)`). `LORRAX_FFI_HAVE_CUFFT`
  builds the mathdx family when the probe finds `cufft.h`, `libcufft`,
  `nvrtc.h` and `libnvrtc`; otherwise the six targets are absent and startup
  refuses with `GATE kconv-target`. `LORRAX_FFI_HAVE_CAL` is §4.
* **Both legs.** `$ORIGIN` is first in `RPATH`. `exports_{cuda,host}.map`
  localise every LORRAX-owned symbol, and the host leg's C entry points carry
  a `_host` suffix (`cpp/common/c_abi.h`), so the two libraries define no
  LORRAX name in common when both are `dlopen`ed `RTLD_GLOBAL`. Each leg
  exports `lorrax_ffi_{cuda,host}_abi_version` and
  `lorrax_ffi_{cuda,host}_build_config`.

`LORRAX_FFI_PLATFORM` is a CMake cache variable; no Python module reads it.

### 2a. Build entry points

| machine | leg | entry |
|---|---|---|
| Perlmutter `lorrax_A` (bare-host CUDA 13) | CUDA | the runtime's recipe outside this repository: `-DLORRAX_FFI_HAVE_CAL=OFF` against the NCCL-native cuSOLVERMp, no device SLATE, MPI from `config/perlmutter/ffi_mpi.sh` |
| Perlmutter | host | `config/perlmutter/build_ffi_host.sh`, bare metal: pins MPI through `ffi_mpi.sh`, loads `cray-hdf5-parallel/1.14.3.7`, captures the cray-fftw path as the hint, and unloads `cray-libsci`, `cray-fftw`, `craype-accel-nvidia80` and `cudatoolkit` before configure |
| Shifter sites | CUDA | `src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh` |
| Frontera | host / CUDA | `config/frontera/build_ffi_host.sh` / `config/frontera/build_ffi.sh` |
| anywhere else | host | `bash src/ffi/cpp/build_host.sh` |

`config/perlmutter/ffi_mpi.sh` pins the one MPI both Perlmutter legs link:
`cray-mpich/9.0.1` (`libmpi_gnu_123.so.12`, the MPI the phdf5 stage and the
SLATE host install need) and the LibSci that links it,
`cray-libsci/25.09.0`. Both legs are loaded into one GPU process, so they
must name the same `libmpi`. To move to another MPI, change that file only.

### 2b. The build contract: `scripts/verify_ffi_build.sh`

Every build path ends in `scripts/verify_ffi_build.sh [--leg host|cuda]
<so>`, and `services/distrib_la/tests/test_so_acceptance.py` checks the exact
handler names out of the loaders' own tables. The verifier checks patterns;
the test checks names. Run both.

| gate | invariant | where it can run |
|---|---|---|
| 0 | every backend in `LORRAX_FFI_EXPECT_BACKENDS` exports a handler and the build stamp agrees. The default is the leg's full set, so a build that lost one fails | anywhere |
| 1 | one MPI runtime in the closure (`gate_one_mpi.sh`: `ldd`, deduplicated by `realpath`) | the run environment |
| 2 | one BLAS vendor and one threading flavour in `DT_NEEDED` | anywhere |
| 3 | the host leg links nothing from the CUDA stack | anywhere |
| 4 | the closure resolves (`ldd -r`) | the run environment |
| 5 | zero undefined `fftw_` symbols and zero `fftw` in `DT_NEEDED` | anywhere |
| 6 | every OpenMP entry in `DT_NEEDED` is `libgomp`, `libiomp5` or `libomp` | anywhere |
| 7 | one HDF5 SOVERSION, and the runtime provides it (`gate_one_hdf5.sh`; `LORRAX_FFI_EXPECT_HDF5_SOVERSION`, `LORRAX_FFI_EXPECT_PEER_SO` for the cross-leg check) | ELF halves anywhere; mapped-object half in the run environment |
| 8 | after one real FFT, exactly one FFTW3 engine is mapped and it is the staged one (`gate_one_fftw.sh`) | host leg, in a process that imports jax (`LORRAX_GATE_FFTW_PY` or `LORRAX_FFTW3_STAGE`) |
| 9 | no LORRAX internal on the dynamic table; every host `lrx_*` entry is suffixed | `build.sh`, `config/perlmutter/build_ffi_host.sh` |
| 10 | a CUDA-capable process with both libraries open completes a host phdf5 read (`gate_one_odr.py`) | a GPU node, both pins set |
| 11 | the exported ABI equals `src/ffi/cpp/common/lorrax_ffi_abi.h` | anywhere |

A gate that cannot run in the current environment prints `GATE COULD NOT
RUN` and is counted apart from passes. `LORRAX_FFI_VERIFY_STRICT=1` makes it
a failure; use it for certification inside an allocation.
`LORRAX_FFI_VERIFY=off` disables the verifier with an announcement, and a
library built that way is not certifiable.

### 2c. The deployable unit is one sealed pair

The two legs become one production provider only through
`src/ffi/cpp/stage/seal_bundle.py`. It publishes a new, non-overwriting
directory holding both libraries under `lib/`, the listed private
redistributables, and one `lorrax_ffi_bundle.json`. The manifest binds the
pair, the handler ABI, the full source revision, each file's size and SHA-256,
each ELF SONAME and `DT_NEEDED` record, and the dependency-first private
closure.

Only cuSOLVERMp, cuBLASMp, CAL, SLATE (and its ScaLAPACK API), BLAS++,
LAPACK++ and NVSHMEM may be sealed as private libraries. MPI, site HDF5, the
system and compiler runtimes, CUDA runtime and driver libraries, and NCCL
belong to the machine runtime; `seal_bundle.py` refuses them as private
inputs. At load,
`lxkit.native_provider` rehashes both legs and the closure, preloads each
private library by exact path (so no run script owns a library search path),
checks the live ABI symbol's origin with `dladdr`, and refuses any mapped
engine-private provider the manifest does not name.

`source.revision` is provenance, not a demand that the active checkout have
the same SHA. Compatibility is the exported ABI, the live feature and target
probes, and the build contract. A different ABI always refuses. An unsealed
build-tree library still loads and prints `LEGACY-UNSEALED` with its hash; it
is not production attestation. The Perlmutter `lorrax_A` module selects one
sealed bundle through `LORRAX_FFI_SO` and `LORRAX_FFI_HOST_SO`.

### 2d. How the loader selects a library

`ffi_loader.get_lib(platform)` (and `distrib_la.loader`, through the same
`lxkit.native_provider` policy):

* **Candidates**, in order: the pin (`LORRAX_FFI_SO` for CUDA,
  `LORRAX_FFI_HOST_SO` for cpu), the in-tree `src/ffi/cpp/build/` or
  `build_host/`, then each `sys.path` directory.
* **Pins.** A pin that is not a file refuses. When a selected leg belongs to
  a sealed bundle, both pins or neither must be set, and both legs must come
  from one manifest; a partial override or mixed providers refuse.
* **ABI.** A stamped library with a different ABI refuses (`FfiAbiMismatch`).
  An unstamped one is announced once and loads, unless
  `LORRAX_FFI_ABI_STRICT=1`.
* **Load order.** In a CUDA-capable process (the first `JAX_PLATFORMS`
  entry, if set, is `cuda` or `gpu`; `CUDA_VISIBLE_DEVICES` is not empty; an
  NVIDIA device node is visible), opening the host library opens the CUDA
  library first, so the CUDA build wins the shared SLATE/BLAS++ SONAMEs.
  After each `dlopen` the loader refuses a process with more than one mapped
  MPI runtime.
* **Probe.** `probe_target(target, platform)` gives one of three reasons:
  unknown target, library could not be loaded, or library loaded but does not
  export the handler. Every gate refusal quotes it.

---

## 3. What each machine provides

| Service | Perlmutter | Frontera |
|---|---|---|
| k-axis FFTs and convolutions, GPU | nvidia-mathdx ([router](#k-convolution-router-and-the-mathdx-family)) | — |
| k-axis FFTs and convolutions, CPU | FFTW3 ABI → cray-fftw (§3c) | FFTW3 ABI → MKL |
| spatial 3-D FFT inside `shard_map` | XLA (cuFFT inside jaxlib) | XLA:CPU |
| host GEMM | Cray LibSci CBLAS | MKL CBLAS |
| distributed dense solvers | cuSOLVERMp and cuBLASMp (GPU); ScaLAPACK from LibSci and host SLATE (CPU) | ScaLAPACK from MKL; host SLATE |
| parallel HDF5 | `cray-hdf5-parallel/1.14.3.7` over `cray-mpich/9.0.1` | phdf5 over Intel MPI |
| runtime | bare-host CUDA 13 (`lorrax_A`); Shifter stages for container sites | apptainer |

Versions and launch recipes: [Perlmutter](../environment/machines/perlmutter.md),
[Frontera](../environment/machines/frontera.md).

### 3a. The dependency matrix

One row per routine LORRAX does not implement itself: who serves it, what
else could, and what proves it built right.

| Routine | Perlmutter | Frontera | Alternatives and refusals | Check |
|---|---|---|---|---|
| **3-D FFT** inside `shard_map` | XLA:GPU `fft` → cuFFT in jaxlib | XLA:CPU `fft` | none: `fft_helpers.local_fftn3` / `local_ifftn3` are `jnp.fft` for code already inside a `shard_map`; no FFI route reaches them | none |
| **k-axis transforms and convolutions** (ζ fit, Σ, COHSEX, BSE, and the flat-k transform on CUDA) | CUDA: nvidia-mathdx. cpu: the FFTW3-ABI host handlers | host handlers | none on CUDA; a missing wheel refuses at startup (`GATE mathdx-headers`) | `tests/multi_device/kconv_router_p4.py` |
| **Flat-k FFT, host leg** | FFTW3 ABI by `dlsym` → cray-fftw | MKL's FFTW3 export, already resident | the candidate ladder (§3c) | GATE 5 (load time), GATE 8 (engine identity) |
| **Host band-block GEMM** (`ffi.gemm`) | LibSci CBLAS; LibSci has no `cblas_?gemm_batch`, so the handler loops plain `cblas_?gemm` | MKL CBLAS, batched entry | any CBLAS provider (MKL, LibSci, OpenBLAS, BLIS); the chosen entry is announced at first use | GATE 2 |
| **Planned axis GEMM** | full range: XLA `dot` → cuBLAS. Active range: `lorrax_cublas_local_active_range_gemm` over pointer views | XLA:CPU `dot` panels | K stays local; only output centroid axes are sharded ([active GEMM ranges](../dev/active_gemm_ranges.md)) | `services/distrib_la/tests/test_local_active_gemm_range.py` |
| **Distributed face GEMM** | cuBLASMp, full and active range | none: a `scalapack` or `slate` face plan refuses by name | `batch_reshard` only through the one-shot `matmul`, when whole matrices fit each device | `test_distrib_la_multiproc.py`, `test_active_gemm_range.py`, `tests/multi_device/active_band_sigma_gate.py` |
| **`eigh`** | default native `jnp.linalg.eigh` → cuSOLVER (GPU), LAPACK (CPU). Distributed: cuSOLVERMp `syevd` (GPU); ScaLAPACK `p?heevd` / `p?syevd` from LibSci (CPU) | default native → LAPACK. Distributed: ScaLAPACK from MKL | `eigh_backend` deck key (`slate` where a SLATE build serves it); refusals in [`distrib_la`](../services/distrib_la.md) | `pytest -m distrib_la`. Nothing observes which vendor answered |
| **Cholesky** | cuSOLVERMp batched `potrf`/`potrs` (GPU); host SLATE `potrf`/`trsm` (CPU) | host SLATE | `native2d` (JAX); no ScaLAPACK `potrf` handler exists | contract tests only |
| **LU** | cuSOLVERMp batched `solve_lu` (GPU); ScaLAPACK `p?getrf`/`p?getrs` (CPU) | ScaLAPACK from MKL | fused `solve_lu` and split `getrf`+`getrs`; resolve refuses if any target is missing | contract tests only |
| **Distributed transport** | NCCL for cuSOLVERMp (§4) and cuBLASMp; an `MPI_COMM_WORLD` split in mesh order for SLATE and ScaLAPACK | Intel MPI | — | the `[lorrax cusolverMp] … comm path:` banner |
| **Parallel HDF5** | `libhdf5_parallel_gnu.so.310` over `libmpi_gnu_123.so.12` | phdf5 over Intel MPI | none: one transport; a deployment that cannot serve it refuses at open | GATE 1, GATE 7 |
| **OpenMP runtime** | `libgomp` | `libiomp5` | `libgomp`, `libiomp5`, `libomp` | GATE 6 |

### 3b. Routines with no check

* The spatial 3-D FFT has no build or run check on either machine.
* No gate observes which vendor answered a distributed `eigh`, Cholesky or
  LU; only the `distrib_la` contract tests exercise them.
* `LORRAX_LU_NO_PIVOT` (a *bool*) disables cuSOLVERMp pivoting from the
  environment with no gate.

### 3c. Which FFT engine the host library binds

One source, `cpp/mklfft/fft_flat_k_ffi.cc`, serves every host FFT through the
FFTW3 advanced interface (`fftw_plan_many_dft`). No FFTW symbol binds at link
time. The engine is resolved at first use:

1. **Already loaded.** `resolve_sym` (`RTLD_DEFAULT`, then `RTLD_NEXT`). On an
   MKL site the ScaLAPACK link line has already loaded MKL's FFTW3 export, so
   the ladder never runs.
2. **The candidate ladder**, `dlopen(RTLD_GLOBAL)`, first hit wins:
   `$LORRAX_FFTW3_SO` → the build's `LORRAX_FFTW3_SO_HINT` (a compile-time
   path) → `libfftw3.so.3` → `libfftw3.so.mpi31.3` → `libmkl_rt.so` →
   `libfftw3.so`.
3. **Refusal.** The handler returns `mklfft: no FFTW3 engine in this
   process`, naming every candidate tried. The startup gate probes only the
   exported handler symbol, so this error arrives at the first host FFT.

On the Perlmutter bare host the hint is cray-fftw's own path. In a Shifter
container `/opt/cray/pe` does not exist, so the engine comes from the
`/lorrax_fftw` stage (`stage/fftw_stage_cray.sh`, mounted by
`run_shifter.sh`).

**Hazard.** CUDA images ship `libcufftw.so`, which exports all three entry
points the ladder binds. Pointing `LORRAX_FFTW3_SO` at it makes the host
handler transform on the GPU, and every FFT check still passes. GATE 8 is
the only check that tells these states apart.

An engine swap is accepted at value-level parity, relative 1e-12 on the Σ
path, never bit-exactness: engines differ in arithmetic order.

---

## 4. The cuSOLVERMp version picks the communication path

cuSOLVERMp changed its grid communicator from CAL (≤ 0.6.x) to NCCL
(≥ 0.7.0). The handler reads the loaded version with `cusolverMpGetVersion`
and selects the path at context creation (`cusolvermp/context.cc`); rank 0
prints `[lorrax cusolverMp] library X.Y.Z, NCCL …, comm path: NCCL|CAL`.

| cuSOLVERMp | ships `cal.h` / `libcal` | comm path | build flag |
|---|---|---|---|
| ≤ 0.6.x | yes | CAL | `-DLORRAX_FFI_HAVE_CAL=ON` (the CMake default) |
| ≥ 0.7.0 | no | NCCL | `-DLORRAX_FFI_HAVE_CAL=OFF` |

* **Every version exports the SONAME `libcusolverMp.so.0`.** A `.so` built
  against one version loads against another without complaint. The run's
  search path must therefore hold exactly one cuSOLVERMp: the sealed bundle
  loads its own by exact path, and `run_shifter.sh` puts exactly one NVHPC
  stage on `LD_LIBRARY_PATH`.
* **A `HAVE_CAL=OFF` build refuses a pre-0.7 library** at context creation. A
  `HAVE_CAL=ON` build carries both paths and `DT_NEEDED` `libcal.so.0`.
* **0.6.x is wrong on a 2-D grid.** With `Px > 1` and `Py > 1` its
  `getrf`/`getrs` return wrong answers, so context creation refuses that
  pairing (`GATE cusolvermp_2d_grid_version`). LORRAX meshes are square, so a
  `HAVE_CAL=ON` build with a pre-0.7 library runs only at P = 2.
* **≥ 0.8 needs NCCL ≥ 2.27** (`ncclCommWindowRegister`); the handler warns
  when the loaded NCCL is older.

**Container stages.** `build.sh` refuses (exit 2) when neither
`LORRAX_NVHPC_ROOT` nor `LORRAX_NVHPC_SUBPATH` names a stage, listing the
stages under `/lorrax_nvhpc` with the flag each needs. It also refuses a stage
without `cal.h` unless `LORRAX_FFI_HAVE_CAL` is set explicitly. The single
source of truth is `LORRAX_NVHPC_SUBPATH` (`config/perlmutter/site_config.sh`,
default `0.7.2_cuda12.9/math_libs/12.9/lib64`). `run_shifter.sh` exports it
with `LORRAX_NVHPC_ROOT` derived from its first component, so a build launched
there agrees with its runs.

---

## 5. Parallel HDF5: the FFI side

[`slab_io.md`](slab_io.md) owns the subsystem: the tile contract, the
launcher requirements, striping, certification, the one-owner-per-file rule
and the measured failure signatures. This section holds only the FFI facts.

* **One transport, no router.** `file_io.slab_io` takes a path, a mode and a
  mesh; a deployment that cannot serve the tile path refuses at open, naming
  the probe that declined ([availability](slab_io.md#availability)).
* **One C++ source, both legs.** The `phdf5/` sources compile into both
  libraries. On the host leg the device staging collapses: the read tail is a
  `memcpy` into the host XLA buffer and the write hands `H5Dwrite` the XLA
  buffer directly. The control-operand stream race
  ([`slab_io.md`](slab_io.md#stream-race)) is therefore CUDA-leg only.
* **The legs' entry points are not interchangeable.** The host leg's C entry
  points end in `_host`, and each leg localises its internals, because one
  `PhdfCtx` name has two struct layouts. A library built without them exports
  both layouts under one name, and the first-loaded library answers for both;
  GATE 9 and GATE 10 catch it.
* **`ffi.io.open_file(path, *, mesh, mode)`** picks the library from the
  mesh's devices and records it per handle, so `close_file` returns through
  the opening library. `mode` has no default. It refuses a mode outside
  `{w, a, r}`, a mesh without both `x` and `y`, and
  `p·q ≠ jax.process_count()`. An already-open path may be opened again only
  when both opens are read-only on the same platform and mesh; they then share
  one native context.

---

## 6. phdf5 defaults

The struct initialisers in `phdf5/ctx.h` are not the effective defaults:
`open_ctx_impl` in `phdf5/context.cc` reassigns every field from the
environment when a file is opened.

| Field | Effective default | Override | Role |
|---|---|---|---|
| `use_collective_read` | `true` | `LORRAX_PHDF5_INDEPENDENT=1` → independent **reads** | tuning; a band-block `read_slabs` is independent regardless ([`slab_io.md`](slab_io.md#tuning)) |
| `use_collective_write` | `true` | `LORRAX_PHDF5_COLLECTIVE_WRITES=0` → independent writes | correctness (§7b) |
| `coll_metadata` | `false` | `LORRAX_PHDF5_COLL_META=1` | non-collective metadata keeps `H5Dcreate`/extend off the collective driver |
| `dedup_replicas` | `true` | `LORRAX_PHDF5_DEDUP_REPLICAS=0` | correctness: one writer per replica group; overlapping selections are undefined under collective writes |
| `align_threshold`, `align_length` | 4 MiB (header: 1 MiB) | `LORRAX_PHDF5_ALIGN_MB` | tuning; independent of the stripe unit |

**Boolean grammar.** `env_flag` (`phdf5/ctx.h`): unset or empty → the
default; otherwise trimmed and lower-cased, and true only for `1`, `true`,
`yes`, `on`. Any other value is false, silently: `=ture` turns collective
writes off. The Python twin, `runtime.env_flags.env_bool`, accepts the same
table but announces an unrecognised value; `tests/test_env_grammar.py` holds
the two in step. Collective-buffering and stripe knobs are in
[`slab_io.md`](slab_io.md#tuning).

---

## 7. Failure modes, and how to tell them apart

### 7a. The PMI-flavour mismatch gives wrong answers

Launched with the wrong PMI for Cray MPICH (`srun --mpi=pmi2` instead of
`cray_shasta`), every rank gets a private singleton `MPI_COMM_WORLD`:
`MPI_Comm_size == 1` while `jax.process_count() == P`. The native checks
cannot see it: `ffi.io.open_file` checks `p·q == jax.process_count()`, and
`shard_index.h::validate_shard_encoding` checks `prod(mesh_shape) ==
ctx->world_size`, where `world_size` *is* `jax.process_count()`. Both compare
JAX to JAX. With disjoint hyperslabs the write completes bit-exact at rc = 0
(sandbox CLAIMS 68); two ranks on one chunk would corrupt silently.

**The guard** (`file_io/_slab_io_ffi._assert_mpi_world`) asks MPI once, at
the first collective open, and compares `MPI_Comm_size(MPI_COMM_WORLD)` with
`jax.process_count()`. The verdict is rank-invariant, so it refuses on every
rank or none.

* A mismatch always refuses. Fix the launcher.
* An MPI world that cannot be probed refuses by default;
  `LORRAX_PHDF5_REQUIRE_MPI_WORLD=0` downgrades that case to a rank-0
  warning.
* `LORRAX_PHDF5_SKIP_MPI_WORLD_CHECK=1` removes the guard. It is a debugging
  escape, never a remedy.

### 7b. The ROMIO collective-buffer OOM

Cray MPICH's collective write can exhaust memory at large per-rank aggregates:

```
Out of memory in .../ad_cray/ad_cray_write_coll.c, line 669
… MPI_Abort … "HDF5: infinite loop closing library"
```

The same line appears when the PMI flavour is wrong (§7a) and collective
writes are on. The two want opposite remedies:

| | genuine collective-buffer OOM | PMI-flavour mismatch |
|---|---|---|
| `MPI_Comm_size(MPI_COMM_WORLD)` | `== jax.process_count()` | `1` on every rank |
| per-rank aggregate | ≳ 1 GB | any |
| `LORRAX_PHDF5_COLLECTIVE_WRITES=0` | fixes it | hides it: silent wrong answers (§7a) |

Check the world size first. `LORRAX_PHDF5_INDEPENDENT=1` changes reads only
and does nothing for a write-side OOM.

### 7c. SONAME aliases that look like two MPIs

In the Shifter container, `stage/phdf5_stage_cray.sh` creates one symlink per
Cray compiler-specific SONAME, `libmpi_gnu_{91,110,123}.so.12`, all pointing
at the container's generic MPICH-ABI `/opt/udiImage/modules/mpich/libmpi.so.12`
(`SHIM_TARGET`). Every variant is one object. On a login node the closure is
incomplete and `ldd` reports several dependencies `not found`, which proves
nothing. Check a library's closure where it runs: inside the container on a
compute node, where `gate_one_mpi.sh` (GATE 1) deduplicates by `realpath`.

### 7d. Bounds-check asymmetry hangs with no traceback

Bounds are tested once, on the logical slab `offset + valid_shape`, which is
replicated, so every rank reaches the same verdict. A test on a rank-local
offset splits the ranks into those that refuse and those that enter the
collective, and the communicator hangs with no HDF5 error. No rank may skip a
collective because of its own error: record it, take part in the teardown,
then raise ([`decisions.md`](decisions.md), 2026-08-04).

---

## 8. Hard invariants

1. **Registered FFI target names and C++ handler symbols do not change.** The
   sets are `_CUDA_TARGET_SYMBOLS` and `_HOST_TARGET_SYMBOLS` in
   `ffi/common/ffi_loader.py` (and `distrib_la.loader`'s table). Refactors
   move files, never a target string.
2. **Env knob spellings do not change.** Add an alias instead.
3. **Library names are `liblorrax_ffi.so` and `liblorrax_ffi_host.so`.** A
   change updates every consumer in the same commit.
4. **The two legs share no LORRAX-owned dynamic symbol** (GATE 9, GATE 10).
5. **A stage or build script refuses an unstated environment fact rather than
   guessing it.** `phdf5_stage_cray.sh` refuses an unset `HDF5_DIR` or
   `MPICH_DIR`; `build.sh` refuses an unstated cuSOLVERMp stage, a CAL
   mismatch, and unset `LORRAX_MPI_INCLUDE_DIR` / `LORRAX_MPICH_LIB_DIR`
   (CMake would otherwise fall back to HPC-X Open MPI). What is staged is what every later build links, and a wrong guess
   surfaces much later as a wrong answer or a hang.
6. **A handler ABI change bumps `src/ffi/cpp/common/lorrax_ffi_abi.h`** and
   its mirror `ffi_loader.LORRAX_FFI_ABI_VERSION` together
   (`tests/test_ffi_abi_stamp.py`), and old bundles then refuse.

---

## k-convolution router and the mathdx family

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
| 1 consumer | ζ fit: `isdf.core.c_q_downfold` (pair); `isdf.core.c_q_from_psi_sm` (parent); `isdf.zeta_mubatch.make_route_g_kernel` (plane, every channel; the current channels' γ̃ as the load's `(perm, phase)`). Σ: `gw.ppm_tau_kernel.get_sigma_spatial_kernel` (τ sweep and static SX/RI: klead `prep` for W, klead unfold for G), `gw.cohsex_sigma._make_static_convolution` (klead). BSE: `bse_stack_matvec._conv_decode`, `bse_ring_comm._make_ring_rung`, and the W_R transforms in `bse_densify.make_w_densifier`, `bse_lanczos`, `davidson_absorption`, `absorption_haydock`, `bse_nontda`, `exciton_bands`. Flat-k transform: `common.fft_helpers.make_flat_k_fft` and its `make_flat_k_ifftn` / `make_flat_k_fftn` / `make_local_flat_k_fftn` wrappers (`gw.w_isdf`, `gw.qsgw_head`, `gw.cohsex_sigma`, `gw.wavefunction_bundle`, `bandstructure.htransform`, `bandstructure.orbital`) |
| 2 router | `ffi/fft.py`: `make_fused_conv_kpair`, `make_fused_conv_kparent`, `make_fused_conv_kplane`, `make_kconv_klead` (→ `KConvStored(prep, apply)`), `make_kconv_klead_unfold`, `make_kconv_lorentz_unfold`, `make_kconv_kminor` / `make_local_kconv_kminor`, `make_kfft_klead` / `make_local_kfft_klead`, `make_kfft_kminor` / `make_local_kfft_kminor`. `common.fft_helpers.get_donated_kfft_kminor` is `make_kfft_kminor` jitted with its input donated, memoised per `(mesh, kgrid, spec, kind, norm)`; the caller drops its own reference after the call |
| 3 gate | `require_kconv`, then `require_fourier_plan`, called by `runtime.initialize_communicator_stack` after the FFT and GEMM gates. `require_kconv` on CUDA: the wheel's headers, every `ffi.fft.KCONV_TARGETS` target, and one probe compile (mode 3, k-grid 2×1×1, disk-cached), so a device the installed cuFFTDx cannot compile for refuses at startup (`GATE mathdx-probe`, naming its compute capability and the wheel); cpu: `lorrax_mklfft_flat_k`. `require_fourier_plan` on CUDA: `lorrax_fourier_plan`; cpu: nothing (XLA ops). Each factory re-probes its own target (a `LocalFourierPlan` that CUDA can lower probes `lorrax_fourier_plan` at construction); operand shapes and dtypes are checked at trace time |
| 4 target | CUDA, in `liblorrax_ffi.so`: the ten `lorrax_mathdx_*` names the router calls (`_kconv_pair`, `_kconv_parent`, `_kconv_plane`, `_kconv_klead`, `_kconv_klead_unfold_rows`, `_kconv_klead_lorentz_rows`, `_kfft_klead`, `_kconv_kminor`, `_kfft_kminor`, `_plane_fft_gather`); the library also keeps `_kconv_klead_unfold` and `_kconv_klead_lorentz`, the same kernels storing every k row, for older source trees. Also `lorrax_fourier_plan` (`cpp/cufft/fourier_plan_cuda_ffi.cc` + `fourier_plan.cu`, nvcc: sm_80 SASS and compute_80 PTX, JIT-compiled by the driver on sm_90 and later). cpu, in `liblorrax_ffi_host.so`: `lorrax_mklfft_flat_k`, `lorrax_mklfft_gw_conv` |
| 5 handler | CUDA: `cpp/cufft/kconv_mathdx_cuda_ffi.cc`, one embedded cuFFTDx source (`kSrc`; mode 10 has its own, `kPlaneSrc`) compiled by NVRTC for the device's own `sm_<cc>` per (CUDA context, mode, `nkx`, `nky`, `nkz`, `ns`, precision) into an in-process cache, backed by the disk cubin cache below (images keyed per sm); `cpp/cufft/fourier_plan_cuda_ffi.cc`, plans cached per (device, attributes, batch). cpu: `cpp/mklfft/fft_flat_k_ffi.cc` (`MklFftFlatKHostFfi`, `MklFftGwConvHostFfi`) |

**Doors.** Pick the door whose k position matches the tile you hold. A caller
never transposes to reach another door.

| door | k axis of the operand | CUDA mode | cpu leg |
|---|---|---|---|
| `make_fused_conv_kpair` | 3-D leading `(nkx, nky, nkz, …)` | 0 | two host flat-k inverse transforms, the spin contraction in XLA, one host forward transform |
| `make_fused_conv_kparent` | parent tables | 1 | the typed parent load in XLA, which materialises `(N_k, ns, μ, ν, ns)` per side, then the pair tail |
| `make_fused_conv_kplane` | route-G plane output `(N_k, g, ns, 2c, ns, p)` | 6 | the Bloch phase, split and transpose in XLA, then the pair tail |
| `make_kconv_klead` | flat leading `(N_k, …)` | `prep` 3, `apply` 2 | `prep` is the identity; `apply` is `lorrax_mklfft_gw_conv`, which transforms W itself and holds the R-space T tile only in per-thread compact chunks |
| `make_kconv_klead_unfold` | raw-parent Green `(n_parent, μ, ns, ν, ns)`; output `(len(store_rows), ns, μ, ns, ν)`, every other k row transformed and never stored | 7 (`prep` of `make_kconv_klead`) | `symmetry_maps.apply_unfold_load_tables_local` in XLA (a full-k copy), then the `make_kconv_klead` apply and the row selection |
| `make_kconv_lorentz_unfold` | the same, with the Lorentz blocks `V (N_k, μ, n_A, ν, n_B)` | 8 | the same composition, the γ̃ block sum in XLA, the row selection |
| `make_kfft_klead` | flat leading `(N_k, …)` | 3 | `lorrax_mklfft_flat_k` |
| `make_kconv_kminor` | flat trailing `(…, N_k)` | 4 | XLA moves k to the front, then host inverse transform, product, host forward transform, and k moves back |
| `make_kfft_kminor` | 3-D trailing `(…, nkx, nky, nkz)` | 5 | the same transpose around one host transform |
| `common.fourier_plan.LocalFourierPlan` | none: ≤ 3 spatial axes with per-axis supports; the entry point for sphere↔box and plane transforms (§ Local Fourier plan) | `lorrax_fourier_plan`; its `in_gather` form is mode 10 | XLA ops: `dot_general` GEMM axes and one `jnp.fft` group |
| `make_plane_fft_gather` | the backend of `LocalFourierPlan(in_gather=…)`: the route-G cylinder `(…, n_col)` → the transformed plane `(…, n_b, n_c)` | 10 | the XLA route: static-run concatenate, then `jnp.fft.fftn` |

- **Sharding.** The pair, parent, plane and `make_local_*` doors are rank-local
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
| 6 plane | mode 0 on the identity plan, loaded from the route-G D-plane FFT output: `P^X = conj(F·D^X)` with the Bloch phase `F[k,g,p]`, L = slots `[0, c)` and R = slots `[c, 2c)` of the `2c` axis, split on load | `D` `(N_k, g, ns, 2c, ns, p)`, `F` `(N_k, g, p)` → `U` `(N_k, c, g·p)` | 3 |
| 7 klead unfold conv | mode 2 read from the raw-parent Green through `symmetry_maps.unfold_load_tables`: per full k the parent row (the transposed partner `Gt` on an antiunitary row), both endpoint gathers and umklapp phases `(mph·G)·nph`, then `U_k G U_k†` in registers; spin-major store | `G`, `Gt` `(n_parent, μ·ns, ν·ns)`, `V_R` `(N_k, μ, ν)` → `U` `(N_k, ns, μ, ns, ν)` | 1 |

- **Flat k** is C order, with `kz` fastest.
- **Dtype.** Modes 0, 1, 6 and 7 are complex128 only. Modes 2–5 take all-complex128
  or all-complex64 operands (the complex64 image serves the fp32-GMRES BSE
  arm) and never cast. The cpu host handlers are complex128 only, so a
  complex64 operand refuses at trace time on a cpu mesh.
- **In place.** Modes 2, 3 and 5, and mode 4 with `out_layout=0`, alias
  operand 0 to the result (`input_output_aliases={0: 0}`). This is safe
  because each block reads all `N_k` values of its rows before it stores any
  of them.
- **Launch geometry.** One 256-thread block per `rb` rows; a row needs
  `banks·16·(N_k|1)` bytes of shared memory (8 per element for complex64).
  Modes 0, 1 and 6 take `rb ≤ 16` within 100 KiB, modes 2–5 and 7 `rb ≤ 64`
  within 48 KiB (three blocks per A100 SM); a row larger than the budget gets
  what the device's opt-in maximum holds. Mode 7 rounds `rb` down to whole
  `ns²` spin groups when at least one fits, and then loads each pair's `ns²`
  sources once for all its spin rows; below one group each bank loads its own
  row with the same arithmetic.
- **Cost.** Each transform is `O(rows·N_k log N_k)` flops. HBM traffic is one
  read of each operand and one write of the result. Modes 6 and 7 read the
  producer's own buffer (the plane FFT output; the parent Green, `n_parent/N_k`
  of the full-k size), so the phased, split or unfolded copy the modes 1 and 2
  operands would need is never written. Mode 7's tables (`lsrc`, `rsrc`
  int32, `mph`, `nph` complex128, each `(N_k, μ·ns)`) are closed-over host
  constants sliced per rank. The CUDA kernels allocate
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
| `GATE mathdx-kconv-residency` | first call (kernel build) | one resident row, `banks·16·(N_k|1)` bytes (8 per element for complex64), exceeds the device's opt-in shared memory per block. On an A100 (166 912 B) that is `N_k > 3477` for modes 0/1/6 and `N_k > 10431` for modes 2–5 and 7 in complex128 | a smaller k-grid; the family has no out-of-core arm |
| `k-leading unfold conv: …` | factory; apply | tables cut for another mesh shape; `G` whose parent count or endpoint widths differ from the tables; `Gt=None` on a plan with antiunitary rows | build the tables from the same plan and mesh as `G` (`plan.unfold_load_tables()`); pass `ParentGreen.transpose` |
| `k-conv plane expects …` | trace | `D` or `F` not complex128 `(N_k, g, ns, 2c, ns, p)` / `(N_k, g, p)` | pass the plane FFT output as laid out |
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
  precision, SM); the whole toolchain that can change an image: the
  cuFFTDx, commonDx, CUTLASS and CCCL version headers, the nvidia-mathdx
  wheel's dist-info name, and the NVRTC version with the loaded libnvrtc's
  real path (its patch level). A version header that reads empty disables the
  disk cache for that build rather than dropping out of the key. Editing the
  kernel source invalidates every image.
- **File.** `kconv_m<mode>_<nkx>x<nky>x<nkz>_ns<ns>[_c64]_sm<XY>_<key>.cubin`,
  with a `LRXKCONV1` header that carries the key and a hash of the payload.
- **Writes and reads.** A write goes to a unique temporary and is `rename`d
  into place, which is atomic on one filesystem, so concurrent ranks each
  publish a whole file. A read re-hashes the payload and checks for an ELF
  image; a torn, foreign or non-ELF file, or one the driver refuses to load,
  is deleted, recompiled once and replaced.
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
more than 1e-3. Modes 6 and 7 must also equal the XLA chains they replace
within 2 ulp of the largest value (bitwise today, reported): mode 6 forms
`F·D` as an XLA:GPU complex multiply (no FMA), and mode 7 spells
`(mph·G)·nph` and `U·G·U†` as the XLA unfold and the spin-rotate kernel
round them. The mode-7 cases include a C3 plan with a general complex spin
action and `q = n/3` phases (`ns` 2 and 4) and `N_k = 196` at `ns = 4`, the
per-bank load. `tests/multi_device/kconv_cubin_cache_check.py` covers the
disk cache's key and refusal paths.

A new mode is added in four steps:

1. Add a kernel entry to the embedded source under its `LRX_MODE` value.
2. Add a handler and `XLA_FFI_DEFINE_HANDLER_SYMBOL` in the same translation
   unit. Register its target in `ffi_loader._CUDA_TARGET_SYMBOLS`,
   `ffi.fft.KCONV_TARGETS` (which `require_kconv` checks at startup) and
   `ffi.cufft.CUDA_TARGETS`/`CUDA_SYMBOLS`.
3. Add a router factory in `ffi/fft.py` that returns the mathdx call on CUDA
   and the plan-route composition on cpu.
4. Add a case, with a red twin, to `tests/multi_device/kconv_router_p4.py`.

### Plane FFT with gather-on-load (mode 10)

Route G transforms planes whose occupied cells arrive as a compact cylinder
`F (…, n_col)`; `plane_from_col (n_b·n_c,)` names each flat cell's column
(`n_col` = empty). The door returns

```text
Y[…, k_b, k_c] = Σ_{b,c} P[…, b, c] e^{-2πi (b k_b/n_b + c k_c/n_c)},   P = F scattered by plane_from_col, 0 elsewhere
```

(`fftn(P, axes=(-2,-1), norm='backward')`) without writing `P`. One block
holds `PB` planes of `(n_b, n_c|1)` in shared memory (`PB ≤ 8` planes within
64 KiB, else 1). It gathers the occupied rows' cells through `gidx (rows,
n_c)` and `row_of (rows,)`, runs the row FFTs on those rows only, runs the
column FFTs on every column with dead rows read as zero, and stores each
plane once, coalesced. HBM traffic is one read of the cylinder and one write
of the plane.

Every line FFT is a cuFFTDx thread FFT (`n ≤ 40`). An axis `n = n1·n2` with
`gcd(n1, n2) = 1` runs as the Good–Thomas two-dimensional DFT: the input sits
at `(n2·i1 + n1·i2) mod n`, output `(k1, k2)` is `X[k]` for `k ≡ k1 (mod n1)`,
`k ≡ k2 (mod n2)`, so there are index maps and no twiddles, and frequency `k`
stays at slot `(n2·(k mod n1) + n1·(k mod n2)) mod n` until the store.
`plane_fft_split` picks the most balanced split, or `(n, 1)` for a prime
power `≤ 40`. Block FFTs are not used because cuFFTDx's fp64 database lacks
45, 54, 75, 90, 150 and 250, which would take Bluestein and a host-built
workspace.

The door decides once, at build, and announces the route by name. Mode 10
serves a plane iff both axes split and the block fits:
`16·n_b·(n_c|1) + 5·n_b + 8·PB + 16 ≤` the device's opt-in shared memory per
block (`ffi.fft.plane_resident_bytes`; the second term is the kernel's static
row tables, and `build()` applies the same bound, refusing as `GATE
mathdx-plane-residency`). Every other plane takes the XLA route: an axis with
no split (a prime above 40 or a prime power above 40: 41, 49, 64, 81, 121,
125, 128, 250, …) or an oversized block. Largest square served: 100 on
sm_80/87 (163 KiB), 78 on sm_86/89/120 (99 KiB, so 80² takes the XLA route),
119 on sm_90/100 (227 KiB). The block runs 512 threads when one block has the
SM (`⌊smem per SM / (PB·plane + 1 KiB)⌋ = 1`, e.g. 72² and up on A100), else
256 with two blocks per SM; the register cap this sets holds on every sm_80+.
F must be complex128 (`GATE plane-fft-dtype`, on both routes) and
`plane_from_col` in `[0, n_col]`. `fn(F, start, size)` transforms the slab
`F[:, start:start+size]` of `F (A, S, …, n_col)` in place, so the ζ loop's
group slice is not copied. The gates are
`tests/multi_device/plane_fft_gather_p4.py` (GPU parity `≤ 1e-13` over the
QE sides 24–250, the routes, a red twin, the slab form) and
`tests/test_plane_fft_gather.py` (a NumPy model of the passes against
`np.fft.fft2`).

### Local Fourier plan (`LocalFourierPlan`)

`common.fourier_plan.LocalFourierPlan(extents, axes, *, sign, norm,
in_support, out_support, out_perm, in_gather, mesh)` computes, on one device,

```text
y = R_out · F_{sign,norm} · E_in · x      over ≤ 3 axes, complex128
```

with `F` exactly `jnp.fft.fftn` (`sign = -1`) or `ifftn` (`+1`) and `jnp.fft`'s
`norm`. `E_in` embeds a compact axis (`in_support[ax]`, indices taken `% N`, a
repeated index refuses); `R_out` restricts one. Supports are separable: a
sphere enters through its tight bounding box.

* **Per-axis backend.** An axis is a GEMM with a Fourier matrix built once in
  float64 (exact integer phase reduction) iff `N` lies in
  `GEMM_CROSSOVER[device kind]` (a full and a supported range); otherwise it
  joins the FFT group. Unknown devices and CPU take the FFT everywhere
  (decisions.md 2026-09-25). Stages run shrinking GEMMs, then the FFT group
  (embed, transform, restrict), then expanding GEMMs.
* **Legs.** Chosen when the call is lowered (`lax.platform_dependent`): on
  CUDA one `lorrax_fourier_plan` custom call (cuBLAS ZGEMM with a stride-0
  matrix, no transposes; one cuFFT Z2Z plan per contiguous run of FFT axes;
  remap kernels for embed/restrict); elsewhere XLA ops. A CPU operand in a GPU
  process therefore takes the XLA leg.
* **Caches.** The CUDA leg caches plans per (device, attributes, batch) for
  the process: device Fourier matrices (`16·N'·K` bytes per GEMM axis), remap
  tables and cuFFT plans without work areas. Work areas and the ping-pong
  intermediates come from XLA's scratch allocator on every call, so nothing
  the plan uses per call lives outside the pool.
* **Refusals.** `GATE fourier-plan-contract` (not complex128, or not 1–3
  axes, at construction on every platform); `GATE fourier-plan-int32` (a cuFFT
  or cuBLAS size past `2³¹−1`); a missing `lorrax_fourier_plan` at
  construction when CUDA can lower the plan.
* **Determinism.** Reruns are bitwise at fixed device and toolkit, and a
  batch slice equals the same rows of a larger batch; nothing is promised
  across architectures.
* **`in_gather=(plane_from_col, n_col)`** is the route-G plane: mode 10
  (§ above) or its XLA route, with the slab form `plan(F, start, size)`.

The service page is [`../dev/fourier_plan.md`](../dev/fourier_plan.md); the
gate is `tests/test_fourier_plan.py` (every leg the platform has).

### Parent-load ISDF pair convolution (mode 1)

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
