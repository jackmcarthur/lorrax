# Building the FFI libraries

LORRAX reaches every vendor library (cuSOLVERMp, cuBLASMp, cuFFT, ScaLAPACK,
SLATE, FFTW, parallel HDF5) through two shared objects built from the one C++
tree `src/ffi/cpp/`. This page is for whoever builds them: what each needs,
how to build and verify a pair on Perlmutter, Frontera or a new site, how a
pair is sealed for other users, how a run selects one, and the ABI rule that
pairs a library with a Python tree. Read [Installation](index.md) first; the
design of the layer (its targets, kernels and failure modes) is
[the FFI layer](../architecture/ffi_layout.md).

## The two legs, and why they must agree

| leg | file | contains |
|---|---|---|
| host | `liblorrax_ffi_host.so` | ScaLAPACK, SLATE (`gpu_backend=none`), a CBLAS GEMM, an FFT through the FFTW3 ABI bound at run time, parallel HDF5. CUDA-free. |
| CUDA | `liblorrax_ffi.so` | cuSOLVERMp, cuBLASMp, cuBLAS, cuSOLVER, cuFFT, NCCL, the NVRTC-built mathdx k-convolution and Fourier-plan kernels, the small nvcc kernels, parallel HDF5. SLATE CUDA handlers only when built against a `gpu_backend=cuda` SLATE. |

Every target and its source file is in the
[kernel catalog](../architecture/ffi_layout.md#kernel-catalog).

A GPU run loads both legs into one process with `dlopen(RTLD_GLOBAL)`. A leg
can therefore be correct alone and still unusable beside its partner. Three
consequences shape every build:

1. **One MPI.** Both legs must link the same `libmpi`. Two MPIs in one
   process bind some MPI calls to one and some to the other, and the run
   hangs in `MPI_Init` or a collective.
2. **One HDF5.** The host leg must link the HDF5 SOVERSION that the run
   environment maps, so that both legs call one HDF5 library (GATE 7).
3. **Shared SONAMEs.** A CUDA-built and a host-built SLATE or BLAS++ have the
   same SONAMEs (`libslate.so.2`, `libblaspp.so.2`); whichever loads first
   serves both legs.

The main hazard of a build is a library that links, loads and registers
every target while delivering less than was asked for: a missing backend, a
second MPI or BLAS, or the wrong HDF5. The verifier ([below](#verify)) turns
each of those into a build failure.

## Dependencies

None of these are declared in `pyproject.toml` except `nvidia-mathdx`.

| dependency | minimum | leg |
|---|---|---|
| NVIDIA GPU | sm_80 | CUDA |
| CUDA toolkit | 12.x or 13.x, the same major as the JAX wheel (the `cuda12` or `cuda13` extra); Perlmutter uses 13.2 | CUDA |
| JAX and jaxlib | 0.9.x ([the JAX contract](index.md#jax)); the XLA FFI headers come from the JAX that will load the library | both |
| cuSOLVERMp | 0.7 (NCCL-native); 0.8 and later need NCCL ≥ 2.27 | CUDA |
| cuBLASMp | the version staged with cuSOLVERMp | CUDA |
| parallel HDF5 | 1.12, built with `HDF5_IS_PARALLEL` | both |
| MPI | the site MPI, named explicitly | both |
| SLATE, BLAS++, LAPACK++ | a `gpu_backend=none` install for the host leg | host |
| BLAS + ScaLAPACK | any one flavour (LibSci, MKL, OpenBLAS + netlib ScaLAPACK) | host |
| FFTW3 ABI | found at run time, never linked | host |
| nvidia-mathdx | 25.6.0, the wheel in the venv; not a build dependency | CUDA (run time) |

- **cuSOLVERMp and cuBLASMp.** One stage script per source:
  `src/ffi/cpp/stage/cusolvermp_stage_pypi.sh` (the standalone wheel),
  `cusolvermp_stage_nvhpc.sh` (an NVHPC SDK install) and
  `cusolvermp_stage_cublasmp_redist.sh` (the cuBLASMp redistributable).
  Every version exports the same SONAME, `libcusolverMp.so.0`, so the stage
  a leg is built against must be the stage its runs load. Versions ≥ 0.7
  communicate through NCCL, ship no `cal.h`, and need
  `-DLORRAX_FFI_HAVE_CAL=OFF`; 0.6.x returns wrong `getrf`/`getrs` results on
  any mesh with both $P_x > 1$ and $P_y > 1$
  ([the communication path](../architecture/ffi_layout.md#4-the-cusolvermp-version-picks-the-communication-path)).
- **Parallel HDF5.** Point CMake at it with `-DHDF5_ROOT=<prefix>` (or
  `$HDF5_ROOT` / `$HDF5_DIR`); CMake refuses an HDF5 without
  `HDF5_IS_PARALLEL`. On Cray, load `cray-hdf5-parallel` and stage it with
  `src/ffi/cpp/stage/phdf5_stage_cray.sh`; on OpenMPI sites use
  `phdf5_stage_openmpi.sh`, an MPI build from conda-forge
  (`hdf5=*=mpi_openmpi_*`) or spack (`hdf5+mpi`).
- **SLATE, BLAS++, LAPACK++.** Build SLATE from source; BLAS++ and LAPACK++
  install under the same prefix. On a Cray PE,
  `src/ffi/cpp/stage/slate_build_perlmutter.sh cpu|gpu` builds either
  reproducibly. Elsewhere:

  ```bash
  git clone --recurse-submodules https://github.com/icl-utk-edu/slate
  cmake -S slate -B slate/build -Dgpu_backend=none -Dblas=openblas \
      -DCMAKE_INSTALL_PREFIX=$HOME/software/slate_builds/cpu/install
  cmake --build slate/build -j && cmake --install slate/build
  ```

  The host leg needs `gpu_backend=none`: a CUDA BLAS++ in the host leg makes
  `get_device_count()` disagree between the legs.
- **FFTW.** The host leg resolves its FFT engine at first use through the
  FFTW3 ABI ([which engine binds](../architecture/ffi_layout.md#3c-which-fft-engine-the-host-library-binds)).
  The FFTW prefix may reach CMake only as a `dlopen` hint, never as a
  `DT_NEEDED` entry, so that an MKL site's already-loaded FFTW3 export wins.
- **nvidia-mathdx.** Header-only; the `cuda12`/`cuda13` extras pin it. The
  k-convolution kernels are compiled from its headers per k-grid by NVRTC at
  run time ([k-convolution](../architecture/kconv.md)), so nothing is linked.

## Building a leg

| site | leg | entry |
|---|---|---|
| Perlmutter | host | `config/perlmutter/build_ffi_host.sh` (needs `slate_build_perlmutter.sh cpu` first) |
| Perlmutter | CUDA | `config/perlmutter/build_ffi_cuda.sh` |
| Perlmutter module | both | `config/perlmutter/module/build.sh`, which runs the two scripts above and seals ([publishing a module](perlmutter.md#publish)) |
| Frontera | host | `config/frontera/build_ffi_host.sh` |
| cloud box with only a driver | both | `config/cloud/build_ffi.sh`, `config/cloud/build_ffi_host.sh` (`config/cloud/README.md`) |
| a Shifter container site | CUDA | `src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh` |
| anywhere else | host | `bash src/ffi/cpp/build_host.sh`, or CMake directly (below) |

The generic `src/ffi/cpp/build_host.sh` hands over to the site recipe on
Perlmutter and Frontera; `LORRAX_FFI_GENERIC_BUILD=1` builds generically
instead.

### With CMake directly

`LORRAX_FFI_PLATFORM` selects the leg; unset, CMake stops with an error
naming both.

```bash
cmake -S src/ffi/cpp -B build_cuda \
    -DLORRAX_FFI_PLATFORM=cuda \
    -DCUSOLVERMP_INCLUDE_DIR=<dir with cusolverMp.h> \
    -DCUSOLVERMP_LIB_DIR=<dir with libcusolverMp.so> \
    -DLORRAX_FFI_HAVE_CAL=OFF \
    -DHDF5_ROOT=<parallel HDF5 prefix> \
    -DLORRAX_MPI_INCLUDE_DIR=<dir with mpi.h> \
    -DLORRAX_MPICH_LIB_DIR=<dir with libmpi> \
    -DCMAKE_CUDA_ARCHITECTURES=<sm of the target GPU>
cmake --build build_cuda -j
scripts/verify_ffi_build.sh --leg cuda build_cuda/liblorrax_ffi.so
```

The host leg is the same command with `-DLORRAX_FFI_PLATFORM=host`,
`-DLORRAX_SLATE_HOST_INSTALL_DIR=<gpu_backend=none prefix>` and no CUDA
options. Add `-DLORRAX_SLATE_INSTALL_DIR=<gpu_backend=cuda prefix>` to give
the CUDA leg SLATE handlers. The CUDA leg reads the XLA FFI headers from
`jax.ffi.include_dir()` of the `python3` on `PATH`; the host leg takes
`-DLORRAX_XLA_FFI_INCLUDE_DIR` or probes the same way.

- `CMAKE_CUDA_ARCHITECTURES` defaults to SASS for sm_80, 86, 89, 90, 100 and
  120 plus compute_80 and compute_120 PTX. It governs only the nvcc
  translation units; NVRTC kernels compile for the running device.
- Without `LORRAX_MPICH_LIB_DIR`, CMake warns and falls back to
  `/opt/hpcx/ompi/lib`, and the library requests `libmpi.so.40`.
  `src/ffi/cpp/build.sh` refuses to run without `LORRAX_MPI_INCLUDE_DIR` and
  `LORRAX_MPICH_LIB_DIR`; `LORRAX_FFI_ALLOW_DEFAULT_MPI=1` lifts that for an
  OpenMPI build. `build.sh` also refuses an unstated cuSOLVERMp stage
  (`LORRAX_NVHPC_ROOT` or `LORRAX_NVHPC_SUBPATH`) and a CAL setting that
  disagrees with it. A build script refuses an unstated fact instead of
  guessing, because a wrong guess surfaces much later as a wrong answer or
  a hang.
- A build-time and run-time MPI mismatch appears as a segfault or a "cannot
  open shared object file" at startup, not as a named refusal.

### Perlmutter {#perlmutter}

`config/perlmutter/ffi_mpi.sh` pins the one MPI both legs link:
`cray-mpich/9.0.1` (`libmpi_gnu_123.so.12`, the MPI the parallel-HDF5 stage
and the SLATE host install also need) and the LibSci that links it,
`cray-libsci/25.09.0`, with darshan unloaded. To move to another MPI, change
that file only.

- **Host leg.** `build_ffi_host.sh` runs bare metal under `PrgEnv-gnu`: it
  pins the MPI, loads `cray-hdf5-parallel/1.14.3.7`, captures the LibSci and
  cray-fftw prefixes and unloads those modules (with `craype-accel-nvidia80`
  and `cudatoolkit`) before CMake runs, so the compiler wrapper cannot inject
  a second BLAS or link FFTW. It passes the LibSci ScaLAPACK link line
  explicitly (CMake's probe expects an MKL layout) with the `_mp` threading
  flavour that matches the SLATE install.
- **CUDA leg.** `build_ffi_cuda.sh` builds against `cudatoolkit/13.2` and the
  clone's `uv sync --extra cuda13` venv, with plain GCC rather than the Cray
  `cc` wrapper (with `craype-accel-nvidia80` loaded the wrapper injects the
  CUDA-12 `libmpi_gtl_cuda`). It stages cuSOLVERMp 0.9.1 and cuBLASMp 0.10
  from their wheels under `LORRAX_BUILD_PREFIX` (default `<clone>/.build`).
  It builds no SLATE, so `slate` on a CUDA mesh refuses when `distrib_la`
  resolves it.
- **Library search path.** Each leg's search path is a `DT_RPATH`
  (`--disable-new-dtags`), which the loader reads before `LD_LIBRARY_PATH`.
  This matters twice: `cudatoolkit/13.2` puts the HPC SDK's cuSOLVERMp 0.8
  and cuBLASMp 0.8 on `LD_LIBRARY_PATH` under the same SONAMEs as the staged
  versions (found first, they fail the first distributed eigh with
  `cusolverMpSyevd status=7`), and `/opt/cray/pe/lib64` maps
  `libhdf5_parallel_gnu.so.310` to the site-default HDF5, which links the
  site-default MPI. The pinned HDF5 and MPICH directories are therefore on
  each leg's RPATH, and a run needs no `LD_LIBRARY_PATH`.
- Build in a zero-GPU compute step
  ([the install page gives the `srun` line](perlmutter.md#build)).

### Frontera

```bash
LORRAX_ROOT=$PWD \
LORRAX_SLATE_HOST_INSTALL_DIR=$WORK/slate_builds/cpu/install \
  config/frontera/build_ffi_host.sh --fresh
```

MKL supplies ScaLAPACK, CBLAS and the FFT. `libmkl_blacs_intelmpi_lp64` must
match the MPI: the wrong BLACS links and fails only inside the first
`blacs_gridinit`. Without `LORRAX_SLATE_HOST_INSTALL_DIR` the recipe builds a
library with parallel HDF5 only and declares that reduced set to the
verifier.

## The verify contract {#verify}

Every build path ends in `scripts/verify_ffi_build.sh [--leg host|cuda] <so>`.

| gate | property | where it can run |
|---|---|---|
| 0 | every backend in `LORRAX_FFI_EXPECT_BACKENDS` exports a handler and the build stamp agrees | anywhere |
| 1 | exactly one MPI runtime in the resolved closure (`libmpi`, `libmpi_gnu`, `libmpi_gnu_<N>`; `gate_one_mpi.sh`, deduplicated by `realpath`) | the run environment |
| 2 | one BLAS vendor and one threading flavour in `DT_NEEDED` | anywhere |
| 3 | the host leg links nothing from the CUDA stack | anywhere |
| 4 | the dependency closure resolves at load time (`ldd -r`) | the run environment |
| 5 | no undefined `fftw_` symbol and no `fftw` in `DT_NEEDED` | anywhere |
| 6 | every OpenMP entry in `DT_NEEDED` is `libgomp`, `libiomp5` or `libomp` | anywhere |
| 7 | one HDF5 SOVERSION, and it is the one the runtime provides (`gate_one_hdf5.sh`) | ELF half anywhere; mapped-object half in the run environment |
| 8 | after one real FFT, exactly one FFTW3 engine is mapped and it is the intended one (`gate_one_fftw.sh`) | a process that imports JAX, never a login node |
| 11 | the exported handler ABI equals `src/ffi/cpp/common/lorrax_ffi_abi.h` | anywhere |

Two gates are properties of the pair, not of one artifact:

| gate | property | where it runs |
|---|---|---|
| 9 | no LORRAX internal on the dynamic table, and every shared `lrx_*` entry carries its leg's suffix | at link time in `src/ffi/cpp/build.sh` and `config/perlmutter/build_ffi_host.sh` |
| 10 | a CUDA-capable process with both libraries open completes host parallel-HDF5 work (`src/ffi/cpp/gate_one_odr.py`) | a GPU node, both pins set |

- `LORRAX_FFI_EXPECT_BACKENDS` defaults to the leg's full backend set, so a
  build that silently loses a backend fails, and a site that builds fewer
  backends must say so.
- A gate that cannot run in the current environment prints `GATE COULD NOT
  RUN` and is counted apart from passes. `LORRAX_FFI_VERIFY_STRICT=1` makes
  that a failure; use it for certification inside an allocation.
- `LORRAX_FFI_VERIFY=off` disables the verifier and announces it on every
  invocation. An unverified library is not deployable.

## Sealing the deployable pair {#seal}

A private build runs unsealed. To publish a pair for other users, seal it
from the checkout that built both legs; each leg's `PROVENANCE` must say
`git_dirty=no` at that checkout's `HEAD`:

```bash
python src/ffi/cpp/stage/seal_bundle.py \
  --cuda path/to/liblorrax_ffi.so \
  --host path/to/liblorrax_ffi_host.so \
  --private-lib path/to/libblaspp.so.2 \
  --private-lib path/to/libslate.so.2 \
  --output path/to/new-bundle-dir
```

The sealer publishes a new directory (it refuses an existing one) holding
both legs and the listed private libraries under `lib/`, and one
`lorrax_ffi_bundle.json`. The manifest binds the pair, the handler ABI, the
full source revision, each file's size and SHA-256, each ELF SONAME and
`DT_NEEDED` list, and the private closure in dependency-first order.

- **Private libraries.** Repeat `--private-lib`, dependency first, for every
  engine-private provider in either leg's `DT_NEEDED` closure. Only
  cuSOLVERMp, cuBLASMp, CAL, SLATE (and its ScaLAPACK API), BLAS++,
  LAPACK++ and NVSHMEM qualify (`lxkit.native_provider.is_private_redistributable`).
  MPI, site HDF5, the CUDA runtime and driver, NCCL, and system and compiler
  libraries belong to the machine runtime and are refused: a copy in the
  bundle would be a second instance of a library the whole process
  shares.
- **At load**, `lxkit.native_provider` rehashes both legs and the closure,
  preloads each private library by exact path (so no run script owns a
  library search path), checks with `dladdr` which file the live ABI symbol
  came from, and refuses any mapped engine-private library the manifest does
  not name.
- **Compatibility.** The manifest's source revision is provenance, not a
  demand that the running checkout have the same commit. Compatibility is the
  exported ABI number, the live feature and target probes, and the build
  contract.

## How a run selects the pair {#loader}

`ffi.common.ffi_loader.get_lib(platform)` (and `distrib_la.loader`, through
the same `lxkit.native_provider` policy) chooses one library per platform:

- **Candidates**, in order: the pin (`LORRAX_FFI_SO` for CUDA,
  `LORRAX_FFI_HOST_SO` for cpu), the in-tree `src/ffi/cpp/build/` or
  `build_host/`, then each `sys.path` directory. A sealed bundle is found
  from its manifest beside the libraries; no variable names the manifest.
- **Pins.** A pin that is not a file refuses. When a selected leg belongs to
  a sealed bundle, both pins or neither must be set and both legs must come
  from one manifest; a partial override or mixed providers refuse.
- **ABI.** A stamped library with a different ABI refuses
  (`HANDLER ABI MISMATCH`). An unstamped one is announced once as
  `LEGACY-UNSEALED` and loads, unless `LORRAX_FFI_ABI_STRICT=1`; an unstamped
  library inside a sealed bundle always refuses.
- **Load order.** In a process that can use CUDA (`JAX_PLATFORMS` does not
  put `cpu` first, and an NVIDIA device is visible), opening the host library
  opens the CUDA library first, so a CUDA-built SLATE and BLAS++, when
  present, win the shared SONAMEs: a host-built BLAS++ reports zero devices
  to a CUDA SLATE. After each `dlopen` the loader refuses a process with more
  than one MPI runtime mapped.
- **Probe.** `probe_target(target, platform)` gives one of three reasons a
  target is unusable: the name is unknown, the library could not be loaded,
  or it loaded without that handler. Every gate refusal quotes it.

## The ABI pairing rule {#abi}

`src/ffi/cpp/common/lorrax_ffi_abi.h` holds one number,
`LORRAX_FFI_ABI_VERSION`. It is compiled into both legs and mirrored by both
Python loaders (`LORRAX_FFI_ABI_VERSION` in `src/ffi/common/ffi_loader.py`);
a drift test compares all three.

Bump it in the same commit as any handler-signature change: adding, removing
or reordering an `Arg` or `Ret`; moving a value between `Attr` and `Arg`;
changing a dtype or rank; or changing the meaning of a positional value while
keeping its type. A new handler needs no bump: an older library does not
export it, and `probe_target` reports that precisely. Every library built
before the bump then refuses with `HANDLER ABI MISMATCH`, naming both
versions and the rebuild command.

## Porting to a new site

Copy the closest site recipe and change values, not structure. The
Perlmutter and Frontera recipes are written section for section in parallel,
so `diff` between them shows only values.

| lever | what it selects | the trap |
|---|---|---|
| `LORRAX_FFI_EXPECT_BACKENDS` | what the build must contain | omitted, it is the full set, so a silent reduction fails |
| the BLAS/ScaLAPACK link line | ScaLAPACK + C-BLACS | CMake's probe expects an MKL layout; elsewhere pass `-DLORRAX_SCALAPACK_LIBRARIES` as a whole link line |
| the BLAS module | which BLAS links | left loaded, the compiler wrapper injects a second flavour (GATE 2) |
| the FFTW module | where the FFT engine lives | it must reach CMake as a `dlopen` hint only; on the link line it becomes `DT_NEEDED` (GATE 5) |
| the HDF5 module (`LORRAX_PM_HDF5` on Cray) | the SOVERSION the host leg links | this, not the stage variable, sets the link |
| the parallel-HDF5 stage (`LORRAX_FFI_PHDF5_DIR`) | the SOVERSION the runtime provides | it feeds only GATE 7's comparison; set it and the HDF5 module to the same version |
| the MPI module | which MPI both legs link | both legs must name the same `libmpi` (GATE 1, GATE 10) |
| the XLA FFI headers | the jaxlib ABI compiled against | take them from the JAX that will load the library |
| the SLATE install | SLATE + BLAS++ + LAPACK++ | the host leg needs `gpu_backend=none` |
| the cuSOLVERMp stage (CUDA leg) | the communication path and correctness | every stage exports the same SONAME; use ≥ 0.7 with `-DLORRAX_FFI_HAVE_CAL=OFF` |
| the MPI include/lib dirs (CUDA leg) | which MPI the library requests | unset, CMake falls back to HPC-X OpenMPI and requests `libmpi.so.40` |
| `CMAKE_CUDA_ARCHITECTURES` (CUDA leg) | the SASS/PTX of the nvcc translation units | NVRTC kernels compile for the device at run time regardless |
