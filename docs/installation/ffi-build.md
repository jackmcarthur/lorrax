# Building the FFI libraries

LORRAX reaches every vendor library (cuSOLVERMp, cuBLASMp, cuFFT, ScaLAPACK,
FFTW, parallel HDF5) through two shared objects built from the one C++
tree `src/ffi/cpp/`. This page is for whoever builds them: what each needs,
how to build and verify a pair on Perlmutter, Frontera or a new site, how a
pair is sealed for other users, how a run selects one, and the ABI rule that
pairs a library with a Python tree. Read [Installation](index.md) first; the
design of the layer (its targets, kernels and failure modes) is
[the FFI layer](../architecture/ffi_layout.md).

## The two legs, and why they must agree

| leg | file | contains |
|---|---|---|
| host | `liblorrax_ffi_host.so` | ScaLAPACK, a CBLAS GEMM, an FFT through the FFTW3 ABI bound at run time, parallel HDF5. CUDA-free. |
| CUDA | `liblorrax_ffi.so` | cuSOLVERMp, cuBLASMp, cuBLAS, cuSOLVER, cuFFT, NCCL, the NVRTC-built mathdx k-convolution and Fourier-plan kernels, the small nvcc kernels, parallel HDF5. |

Every target and its source file is in the
[kernel catalog](../architecture/ffi_layout.md#kernel-catalog).

A GPU run loads both legs into one process with `dlopen(RTLD_GLOBAL)`. A leg
can therefore be correct alone and still unusable beside its partner. Two
consequences shape every build:

1. **One MPI.** Both legs must link the same `libmpi`. Two MPIs in one
   process bind some MPI calls to one and some to the other, and the run
   hangs in `MPI_Init` or a collective.
2. **One HDF5.** The host leg must link the HDF5 SOVERSION that the run
   environment maps, so that both legs call one HDF5 library (GATE 7).

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
- **FFTW.** The host leg resolves its FFT engine at first use through the
  FFTW3 ABI ([which engine binds](../architecture/ffi_layout.md#3c-which-fft-engine-the-host-library-binds)).
  The FFTW prefix may reach CMake only as a `dlopen` hint, never as a
  `DT_NEEDED` entry, so that an MKL site's already-loaded FFTW3 export wins.
- **nvidia-mathdx.** Header-only; the `cuda12`/`cuda13` extras pin it. NVRTC
  compiles the k-convolution and Fourier-pair kernels from its headers at run
  time ([compilation §2](../architecture/compilation.md#2-native-kernels-and-nvrtc-images)),
  so nothing is linked.

## Building a leg

| site | leg | entry |
|---|---|---|
| Perlmutter | host | `config/perlmutter/build_ffi_host.sh` |
| Perlmutter | CUDA | `config/perlmutter/build_ffi_cuda.sh` |
| Perlmutter module | both | `config/perlmutter/module/build.sh`, which runs the two scripts above and seals ([publishing a module](perlmutter.md#publish)) |
| Frontera | host | `config/frontera/build_ffi_host.sh` |
| a workstation, laptop or rented GPU box with only a driver | both | `config/local/build_ffi.sh`, `config/local/build_ffi_host.sh` (`config/local/README.md`) |
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
    -DLORRAX_MPICH_LIB_DIR=<dir with libmpi>
cmake --build build_cuda -j
scripts/verify_ffi_build.sh --leg cuda build_cuda/liblorrax_ffi.so
```

The host leg is the same command with `-DLORRAX_FFI_PLATFORM=host` and no
CUDA options. The CUDA leg reads the XLA FFI headers from
`jax.ffi.include_dir()` of the `python3` on `PATH`; the host leg takes
`-DLORRAX_XLA_FFI_INCLUDE_DIR` or probes the same way.

- `CMAKE_CUDA_ARCHITECTURES` defaults to SASS for sm_80, 86, 89, 90, 100 and
  120 plus compute_80 and compute_120 PTX, so the example leaves it unset. It
  governs only the nvcc translation units; NVRTC kernels compile for the
  running device.
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
also needs) and the LibSci that links it,
`cray-libsci/25.09.0`, with darshan unloaded. To move to another MPI, change
that file only.

- **Host leg.** `build_ffi_host.sh` runs bare metal under `PrgEnv-gnu`: it
  pins the MPI, loads `cray-hdf5-parallel/1.14.3.7`, captures the LibSci and
  cray-fftw prefixes and unloads those modules (with `craype-accel-nvidia80`
  and `cudatoolkit`) before CMake runs, so the compiler wrapper cannot inject
  a second BLAS or link FFTW. It passes the LibSci ScaLAPACK link line
  explicitly (CMake's probe expects an MKL layout) with the measured `_mp`
  threading flavour (`LORRAX_PM_LIBSCI_FLAVOUR`).
- **CUDA leg.** `build_ffi_cuda.sh` builds against `cudatoolkit/13.2` and the
  clone's `uv sync --extra cuda13` venv, with plain GCC rather than the Cray
  `cc` wrapper (with `craype-accel-nvidia80` loaded the wrapper injects the
  CUDA-12 `libmpi_gtl_cuda`). It stages cuSOLVERMp 0.9.1 and cuBLASMp 0.10
  from their wheels under `LORRAX_BUILD_PREFIX` (default `<clone>/.build`).
- **Library search path.** Each leg's only search path is `$ORIGIN`, a
  `DT_RUNPATH`, so the run environment decides every machine library.
  `/opt/cray/pe/lib64` maps `libhdf5_parallel_gnu.so.310`, the LibSci
  SONAMEs and `libmpi_gnu.so.12` to the site defaults, which link a second
  MPI; `config/perlmutter/ffi_mpi.sh` therefore lists the pinned HDF5,
  cray-mpich and LibSci directories (`LORRAX_PM_RUNTIME_LIB_DIRS`), and the
  module puts them on `LD_LIBRARY_PATH` ahead of it, as both legs' verify
  steps do. `cudatoolkit/13.2` also puts the HPC SDK's cuSOLVERMp 0.8 and
  cuBLASMp 0.8 on `LD_LIBRARY_PATH` under the same SONAMEs as the staged
  0.9.1 / 0.10 (found first, they fail the first distributed eigh with
  `cusolverMpSyevd status=7`). A sealed bundle carries the staged pair
  beside the legs and preloads it by exact path, so a run should use a
  sealed pair; an unsealed build finds its private providers only on
  `LD_LIBRARY_PATH`.
- Build in a zero-GPU compute step
  ([the install page gives the `srun` line](perlmutter.md#build)).

### Frontera

```bash
LORRAX_ROOT=$PWD LORRAX_HOST_SCALAPACK=1 \
  config/frontera/build_ffi_host.sh --fresh
```

MKL supplies ScaLAPACK, CBLAS and the FFT. `libmkl_blacs_intelmpi_lp64` must
match the MPI: the wrong BLACS links and fails only inside the first
`blacs_gridinit`. Without `LORRAX_HOST_SCALAPACK=1` the recipe builds a
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

## The dependency closure: recorded at build, enforced at start-up {#closure}

A leg's search path is `$ORIGIN` alone, a `DT_RUNPATH`, which
`LD_LIBRARY_PATH` outranks. Without a record, the launch shell would decide which
file serves each SONAME, and it has: on Perlmutter `cudatoolkit/13.2` puts the
HPC SDK's cuSOLVERMp 0.8 under the SONAME of the staged 0.9.1 the CUDA leg is
compiled against, and the site-default LibSci/HDF5 map cray-mpich 9.1.0 beside
the pinned 9.0.1. So every build records the leg's dependency closure and
every start-up enforces it, on every route (a clone, the module, a sealed
bundle) and for every leg (CUDA, ROCm, host).

**Recorded.** Every build script ends by writing `<leg>.closure.json` beside the
leg (`src/ffi/cpp/stage/write_checkout_closure.py`): the leg's SHA-256 and every
library of its `DT_NEEDED` closure, transitively, with path, size, mtime,
SHA-256, `DT_SONAME` and family. Each name resolves from the first of:

1. the engine-private stage (cuSOLVERMp/cuBLASMp; ELPA on ROCm);
2. the directories the venv's JAX plugin loads its GPU libraries from (its
   RPATH/RUNPATH: the CUDA wheels, or the ROCm SDK wheels). JAX maps those
   before any leg opens, so the leg must bind the same copy;
3. the pinned machine directories: MPI, parallel HDF5, LibSci (`ffi_mpi.sh`),
   and on Perlmutter the toolkit's math and NCCL libraries, which serve a venv
   without CUDA wheels such as the module's;
4. the needing library's own RPATH/RUNPATH.

The process, compiler and driver runtime (libc, libstdc++, libgomp, libcuda)
and libraries reached only through the system search path (libfabric, libz)
are listed as `external` and not enforced. An MPI, HDF5, LibSci, GPU
math/collective or engine-private library found in none of the four refuses
the build. `PROVENANCE` carries the record's `closure_sha256`, and
`seal_bundle.py` copies both records into the manifest ([sealing](#seal)).

**Enforced.** `lxkit.native_provider.open_and_attest`, before the leg can run
anything:

1. refuses a leg without a record, a record of another schema, a record
   written for other leg bytes, and a recorded file that is gone, has another
   size, or (when its mtime moved) another SHA-256;
2. opens every recorded file by exact path, `RTLD_GLOBAL`, dependencies first,
   so the recorded file holds each SONAME before anything searches for it;
3. opens the leg, then reads `/proc/self/maps` and groups the mapped objects by
   family (from their `DT_SONAME`). Every recorded family must be mapped
   exactly once, at the recorded path or byte-identical to the recorded file.

A wrong file, a second copy of a family (the one-MPI check, generalized to
every recorded family) or a recorded family that is not mapped refuses. There
is no fallback and no warning-and-continue. MPI is one family whatever the
SONAME (`libmpi`, `libmpi_gnu`, `libmpi_gnu_123`, `libmpich`), as are LibSci,
LibSci-MPI and HDF5 across their compiler and threading variants; any other
library is its SONAME up to `.so`. h5py's auditwheel `libhdf5-<hash>.so`, a
private SONAME that no leg can bind, is therefore not a second HDF5.

One process maps one file per family across its legs, and any disagreement
between the legs' records refuses. "First" is the order the legs' `dlopen`s
succeeded, recorded at that moment in one registry that every loader
(`ffi.common.ffi_loader`, `distrib_la.loader`) shares, not the order their
checks finished.

**The refusal and its remedy.** Every refusal is one line:

```text
GATE native-closure: got <what is mapped or found>; want <what the record says>; fix: <remedy>
```

| got | cause | fix |
|---|---|---|
| `<path> serving libcusolverMp.so.0` with the staged file wanted | something mapped another copy before the leg opened: an `LD_PRELOAD`, a Python extension, a package that loads a module's library | remove what loads it; `LD_LIBRARY_PATH` neither causes nor fixes this. If the recorded file is the one that should change, rebuild the leg in the environment you run in |
| `2 copies of mpi mapped: …` (or of HDF5, NCCL, cuBLAS, rocBLAS, …) | a second copy beside the recorded one, for example mpi4py or a site LibSci linked against another cray-mpich | the same |
| `no closure record for …` | a leg built before this rule, or by a script that does not record | rebuild with the site's build script |
| `stale closure record …` | the leg was rebuilt without its record, or a recorded file moved, was purged or was replaced (a module upgrade, a `uv sync` that changed the CUDA or ROCm wheels, a purged `.build`) | rebuild the leg |
| `<leg> records mpi as … but <leg> (opened first) records …` | the two legs were built against different library sets | rebuild both legs in one environment |
| `libcudart.so on LD_LIBRARY_PATH resolves to <file> …: a second copy of libcudart` (multi-node) | the site transport would dlopen a runtime other than the recorded one (below) | source the site's launch script (`config/perlmutter/gpu_env.sh`, `config/frontier/rocm_env.sh`) before `srun` |

**Before the first collective.** A GPU process opens its device leg as
start-up step 5e, after the backend exists and before the mesh creates the
first NCCL/RCCL communicator. That is when the collective library dlopens the
site network plugin, and the plugin binds its own runtime by SONAME
(Perlmutter's aws-ofi-nccl: `NEEDED libcudart.so.13`, no RUNPATH). With the
closure already mapped it binds the recorded copy.

**Launch libraries.** The site transport also dlopens some libraries by an
unversioned name, which matches no loaded SONAME and so resolves only through
the launch's `LD_LIBRARY_PATH`: libfabric's CUDA memory interface opens
`libcudart.so`, its ROCr one `libhsa-runtime64.so`. `site.toml` lists them as
`launch_libraries`. The site's launch script links each name to the run
python's wheel runtime, first on `LD_LIBRARY_PATH` (`gpu_env.sh`,
`rocm_env.sh`); a python without that wheel (the Perlmutter module, which
records the toolkit runtime) leaves the path as the module sets it. On a
multi-node step, step 5e refuses when a name's first match is not the recorded
file of its family (`native_provider.check_launch_libraries`), before the
transport can map a second runtime. This is the one place where
`LD_LIBRARY_PATH` decides a mapped library, so it is checked rather than
trusted.

The check runs once per leg at start-up, and once more at the first
`native_closure_identity()` call. It costs stat calls and the ELF headers of
the mapped objects, plus a SHA-256 only on a mismatch. It touches no device
and sets no variable.

**Identity and caches.** `lxkit.native_provider.native_closure_identity(platform)`
is the SHA-256 of the enforced record (paths and content hashes) and the files
verified mapped for it, computed once at enforcement. It keys every
persistent cache that depends on native libraries, so a different library set
never reuses an entry:

- the JAX compile-cache namespace (`…_ffi-<12 hex>_k2`); an explicit
  `ISDF_JAX_CACHE_DIR` is split by the same tag;
- the k-convolution NVRTC/hipRTC image store and the rocFFT kernel store;
- the sealed `bundle_id` and the build's `PROVENANCE`.

`python -m lxkit.closure verify <leg>` checks a record without loading
the leg and prints the identity it promises (exit 3 on a refusal), for
build-reuse tooling ([the FFI source key](../architecture/startup_environment.md#ffi-source-key)).

## Sealing the deployable pair {#seal}

A private build runs unsealed. To publish a pair for other users, seal it
from the checkout that built both legs; each leg's `PROVENANCE` must say
`git_dirty=no` at that checkout's `HEAD`:

```bash
python src/ffi/cpp/stage/seal_bundle.py \
  --cuda path/to/liblorrax_ffi.so \
  --host path/to/liblorrax_ffi_host.so \
  --private-lib path/to/libcublasmp.so.0 \
  --private-lib path/to/libcusolverMp.so.0 \
  --output path/to/new-bundle-dir
```

The sealer publishes a new directory (it refuses an existing one) holding
both legs and the listed private libraries under `lib/`, and one
`lorrax_ffi_bundle.json`. It refuses a leg whose search path is not one
`DT_RUNPATH` starting with `$ORIGIN`, and any library in the bundle that
searches a relative directory, a scratch directory (`/pscratch`, `/tmp`,
`$SCRATCH`, through symlinks too) or a build tree (`.build`, `build`,
`build_host`, `CMakeFiles`, or the sealing checkout). The manifest binds the pair, the handler ABI, the
full source revision, each file's size and SHA-256, each ELF SONAME and
`DT_NEEDED` list, and the private closure in dependency-first order.

- **Private libraries.** Repeat `--private-lib`, dependency first, for every
  engine-private provider in either leg's `DT_NEEDED` closure. Only
  cuSOLVERMp, cuBLASMp, CAL, NVSHMEM and ELPA qualify (`lxkit.native_provider.is_private_redistributable`).
  MPI, site HDF5, the CUDA runtime and driver, NCCL, and system and compiler
  libraries belong to the machine runtime and are refused: a copy in the
  bundle would be a second instance of a library the whole process
  shares.
- **Closure.** Each leg's build record is copied into the manifest's
  `closures` (schema 2). Its private entries point at the sealed copies, which
  must be byte-identical to the files the build recorded. A machine or JAX
  plugin file the record names may not live under scratch or a build tree. A
  `--private-lib` that neither record names refuses. A schema-1 bundle (no
  closures) refuses at load: reseal.
- **At load**, `lxkit.native_provider` rehashes both legs and the private
  libraries, enforces the leg's recorded closure ([the dependency
  closure](#closure)) exactly as for a checkout leg, checks with `dladdr`
  which file the live ABI symbol came from, and refuses any mapped
  engine-private library the manifest does not name.
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
- **Load order.** In a process that can use CUDA (`JAX_PLATFORMS` unset or
  naming `cuda` or `gpu` first, and an NVIDIA device is visible), opening the host library
  opens the CUDA library first (kept pending re-measurement). Each leg opens over its recorded dependency closure, and
  after each `dlopen` the loader refuses a process that maps a wrong file or a
  second copy of any recorded family ([the dependency closure](#closure)),
  more than one MPI runtime included.
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
| the cuSOLVERMp stage (CUDA leg) | the communication path and correctness | every stage exports the same SONAME; use ≥ 0.7 with `-DLORRAX_FFI_HAVE_CAL=OFF` |
| the MPI include/lib dirs (CUDA leg) | which MPI the library requests | unset, CMake falls back to HPC-X OpenMPI and requests `libmpi.so.40` |
| `CMAKE_CUDA_ARCHITECTURES` (CUDA leg) | the SASS/PTX of the nvcc translation units | NVRTC kernels compile for the device at run time regardless |
