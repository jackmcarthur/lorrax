# FFI native libraries

The distributed code paths (cuSolverMp `eigh`, cuBLASMp matrix operations, sharded
parallel-HDF5 I/O, and SLATE Cholesky / trsm / heev) call into the native stack through
`src/ffi/`. A single
`liblorrax_ffi.so` exposes all of them. None of the native libraries are declared in
`pyproject.toml`; you obtain an ABI-matched stack separately, then build the `.so` against
it. The wheel includes the CMake, C++/CUDA, header, and staging-script sources needed for
that build, but never a prebuilt `.so` or vendor library.

!!! warning "Fresh installs have no `liblorrax_ffi.so`"
    It is a platform-specific build artifact. The pure-JAX path never needs it; the first
    distributed / FFI-I/O call without it raises `FileNotFoundError` and reports the
    packaged `build.sh` path.

!!! warning "CUDA support boundary"
    The Python package defaults to `jax[cuda13]>=0.9.0`, and its pure-JAX path is the
    current CUDA-13 install track. The native FFI stack below is validated on Perlmutter
    with CUDA 12.9. A CUDA-13 FFI build must use CUDA-13-compatible cuSolverMp,
    cuBLASMp, CAL/NCCL, and compiler/runtime components throughout; the CUDA-12 stage
    described below must not be loaded into that process. CUDA-13 native FFI remains an
    unvalidated porting target.

!!! note "Portability status"
    These recipes are assembled from the portability and dependency-architecture reviews.
    The provider-agnostic (non-Cray) variants are documented but **untested** on a
    non-NERSC cluster. The Cray-PE path (stage scripts) is the validated one on Perlmutter.

## 1. cuSolverMp + CAL

**Validated CUDA-12.9 baseline: PyPI wheel.** The current reference is
`nvidia-cusolvermp-cu12==0.7.2.888` plus the matching `nvidia-cal-cu12`. Version matters:
0.6.0 silently returns **wrong answers** on $P_x>1$ and $P_y>1$ meshes; 0.7.2 includes the
CAL→NCCL ABI fix and the race-condition follow-up (see comments in
`config/perlmutter/site_config.sh`).

```bash
pip install nvidia-cusolvermp-cu12==0.7.2.888 nvidia-cal-cu12
# then point CMake at the wheel's include/lib (see the build step below):
#   -DCUSOLVERMP_INCLUDE_DIR=<site-packages>/nvidia/.../include
#   -DCUSOLVERMP_LIB_DIR=<site-packages>/nvidia/.../lib
```

**Alternative: NVHPC SDK** (spack `nvhpc`, or the tarball from developer.nvidia.com) → use
`src/ffi/cusolvermp/scripts/stage_nvhpc.sh` with `NVHPC_ROOT` pointed at the install.

For a CUDA-13 port, install a CUDA-13 cuSolverMp/cuBLASMp/CAL distribution from one
provider, point CMake at those headers and libraries explicitly, and run the FFI contract
tests before enabling a distributed backend. No pinned CUDA-13 native recipe is claimed
here yet.

## 2. Parallel HDF5

The CMake check enforces `HDF5_IS_PARALLEL`; a serial HDF5 will be rejected with a clear
error.

**Preferred (non-Cray): conda-forge or spack.**

```bash
# conda-forge, OpenMPI-flavored build:
conda install -c conda-forge 'hdf5=*=mpi_openmpi_*' openmpi
# or spack:
spack install hdf5+mpi
```

Build the FFI against it directly with `-DHDF5_ROOT=<prefix>` — no staging / SONAME
shimming needed off-container.

**Cray:** load `cray-hdf5-parallel` and stage it with
`src/ffi/phdf5/scripts/stage_cray.sh`. The OpenMPI stage script
(`src/ffi/phdf5/scripts/stage_openmpi.sh`) is the portable stack for non-Cray clusters.

## 3. SLATE (+ blaspp + lapackpp)

Build from source (blaspp/lapackpp build as part of the superbuild and land under the same
prefix):

```bash
git clone --recurse-submodules https://github.com/icl-utk-edu/slate
cd slate
cmake -B build -S . \
    -Dgpu_backend=cuda \
    -Dblas=openblas \
    -DCMAKE_INSTALL_PREFIX=$HOME/software/slate/install
cmake --build build -j && cmake --install build
```

!!! note
    The BLAS backend is a free choice — Cray uses `libsci`; elsewhere OpenBLAS or MKL is
    fine. SLATE does not require libsci (that is a Perlmutter performance choice). Point the
    build at the install prefix with `-DLORRAX_SLATE_INSTALL_DIR=$HOME/software/slate/install`.

## 4. Build `liblorrax_ffi.so` (non-Shifter)

From a Git checkout or source distribution the build script is
`src/ffi/common/cpp/build.sh`. In a wheel install, locate the same packaged source with:

```bash
python -c 'from pathlib import Path; import ffi; print(Path(ffi.__file__).parent / "common/cpp/build.sh")'
```

On NERSC the launcher `src/ffi/common/cpp/run_shifter.sh` sets the MPI env vars and runs
`build.sh` inside the container (see [Perlmutter](perlmutter.md)). Off-container, drive
CMake directly with explicit `-D` overrides — the CMake config already supports pointing at
arbitrary install locations:

```bash
cd src/ffi/common/cpp
cmake -B build -S . \
    -DCUSOLVERMP_INCLUDE_DIR=<cusolvermp-include> \
    -DCUSOLVERMP_LIB_DIR=<cusolvermp-lib> \
    -DHDF5_ROOT=<hdf5-parallel-prefix> \
    -DLORRAX_SLATE_INSTALL_DIR=$HOME/software/slate/install \
    -DLORRAX_MPI_INCLUDE_DIR=<mpi-include> \
    -DLORRAX_MPICH_LIB_DIR=<mpi-lib>
cmake --build build -j
# -> build/liblorrax_ffi.so ; the loader (ffi_loader.py) searches cpp/build/,
#    $LORRAX_FFI_SO, and sys.path.
```

For an OpenMPI / non-Cray build, `build.sh` accepts `LORRAX_FFI_ALLOW_DEFAULT_MPI=1` so it
does not hard-require the Cray-MPICH env vars; set `LORRAX_PHDF5_MPI_STACK=openmpi`
accordingly (this variable is consumed by the build pipeline — run_shifter.sh / CMake — not
by `build.sh` itself). The build-time and runtime MPI stacks **must match** — a mismatch surfaces as
a runtime SONAME error / segfault, not a clean message.

## See also

- `src/ffi/PORTING.md` — the full FFI porting checklist (in the source distribution and wheel)
- `src/ffi/AGENTS.md` — FFI subpackage entry points
- [`docs/ENVIRONMENT_COMPREHENSIVE.md` §5](../ENVIRONMENT_COMPREHENSIVE.md) — the FFI stack
  reference (bind-mounts, staging, MPI override)
