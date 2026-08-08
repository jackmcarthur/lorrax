# FFI native libraries

The distributed code paths (cuSolverMp `eigh`, sharded parallel-HDF5 I/O, SLATE Cholesky /
trsm / heev) call into three native libraries through `src/ffi/`. A single
`liblorrax_ffi.so` exposes all of them. None of these are declared in `pyproject.toml`; you
obtain them separately, then build the `.so` against them.

!!! warning "Fresh clone has no `liblorrax_ffi.so`"
    It is a gitignored build artifact. The pure-JAX path never needs it; the first
    distributed / FFI-I/O call without it fails with
    a `FileNotFoundError` naming the platform's build recipe:
    `src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh` for the CUDA
    library, `bash src/ffi/cpp/build_host.sh` for the host one.

!!! note "TODO"
    These recipes are assembled from the portability and dependency-architecture reviews.
    The provider-agnostic (non-Cray) variants are documented but **untested** on a
    non-NERSC cluster. The Cray-PE path (stage scripts) is the validated one on Perlmutter.

## 1. cuSolverMp + CAL

**Preferred (any cluster): PyPI wheel.** The validated baseline is
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
`src/ffi/cpp/stage/cusolvermp_stage_nvhpc.sh` with `NVHPC_ROOT` pointed at the install.

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

Build the FFI against it directly with `-DHDF5_ROOT=<prefix>` — no staging and no
library-name shimming needed off-container.

**Cray:** load `cray-hdf5-parallel` and stage it with
`src/ffi/cpp/stage/phdf5_stage_cray.sh`. The OpenMPI stage script
(`src/ffi/cpp/stage/phdf5_stage_openmpi.sh`) is the portable stack for non-Cray clusters.

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

On NERSC the launcher `src/ffi/cpp/run_shifter.sh` sets the MPI env vars and runs
`build.sh` inside the container (see [Perlmutter](perlmutter.md)). Off-container, drive
CMake directly with explicit `-D` overrides — the CMake config already supports pointing at
arbitrary install locations:

```bash
cd src/ffi/cpp        # the tree's ONE CMakeLists.txt lives here
cmake -B build -S . \
    -DCUSOLVERMP_INCLUDE_DIR=<cusolvermp-include> \
    -DCUSOLVERMP_LIB_DIR=<cusolvermp-lib> \
    -DHDF5_ROOT=<hdf5-parallel-prefix> \
    -DLORRAX_SLATE_INSTALL_DIR=$HOME/software/slate/install \
    -DLORRAX_MPI_INCLUDE_DIR=<mpi-include> \
    -DLORRAX_MPICH_LIB_DIR=<mpi-lib>
cmake --build build -j
# -> build/liblorrax_ffi.so ; the loader (ffi_loader.py) reads $LORRAX_FFI_SO
#    FIRST, then falls back to the in-tree cpp/build/ (cpp/build_host/ for the
#    host library).  A set-but-missing pin REFUSES rather than falling through.
#    services/distrib_la opens the same two .so files through its own loader —
#    see docs/distributed_linalg.md.
```

For an OpenMPI / non-Cray build, `build.sh` accepts `LORRAX_FFI_ALLOW_DEFAULT_MPI=1` so it
does not hard-require the Cray-MPICH env vars; set `LORRAX_PHDF5_MPI_STACK=openmpi`
accordingly (this variable is consumed by the build pipeline — run_shifter.sh / CMake — not
by `build.sh` itself). The build-time and runtime MPI stacks **must match** — a mismatch surfaces as a
segfault or a "cannot open shared object file" at start-up, not a clean message.

## See also

- `src/ffi/PORTING.md` — the full FFI porting checklist (in the repo)
- `src/ffi/AGENTS.md` — FFI subpackage entry points
- [`docs/environment/machines/perlmutter.md` §2](../environment/machines/perlmutter.md) — the FFI stack
  reference (bind-mounts, staging, MPI override)
