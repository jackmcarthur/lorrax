# FFI native libraries

This page lists where each native dependency of the FFI pair comes from on a
site with no LORRAX module. Read [Building the FFI libraries](../building_ffi.md)
first: it owns the verify contract, the sealing step and the porting levers.
None of these dependencies are declared in `pyproject.toml` except
`nvidia-mathdx`.

| dependency | minimum |
|---|---|
| NVIDIA GPU | sm_80 (the CUDA leg's nvcc TUs carry SASS through sm_120) |
| CUDA toolkit | 13.0; the library's CUDA major must match the JAX wheel's |
| JAX and jaxlib | 0.9.x, one generation (`pyproject.toml`) |
| cuSOLVERMp | 0.7 (NCCL-native); 0.8 and later need NCCL 2.27 |
| parallel HDF5 | 1.12 |
| nvidia-mathdx | 25.6.0 (the wheel; not a build dependency) |

## 1. cuSOLVERMp and cuBLASMp (CUDA leg)

Stage scripts, one per source:

- `src/ffi/cpp/stage/cusolvermp_stage_pypi.sh`: the standalone
  `nvidia-cusolvermp-cu12` wheel (`CUSOLVERMP_VERSION`, default 0.7.2.888).
- `src/ffi/cpp/stage/cusolvermp_stage_nvhpc.sh`: from an NVHPC SDK install.
- `src/ffi/cpp/stage/cusolvermp_stage_cublasmp_redist.sh`: the cuBLASMp
  redistributable.

Every stage exports the same SONAME, `libcusolverMp.so.0`, so the stage a leg
is built against must be the stage its runs load. `src/ffi/cpp/build.sh`
refuses until `LORRAX_NVHPC_ROOT` (or `LORRAX_NVHPC_SUBPATH`) names it.
cuSOLVERMp 0.6.0 returns wrong `getrf`/`getrs` results on any mesh with both
P_x > 1 and P_y > 1; use ≥ 0.7. Versions ≥ 0.7 are NCCL-native, ship no
`cal.h`, and need `-DLORRAX_FFI_HAVE_CAL=OFF`. Versions ≥ 0.8 need NCCL ≥ 2.27.

## 2. Parallel HDF5 (both legs)

CMake refuses an HDF5 without `HDF5_IS_PARALLEL`. Point it at the install with
`-DHDF5_ROOT=<prefix>` (or `$HDF5_ROOT` / `$HDF5_DIR`).

- Cray: load `cray-hdf5-parallel` and stage it with
  `src/ffi/cpp/stage/phdf5_stage_cray.sh`.
- OpenMPI sites: `src/ffi/cpp/stage/phdf5_stage_openmpi.sh`, or an MPI build
  from conda-forge (`hdf5=*=mpi_openmpi_*`) or spack (`hdf5+mpi`).

The host leg must link the same HDF5 SOVERSION the runtime provides (GATE 7).

## 3. SLATE, BLAS++ and LAPACK++

Build SLATE from source; BLAS++ and LAPACK++ install under the same prefix. The
host leg needs a `gpu_backend=none` install (`LORRAX_SLATE_HOST_INSTALL_DIR`,
default `$HOME/software/slate_builds/cpu/install`). The CUDA leg builds
without SLATE (the `lorrax_A` bundle has none); a `gpu_backend=cuda` install
at `LORRAX_SLATE_INSTALL_DIR` adds its CUDA handlers. On a Cray PE,
`src/ffi/cpp/stage/slate_build_perlmutter.sh cpu|gpu` builds both
reproducibly. Elsewhere:

```bash
git clone --recurse-submodules https://github.com/icl-utk-edu/slate
cmake -S slate -B slate/build -Dgpu_backend=none -Dblas=openblas \
    -DCMAKE_INSTALL_PREFIX=$HOME/software/slate_builds/cpu/install
cmake --build slate/build -j && cmake --install slate/build
```

Any BLAS works (LibSci, OpenBLAS, MKL), provided the host leg links exactly one
BLAS flavour (GATE 2).

## 3a. FFTW and nvidia-mathdx

- The host leg resolves FFTW at run time. The FFTW prefix reaches CMake only as
  a dlopen hint, never as `DT_NEEDED` (GATE 5); on Cray,
  `src/ffi/cpp/stage/fftw_stage_cray.sh` stages it.
- The CUDA leg's k-convolution router needs the Python package
  `nvidia-mathdx==25.6.0`, pinned in the `cuda12`/`cuda13` extras of
  `pyproject.toml`. It is header-only and nothing is linked: kernels are
  compiled per k-grid with NVRTC at run time.

## 4. Build `liblorrax_ffi.so` with CMake

Drive CMake on `src/ffi/cpp/` with the platform selector and explicit paths.
`LORRAX_FFI_PLATFORM` is required; unset, CMake stops with an error naming both
legs. The XLA FFI headers must come from the JAX that will load the library:
the CUDA leg reads `jax.ffi.include_dir()` from the `python3` on `PATH`, and the
host leg takes `-DLORRAX_XLA_FFI_INCLUDE_DIR` or probes the same way.

```bash
cmake -S src/ffi/cpp -B build_cuda \
    -DLORRAX_FFI_PLATFORM=cuda \
    -DCUSOLVERMP_INCLUDE_DIR=<dir with cusolverMp.h> \
    -DCUSOLVERMP_LIB_DIR=<dir with libcusolverMp.so> \
    -DLORRAX_FFI_HAVE_CAL=OFF \
    -DHDF5_ROOT=<parallel HDF5 prefix> \
    -DLORRAX_SLATE_INSTALL_DIR=<gpu_backend=cuda SLATE prefix> \
    -DLORRAX_MPI_INCLUDE_DIR=<dir with mpi.h> \
    -DLORRAX_MPICH_LIB_DIR=<dir with libmpi> \
    -DCMAKE_CUDA_ARCHITECTURES=<sm of the target GPU>
cmake --build build_cuda -j
scripts/verify_ffi_build.sh --leg cuda build_cuda/liblorrax_ffi.so
```

The host leg is the same command with `-DLORRAX_FFI_PLATFORM=host`,
`-DLORRAX_SLATE_HOST_INSTALL_DIR=<gpu_backend=none prefix>` and no CUDA
options. Then seal the two legs
([Building the FFI libraries](../building_ffi.md#seal-the-deployable-pair)).

- Without `LORRAX_MPICH_LIB_DIR`, CMake warns and falls back to
  `/opt/hpcx/ompi/lib`, and the library requests `libmpi.so.40`.
- `src/ffi/cpp/build.sh` refuses to run without `LORRAX_MPI_INCLUDE_DIR` and
  `LORRAX_MPICH_LIB_DIR`; `LORRAX_FFI_ALLOW_DEFAULT_MPI=1` lifts that for an
  OpenMPI build.
- The build-time and run-time MPI must match. A mismatch appears as a
  segfault or a "cannot open shared object file" error at startup, not as a
  named refusal.

## See also

- [Kernel catalog](../architecture/ffi_layout.md#kernel-catalog): every
  target, its source file, door, selection rule and gate.
