# Installation

LORRAX has two layers: a **pure-JAX core** (centroids, wavefunction loading, serial GW)
that needs only Python + JAX, and a **native FFI stack** (cuSolverMp, parallel HDF5, SLATE)
required for distributed `eigh`, sharded HDF5 I/O, and SLATE linear algebra. The public
Python package defaults to JAX >=0.9 with CUDA 13; the validated native reference remains
the CUDA-12.9 Perlmutter container. Pick a track from the matrix below.

!!! note "TODO"
    This page is a Tier-2 scaffold assembled from the portability and
    dependency-architecture reviews. The support-matrix rows below marked *untested*
    have not been validated on a non-NERSC cluster; treat them as targets, not guarantees.

## Support matrix

| Config | OS | CUDA | JAX | MPI | parallel HDF5 | Runtime | FFI features | Tested |
|---|---|---|---|---|---|---|---|---|
| **Public package, pure JAX** | Linux x86_64 | 13 | >=0.9 | - | - | bare venv | none (serial only) | package default; FFI not exercised |
| **CPU execution, pure JAX** | Linux / macOS | none | >=0.9 | - | - | bare venv | none (serial only) | CPU tests |
| **NERSC Perlmutter (native reference)** | SLES 15 | 12.9 | NVIDIA JAX 25.04 image | Cray MPICH | cray-hdf5-parallel | Shifter | all | 1-4 nodes x 4 A100 |
| **Generic Cray EX native port** | - | site-matched | site-matched | Cray MPICH | parallel HDF5 | Apptainer | target: all | untested |
| **Generic SLURM + OpenMPI native port** | Linux x86_64 | site-matched | site-matched | OpenMPI 4/5 + UCX | parallel HDF5 | Apptainer / bare venv | target: all | untested |

Only the pure-JAX path works with **zero native libs**. Everything distributed needs the
[FFI native-library stack](ffi-native-libs.md).

!!! warning "The `liblorrax_ffi.so` build cliff"
    A fresh install has **no** `liblorrax_ffi.so` (it is a platform-specific build artifact).
    The pure-JAX path never touches it (all FFI imports are lazy), but the first time you
    run a distributed / FFI-I/O code path the loader reports the packaged `build.sh` path.
    Build the native library per [FFI native libraries](ffi-native-libs.md) before
    using those features.

## Track 1 - pip / uv (CUDA 13 default, no FFI)

Gives you centroids, wavefunction loading, and serial GW. **Not** distributed `eigh`,
sharded HDF5, or SLATE.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # one-time
git clone <lorrax-repo-url> && cd lorrax
uv sync                                            # installs jax[cuda13]>=0.9
uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in
```

The same Python code can run on JAX's CPU backend. The project dependency still installs
the CUDA-13 plugin on supported Linux systems, so `JAX_PLATFORMS=cpu` selects CPU execution
but is not a minimal CPU-only dependency set:

```bash
JAX_PLATFORMS=cpu uv run python -m gw.gw_jax -i <cohsex.in>
```

!!! warning "CUDA 13 Python does not imply CUDA 13 FFI validation"
    `jax[cuda13]>=0.9.0` is the public-package default for the pure-JAX layer. The
    cuSolverMp/cuBLASMp/parallel-HDF5/SLATE bridge has only been validated in the
    Perlmutter CUDA-12.9 environment. Do not mix its CUDA-12 staged libraries into a
    CUDA-13 process; port and rebuild the complete native stack against one matching ABI.

## Track 2 — container

The validated native configuration runs inside the NVIDIA JAX image
(`nvcr.io/nvidia/jax:25.04-py3`). On NERSC the
container runtime is **Shifter** (see [Perlmutter](perlmutter.md)); on other clusters use
**Apptainer** or **Singularity**. The container provides JAX + CUDA; LORRAX `src/` and the
staged native libraries are bind-mounted in.

!!! note "TODO"
    A worked, copy-paste Apptainer/Singularity invocation (image pull, `--nv`, the
    `--bind <nvhpc>,<phdf5>,<slate>` contract, and how host MPI gets in) is not yet written.
    Today only the Shifter path is exercised — see [Perlmutter](perlmutter.md). The
    bind-mount contract and the per-runtime flag mapping are described in
    `src/ffi/PORTING.md`.

## Track 3 — from source (the native stack)

The full distributed build: obtain ABI-matched cuSolverMp/cuBLASMp, parallel HDF5, MPI,
and SLATE, then build `liblorrax_ffi.so` and point LORRAX at it. CUDA-13 native support is
a porting target, not a validated release row. The process is documented separately:

➡ **[FFI native libraries](ffi-native-libs.md)** — acquisition recipes + the non-Shifter
`cmake -D…` build.

## See also

- [`docs/ENVIRONMENT_COMPREHENSIVE.md`](../ENVIRONMENT_COMPREHENSIVE.md) — the full
  environment / JAX-config / troubleshooting reference this page summarizes
- [Perlmutter](perlmutter.md) — the NERSC reference cluster (module, `lxrun`/`lxpre`)
- `src/ffi/PORTING.md` — the FFI porting checklist (included in source and wheel installs)
