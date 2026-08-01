# Installation

LORRAX has two layers: a **pure-JAX core** (centroids, wavefunction loading, serial GW)
that needs only Python + JAX, and a **native FFI stack** (cuSolverMp, parallel HDF5, SLATE)
required for distributed `eigh`, sharded HDF5 I/O, and SLATE linear algebra. Pick a track
from the matrix below; the pure-JAX core works with zero native libraries.

!!! note "TODO"
    This page is a Tier-2 scaffold assembled from the portability and
    dependency-architecture reviews. The support-matrix rows below marked *untested*
    have not been validated on a non-NERSC cluster; treat them as targets, not guarantees.

## Support matrix

| Config | OS | CUDA | JAX | MPI | parallel HDF5 | Runtime | FFI features | Tested |
|---|---|---|---|---|---|---|---|---|
| **Pure-JAX (no FFI)** | any | 13 or none | ≥0.9 | — | — | bare venv | none (serial only) | CI / CPU |
| **NERSC Perlmutter (reference)** | SLES 15 | 12.9 (staged native libs) | ≥0.9 | Cray MPICH | cray-hdf5-parallel | Shifter | all | 1–4 nodes × 4 A100 |
| **Generic Cray EX** | — | 12.x/13.x | ≥0.9 | Cray MPICH | cray-hdf5-parallel | Apptainer | all | untested |
| **Generic SLURM + OpenMPI** | Linux x86_64 | 12.x/13.x | ≥0.9 | OpenMPI 4/5 + UCX | conda-forge `hdf5=*mpi_openmpi*` | Apptainer / bare venv | all | untested |

Only the pure-JAX path works with **zero native libs**. Everything distributed needs the
[FFI native-library stack](ffi-native-libs.md).

!!! warning "The `liblorrax_ffi.so` build cliff"
    A fresh `git clone` has **no** `liblorrax_ffi.so` (it is a gitignored build artifact).
    The pure-JAX path never touches it (all FFI imports are lazy), but the first time you
    run a distributed / FFI-I/O code path you will hit
    `FileNotFoundError … Build with: bash src/ffi/cpp/build.sh`.
    Build the native library per [FFI native libraries](ffi-native-libs.md) before
    using those features.

## Track 1 — pip / uv (pure JAX, no FFI)

Gives you centroids, wavefunction loading, and serial GW. **Not** distributed `eigh`,
sharded HDF5, or SLATE.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # one-time
git clone <lorrax-repo-url> && cd lorrax
uv sync                                            # editable install; puts src/ on sys.path
uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in
```

`uv sync` installs the GPU build by default: the package pins `jax[cuda13]>=0.9.0`,
and the CUDA-13 wheels bundle the CUDA runtime (no system CUDA install needed; a
recent NVIDIA driver is required). A CUDA-12 row on the same JAX-0.9 line is a
plausible future support-matrix entry but is not offered as an extra — see the
note in `pyproject.toml`, which is authoritative for all Python-side pins.

## Track 2 — container

LORRAX runs inside the NVIDIA JAX image (`nvcr.io/nvidia/jax:25.04-py3`). On NERSC the
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

The full distributed build: obtain cuSolverMp, parallel HDF5, and SLATE on a system with
no Cray PE, then build `liblorrax_ffi.so` and point LORRAX at it. This is the largest gap
for non-NERSC users and is documented separately:

➡ **[FFI native libraries](ffi-native-libs.md)** — acquisition recipes + the non-Shifter
`cmake -D…` build.

## See also

- [`docs/ENVIRONMENT_COMPREHENSIVE.md`](../ENVIRONMENT_COMPREHENSIVE.md) — the full
  environment / JAX-config / troubleshooting reference this page summarizes
- [Perlmutter](perlmutter.md) — the NERSC reference cluster (module, `lxrun`/`lxpre`)
- `src/ffi/PORTING.md` — the FFI porting checklist (in the repo, not the site)
