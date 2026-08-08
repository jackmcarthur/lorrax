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
| **Pure-JAX (no FFI)** | any | 13 or none | ≥0.7,<0.10 | — | — | bare venv | none (single-process only — see the note below) | CI / CPU |
| **NERSC Perlmutter (reference)** | SLES 15 | 12.9 (staged native libs) | **0.7.0** (container `ghcr.io/nvidia/jax:jax-2025-07-21`; inside the declared window — see below) | Cray MPICH | cray-hdf5-parallel | Shifter | all | 1–4 nodes × 4 A100 |
| **Generic Cray EX** | — | 12.x/13.x | ≥0.7,<0.10 | Cray MPICH | cray-hdf5-parallel | Apptainer | all | untested |
| **Generic SLURM + OpenMPI** | Linux x86_64 | 12.x/13.x | ≥0.7,<0.10 | OpenMPI 4/5 + UCX | conda-forge `hdf5=*mpi_openmpi*` | Apptainer / bare venv | all | untested |

Only the pure-JAX path works with **zero native libs**, and only for a
single-process run: **since the 2026-08-01 ruling the FFI layer is REQUIRED**,
enforced at startup step 6b (`runtime.__init__._enforce_required_ffi` →
`ffi.gate.enforce`), so a distributed or CPU-mesh run without a loadable `.so`
refuses rather than falling back. Everything distributed needs the
[FFI native-library stack](ffi-native-libs.md).

!!! note "The reference platform's JAX version — resolved 2026-08-06"
    This section used to warn that the reference platform ran *below* the declared
    floor: `pyproject.toml` said `jax>=0.9.0` while the Perlmutter container shipped
    the 0.5.3 line, and nothing enforced it. Both halves are now closed, in opposite
    directions, and the reason the fix was not simply "raise the container" is worth
    keeping:

    **No NVIDIA JAX container exists with both JAX ≥ 0.9 and CUDA 12** — ten tags
    probed 2026-08-06, and the CUDA 12→13 flip lands three minor versions *before*
    JAX reaches 0.9. Everything staged for Perlmutter (cuSOLVERMp 0.7.2_cuda12.9,
    the CUDA-12 `libmpi_gtl_cuda`, the device `liblorrax_ffi.so` itself) is CUDA 12,
    so a 0.9 floor was unsatisfiable on the reference platform *by construction*.

    So the container moved **up** to the last CUDA-12 image,
    `ghcr.io/nvidia/jax:jax-2025-07-21` (JAX **0.7.0**, CUDA 12.9), and the declared
    range moved **down and closed**, to `jax>=0.7.0,<0.10.0` in all three
    `pyproject.toml` sites. 0.7.0 is not an arbitrary floor: it is where
    `jax.shard_map`, `lax.pvary` and varying-manual-axes tracking inside `shard_map`
    all arrive, and where every `jax._src` private this tree patches reaches the
    shape it has on 0.9.

    **It is now enforced.** `runtime.jax_support.enforce()` runs at step 5b of
    `runtime.initialize_communicator_stack` — after the backend exists, before the
    first `jit` — and refuses a JAX outside the window or with an unexpected
    `jax._src` shape, naming the fix. `LORRAX_JAX_UNSUPPORTED_OK=1` is the one
    declared, announced escape.

    The symptom this replaces, for anyone who meets it on an old image: a
    `UserWarning` on the first jit and a **silently dead persistent compile cache** —

    ```text
    /opt/jax/jax/_src/compiler.py:723: UserWarning: Error reading persistent compilation cache
    entry for 'jit_convert_element_type': AttributeError: module 'jax._src.config' has no
    attribute 'compilation_cache_check_contents'
    ```

    printed *immediately after* a startup block saying the cache is enabled.
    Numerical results were never implicated. On the 0.7.0 image the same probe writes
    9 entries cold and takes a full warm hit with 0 recompiles, at P=1 and at P=2.
    [`environment/overview.md` §2](../environment/overview.md) owns the measurements.

!!! warning "The `liblorrax_ffi.so` build cliff"
    A fresh `git clone` has **no** `liblorrax_ffi.so` (it is a gitignored build artifact).
    The pure-JAX path never touches it (all FFI imports are lazy), but the first time you
    run a distributed / FFI-I/O code path you will hit
    a `FileNotFoundError` naming the platform's build recipe
    (`src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh` for CUDA,
    `bash src/ffi/cpp/build_host.sh` for the host library).
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

`uv sync` installs the GPU build by default: the package offers
`jax[cuda12]` and `jax[cuda13]` extras, both pinned to the same window
`>=0.7.0,<0.10.0`, and the wheels bundle the CUDA runtime (no system CUDA install
needed; a recent NVIDIA driver is required). Note this whole track is a *bare-host*
install and is **not** how the Perlmutter GPU leg runs — that leg takes JAX and its
CUDA plugin from the Shifter image (Track 2) and pip-installs nothing. See the note
in `pyproject.toml`, which is authoritative for all Python-side pins.

## Track 2 — container

LORRAX runs inside the NVIDIA JAX image (`ghcr.io/nvidia/jax:jax-2025-07-21`, JAX 0.7.0
/ CUDA 12.9; `config/perlmutter/site_config.sh` owns the tag). On NERSC the
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

- [`docs/environment/overview.md`](../environment/overview.md) — the full
  environment / JAX-config / troubleshooting reference this page summarizes
- [Perlmutter](perlmutter.md) — the NERSC reference cluster (`lx`, the module, `lxpre`)
- `src/ffi/PORTING.md` — the FFI porting checklist (in the repo, not the site)
