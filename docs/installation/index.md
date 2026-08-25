# Installation

LORRAX needs Python + JAX **and** a native FFI stack (FFT/GEMM on the host library;
cuSolverMp, parallel HDF5 and SLATE for the distributed features). Pick a track from the
matrix below.

!!! danger "The 'pure-JAX, zero native libs' track described below no longer exists"
    This page was written when the FFI layer was optional and the JAX fallbacks were
    live. [Design decisions, 2026-08-01](../architecture/decisions.md) reversed that:
    where a certified FFI path exists the JAX path "is not maintained and may be
    deleted", and a missing library "is a refusal at startup, not a silent demotion".
    The same ruling is explicit that this binds at **P=1** — "the code must still run on
    one process: the required libraries must build and load for P=1".

    Verified 2026-08-10 at `88f28325` on a CPU-only machine, a fresh `uv sync` clone
    running the bundled single-process CPU fixture refuses in
    `runtime.initialize_communicator_stack` → `ffi.gate.enforce`, before the first
    `jit`, with `Could not locate liblorrax_ffi_host.so`. The `Pure-JAX (no FFI)` row
    below and the "zero native libs" sentence under the matrix are **stale**; they are
    left in place only because the rows around them are still accurate about OS/CUDA/MPI
    pairings. Treat "FFI features: none (serial only)" as "will not start".

!!! note "TODO"
    This page is a Tier-2 scaffold assembled from the portability and
    dependency-architecture reviews. The support-matrix rows below marked *untested*
    have not been validated on a non-NERSC cluster; treat them as targets, not guarantees.

## Support matrix

| Config | OS | CUDA | JAX | MPI | parallel HDF5 | Runtime | FFI features | Tested |
|---|---|---|---|---|---|---|---|---|
| **Pure-JAX (no FFI)** | any | 13 or none | ≥0.9,<0.10 | — | — | bare venv | none (serial only) | CI / CPU |
| **NERSC Perlmutter (reference)** | SLES 15 | 13.2 | **0.9.1** (`lorrax_A`) | Cray MPICH / NCCL | cray-hdf5-parallel | bare host | CUDA FFI + pinned PHDF5 build | 1–4 nodes × 4 A100 |
| **Generic Cray EX** | — | 12.x/13.x | ≥0.9,<0.10 | Cray MPICH | cray-hdf5-parallel | Apptainer | all | untested |
| **Generic SLURM + OpenMPI** | Linux x86_64 | 12.x/13.x | ≥0.9,<0.10 | OpenMPI 4/5 + UCX | conda-forge `hdf5=*mpi_openmpi*` | Apptainer / bare venv | all | untested |

Only the pure-JAX path works with **zero native libs**. Everything distributed needs the
[FFI native-library stack](ffi-native-libs.md).

!!! note "The reference platform's JAX version — resolved 2026-08-25"
    LORRAX requires the JAX/JAXLIB 0.9 series everywhere.  Perlmutter's
    `lorrax_A` module supplies 0.9.1 with CUDA 13.2, and the tracked lock file
    resolves the same patch release.  `tools/require_jax09.py` checks package
    metadata before the first driver import; `runtime.jax_support.enforce()`
    independently checks both packages and the private compile-cache API before
    the first physics `jit`.  There is no unsupported-version escape hatch.

    The former CUDA-12/JAX-0.7 Shifter lane is historical evidence, not an
    installation option.  The legacy installer and FFI header-staging scripts
    no longer default to that image and explicitly refuse its known tags.

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
    Since the 2026-08-01 ruling this is a cliff you hit on the **first run of anything**,
    not on the first distributed run: the gate refuses at startup naming the library and
    the build script (`build_host.sh` → `liblorrax_ffi_host.so` for the host leg,
    `build.sh` → `liblorrax_ffi.so` for the CUDA leg).

    Two things the refusal does not tell you, both measured 2026-08-10:
    `build_host.sh` requires a SLATE `gpu_backend=none` install and refuses without one,
    naming `src/ffi/cpp/stage/slate_build_perlmutter.sh cpu`; and on Perlmutter
    `build.sh` refuses until `LORRAX_NVHPC_ROOT` names a staged CUDA tree, printing the
    stages it can see. With that variable set, the CUDA leg builds in ~40 s on a compute
    node (`LORRAX_NVHPC_ROOT=/lorrax_nvhpc/25.5_cuda12.9 lx run bash src/ffi/cpp/build.sh`);
    on a **login** node it fails in Shifter's bind-mount setup instead, which is a
    node-type error rather than a build error.
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
`>=0.9.0,<0.10.0`, and the wheels bundle the CUDA runtime (no system CUDA install
needed; a recent NVIDIA driver is required). Note this whole track is a *bare-host*
install and is **not** how the Perlmutter GPU leg runs — that leg takes JAX and its
CUDA plugin from the Shifter image (Track 2) and pip-installs nothing. See the note
in `pyproject.toml`, which is authoritative for all Python-side pins.

## Track 2 — container

A container deployment must explicitly choose an image carrying JAX/JAXLIB
0.9 and a CUDA major matching its native FFI build.  The tracked Perlmutter
Shifter template deliberately has no image default; Perlmutter production uses
the `lorrax_A` bare-host CUDA-13 module instead.  On other clusters use
**Apptainer** or **Singularity** and retain the same preflight/runtime gates.

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
