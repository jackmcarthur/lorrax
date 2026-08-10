# Perlmutter (NERSC) — reference cluster

NERSC Perlmutter is LORRAX's reference / most-tested platform: 4× A100 GPU nodes, Cray
MPICH, and the Shifter container runtime. The site-specific paths, SLURM defaults, and the
literal NERSC values (`m2651`, `interactive` QOS, `/opt/udiImage`, …) live here, kept out of
the generic [Installation](index.md) tracks.

!!! note "Authoritative sources"
    This page is a short orientation. The authoritative, maintained references are
    [`config/README.md`](../../config/README.md) (module, `lxalloc`/`lxrun`/`lxpre`,
    bind-mounts, porting) and
    [`docs/environment/machines/perlmutter.md`](../environment/machines/perlmutter.md) (Lmod
    module, JAX config, FFI stack).

## One-time install

```bash
vi config/perlmutter/site_config.sh      # edit account, QoS, paths
bash config/perlmutter/install.sh        # patches + installs the Lmod module
```

To install several checkouts side-by-side, set a distinct `LORRAX_MODULE_NAME`
(see [`config/README.md`](../../config/README.md)).

## Every session

**`lx` is the entry point.** Never a login node, never `sbatch`:

```bash
lx run python3 -u -m gw.gw_jax -i cohsex.in   # one step on a compute node
lx run -G 4 python3 -u -m gw.gw_jax -i cohsex.in   # whole node, 4 GPUs
lx test                                       # the default gate, on a compute node, in cwd
lx test --census                              # the full census (see docs/contributing.md)
lx doctor                                     # which checkout will actually run
```

`lx` allocates if nothing is live and attaches if something is. It takes the
checkout it runs from a **base module**, not from `cwd` — run `lx doctor`
before trusting that you are exercising your own worktree. The full command
set, the `LX_BASE_MODULE` override, and why `module load lorrax` is *not* the
recommended entry point are in
[`environment/machines/perlmutter.md` §1](../environment/machines/perlmutter.md#1-entry-point-lx).

The older `module load lorrax` + `lxalloc`/`lxrun`/`lxpre` workflow still
exists. Measured 2026-08-06, the module installed on this machine resolves
`LORRAX_ROOT` to an **older worktree** and exports
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, which
[`environment/overview.md` §2.1](../environment/overview.md#21-the-three-allocators)
identifies as the setting that starves NCCL into `cusolverMpSyevd: status=7`.
Prefer `lx`.

`lxpre <deck> <N>` bundles the three preprocessing steps:

1. `python3 -m centroid.kmeans_cli <N> --seed 42` → `centroids_frac_<N>.txt`
2. `python3 -m psp.get_dipole_mtxels -i <in>` → `dipole.h5`
3. `python3 -m gw.kin_ion_io -i <in>` → `kin_ion.h5`

**Three of those steps have defects that bite on a first run** — the
wavefunction filename step 1 reads is hardcoded, the filename it writes does
not carry the `N` you asked for, and step 2's output cannot satisfy the
consistency guard step 4 applies to it. See
[Quickstart → Your first real calculation](../quickstart.md#your-first-real-calculation)
before running them.

## Native FFI stack on Perlmutter

The three native trees (cuSolverMp, parallel HDF5, SLATE) are staged once under
`$HOME/software/lorrax_{nvhpc,phdf5_cray/stage,slate_cray/stage}` and bind-mounted into the
container. Then `liblorrax_ffi.so` is built inside Shifter:

```bash
src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh
```

See [`docs/environment/machines/perlmutter.md` §2](../environment/machines/perlmutter.md) for staging
details and the Cray-MPICH GPU-Direct knobs, and
[FFI native libraries](ffi-native-libs.md) for the off-NERSC acquisition recipes.
