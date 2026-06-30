# Perlmutter (NERSC) — reference cluster

NERSC Perlmutter is LORRAX's reference / most-tested platform: 4× A100 GPU nodes, Cray
MPICH, and the Shifter container runtime. The site-specific paths, SLURM defaults, and the
literal NERSC values (`m2651`, `interactive` QOS, `/opt/udiImage`, …) live here, kept out of
the generic [Installation](index.md) tracks.

!!! note "Authoritative sources"
    This page is a short orientation. The authoritative, maintained references are
    [`config/README.md`](../../config/README.md) (module, `lxalloc`/`lxrun`/`lxpre`,
    bind-mounts, porting) and
    [`docs/ENVIRONMENT_COMPREHENSIVE.md`](../ENVIRONMENT_COMPREHENSIVE.md) §4–§5 (Lmod
    module, JAX config, FFI stack).

## One-time install

```bash
vi config/perlmutter/site_config.sh      # edit account, QoS, paths
bash config/perlmutter/install.sh        # patches + installs the Lmod module
```

To install several checkouts side-by-side, set a distinct `LORRAX_MODULE_NAME`
(see [`config/README.md`](../../config/README.md)).

## Every session

```bash
module load lorrax
lxalloc                                  # 1 node / 4 GPUs / 2 h, exports SLURM_JOBID
lxpre cohsex.in 640                      # 3 preprocessing steps (centroids, dipole, kin_ion)
lxrun python3 -u -m gw.gw_jax -i cohsex.in   # 4-GPU GW
```

`lxpre` runs, in order:

1. `python3 -m centroid.kmeans_cli <N> --seed 42` → `centroids_frac_<N>.txt`
2. `python3 -m psp.get_dipole_mtxels -i <in>` → `dipole.h5`
3. `python3 -m gw.kin_ion_io -i <in>` → `kin_ion.h5`

## Native FFI stack on Perlmutter

The three native trees (cuSolverMp, parallel HDF5, SLATE) are staged once under
`$HOME/software/lorrax_{nvhpc,phdf5_cray/stage,slate_cray/stage}` and bind-mounted into the
container. Then `liblorrax_ffi.so` is built inside Shifter:

```bash
src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
```

See [`docs/ENVIRONMENT_COMPREHENSIVE.md` §5](../ENVIRONMENT_COMPREHENSIVE.md) for staging
details and the Cray-MPICH GPU-Direct knobs, and
[FFI native libraries](ffi-native-libs.md) for the off-NERSC acquisition recipes.
