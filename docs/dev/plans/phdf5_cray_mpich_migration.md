# Plan — migrate `ffi.phdf5` from OpenMPI/conda-forge HDF5 → Cray MPICH stack

## Motivation

Our current phdf5 stack uses conda-forge HDF5 1.14 linked against the
JAX container's HPC-X OpenMPI. Performance: 4.45 GB/s at 16 GPUs. The
NERSC / Cray MPICH stack has Lustre-aware collective buffering that
could buy another 1.5–3×. It is also the "default" pattern for novice
users on DOE clusters — `module load cray-hdf5-parallel` exists on
every Cray machine. Staying in lockstep with it reduces the
per-cluster porting cost.

## Findings from environment probe

Host side on Perlmutter login node:

- `cray-hdf5-parallel/1.12.2.9/gnu/12.3` at
  `/opt/cray/pe/hdf5-parallel/1.12.2.9/gnu/12.3/` (17 MB total, 16 MB
  in `lib`). HDF5 **1.12** (SONAME 200), not 1.14. All APIs we use
  (`H5Pset_fapl_mpio`, `H5Pset_coll_metadata_write`, `H5Pset_fill_time`,
  `H5Pset_alloc_time`, `H5Pset_alignment`, `H5F_LIBVER_LATEST`) exist
  in 1.10+, so 1.12 is sufficient.
- `cray-mpich/9.0.1/ofi/gnu/12.3` at
  `/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/`. MPICH 4.1.2, SONAME 12.
- Host `libhdf5.so` NEEDS `libmpi_gnu_123.so.12` + PMI / libfabric /
  libcxi from `/opt/cray/pe/lib64/`.

Container side (`nvcr.io/nvidia/jax:25.04-py3`):

- HPC-X OpenMPI at `/opt/hpcx/ompi/` (libmpi.so.40). We'll stop using
  it — JAX/XLA/cuSOLVERMp/libcal don't link MPI at all, so the
  container has no _hard_ OpenMPI dependency for our workload.
- Shifter flag `--module=mpich` bind-mounts a Cray MPICH runtime at
  `/opt/udiImage/modules/mpich/`, providing `libmpi.so.12`,
  `libmpi_gnu_91.so.12`, `libmpi_cray.so.12`, `libmpi_gtl_cuda.so`,
  plus PMI and libfabric deps under `dep/`. Does _not_ ship
  `libmpi_gnu_123.so.12` (the Cray-PE GCC 12.3 variant).
- No `mpi.h` inside the container — the module is runtime-only.

Upshot: we **cannot** simply bind-mount the host's `libhdf5.so` because
its `NEEDED` points at a GCC-12-specific libmpi that isn't in the
shifter mpich module. We must link HDF5 against the generic
`libmpi.so.12` (MPICH ABI) that the shifter module does provide.

## Proposed design

**Stage a Cray-MPICH-backed HDF5 build in `$HOME/software/`**, built
once inside the shifter+mpich container from upstream HDF5 source.

```
$HOME/software/lorrax_phdf5_mpich/
├── stage/               ← built HDF5 1.14.x with Cray MPICH ABI
│   ├── bin/ include/ lib/
├── build.sh             ← reproducible rebuild script
├── src/
│   └── hdf5-1.14.X.tar.gz + extracted source
└── README.md            ← this plan, short version
```

The stage is keyed off Cray MPICH's generic ABI (`libmpi.so.12`), so
at runtime the shifter `--module=mpich` bind-mount satisfies it
regardless of GCC version.

MPICH headers (`mpi.h` and `mpio.h`) come from the host Cray MPICH
install. Rather than bind-mount `/opt/cray` (fragile, system path),
we pre-stage `mpi.h` + `mpio.h` + a handful of MPICH-specific headers
into `$HOME/software/lorrax_phdf5_mpich/mpi_headers/`. One-time copy.
Total ~1 MB. These are stable across Cray MPICH minor versions.

## Migration steps

### Phase A — stage new tree (login-node, no GPU needed)

1. `mkdir -p ~/software/lorrax_phdf5_mpich/{src,stage,mpi_headers}`
2. Copy host MPICH headers:
   ```
   cp /opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include/{mpi.h,mpio.h,mpicxx.h} \
      ~/software/lorrax_phdf5_mpich/mpi_headers/
   ```
   (plus any additional headers revealed by the build; they're small.)
3. Reuse the existing HDF5 1.14.3 source tarball from
   `/pscratch/sd/j/jackm/lorrax_phdf5/src/hdf5-1.14.3.tar.gz` (copy it
   to our new `src/` so scratch can be purged).
4. Write `build.sh` — enters shifter+mpich container, `configure
   --enable-parallel --enable-threadsafe CC=gcc CFLAGS="-I../mpi_headers"
   LIBS="-L/opt/udiImage/modules/mpich -lmpi" --prefix=.../stage`,
   then `make install`. Matches the env the FFI will run in.
5. Run `build.sh` on a login node (no GPU needed). ~5–10 min.
6. `ldd ~/software/lorrax_phdf5_mpich/stage/lib/libhdf5.so` should
   show `libmpi.so.12 => not found` (expected: it will resolve at
   runtime via the shifter mpich module). Confirm SONAME is
   `libhdf5.so.310` (HDF5 1.14), NEEDED is `libmpi.so.12` (MPICH
   ABI, not `libmpi.so.40`).

### Phase B — launcher + build plumbing

7. Update
   [`src/ffi/common/cpp/run_shifter.sh`](/global/u2/j/jackm/software/lorrax/src/ffi/common/cpp/run_shifter.sh):
   - Add `--module=mpich` alongside `--module=gpu`.
   - Change `LORRAX_FFI_PHDF5_DIR` default from
     `/pscratch/sd/j/jackm/lorrax_phdf5/stage` to
     `/global/homes/j/jackm/software/lorrax_phdf5_mpich/stage`.
   - Drop `/opt/hpcx/ompi/lib` from the `LD_LIBRARY_PATH` chain — the
     shifter mpich module owns MPI now. Keep `/lorrax_phdf5/lib`.
   - Add `--mpi=pmix` is already set for srun; remains correct.
   - Bind-mount stays `--volume=$LORRAX_FFI_PHDF5_DIR:/lorrax_phdf5`.

8. Update
   [`src/ffi/common/cpp/CMakeLists.txt`](/global/u2/j/jackm/software/lorrax/src/ffi/common/cpp/CMakeLists.txt):
   - `find_package(HDF5 ... COMPONENTS C REQUIRED)` with
     `HDF5_PREFER_PARALLEL=ON`, `HDF5_ROOT=/lorrax_phdf5` should Just
     Work.
   - For MPI linking: the container now provides libmpi via the
     shifter module's LD_LIBRARY_PATH; at compile time we need the
     header path. Add a CMake option
     `-DLORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/../mpi_headers` or bake
     into a defaulted path.
   - Change `target_link_libraries(... mpi)` — the `libmpi.so.12`
     soname will be picked up at link time from the shifter bind-
     mount (verified during test build).

### Phase C — verification

9. Rebuild: `src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh`
   inside the container, no GPU needed.
10. Sanity: `ldd build/liblorrax_ffi.so` should show `libhdf5.so.310
    => /lorrax_phdf5/lib/libhdf5.so.310` and `libmpi.so.12 =>
    /opt/udiImage/modules/mpich/libmpi.so.12`.
11. 4-GPU round-trip: `common.phdf5_write_test` — existing test should
    pass unchanged; `H5Dwrite` API is identical.
12. 16-GPU bench: `common.phdf5_vs_gather_bench -n 16384 --iters 3`.
    Expected improvement: 4.45 GB/s → 6–10 GB/s based on the Cray
    MPICH Lustre-aware CB claim. If no change, diagnose with
    `MPICH_MPIIO_STATS=1`.

### Phase D — cleanup

13. Delete `/pscratch/sd/j/jackm/lorrax_phdf5/` (conda-forge stage) once
    verified. Scratch reclaims itself anyway, but explicit is clearer.
14. Update
    [`src/ffi/PORTING.md`](/global/u2/j/jackm/software/lorrax/src/ffi/PORTING.md):
    document the new build/run convention. Other clusters: instructions
    become "point `LORRAX_FFI_PHDF5_DIR` at your cluster's parallel
    HDF5 module's install prefix, and load an MPICH-ABI MPI module
    (cray-mpich, MPICH, MVAPICH2, Intel MPI all work)".
15. Update
    [`src/ffi/AGENTS.md`](/global/u2/j/jackm/software/lorrax/src/ffi/AGENTS.md)
    with the new stage dir path and shifter flag.

## Risks / known unknowns

- **Shifter mpich module interaction with `LD_LIBRARY_PATH`**: the
  module sets its own LD_LIBRARY_PATH before our launcher's override.
  If our explicit LD_LIBRARY_PATH omits `/opt/udiImage/modules/mpich`,
  libmpi won't resolve. Fix: prepend the module's paths to our chain,
  or don't override LD_LIBRARY_PATH at all (let shifter's inherited
  value win).
- **PMI bootstrap**: Cray MPICH uses Cray PMI (libpmi/libpmi2), which
  the shifter module bind-mounts. `--mpi=pmix` is MPI-agnostic on
  Slurm side and should keep working.
- **Other MPI users in container**: none. HPC-X OpenMPI is unused by
  JAX/XLA/cuSOLVERMp. Verified.
- **libmpi_gtl_cuda**: Cray MPICH's GPU transport layer. Auto-loaded
  when `MPICH_GPU_SUPPORT_ENABLED=1`. For our use case (H5Dwrite
  with host pinned buffers), this shouldn't be needed, but we can
  flip the env knob if bench shows surprising results.
- **Regression in the cuSOLVERMp FFI**: doesn't touch MPI, but sits
  in the same `liblorrax_ffi.so` that we're rebuilding. The
  `tests/bench/cusolvermp_eigh_test.py` must still pass after migration.
- **Build-time gcc version mismatch**: host Cray-PE is gnu/12.3, but
  the container's gcc is whatever Ubuntu 22.04 ships
  (probably gcc-11). Shouldn't matter as long as we build inside the
  container and target the MPICH ABI. Confirm by running the bench
  before declaring done.
- **HDF5 version change**: we're staying on 1.14.x (not downgrading to
  host's 1.12). Files written with our 1.14+LATEST will be readable
  by other 1.14+ readers. Not a regression.

## Scope not included

- Cray MPICH on non-NERSC clusters. Handled by the "other clusters"
  note in PORTING.md — users stage their local MPICH-ABI HDF5.
- Additional H5Pset_meta_block_size tweak from last session's
  research. Separate commit, uncomplicated by MPI layer.
- GDS / cuFile VFD. Out of scope; Tier-D in the research punch list.

## Rollback

If the migration fails verification, `git revert` the launcher +
CMake commits and the old conda-forge stage at
`/pscratch/sd/j/jackm/lorrax_phdf5/stage` is still intact until
Phase D cleanup. Full round-trip: <5 minutes.

## Estimated effort

Roughly half a day: 1 hr staging + HDF5 rebuild, 1 hr launcher/
CMake plumbing, 1 hr verification on a 4-GPU alloc, 1 hr docs.
