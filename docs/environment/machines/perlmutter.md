# Perlmutter (NERSC)

*The Shifter/GPU reference platform. Authoritative for module mechanics
and porting knobs: [`config/README.md`](../../../config/README.md). This
page holds the runtime environment: the Lmod module, the FFI staging
contract, multi-host topology — and an honest statement of what has and
has not been exercised recently.*

## 0. Test status — honest

* The GPU FFI stack (cuSOLVERMp eigh, phdf5 slab I/O, SLATE) and the
  `lxalloc`/`lxrun` workflow below were production-certified on
  Perlmutter (1–4 nodes × 4 A100).
* CPU multi-process MPI runs were validated end-to-end on Milan (§5):
  Si 4×4×4 μ=384, x_only + full COHSEX, 1 node, 4 ranks × 8 threads —
  in the **gloo/Cray-MPICH era**.
* The 2026-07 campaign (the `impl=mpi` collectives migration, the
  MPIwrapper, the mpi4py overlay, the runtime bundle, the host FFI
  `.so`) ran **on Frontera**. None of that layered CPU stack has a
  Perlmutter build or a Perlmutter measurement; on Cray the analogous
  pieces (Cray MPICH thread grants, PMI, `sbcast`-vs-Lustre trade-offs)
  would need their own bring-up. Treat
  [transports](../transports.md) claims as Frontera-measured unless a
  jobid says otherwise.

## 1. Module: install and session

```bash
vi config/perlmutter/site_config.sh          # account, QoS, paths
bash config/perlmutter/install.sh            # or LORRAX_MODULE_NAME=<name> bash …
```

```bash
module load lorrax
lxalloc                    # 1 node / 4 GPUs / 2 h, exports SLURM_JOBID
lxalloc 4                  # 4 nodes / 16 GPUs
lxpre cohsex.in 640        # all 3 preprocessing steps (single-GPU)
lxrun python3 -u -m gw.gw_jax -i cohsex.in        # 4-GPU GW
LORRAX_NGPU=1 lxrun …      # single-GPU override
lxshell                    # interactive single-rank shell in container
lxkill                     # cancel allocation
```

`lxrun` expands to `srun --mpi=cray_shasta … select_gpu.sh shifter …
in_container.sh "$@"`: each rank sees exactly one GPU as device 0 via
`CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` (**not** `--gpus-per-task=1`, which
breaks JAX's topology sync). Batch template:
`config/perlmutter/run_gw.slurm`; multi-node adds
`LORRAX_NNODES=2 LORRAX_NGPU=8`.

Per-invocation cost: ~7 s single-rank, 10–15 s multi-rank (srun step 2–5 s,
Shifter bring-up ~5 s, `jax.distributed` handshake 3–5 s). `lxshell` and a
persistent `JAX_COMPILATION_CACHE_DIR` are the fast-iteration knobs.

## 2. FFI stack: staging and bind-mounts

One `liblorrax_ffi.so` calls three native stacks not present in the JAX
container:

| subpackage | library | use |
|---|---|---|
| `cusolvermp` | cuSOLVERMp + CAL/NCCL | distributed `eigh` (syevd) |
| `phdf5` | parallel HDF5 via MPI-IO | sharded slab read/write |
| `slate` | SLATE + libsci | distributed Cholesky, trsm, heev |

Staged once per cluster (idempotent, each ends with a `readelf -d` check;
staging is mandatory because Shifter forbids `--volume` sources under
`/opt/*` or `$HOME`):

```bash
src/ffi/cpp/stage/cusolvermp_stage_nvhpc.sh   # cuSolverMp + CAL
src/ffi/cpp/stage/phdf5_stage_cray.sh         # Cray HDF5 (canonical here)
src/ffi/cpp/stage/phdf5_stage_openmpi.sh      # portable non-Cray stack
src/ffi/cpp/stage/slate_stage_cray.sh         # libsci + GTL + xpmem
```

Bind-mounts (host dir → container mount): `$LORRAX_FFI_NVHPC_DIR` →
`/lorrax_nvhpc`, `$LORRAX_FFI_PHDF5_DIR` → `/lorrax_phdf5`,
`$LORRAX_FFI_SLATE_DIR` → `/lorrax_slate`; `LORRAX_NVHPC_SUBPATH`,
`LORRAX_MPICH_CONTAINER_DIR`, `LORRAX_DARSHAN_LIB_DIR` are patched from
`site_config.sh`.

Build (needs staged libs + a GPU allocation):

```bash
src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh
```

Off-Shifter builds drive CMake directly with `-D` overrides —
`src/ffi/PORTING.md` and
[installation/ffi-native-libs](../../installation/ffi-native-libs.md).

MPI stack override: `LORRAX_MPI_TYPE=cray_shasta` (default) | `none` |
`pmix` (legacy, has hung non-FFI workloads — never set unconditionally).
GPU-aware Cray MPICH: the module sets `MPICH_GPU_SUPPORT_ENABLED=1` and
preloads `libmpi_gtl_cuda.so.0` — Cray-specific; no OpenMPI/UCX
equivalent exists for these two knobs.

## 3. Multi-host topology

`SLURM_NTASKS > 1` auto-triggers `jax.distributed.initialize()` (via
`runtime.initialize_communicator_stack()`; a sentinel guards re-import).
Expected in-job topology: `jax.local_devices()` = `[cuda:0]` per rank,
`len(jax.devices())` = ranks × nodes.

## 4. Generic-cluster porting

Everything cluster-specific funnels through `site_config.sh`
(`LORRAX_SLURM_*`, `LORRAX_GPUS_PER_NODE`, `LORRAX_MPI_TYPE_DEFAULT`,
the three `LORRAX_FFI_*_DIR` stage roots, …) — the full table is in
[`config/README.md`](../../../config/README.md). For non-Shifter runtimes
swap the `shifter` invocation in `lxrun`/`lxshell`/`lxpre`;
`select_gpu.sh`, `in_container.sh` and the SLURM defaults are portable.

## 5. CPU multi-process runs (Milan) — validated recipe, gloo-era

Validated end-to-end at Si 4×4×4 μ=384 (1 node, 4 ranks × 8 threads);
the same deck runs on GPU and CPU, with the FFI flags auto-routed from
`jax.default_backend()` (`gw.gw_config.LorraxConfig.from_input_file`).
One-time deps inside the venv — mpi4py and h5py built against
`cray-hdf5-parallel` + `cray-mpich` (`--no-binary`, `HDF5_MPI=ON`);
verify `h5py.get_config().mpi`. Launch: `salloc -C cpu`, then

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8
srun --jobid=$SLURM_JOBID -N 1 -n 4 -c 8 --cpu-bind=cores \
     $LORRAX_VENV/bin/python -u -m gw.gw_jax -i cohsex.in
```

What auto-routes on CPU: `slab_io=auto` → the capability router
(FFI write handler → `PHDF5_HOST` → allgather); `distributed_cholesky` /
`distributed_lu` `auto` → `off`; `pair_density_slots` 3 → 4. The CPU
path writes synchronously — the FFI's threaded design deadlocks at
`H5Fclose` under Cray MPICH's default `MPI_THREAD_SINGLE`
(`file_io/_slab_io_mpi_host.py` docstring). Note this pre-dates the
`impl=mpi` migration; a modern multi-process CPU run on Perlmutter
should expect the [transports](../transports.md) bring-up work first.
