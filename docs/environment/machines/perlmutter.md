# Perlmutter (NERSC)

*The Shifter/GPU reference platform. Authoritative for module mechanics
and porting knobs: [`config/README.md`](../../../config/README.md). This
page holds the runtime environment: the Lmod module, the FFI staging
contract, multi-host topology — and an honest statement of what has and
has not been exercised recently.*

## 0. Test status — honest

* The GPU FFI stack (cuSOLVERMp eigh, phdf5 slab I/O, SLATE) and the
  `lxalloc`/`lxrun` module workflow were production-certified on
  Perlmutter (1–4 nodes × 4 A100). `lx` (§1) drives the same
  `select_gpu.sh` + Shifter + `in_container.sh` composition.
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

## 1. Entry point: `lx` {#1-entry-point-lx}

**`lx` is how you run things on Perlmutter.** Never on a login node, never
`sbatch`. It allocates if nothing is live, attaches if something is, and
calling it twice never double-allocates.

```bash
lx run python3 -u -m gw.gw_jax -i cohsex.in   # one step on a compute node
lx test                                       # the default gate, on a compute node, in cwd
lx test --census                              # the full census (see docs/contributing.md)
lx status                                     # who is running where
lx doctor                                     # verify site, module, helpers
lx shell -G 4                                 # interactive pty on a compute node
lx alloc -N 4 --time 04:00:00                 # allocate deliberately (idempotent)
lx release                                    # cancel only what lx created
```

Defaults are **one GPU per step** (so several steps share a node), and a new
allocation is 4 nodes / 4 h. Ask for `-G 4` when you want a whole node,
`-N n` for multiple nodes, `--cpu` for the Milan CPU partition. `--dry-run`
prints the `srun` line and exits — the fastest way to see what your step will
actually inherit.

> ### `lx` runs the checkout its *base module* names, not the one you are standing in
>
> **Re-measured 2026-08-10 — `lx` now prefers the checkout you are standing
> in, and this warning used to say the opposite.** The resolution order is
> `LORRAX_CHECKOUT`, then the checkout `cwd` sits inside, then the base
> module's tree, where "a checkout" means a directory holding both
> `src/gw/__init__.py` and `tests/`. `lx` announces which one it picked on
> every invocation, tagged with the reason:
>
> ```text
> [lx] source tree: /pscratch/sd/j/jackm/lorrax_pipehealth/src  [cwd]
> [lx] source tree: /pscratch/sd/j/jackm/lorrax_pipehealth/src  [LORRAX_CHECKOUT]
> [lx] source tree: /global/u2/j/jackm/software/lorrax_P/src    [module default (cwd is not in a checkout)]
> ```
>
> The trap the old warning was reaching for is still real, but it is narrower
> and it bites in the opposite place: **a data directory is not a checkout.**
> Real calculations run in a scratch directory holding `WFN.h5` and a deck, so
> `lx run` there silently resolves to the base module's tree — which on this
> machine is `lorrax_P`, measured 2026-08-10 sitting **624 commits behind
> `main`** and old enough to predate the `jax_support` window check and to have
> no `gw/downfold_cli.py` at all. Running a current pipeline from a data
> directory therefore runs old code without saying so beyond that one line.
>
> `LORRAX_CHECKOUT=/path/to/tree` is the fix, and it is the only one that works
> from a data directory; `LX_BASE_MODULE=lorrax_B|_C|_J070|…` swaps the base
> module (and with it the container image), and `LORRAX_DEFAULT_BASE` sets the
> default. Two other things stay visible: `lx doctor` prints `base module` and
> `LORRAX_ROOT` before you run anything, and every run's
> [startup block](../overview.md#startup-block) prints the path of the `.so` it
> loaded. `lx test` is the exception that has always used `cwd`: it runs pytest
> in the current directory, and only falls back to `$LORRAX_ROOT/tests` when
> `cwd` has no `tests/`.
>
> One pairing is worth writing down because it costs an afternoon to find. The
> default base module's image ships JAX 0.5.3, which current `main` **refuses**
> (`REFUSED: jax-support.version … want jax >= 0.7.0, < 0.10.0`), so a current
> checkout needs `LX_BASE_MODULE=lorrax_J070` for the
> `ghcr.io/nvidia/jax:jax-2025-07-21` image. That module swaps the image only —
> its `LORRAX_ROOT` still points at `lorrax_P` — so the working combination for
> running current source on this machine is both variables together, plus an
> FFI library built from that same checkout:
>
> ```bash
> export LX_BASE_MODULE=lorrax_J070
> export LORRAX_CHECKOUT=/path/to/your/checkout
> ```

### Do not start from `module load lorrax`

The older `module load lorrax` + `lxalloc`/`lxrun`/`lxpre`/`lxshell`/`lxkill`
workflow still exists, and the module is still what *produces* the Shifter
string `lx` consumes. But it is not the entry point to hand a newcomer, for
two measured reasons:

**It resolves the wrong checkout.** The base module derives `LORRAX_ROOT` by
matching its own path against `<root>/config/modulefiles/lorrax/`; an
*installed* (copied, not symlinked) modulefile always fails that match and
falls back to a hardcoded value. Measured on this machine 2026-08-06,
`module show lorrax` sets `LORRAX_ROOT=$HOME/software/lorrax_C` — an older
worktree — no matter which tree you are standing in.

**The installed module exports the allocator setting
[§2.1](../overview.md#21-the-three-allocators) condemns.** Measured
2026-08-06, `module show lorrax` sets:

| it sets | consequence |
|---|---|
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` | with the allocator left unset (BFC) this pre-grabs 95 % of the card, **starving NCCL** — the failure surfaces as `cusolverMpSyevd: status=7` (§2.1) |
| `TF_GPU_ALLOCATOR=cuda_malloc_async` | **inert.** A TensorFlow variable; it selects nothing for JAX, and the startup block says so outright |

Note the repo *template* `config/modulefiles/lorrax/0.1.0.lua` has since been
changed to set `XLA_PYTHON_CLIENT_PREALLOCATE=false` +
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` instead, and carries no
`MEM_FRACTION`. **The installed module on this machine predates that
change.** Which one you get therefore depends on when `install.sh` was last
run — which is the whole argument for not routing newcomers through it.
`lx` reads the module for the Shifter string only and sets its own allocator
pair (`preallocate=false`, `allocator=platform`); what any given run actually
resolved is in its [startup block](../overview.md#startup-block).

`lxrun` expands to `srun --mpi=cray_shasta … select_gpu.sh shifter …
in_container.sh "$@"`: each rank sees exactly one GPU as device 0 via
`CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` (**not** `--gpus-per-task=1`, which
breaks JAX's topology sync). Batch template:
`config/perlmutter/run_gw.slurm`; multi-node adds
`LORRAX_NNODES=2 LORRAX_NGPU=8`.

Per-invocation cost: ~7 s single-rank, 10–15 s multi-rank (srun step 2–5 s,
Shifter bring-up ~5 s, `jax.distributed` handshake 3–5 s). `lx shell` and a
persistent compile-cache directory are the fast-iteration knobs.

### One-time install

```bash
vi config/perlmutter/site_config.sh          # account, QoS, paths
bash config/perlmutter/install.sh            # or LORRAX_MODULE_NAME=<name> bash …
```

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

## 5. CPU multi-process runs (Milan)

The former gloo-era recipe and its mpi4py/parallel-h5py SlabIO tier are
retired. Current CPU runs require the host native FFI and use the same single
parallel-HDF5 transport as GPU runs; the `slab_io` and `use_ffi_io` deck keys
are refused. Use `lx run` for compute-node execution and follow the current
[transport bring-up](../transports.md) before treating a multi-process CPU run
as certified.
