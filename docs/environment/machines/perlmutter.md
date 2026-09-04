# Perlmutter (NERSC)

*The CUDA-13/JAX-0.9 reference platform. Authoritative for module mechanics
and porting knobs: [`config/README.md`](../../../config/README.md). This
page holds the runtime environment: the Lmod module, the FFI staging
contract, multi-host topology — and an honest statement of what has and
has not been exercised recently.*

## 0. Test status — honest

* The GPU FFI stack (cuSOLVERMp eigh, phdf5 slab I/O, SLATE) is exercised on
  Perlmutter at 1–4 nodes × 4 A100. `lx` drives the tracked
  `select_gpu.sh` + environment + `in_container.sh` composition.
* CPU multi-process `impl=mpi` is now validated on Milan with Cray MPICH
  9.0.1.498 and JAX/JAXLIB 0.9.1.  A P=4 collective probe passed on one and
  two nodes, including all three 2-D mesh cliques and exact
  reduce-scatter; the two-node tracked-recipe proof is allocation 57261316,
  step `lx-Xg1-221900-433833-3385`.
* The same tracked recipe passed at P=16 on four nodes with a 4×4 process
  mesh and 16 logical CPU affinity slots per rank (256 across the step, not
  256 physical cores or full-node occupancy), exact on every
  allreduce/reduce-scatter segment and cleanly finalized: allocation
  57261316, step `lx-Xg1-222848-493981-5730`.
* A frozen two-node P=4 GN-PPM GW calculation then completed in 109.907 s and
  matched all 2,484 reference Sigma cells exactly (allocation 57261316,
  step `lx-Xg1-222309-460416-9952`).  This certifies the GW path at P=4;
  it is not yet a large-physics-run scaling or performance claim.

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
lx shell                                      # one-task, one-GPU interactive pty
lx alloc -N 4 --time 04:00:00                 # allocate deliberately (idempotent)
lx release                                    # cancel only what lx created
```

Defaults are **one GPU per step** (so several steps share a node), and a new
allocation is 4 nodes / 4 h. Ask for `-G 4` when you want a whole node,
`-N n` for multiple nodes, `--cpu` for the Milan CPU partition. `--dry-run`
prints the `srun` line and exits — the fastest way to see what your step will
actually inherit.

### Required GPU task geometry

Use one task/rank per GPU. Keep `lx`'s default rank count or set it explicitly:

```bash
lx run -N 1 -G 4 -n 4 python3 -u -m gw.gw_jax -i cohsex.in
```

`-N 1 -G 4 -n 1` violates the process/collective contract and is not P=4
evidence. `lx shell` is one task/one GPU. Report GPU count, rank count, and
runtime mesh.

> ### Source selection is explicit and attested
>
> `lx` resolves source in this order:
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
> A data directory is not a checkout, so production run directories must set
> `LORRAX_CHECKOUT`. `LX_BASE_MODULE` chooses the machine environment; it does
> not choose the source. Startup refuses a requested checkout that disagrees
> with the imported core or first-party services.
>
> `lx doctor` prints the base environment, and every run records its actual
> Python source and native-library origins. `lx test` uses the current checkout
> when `cwd` contains `tests/`.
>
> Current source requires the 0.9 series for both JAX and JAXLIB.  Select the
> deployed bare-host CUDA-13 lane and pin the source checkout independently:
>
> ```bash
> export LX_BASE_MODULE=lorrax_A
> export LORRAX_CHECKOUT=/path/to/your/checkout
> ```
>
> `tools/require_jax09.py` must run before the first driver import.  The driver
> repeats the version check internally, so an old module or copied launcher
> refuses even if the preflight was omitted.

### The module is a descriptor, not a launcher

`lx` loads the named module in a throwaway shell only to obtain
`LORRAX_ROOT` and the assembled native/container capability string. The
tracked module defines no run or allocation functions and sets no JAX,
allocator, HDF5, compile-cache, or profiling policy. Those defaults are set
and reported by `runtime` before JAX/HDF5 import; documented experiments may
override them explicitly.

### One-time install

```bash
vi config/perlmutter/site_config.sh          # image and dependency paths
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
staging is mandatory because Shifter cannot mount the vendor `/opt/*` trees
directly):

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
`len(jax.devices())` = total ranks = nodes × GPUs per node.

## 4. Generic-cluster porting

The image and native stage roots funnel through `site_config.sh`; the full
ownership map is in [`config/README.md`](../../../config/README.md). A
non-Shifter site needs its own environment descriptor. Do not copy the
retired shell launch functions as a portability layer.

## 5. CPU multi-process runs (Milan)

The former gloo-era recipe and its mpi4py/parallel-h5py SlabIO tier are
retired. Current CPU runs require the host native FFI and use the same single
parallel-HDF5 transport as GPU runs; the `slab_io` and `use_ffi_io` deck keys
are refused.

Build the small MPI ABI adapter once on a CPU compute node.  This builds the
exact pinned, **unmodified** upstream MPIwrapper against versioned Cray
wrappers; Cray MPICH remains the MPI implementation. The candidate is gated
in isolation and an immutable, content-addressed release becomes `current`
atomically, so a failed rebuild cannot delete the active adapter.

```bash
config/perlmutter/build_mpiwrapper.sh --fresh
```

For every core-driver multi-process CPU step, source the tracked launch
prelude before Python/JAX:

```bash
export LORRAX_CHECKOUT=/path/to/lorrax
export LORRAX_ROOT="$LORRAX_CHECKOUT"
export CPU_JAX_VENV=/path/to/jax-0.9.1-venv
export LX_BASE_MODULE=lorrax_A
export LORRAX_CPUS_PER_TASK=16
export PYTHONPATH="$LORRAX_CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}"
lx run --cpu -N 2 -n 4 -- bash -c '
  set -euo pipefail
  export PATH="$CPU_JAX_VENV/bin:$PATH"
  export LORRAX_CPU_SKIP_GPU_PLUGINS=1
  export OMP_NUM_THREADS=14
  . "$LORRAX_CHECKOUT/config/perlmutter/cpu_mpi_env.sh"
  python3 -u "$LORRAX_CHECKOUT/tools/require_jax09.py"
  python3 -u -m gw.gw_jax -i gw.in
'
```

Both pins are required. `LORRAX_CHECKOUT` prevents `lx` from falling back to
the base module's source when invoked from a data directory, while
`CPU_JAX_VENV` selects the tested CPU JAX environment inside the rank shell.
The tracked preflight checks JAX and jaxlib before either is imported.

The source inside the `lx` rank shell is the load-bearing one.  It sets the
four-part Cray contract before JAX import:

| setting | why |
|---|---|
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` | selects JAX's non-corrupting MPI backend |
| `MPITRAMPOLINE_LIB=<unmodified Cray-built MPIwrapper>` | supplies the ABI adapter expected by JAX's bundled MPItrampoline |
| `LD_PRELOAD=/opt/cray/pe/lib64/libpmi.so.0` | loads the verified Cray PMI implementation before `jax.distributed` starts coordination threads; omitting it segfaults in `_pmi_spawn_init`; `libpmi2.so.0` is a measured negative |
| `MPICH_ASYNC_PROGRESS=1` | HPE's public control; promotes XLA's explicit FUNNELED request to `MPI_THREAD_MULTIPLE`, required when XLA executor threads coexist with native MPI I/O/linalg |

Async progress consumes a progress thread. The example requests two fewer
OpenMP threads than affinity slots, but this is not a certified reservation:
XLA workers do not obey `OMP_NUM_THREADS`, and progress-thread placement was
not measured. The prelude also forces
`MPICH_GPU_SUPPORT_ENABLED=0` and unsets the retired
`LORRAX_MPI_FORCE_THREAD_MAIN`; communicator creation is owned by
`common.collectives.warm_mesh_cliques()`.

### CPU rank threads: one affinity mask, several thread populations

`-c`/`LORRAX_CPUS_PER_TASK` gives each MPI rank an **allowed set of logical
CPUs**. It does not divide that set among the work performed by the rank. A
CPU rank can contain all of these at once:

| population | controlled by | important limitation |
|---|---|---|
| XLA CPU workers for compiled `jax.numpy` operations | XLA and the rank affinity mask | `jax.numpy` is lowered by XLA, not executed by NumPy; one JAX device per rank is not one CPU thread; `OMP_NUM_THREADS` does not cap this pool |
| LibSci/BLAS/SLATE OpenMP teams | `OMP_NUM_THREADS` and handler-specific controls printed in the startup block | the team is not assigned a private subset of the rank's CPUs |
| Cray MPICH progress thread | `MPICH_ASYNC_PROGRESS=1` | required by the current route to obtain `MPI_THREAD_MULTIPLE`; it is not automatically given a reserved CPU |
| Python/runtime and asynchronous-I/O threads | the OS within the same affinity mask | usually small, but can overlap compilation, I/O, and native calls |

Unless a launcher supplies a measured binding policy, these threads inherit
the same rank affinity mask and may migrate or contend on the same logical
CPUs. `OMP_NUM_THREADS=14` with `-c16` therefore leaves *nominal headroom*; it
does not prove that two CPUs are reserved for XLA and MPI. Synchronous native
calls often leave other XLA workers idle, so oversubscription is also not
proved merely by counting threads.

The measured P=16/four-node smoke used `-c16`, `OMP_NUM_THREADS=14`, one JAX
device per rank, and a live `MPI_THREAD_MULTIPLE` grant. It proved collective
correctness, not thread placement or application scaling. The P=4/two-node
GN-PPM calculation and P=4 ScaLAPACK, SLATE, and PHDF5 controls passed; GN-PPM
selected its in-tree per-q solve and did not exercise ScaLAPACK. Do not turn
off async progress for a performance experiment under this recipe: the live
thread gate will correctly refuse the resulting FUNNELED grant.

The next CPU-performance pass should, in order:

1. record every thread's affinity and CPU residency after XLA initialization
   and during one representative XLA kernel, one LibSci call, and MPI traffic;
2. measure ranks-per-node, CPUs-per-rank, and OpenMP-team-size as a matrix,
   reporting cold compile separately from warm execution;
3. only then add `OMP_PLACES`/`OMP_PROC_BIND`, a supported XLA worker-pool cap,
   or a dedicated progress-thread placement—do not guess a binding policy;
4. repeat with an application at P=16 and then P=64/P=256 before making a
   production-scale or multi-terabyte-workspace claim.

The two unavoidable Perlmutter-specific seams remain the early
`libpmi.so.0` preload and the async-progress request. Everything else is
tracked, fail-closed build/activation plumbing around an unmodified ABI
adapter; MPIwrapper is not a second MPI implementation. A fresh-machine
reproduction still needs the pinned JAX 0.9.1 venv and the versioned Cray
modules named by the builder.

The PMI diagnosis and controls are allocation 57261316: no preload crashes in
`PMI2_Init` (`lx-Xg1-203405-2056047-6411`), the stable SONAME preload passes
(`lx-Xg1-204021-2096105-1906`), and `libpmi2.so.0` still crashes
(`lx-Xg1-203924-2089729-3659`).  Full mechanism and machine split:
[collective transports](../transports.md) and
[`docs/dev/mpi_collectives.md`](../../dev/mpi_collectives.md).
