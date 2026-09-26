# JAX CPU collectives on MPI (`impl=mpi`) and the MPIwrapper adapter

Multi-process CPU runs use `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`.
[Collective transports](../environment/transports.md) is the map of which
transport runs where; this page owns the mechanism: the jaxlib guard and the
clique warm-up that satisfies it, the thread-level requirement, the
MPIwrapper adapter per machine, and the launch compositions.

## Why not gloo

jaxlib 0.9.1 offers `gloo` (its default) and `mpi`.

* **gloo's reduce-scatter silently corrupts.** `lax.psum_scatter` over a 2-D
  mesh returns wrong data with rc = 0 in about 5 % of executions (always
  output segment 0, error of order the answer), reproducible with no LORRAX
  imports. `impl=mpi` on the identical program is clean in 504/504
  executions, with a gloo positive control corrupting 4 of 4 process
  lifetimes in the same allocations.
* **mpi is faster.** On identical payloads (1.12 GB all-reduce, 2.24 GB
  all-gather, 1.12 GB reduce-scatter) mpi takes 0.83 / 1.05 / 0.63 s and gloo
  14.99 / 31.11 / 11.98 s; collective-bound stages are 1.4–8.2× faster.
* **gloo here has no non-TCP transport**; `GLOO_SOCKET_IFNAME` is inert.

`runtime.announce_cpu_collectives()` prints the resolved implementation once
from rank 0 and warns when a multi-process CPU run is on gloo.

## The jaxlib guard, and the warm-up that satisfies it

`xla::cpu::MpiCollectives::CreateCommunicators()` refuses with
"Communicator requested from a thread that is not the one MPI was initialized
from" unless `MPI_Is_thread_main` is true, and only then calls
`MPI_Comm_split(MPI_COMM_WORLD, …)`. Three properties decide the design:

* It is a thread-identity test, not a thread-level test: `MPI_Is_thread_main`
  is false on every non-initialising thread even under `MPI_THREAD_MULTIPLE`.
* It fires only on communicator **creation**. `xla::cpu::AcquireCommunicator`
  caches communicators in a process-global map keyed only by the
  participating-device set, and the collectives themselves carry no check.
* Whether a program trips it depends on XLA:CPU's executor.
  `ThunkExecutor::ExecuteSequential` runs thunks inline on the caller thread
  (small programs pass); the parallel executor dispatches to intra-op pool
  workers (real programs fail). No XLA flag forces the sequential executor.

`common.collectives.warm_mesh_cliques(mesh)` therefore creates every clique
the mesh will use (one per mesh axis **and** the world clique; any subset
fails) from the main thread, inside a jit small enough (one 8-byte buffer,
≤ 8 thunks) to run sequentially. Every later acquisition, including from a
pool worker, is a cache hit. Cost: three 8-byte `psum`s once per process,
independent of μ, k, q and P; it changes no compiled HLO. Creating all cliques
from one thread in a fixed order also removes the cross-rank ordering hazard
of calling the world-collective `MPI_Comm_split` from arbitrary pool workers.

Contract: call `common.collectives.prepare_mesh()` (`resolve_mesh` +
`warm_mesh_cliques` + `runtime.nccl_warmup`) once per mesh before any jit,
synchronously on every rank. The two warm-ups are deliberately separate: the
CPU one works because its program is small enough to run inline, and the NCCL
one exists to force `ncclCommInitRank` topology discovery. Both are no-ops off
their platform, at P = 1, and on an already-warmed mesh.
`contract_bands_block_reshard` also warms its mesh at factory time.

## Thread level: MULTIPLE is required

XLA's `MpiCollectives::Init()` requests `MPI_THREAD_FUNNELED` and never reads
`provided`. XLA's collectives run on pool threads while native parallel HDF5
and distributed linalg use MPI from other threads, which is undefined behaviour
below `MPI_THREAD_MULTIPLE` (measured on Intel MPI: segfaults and hangs at the
ζ-write/`V_q` boundary in 4 of 14 P = 16 runs, two threads of one rank inside
`MPID_Progress_wait`). Two checks enforce it:

* multi-process CPU/MPI startup queries the live grant through
  `MPIABI_Query_thread` on `MPITRAMPOLINE_LIB` and refuses below MULTIPLE,
  before any XLA clique exists;
* `ffi/cpp/common/mpi_thread_guard.h` (phdf5, SLATE) calls
  `MPI_Init_thread(MULTIPLE)` only when nothing initialised MPI first, and
  `MPI_Abort`s the world before its first collective when the grant is below
  MULTIPLE.

## The adapter

JAX's bundled MPItrampoline loads the library named by `MPITRAMPOLINE_LIB`,
which must be an MPIwrapper built for the site MPI (not the vendor `libmpi`).
Both machines pin upstream MPIwrapper v2.11.1 by commit SHA. How MULTIPLE is
obtained differs:

| machine | adapter | MULTIPLE comes from |
|---|---|---|
| Frontera / Intel MPI | `config/frontera/build_mpiwrapper.sh`: upstream plus `config/frontera/mpiwrapper/lorrax_thread.patch` | the patch forwards every `MPI_Init`/`MPI_Init_thread` to `PMPI_Init_thread(…, MPI_THREAD_MULTIPLE, …)`; requests are upgraded, never downgraded |
| Perlmutter / Cray MPICH | `config/perlmutter/build_mpiwrapper.sh`: unmodified upstream, refuses a dirty checkout | `MPICH_ASYNC_PROGRESS=1`, which makes Cray MPICH grant MULTIPLE to XLA's explicit FUNNELED request (it adds a progress thread per rank) |

`I_MPI_THREAD_LEVEL_DEFAULT` and `MPIR_CVAR_DEFAULT_THREAD_LEVEL` do not work:
MPICH grants the explicit request, not the default. mpi4py, h5py and the FFI
host `.so` link the site `libmpi` directly and never see the adapter.

The Frontera patch also carries an `MPI_Is_thread_main` override gated on
`LORRAX_MPI_FORCE_THREAD_MAIN`; production leaves it unset (the Perlmutter
prelude unsets it), because the warm-up satisfies the guard and setting it
would only hide a missing warm-up call site.

**Build verification.** A wrapper that grants FUNNELED loads exactly like a
good one, so the Frontera build checks machine code: it disassembles
`MPIABI_Init_thread` and asserts `required` is hard-set to 3, and checks that
`MPIABI_Is_thread_main` falls through to `PMPI_Is_thread_main` when the gate is
unset. `LORRAX_MPIWRAPPER_REFERENCE_SO` compares `.text` against a known-good
build. The Perlmutter build uses Cray `cc`/`CC`/`ftn`, checks the MPItrampoline
ABI exports, rejects CUDA-GTL and Darshan dependencies, and runs the one-MPI
dynamic-closure gate; release names carry the adapter content hash and the
recipe hashes.

```bash
export LORRAX_ROOT=/path/to/lorrax
config/frontera/build_mpiwrapper.sh --fresh      # Intel MPI, patched
config/perlmutter/build_mpiwrapper.sh --fresh    # Cray MPICH, upstream
```

## Launch

### Perlmutter

Build on a CPU compute node, and source `config/perlmutter/cpu_mpi_env.sh` in
every rank shell before Python. The prelude validates the adapter's source pin,
MPI ABI and SHA-256 manifest; rejects stale overlays and conflicting MPI/PMI
preloads; forces CPU, one JAX device per rank and `impl=mpi`; sets
`LD_PRELOAD=/opt/cray/pe/lib64/libpmi.so.0` (Cray PMI must initialise before
JAX coordination threads exist, otherwise MPI init segfaults in
`PMI2_Init`; `libpmi2.so.0` does not fix it) and `MPICH_ASYNC_PROGRESS=1`;
disables Cray GPU support. It does not set an OpenMP team size.

```bash
export LORRAX_CHECKOUT=/path/to/lorrax
export LORRAX_ROOT="$LORRAX_CHECKOUT"
export CPU_JAX_VENV=/path/to/jax-0.9.1-venv
export LORRAX_CPUS_PER_TASK=16    # read by lx: srun -c for --cpu steps (default 8)
export PYTHONPATH="$LORRAX_CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}"
lx run --cpu --pool POOL -N 2 -n 4 -- bash -c '
  set -euo pipefail
  export PATH="$CPU_JAX_VENV/bin:$PATH"
  export LORRAX_CPU_SKIP_GPU_PLUGINS=1
  export OMP_NUM_THREADS=14
  . "$LORRAX_CHECKOUT/config/perlmutter/cpu_mpi_env.sh"
  python3 -u "$LORRAX_CHECKOUT/tools/require_jax09.py"
  python3 -u -m gw.gw_jax -i gw.in
'
```

`tools/require_jax09.py` refuses a JAX/JAXLIB series other than 0.9 without
importing JAX ([JAX support](jax_support.md)). XLA workers ignore
`OMP_NUM_THREADS`; the progress-thread/XLA-thread affinity is not yet a
certified policy.

### Frontera

`config/frontera/templates/gw_dev.sbatch` is the executable launch block.
Its load-bearing pieces:

| piece | omit it and |
|---|---|
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` | the run is on gloo's corrupting reduce-scatter |
| `MPITRAMPOLINE_LIB` → the patched adapter | MPItrampoline refuses at startup |
| `. config/frontera/mpi_transport_env.sh` | PMI2 glue, fabrics and the provider block are missing; provider policy is in [transports](../environment/transports.md#3-the-intel-mpi-provider-layer-frontera) |
| the overlay (`config/frontera/build_mpi_overlay.sh`: mpi4py, parallel h5py, `sitecustomize`) with `LORRAX_MPI_FINALIZE_FIX=skip_atexit` | post-atexit teardown makes an MPI call after finalize and a successful run exits rc = 1 |

`MPITRAMPOLINE_LIB` is never defaulted from `src/`: it names a build artifact
outside the repo, and the choice must stay visible in the launcher.

Every core driver's CLI boundary runs through `runtime.run_main_and_finalize`,
which enters the ordered MPI/JAX/FFI shutdown on normal exit and, at P > 1,
exits immediately without collective teardown on an unexpected exception.
