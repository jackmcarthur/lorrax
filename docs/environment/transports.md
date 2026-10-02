# Collective transports

A multi-process LORRAX run moves data through three kinds of channel: XLA's
collectives between JAX processes, the communication inside the distributed
vendor libraries, and the MPI-IO of parallel HDF5. This page states which
transport each uses on each platform and why, and the mechanism that makes
JAX's CPU collectives run on MPI: the jaxlib guard and the clique warm-up
that satisfies it, the thread level, and the MPIwrapper adapter. It is for
anyone launching a CPU run or porting to a new MPI. The launch recipes are on
the machine pages ([Perlmutter](machines/perlmutter.md#cpu),
[Frontera](machines/frontera.md)).

## The map

```
CPU collectives (JAX_CPU_COLLECTIVES_IMPLEMENTATION)
└── mpi   via MPItrampoline → MPIwrapper ABI adapter
          ├── Frontera:   patched adapter → Intel MPI → libfabric mlx
          └── Perlmutter: unmodified adapter → Cray MPICH → Slingshot
GPU collectives — NCCL through XLA; cuSOLVERMp and cuBLASMp also use NCCL
Native MPI (parallel HDF5, SLATE, ScaLAPACK) — the site MPI, linked directly
```

| run | JAX collectives |
|---|---|
| one process | none |
| several CPU processes | `impl=mpi`; startup refuses gloo and an unset or missing `MPITRAMPOLINE_LIB` |
| GPU | NCCL through XLA ([Perlmutter network transport](machines/perlmutter.md#network-transport-at-startup)) |

## Why CPU collectives run on MPI {#why-mpi}

jaxlib 0.9.1 offers two CPU collective implementations, `gloo` (its default)
and `mpi`. LORRAX refuses gloo for three measured reasons:

- **gloo's reduce-scatter silently corrupts.** `lax.psum_scatter` over a 2-D
  mesh returns wrong data with rc = 0 in about 5 % of executions (always
  output segment 0, with an error of the size of the answer), reproducible
  with no LORRAX code. `impl=mpi` on the same program was clean in 504 of 504
  executions, while a gloo control in the same allocations corrupted 4 of 4
  process lifetimes.
- **mpi is faster.** On the same payloads (1.12 GB all-reduce, 2.24 GB
  all-gather, 1.12 GB reduce-scatter) mpi takes 0.83 / 1.05 / 0.63 s and gloo
  14.99 / 31.11 / 11.98 s; collective-bound stages run 1.4–8.2× faster.
- **gloo here is TCP-only**; `GLOO_SOCKET_IFNAME` has no effect.

`runtime.announce_cpu_collectives()` prints the resolved implementation once
from rank 0.

## What `impl=mpi` requires {#requirements}

| requirement | missing it |
|---|---|
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` | gloo, refused at startup |
| `MPITRAMPOLINE_LIB` = an MPIwrapper built for the site MPI | refused at startup; MPItrampoline cannot load a vendor `libmpi.so` directly, because it expects MPIwrapper's ABI symbols |
| every mesh's cliques created by `common.collectives.prepare_mesh()` before any `jit` ([below](#warm-up)) | a clique first created inside a `jit` dies on every rank with jaxlib's communicator refusal |
| a live `MPI_THREAD_MULTIPLE` grant ([below](#thread-level)) | startup refuses before XLA builds its cliques; the native FFI aborts the MPI world before its first collective |
| `runtime.run_main_and_finalize()` at every driver boundary | interpreter teardown can call MPI after XLA finalized it, turning a successful run into rc = 1 |

`MPITRAMPOLINE_LIB` has no default in `src/`: it names a site build artifact,
and that choice stays visible in the launcher.

## The jaxlib guard, and the warm-up that satisfies it {#warm-up}

`xla::cpu::MpiCollectives::CreateCommunicators()` refuses with
"Communicator requested from a thread that is not the one MPI was initialized
from" unless `MPI_Is_thread_main` is true, and only then calls
`MPI_Comm_split(MPI_COMM_WORLD, …)`. Three properties of the guard decide the
design:

- It tests thread identity, not thread level: `MPI_Is_thread_main` is false
  on every thread but the initializing one, even under
  `MPI_THREAD_MULTIPLE`.
- It fires only when a communicator is **created**.
  `xla::cpu::AcquireCommunicator` caches communicators in a process-global
  map keyed by the participating-device set, and the collectives themselves
  carry no check.
- Whether a program trips it depends on XLA:CPU's executor.
  `ThunkExecutor::ExecuteSequential` runs thunks inline on the calling thread
  (small programs pass); the parallel executor dispatches to intra-op pool
  workers (real programs fail). No XLA flag forces the sequential executor.

`common.collectives.warm_mesh_cliques(mesh)` therefore creates every clique
the mesh will use (one per mesh axis **and** the world clique; any subset
fails) from the main thread, inside a `jit` small enough (one 8-byte buffer,
at most 8 thunks) to run sequentially. Every later acquisition, from any pool
worker, is a cache hit. The cost is three 8-byte `psum`s once per process,
independent of μ, k, q and P, and no compiled HLO changes. Creating all
cliques from one thread in a fixed order also removes the cross-rank ordering
hazard of calling the world-collective `MPI_Comm_split` from arbitrary pool
workers.

Contract: call `common.collectives.prepare_mesh()` (`resolve_mesh`, then
`warm_mesh_cliques`, then `runtime.nccl_warmup`) once per mesh before any
`jit`, synchronously on every rank. The two warm-ups stay separate: the CPU
one works because its program is small enough to run inline, and the NCCL
one exists to force `ncclCommInitRank` topology discovery. Both are no-ops
off their platform, at P = 1 and on an already-warmed mesh.

## Thread level: `MPI_THREAD_MULTIPLE` {#thread-level}

XLA's `MpiCollectives::Init()` requests `MPI_THREAD_FUNNELED` and never reads
the level it was granted. But XLA's collectives run on pool threads while
parallel HDF5 and the distributed linear algebra call MPI from other threads,
which is undefined behaviour below `MPI_THREAD_MULTIPLE`; on Intel MPI it
segfaulted or hung at the ζ-write / $V_q$ boundary in 4 of 14 P = 16 runs,
with two threads of one rank inside `MPID_Progress_wait`. Two checks enforce
MULTIPLE:

- multi-process CPU startup queries the live grant through
  `MPIABI_Query_thread` on `MPITRAMPOLINE_LIB` and refuses below MULTIPLE,
  before any XLA clique exists;
- `src/ffi/cpp/common/mpi_thread_guard.h` (parallel HDF5, SLATE)
  calls `MPI_Init_thread(MULTIPLE)` only when nothing initialized MPI first,
  and `MPI_Abort`s the world before its first collective when the grant is
  below MULTIPLE.

`I_MPI_THREAD_LEVEL_DEFAULT` and `MPIR_CVAR_DEFAULT_THREAD_LEVEL` do not
help: MPICH grants the explicit request, not the default. Each machine
obtains MULTIPLE through its adapter ([below](#adapter)).

## The MPIwrapper adapter {#adapter}

JAX's bundled MPItrampoline loads the library `MPITRAMPOLINE_LIB` names,
which must be an MPIwrapper built for the site MPI. Both machines pin
upstream MPIwrapper v2.11.1 by commit SHA, because a tag can be moved.

| machine | build | MULTIPLE comes from |
|---|---|---|
| Frontera, Intel MPI | `config/frontera/build_mpiwrapper.sh`: upstream plus `config/frontera/mpiwrapper/lorrax_thread.patch` | the patch forwards every `MPI_Init` / `MPI_Init_thread` to `PMPI_Init_thread(…, MPI_THREAD_MULTIPLE, …)`; requests are raised, never lowered |
| Perlmutter, Cray MPICH | `config/perlmutter/build_mpiwrapper.sh`: unmodified upstream; refuses a dirty checkout | `MPICH_ASYNC_PROGRESS=1`, which makes Cray MPICH grant MULTIPLE to XLA's explicit FUNNELED request (it adds one progress thread per rank) |

mpi4py, h5py and the host FFI library link the site `libmpi` directly and
never see the adapter.

The Frontera patch also carries an `MPI_Is_thread_main` override gated on
`LORRAX_MPI_FORCE_THREAD_MAIN`. Production leaves it unset (the Perlmutter
prelude unsets it): the warm-up already satisfies the guard, and the override
would only hide a missing warm-up call site.

**Build verification.** A wrapper that grants only FUNNELED loads exactly
like a good one, so the builds check the artifact itself. The Frontera build
disassembles `MPIABI_Init_thread` and asserts that `required` is hard-set to
3, and checks that `MPIABI_Is_thread_main` falls through to
`PMPI_Is_thread_main` when the gate is unset; `LORRAX_MPIWRAPPER_REFERENCE_SO`
compares `.text` against a known-good build. The Perlmutter build uses the
Cray `cc`/`CC`/`ftn` wrappers, checks the MPItrampoline ABI exports, rejects
CUDA-GTL and Darshan dependencies, runs the one-MPI dynamic-closure gate, and
names each release by the adapter's content hash and the recipe hashes.

```bash
export LORRAX_ROOT=/path/to/lorrax
config/frontera/build_mpiwrapper.sh --fresh      # Intel MPI, patched
config/perlmutter/build_mpiwrapper.sh --fresh    # Cray MPICH, upstream
```

## The Intel MPI provider layer (Frontera) {#intel-mpi-provider}

`config/frontera/mpi_transport_env.sh` owns the PMI2 glue, fabrics and
provider selection; source it, never copy its exports.

| `LORRAX_MPI_PROVIDER` | provider | latency / bandwidth (Frontera CLX) |
|---|---|---|
| `auto` (default; `FI_PROVIDER` unset) | Intel MPI selects mlx (UCX/RDMA) | 1.07 µs / 11.4 GB/s |
| `tcp` | IPoIB via `ib0`; for ConnectX-3 nodes only | 10.9 µs / 2.15 GB/s |

The provider sets distributed linear-algebra cost directly: `pzheevd` at
n = 2448, P = 144 takes 0.5–0.9 s per q on mlx and about 12 s on tcp. Read
the `I_MPI_DEBUG>=4` `libfabric provider:` banner to see which one ran;
`fi_info` reports mlx as absent even when it works.

## The FFI libraries' own MPI {#ffi-mpi}

The native libraries link the site MPI directly, not through MPItrampoline,
and initialize it only when nothing else has (`mpi_thread_guard.h`), so they
coexist with XLA's initialization. SLATE and ScaLAPACK split
`MPI_COMM_WORLD` in mesh order. On GPU runs no LORRAX path hands a device
buffer to MPI: parallel HDF5 moves host buffers, and cuSOLVERMp and cuBLASMp
communicate through NCCL. That is why Cray MPICH GPU support can stay off on
Perlmutter ([why it must](machines/perlmutter.md#mpich-gpu-support)).

## Launch {#launch}

On Perlmutter every CPU rank sources `config/perlmutter/cpu_mpi_env.sh`
([the recipe and what it sets](machines/perlmutter.md#cpu)).

On Frontera `config/frontera/templates/gw_dev.sbatch` is the launch block.
Its load-bearing pieces:

| piece | omit it and |
|---|---|
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` | the run is on gloo's corrupting reduce-scatter |
| `MPITRAMPOLINE_LIB` → the patched adapter | MPItrampoline refuses at startup |
| `. config/frontera/mpi_transport_env.sh` | the PMI2 glue, fabrics and provider block ([above](#intel-mpi-provider)) are missing |
| the overlay (`config/frontera/build_mpi_overlay.sh`: mpi4py, parallel h5py, `sitecustomize`) with `LORRAX_MPI_FINALIZE_FIX=skip_atexit` | post-atexit teardown makes an MPI call after finalize, and a successful run exits rc = 1 |

Every core driver's command-line boundary runs through
`runtime.run_main_and_finalize`, which enters the ordered MPI/JAX/FFI
shutdown on a normal exit and, at P > 1, exits immediately without collective
teardown on an unexpected exception, so a failing rank cannot hang the
others in a teardown collective.
