# Collective transports

How LORRAX moves collective data on each platform, and why. The wrapper
mechanism (the jaxlib thread guard, the MPIwrapper patch) is owned by
[`docs/dev/mpi_collectives.md`](../dev/mpi_collectives.md).

## The map

```
CPU collectives (JAX_CPU_COLLECTIVES_IMPLEMENTATION)
└── mpi   via MPItrampoline → MPIwrapper ABI adapter
          ├── Frontera:   patched adapter → Intel MPI → libfabric mlx
          └── Perlmutter: unmodified adapter → Cray MPICH → Slingshot
GPU collectives — NCCL through XLA; cuSOLVERMp and cuBLASMp also use NCCL
Native MPI (phdf5, SLATE, ScaLAPACK) — the site MPI, linked directly
```

| run | collectives |
|---|---|
| single process | none |
| multi-process CPU | `impl=mpi`; startup refuses gloo and an unset or missing `MPITRAMPOLINE_LIB` |
| GPU | NCCL via XLA ([Perlmutter transport](machines/perlmutter.md#network-transport-at-startup)) |

## 1. Why `impl=mpi`

- **gloo's reduce-scatter corrupts silently.** `jax.lax.psum_scatter` over a
  2-D CPU mesh intermittently returns wrong data with rc=0: about 5 % of
  executions and 80 % of process lifetimes, always output segment 0, with an
  error of order the answer, reproducible with no LORRAX imports. The
  identical program under `impl=mpi` was clean in 504/504 executions while a
  gloo control in the same allocations corrupted 4/4 lifetimes.
- **gloo is also slower.** On 1.12 GB all-reduce / 2.24 GB all-gather /
  1.12 GB reduce-scatter payloads: mpi 0.83 / 1.05 / 0.63 s, gloo 14.99 /
  31.11 / 11.98 s; end to end, 1.18× at P=16.
- **gloo in jaxlib 0.9.1 is TCP-only;** `GLOO_SOCKET_IFNAME` is inert.

## 2. What `impl=mpi` requires

| requirement | missing it |
|---|---|
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` | gloo, refused at startup |
| `MPITRAMPOLINE_LIB` = an MPIwrapper built for the site MPI | refused at startup; MPItrampoline cannot load a vendor `libmpi.so` directly (it expects MPIwrapper ABI symbols) |
| `common.collectives.warm_mesh_cliques()` on every mesh (called by `collectives.prepare_mesh()` and the mesh factories) | a clique first created inside a jit dies on every rank with jaxlib's communicator refusal |
| `runtime.run_main_and_finalize()` at every driver boundary | interpreter teardown can call MPI after XLA finalized, turning a successful run into rc=1 |
| a live `MPI_THREAD_MULTIPLE` grant | startup refuses before XLA builds its cliques; the native FFI aborts the MPI world before its first collective |

The thread grant is machine-specific: Frontera's patched MPIwrapper upgrades
the request; Perlmutter's `config/perlmutter/cpu_mpi_env.sh` sets
`MPICH_ASYNC_PROGRESS=1` (which promotes XLA's FUNNELED request) and preloads
`/opt/cray/pe/lib64/libpmi.so.0` so Cray PMI initializes before JAX's
coordination threads exist. `MPITRAMPOLINE_LIB` has no default in `src/`: it
names a site build artifact.

## 3. The Intel MPI provider layer (Frontera)

`config/frontera/mpi_transport_env.sh` owns the PMI2 glue, fabrics and
provider selection; source it, never copy its exports.

| `LORRAX_MPI_PROVIDER` | provider | latency / bandwidth |
|---|---|---|
| `auto` (default; `FI_PROVIDER` unset) | Intel MPI selects mlx (UCX/RDMA) | 1.07 µs / 11.4 GB/s |
| `tcp` | IPoIB via `ib0`; for ConnectX-3 nodes only | 10.9 µs / 2.15 GB/s |

The provider decides distributed linear-algebra cost directly: `pzheevd` at
n=2448, P=144 takes 0.5–0.9 s per q on mlx and about 12 s on tcp. Read the
`I_MPI_DEBUG>=4` `libfabric provider:` banner; `fi_info` reports mlx as absent
when it works.

## 4. The FFI libraries' own MPI

The native libraries link the site MPI directly, not through MPItrampoline.
`ffi/cpp/phdf5/context.cc` and `ffi/cpp/slate/context.cc` call
`MPI_Init_thread(MULTIPLE)` only when nothing initialized MPI first, so they
coexist with XLA's initialization. On Perlmutter Cray MPICH GPU support is
off (`MPICH_GPU_SUPPORT_ENABLED=0`, its GTL is built for CUDA 12): phdf5
moves host buffers, and the GPU distributed libraries communicate through
NCCL.
