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

gloo's CPU reduce-scatter corrupts silently (rc=0), gloo is slower on every
collective, and jaxlib 0.9.1's gloo is TCP-only. The measured verdict is owned
by [MPI collectives § Why not gloo](../dev/mpi_collectives.md#why-not-gloo).

## 2. What `impl=mpi` requires

| requirement | missing it |
|---|---|
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` | gloo, refused at startup |
| `MPITRAMPOLINE_LIB` = an MPIwrapper built for the site MPI | refused at startup; MPItrampoline cannot load a vendor `libmpi.so` directly (it expects MPIwrapper ABI symbols) |
| `common.collectives.warm_mesh_cliques()` on every mesh (called by `collectives.prepare_mesh()` and the mesh factories) | a clique first created inside a jit dies on every rank with jaxlib's communicator refusal |
| `runtime.run_main_and_finalize()` at every driver boundary | interpreter teardown can call MPI after XLA finalized, turning a successful run into rc=1 |
| a live `MPI_THREAD_MULTIPLE` grant | startup refuses before XLA builds its cliques; the native FFI aborts the MPI world before its first collective |

How each machine obtains the MULTIPLE grant, and the adapter builds:
[MPI collectives § The adapter](../dev/mpi_collectives.md#the-adapter).
`MPITRAMPOLINE_LIB` has no default in `src/`: it names a site build artifact.

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
