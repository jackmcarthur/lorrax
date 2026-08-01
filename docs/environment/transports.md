# Collective transports

*The three ways LORRAX moves collective data, with the measured verdicts
that chose between them. Deep mechanism (the jaxlib thread guard, the
MPIwrapper patch, the falsified alternatives):
`docs/dev/mpi_collectives.md` — that page is the authority on the wrapper;
this one is the map.*

## The map

```
CPU collectives (JAX_CPU_COLLECTIVES_IMPLEMENTATION)
├── gloo   (jaxlib default)  — TCP only in this jaxlib; RETIRED for
│                              multi-process production (silent corruption)
└── mpi    (LORRAX production) — via MPItrampoline → LORRAX MPIwrapper
                                 → Intel MPI → libfabric provider:
                                 ├── mlx  (UCX/RDMA — the default; fast)
                                 └── tcp  (rtx/mlx4 escape hatch only)
GPU collectives — NCCL (XLA:GPU), plus Cray-MPICH GTL for the FFI
                  libraries' own MPI on Perlmutter
```

## 1. gloo vs `impl=mpi` — the measured verdict

LORRAX ran on gloo through 2026-07-27. Three results moved it off, all
recorded with their controls in `docs/dev/mpi_collectives.md`:

1. **gloo's `reduce-scatter` silently corrupts.** Under gloo,
   `jax.lax.psum_scatter` over a 2-D mesh intermittently returns wrong data
   with no error and rc=0 — ~5 % of executions, ~80 % of process lifetimes,
   always output segment 0, at a magnitude of order the correct answer.
   Reproduces with no LORRAX imports. `impl=mpi` on the identical program:
   clean in **504/504** executions, with a gloo positive control corrupting
   4 of 4 process lifetimes in the same allocations.
2. **The performance case evaporated.** `impl=mpi` is 1.18× end-to-end at
   P=16 against gloo on its ib0 pin; collective-bound stages 1.4–8.2×. On
   identical payloads (1.12 GB all-reduce / 2.24 GB all-gather / 1.12 GB
   reduce-scatter): mpi 0.83 / 1.05 / 0.63 s, gloo 14.99 / 31.11 / 11.98 s.
3. **gloo in jaxlib 0.9.1 has no non-TCP transport.** `GLOO_SOCKET_IFNAME`
   is inert — the string appears in no shipped `.so` (scorecard AF.5).

When each is used today:

* **Single-process runs** — no collectives; the implementation is
  irrelevant.
* **Multi-process CPU** — `impl=mpi`, always. `runtime` warns at startup
  when a multi-process CPU run is using gloo
  (`announce_cpu_collectives`).
* **GPU** — NCCL via XLA; none of this page's CPU machinery is involved
  (see `runtime.nccl_warmup`).

## 2. What `impl=mpi` requires

All four are load-bearing; the certified composition is
`config/frontera/templates/gw_dev.sbatch`:

| requirement | omit it and |
|---|---|
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` | you are on gloo, i.e. on the corrupting reduce-scatter |
| `MPITRAMPOLINE_LIB` → the patched MPIwrapper (`config/frontera/build_mpiwrapper.sh`) | MPItrampoline refuses loudly at startup; an **unpatched** wrapper loads silently and restores the ~29 % multi-node crash/hang class (AS.4b: XLA collectives on pool threads vs h5py/mpi4py MPI-IO on the main thread under a FUNNELED grant) |
| a `warm_mesh_cliques()` call on every mesh (a **code call site**, owned by `collectives.prepare_mesh()` and the mesh factories) | any clique first created inside a real jit dies on every rank with jaxlib's communicator refusal — 32 refusals at P=16 killed the BSE TDA Lanczos (job 7879458 / gate 7881216) |
| `LORRAX_MPI_FINALIZE_FIX=skip_atexit` + the overlay `sitecustomize` | every run exits rc=1 **after succeeding** ("MPI routine after finalizing MPICH") |

`MPITRAMPOLINE_LIB` is deliberately **not** auto-defaulted from `src/`:
it names a build artifact outside the repo, and the hazardous-vs-good
choice must stay visible in the harness.

The wrapper's scope has shrunk to one override: the always-on
`MPI_THREAD_MULTIPLE` upgrade. The `MPI_Is_thread_main` override
(`LORRAX_MPI_FORCE_THREAD_MAIN`) is **superseded** by
`common.collectives.warm_mesh_cliques()` — leave it unset; setting it only
masks a missing warm-up call site. Mechanism, evidence and the falsified
alternatives: `docs/dev/mpi_collectives.md`.

## 3. The Intel MPI provider layer (Frontera)

Under `impl=mpi` (and for phdf5 MPI-IO and mpi4py) the transport is Intel
MPI's libfabric provider. The measured provider policy, recorded in
`config/frontera/README.md`:

* **Leave `FI_PROVIDER` unset** (`LORRAX_MPI_PROVIDER=auto`, the
  `mpi_transport_env.sh` default). Intel MPI then auto-selects the native
  **mlx** (UCX/RDMA) provider — measured 1.07 µs / 11.4 GB/s.
* The old `FI_PROVIDER=tcp` seed measured 10.9 µs / 2.15 GB/s and was the
  root cause of the 30-minute pzheevd era (n=2448 P=144: ~12 s/q under tcp
  vs 0.5–0.9 s/q under mlx; scorecard AP, seed deleted by AU).
* `LORRAX_MPI_PROVIDER=tcp` remains **only** as the rtx/mlx4 (ConnectX-3)
  escape hatch.
* Trust the `I_MPI_DEBUG≥4` `libfabric provider:` banner, never `fi_info`
  (it false-negatives on mlx).

All PMI2 glue, fabrics, the provider case-block and the UCX setdefaults
live in `config/frontera/mpi_transport_env.sh` — source it, never
hand-copy exports.

## 4. Coexistence with the FFI libraries' own MPI

mpi4py, h5py and the FFI host `.so` link Intel `libmpi.so.12` directly and
never route through MPItrampoline — they see neither wrapper override.
`ffi/cpp/phdf5/context.cc` and `ffi/cpp/slate/context.cc` call
`MPI_Init_thread(MULTIPLE)` only when nothing initialized MPI first, so
they coexist with XLA's init by construction. The phdf5 open **warns when
the granted thread level is below MULTIPLE** — that warning firing means
the wrapper is not on the path and the ~29 % race regime is back.

On Perlmutter, GPU-aware Cray MPICH (`MPICH_GPU_SUPPORT_ENABLED=1` +
`libmpi_gtl_cuda.so.0` preload) serves the FFI libraries' collectives —
Cray-specific knobs with no OpenMPI/UCX equivalent
([Perlmutter](machines/perlmutter.md)).
