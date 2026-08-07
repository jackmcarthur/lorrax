# `src/ffi/` — XLA FFI bridge

Compiled-library call-sites for JAX: one shared object
(`liblorrax_ffi.so`), one ctypes loader, one Shifter launcher.  Currently
ships five targets, all validated on NERSC Perlmutter (1–4 nodes × 4×A100):

| Subpackage | Library | Process model | Smoke test | Status |
|---|---|---|---|---|
| `cusolvermp` | cuSOLVERMp (multi-proc multi-GPU, NCCL-backed CAL) | 1 proc per GPU | `services/distrib_la/tests/test_distrib_la_contract.py` (`pytest -m distrib_la`) | potrf/potrs/getrf+getrs 1e-16–1e-14 on 1×1/2×2/4×1/1×4; syevd square meshes only (rect mesh DEADLOCKS — wrapper rejects) |
| `cublasmp` | cuBLASMp (batched gemm + fused W-solve) | 1 proc per GPU | `common.cublasmp_gemm_test`, `common.cublasmp_w_solve_test`, contract tests | 1e-16–1e-14 on all meshes.  Comm ABI must match the LOADED cuBLASMp generation (≥0.5.0 = NCCL) — see `cpp/stage/cusolvermp_stage_cublasmp_redist.sh` |
| ~~`slate`~~ | **moved** — SLATE now lives in `services/distrib_la/` (`distrib_la._slate`, reached through `distrib_la.backend_module("slate")`).  The python package here was a re-export shim and was deleted by the wave-0 replumb; the C++ tree under `cpp/slate/` stays.  Its notes are [`services/distrib_la/docs/slate_backend.md`](../../services/distrib_la/docs/slate_backend.md). | | | |
| `phdf5`      | parallel HDF5 via MPI-IO (read + write sharded slabs) | 1 proc per GPU | `common.phdf5_write_test`, `common.phdf5_multi_offset_test` | 0.000e+00 round-trip; 4 / 9 GB/s write / read @ 16 GPUs. See [`phdf5/ARCHITECTURE.md`](phdf5/ARCHITECTURE.md) for the async-design rationale and the non-obvious pitfalls encountered along the way. |
| ~~`scalapack`~~ | **moved** — same story: `distrib_la._scalapack`, via `distrib_la.backend_module("scalapack")`.  Still the **ScaLAPACK API** (HOST platform, JAX CPU backend) supplied by whichever implementation the host lib was linked against (Cray LibSci on Perlmutter, Intel MKL on Frontera); pXgetrf+pXgetrs fused per-q LU + pXheevd eigh, square + 1-D meshes, zero extra link deps.  Its cells are `services/distrib_la/tests/test_distrib_la_contract.py` (`scalapack_*`). | | | |

> **Which of these families are vendor-swappable and which are not** —
> with the API each one actually calls, and the evidence — is
> [PORTING.md §0](PORTING.md). Short version: ScaLAPACK, CBLAS and HDF5
> are published APIs with several implementations, so porting them is a
> link line; SLATE, cuSOLVERMp, cuBLASMp, cuFFT and MKL's DFTI have one
> implementation each. `mklblas`/`mklfft` are historical names, not
> vendor assertions.
>
> **SLATE can answer to ScaLAPACK names — that is the portability route,
> and it is one permutation away from working.** SLATE's optional
> `libslate_scalapack_api` re-defines 6 of LORRAX's 11 ScaLAPACK names
> (measured: every compute routine, none of the grid/descriptor tools,
> which every platform's own ScaLAPACK supplies anyway) and forwards them
> to `slate::`, whose CPU/CUDA/ROCm backend is fixed by the *build*. So
> `pzheevd_` can be one symbol over three platforms. Two things stand in
> the way today, both measured: the shims hard-code `MPI_COMM_WORLD`, so
> under the mesh device order LORRAX ships they are **wrong by ~15 % on any
> 2-D mesh** (a Fortran-order mesh swaps which provider is right — the fix
> is LORRAX-side, no upstream patch); and the target defaults to
> `HostTask`, so a GPU build silently runs on the CPU.
> `cpp/scalapack/blacs_grid.h` detects the interposition and refuses by
> default — `ffi_loader` keys only on LORRAX's own handler symbols and
> cannot see it. Full measurement, the fix and its cost:
> [PORTING.md §0b](PORTING.md); what each user's machine gets: §0c.

Multi-process targets share the same bootstrap pattern (KV-store broadcast
of a unique handle → `cal_comm_create` / `H5Fcreate`) — the scaffold for
any future distributed solver (ELPA, `H5Dwrite_async`, etc.).

## Cold start (fresh clone on Perlmutter)

```bash
cd sources/lorrax

src/ffi/cpp/stage/cusolvermp_stage_nvhpc.sh       # ~100 MB → /pscratch, one-time
src/ffi/cpp/stage/phdf5_stage_openmpi.sh          #  ~40 MB → /pscratch, one-time

lxalloc                                          # 1 node × 4 GPUs
export SLURM_JOBID=<from lxalloc output>
src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh
```

Staging copies are mandatory because Shifter's `udiRoot.conf` on Perlmutter
forbids `--volume` sources under `/opt/*` or `$HOME` — only `/pscratch` is
bind-mountable.  Both stage scripts are idempotent, cache downloads next
to themselves, and print a `readelf -d` verification at the end.

For the Cray MPICH stack (opt-in; currently unstable for large collective
writes) or non-NERSC clusters, see [PORTING.md](PORTING.md).

## Layout

```
ffi/                       (full design: docs/architecture/ffi_layout.md)
├── AGENTS.md            ← you are here
├── PORTING.md           per-cluster + MPI stack notes, known issues
├── TEMPLATE.md          skeleton for a new target
├── gate.py  fft.py  gemm.py    facade modules (gate grammar; flat-k FFT
│                        serving BOTH platforms; vendor-CBLAS batched GEMM)
├── linalg/              resolve/plan/dispatch facade over the linalg backends
├── common/
│   ├── ffi_loader.py    ctypes-loads the per-platform .so's, registers
│   │                    handlers (CUDA → liblorrax_ffi.so; cpu →
│   │                    liblorrax_ffi_host.so; same target names, jaxlib-style)
│   └── broadcast.py     JAX-KV-store broadcast helpers
├── cusolvermp/ cublasmp/ phdf5/
│   │                    pure-python service packages (shard_map wrappers,
│   │                    context/bootstrap); mklfft/ mklblas/ cufft/ are
│   │                    re-export shims onto fft.py / gemm.py.
│   │                    slate/ and scalapack/ used to sit here and are
│   │                    services/distrib_la/ now — their C++ is still
│   │                    cpp/slate/ and cpp/scalapack/, which is the point:
│   │                    the SO is one build, the python is per service.
├── _services.py         puts services/*/src on sys.path.  TRANSITIONAL —
│                        see its docstring; it goes with the owner's
│                        install/PYTHONPATH decision.
└── cpp/                 THE one C++ tree (both platform legs)
    ├── CMakeLists.txt   ONE entry point; -DLORRAX_FFI_PLATFORM=cuda|host
    │                    selects the leg, refuses when unset
    ├── build.sh         CUDA leg via cmake inside Shifter → cpp/build/
    ├── build_host.sh    host leg, no container → cpp/build_host/
    ├── run_shifter.sh   Shifter launcher + MPI stack switch
    ├── stage/           vendor stage scripts (cusolvermp_stage_nvhpc.sh,
    │                    phdf5_stage_{openmpi,cray}.sh, slate_*.sh, …)
    ├── common/          ffi_helpers.h, mkl_thread_pin.h, api.cc (ctypes ABI)
    ├── cusolvermp/      ctx.h, context.cc, eigh_ffi.cc, batched_*.cc
    └── phdf5/           ctx.h, context.cc, phdf5_interface.h,
                         {write,read}_ffi.cc   (+ mklfft/ mklblas/ cufft/
                         scalapack/ slate/ cublasmp/ same pattern)
```

## Build

```bash
src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh
```

Output: `src/ffi/cpp/build/liblorrax_ffi.so`.  CMake prints the
resolved HDF5 + MPI paths so build logs confirm the right stack.  To
build against the opt-in Cray MPICH stack, prefix with
`LORRAX_PHDF5_MPI_STACK=mpich`.

## Smoke tests

```bash
# cusolvermp — 4 ranks × 4 GPUs
LORRAX_NGPU=4 src/ffi/cpp/run_shifter.sh env \
    CUSOLVERMP_FORCE_NCCL=1 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    python3 -u -m common.cusolvermp_eigh_test --grid 2 2

# phdf5 round-trip (write + parallel read, exact equality)
LORRAX_NGPU=4 src/ffi/cpp/run_shifter.sh env \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async HDF5_USE_FILE_LOCKING=FALSE \
    python3 -u -m common.phdf5_write_test

# phdf5 bench @ 16 GPUs / 4 nodes (write or read; pick one)
LORRAX_NNODES=4 LORRAX_NGPU=4 LORRAX_NTASKS=16 \
    src/ffi/cpp/run_shifter.sh env \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async HDF5_USE_FILE_LOCKING=FALSE \
    python3 -u -m common.phdf5_vs_gather_bench -n 16384 --iters 3
```

## Required env vars

| Var | Target | Why |
|---|---|---|
| `LORRAX_NGPU` / `LORRAX_NNODES` / `LORRAX_NTASKS` | all | `run_shifter.sh` forwards to `srun --gres=gpu -N -n`. |
| `CUSOLVERMP_FORCE_NCCL=1` | cusolvermp | Routes libcal collectives via NCCL instead of UCC+IB. |
| `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` | cusolvermp, phdf5 | CUDA async mempool so JAX and `cudaMalloc` (libcal, pinned staging) share VRAM. |
| `HDF5_USE_FILE_LOCKING=FALSE` | phdf5 | Lustre doesn't support advisory locks; HDF5 defaults to acquiring them. |
| `LORRAX_PHDF5_MPI_STACK={openmpi,mpich}` | phdf5 | Pick MPI backend (default `openmpi`; see PORTING.md). |

## How it works

Each target's bootstrap + handler flow is self-contained in its `cpp/`:

- **cusolvermp** (`cpp/context.cc`, `cpp/eigh_ffi.cc`): rank 0 creates an
  `ncclUniqueId`, all ranks receive it via JAX's KV store, then
  `cal_comm_create` wraps NCCL allgather callbacks — NVIDIA's documented
  non-MPI CAL bootstrap.  `cusolverMpSyevd` on the resulting grid.
- **phdf5** (`cpp/context.cc`, `cpp/write_ffi.cc`, `cpp/read_ffi.cc`):
  lazy `MPI_Init_thread` on first `open_file`; FAPL cached with
  `H5Pset_fapl_mpio` + I/O tuning (`cb_*` left at ROMIO's automatic
  policy since 2026-07-27; `striping_factor`/`striping_unit` from
  `stripe_policy_count`/`_unit`, which transcribe Python's
  `_stripe_policy(nranks)`; 4 MiB alignment,
  `H5D_FILL_TIME_NEVER`).  Handler: D2H-to-pinned on a private
  CUDA stream, then blocking `H5Dwrite`/`H5Dread` — host thread parks
  for the IO, XLA's device stream stays free for queued compute.  One
  write/read in flight at a time (shared pinned buffer).

## Adding a new target

1. `src/ffi/cpp/<lib>/<feature>_ffi.cc`: `XLA_FFI_DEFINE_HANDLER_SYMBOL`.
   Templates on dtype: follow `cpp/phdf5/write_ffi.cc`.  Stateful
   context: follow `cpp/cusolvermp/eigh_ffi.cc`.
2. Append the `.cc` to `LORRAX_FFI_SOURCES` (CUDA leg) and/or
   `LORRAX_FFI_HOST_SOURCES` (host leg) in `src/ffi/cpp/CMakeLists.txt`;
   add any `-l<lib>` to `target_link_libraries`.
3. Register the target in `_FFI_TARGET_SYMBOLS` in `common/ffi_loader.py`;
   add ctypes decls for any lifecycle C entry points.
4. Communicator bootstrap: reuse `cusolvermp/context.py` (NCCL / CAL) or
   `cpp/phdf5/context.cc`'s `ensure_mpi_initialized` (MPI).
5. Host-file deps (SDK, module install): `src/ffi/cpp/stage/<lib>_stage_*.sh`
   following `cusolvermp_stage_nvhpc.sh` / `phdf5_stage_openmpi.sh`.  Extend
   `run_shifter.sh` to bind-mount the stage to a stable container path.
6. Python wrapper `src/ffi/<lib>/<feature>.py`: `shard_map` →
   `jax.ffi.ffi_call`.  Re-export from `src/ffi/<lib>/__init__.py`.
7. Rebuild.

See [TEMPLATE.md](TEMPLATE.md) for a fuller walkthrough.
