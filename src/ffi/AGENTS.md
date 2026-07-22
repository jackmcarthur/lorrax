# `src/ffi/` — XLA FFI bridge

Compiled-library call-sites for JAX: one shared object
(`liblorrax_ffi.so`), one ctypes loader, one Shifter launcher.  Currently
ships five targets, all validated on NERSC Perlmutter (1–4 nodes × 4×A100):

| Subpackage | Library | Process model | Smoke test | Status |
|---|---|---|---|---|
| `cusolvermp` | cuSOLVERMp (multi-proc multi-GPU, NCCL-backed CAL) | 1 proc per GPU | `common.cusolvermp_eigh_test`, `common.cusolvermp_batched_test`, `tests/test_ffi_linalg_contract.py` | potrf/potrs/getrf+getrs 1e-16–1e-14 on 1×1/2×2/4×1/1×4; syevd square meshes only (rect mesh DEADLOCKS — wrapper rejects) |
| `cublasmp` | cuBLASMp (batched gemm + fused W-solve) | 1 proc per GPU | `common.cublasmp_gemm_test`, `common.cublasmp_w_solve_test`, contract tests | 1e-16–1e-14 on all meshes.  Comm ABI must match the LOADED cuBLASMp generation (≥0.5.0 = NCCL) — see `scripts/stage_cublasmp_redist.sh` |
| `slate` | SLATE (MPI + GPU tile linalg; AMD-portable path) | 1 proc per GPU | `common.slate_cholesky_trsm_test`, `common.slate_batched_test`, contract tests | potrf/trsm/heev ~1e-16/1e-14 on p==q or 1-D meshes; see [`slate/README.md`](slate/README.md) |
| `phdf5`      | parallel HDF5 via MPI-IO (read + write sharded slabs) | 1 proc per GPU | `common.phdf5_write_test`, `common.phdf5_multi_offset_test` | 0.000e+00 round-trip; 4 / 9 GB/s write / read @ 16 GPUs. See [`phdf5/ARCHITECTURE.md`](phdf5/ARCHITECTURE.md) for the async-design rationale and the non-obvious pitfalls encountered along the way. |
| `scalapack` | ScaLAPACK from Cray LibSci (HOST platform — JAX CPU backend) | 1 proc per rank | `tests/test_ffi_linalg_contract.py` (`scalapack_*` cells) | pXgetrf+pXgetrs fused per-q LU (`distributed_lu = scalapack`); square + 1-D meshes (square-block requirement); zero extra link deps (libsci already in `liblorrax_ffi_host.so`) |

Multi-process targets share the same bootstrap pattern (KV-store broadcast
of a unique handle → `cal_comm_create` / `H5Fcreate`) — the scaffold for
any future distributed solver (ELPA, `H5Dwrite_async`, etc.).

## Cold start (fresh clone on Perlmutter)

```bash
cd sources/lorrax

src/ffi/cusolvermp/scripts/stage_nvhpc.sh       # ~100 MB → /pscratch, one-time
src/ffi/phdf5/scripts/stage_openmpi.sh          #  ~40 MB → /pscratch, one-time

lxalloc                                          # 1 node × 4 GPUs
export SLURM_JOBID=<from lxalloc output>
src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
```

Staging copies are mandatory because Shifter's `udiRoot.conf` on Perlmutter
forbids `--volume` sources under `/opt/*` or `$HOME` — only `/pscratch` is
bind-mountable.  Both stage scripts are idempotent, cache downloads next
to themselves, and print a `readelf -d` verification at the end.

For the Cray MPICH stack (opt-in; currently unstable for large collective
writes) or non-NERSC clusters, see [PORTING.md](PORTING.md).

## Layout

```
ffi/
├── AGENTS.md            ← you are here
├── PORTING.md           per-cluster + MPI stack notes, known issues
├── TEMPLATE.md          skeleton for a new target
├── common/
│   ├── ffi_loader.py    ctypes-loads the per-platform .so's, registers
│   │                    handlers (CUDA → liblorrax_ffi.so; cpu →
│   │                    liblorrax_ffi_host.so; same target names, jaxlib-style)
│   ├── broadcast.py     JAX-KV-store broadcast helpers
│   └── cpp/
│       ├── CMakeLists.txt   single build producing liblorrax_ffi.so
│       ├── build.sh         invokes cmake + make inside Shifter
│       ├── run_shifter.sh   Shifter launcher + MPI stack switch
│       ├── ffi_helpers.h    LORRAX_*_CHECK + FFI_RETURN_IF_ERROR
│       ├── api.cc           extern "C" ABI for ctypes
│       └── host/            CUDA-free liblorrax_ffi_host.so (slate host
│           │                handlers; JAX CPU backend)
│           ├── CMakeLists.txt
│           └── build_host.sh    host-side Cray PE build (no container)
├── cusolvermp/
│   ├── {__init__,context,eigh}.py     public API + NCCL bootstrap + shard_map
│   ├── scripts/stage_nvhpc.sh         copy cuSOLVERMp/libcal to /pscratch
│   └── cpp/{ctx.h,context.cc,eigh_ffi.cc}
│   ├── {__init__,eigh}.py
└── phdf5/
    ├── {__init__,context,write,read}.py    open/close/write/read_sharded_slab
    ├── scripts/{stage_openmpi,stage_cray}.sh
    └── cpp/
        ├── ctx.h                 PhdfCtx (MPI_Comm + cached HDF5 plists + pinned buf)
        ├── context.cc            MPI_Init_thread + H5F*/H5D* + NERSC MPI-IO hints
        ├── phdf5_interface.h     dtype → hid_t template trait
        └── {write,read}_ffi.cc   XLA_FFI_DEFINE_HANDLER_SYMBOL(Phdf{Write,Read}Ffi)
```

## Build

```bash
src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
```

Output: `src/ffi/common/cpp/build/liblorrax_ffi.so`.  CMake prints the
resolved HDF5 + MPI paths so build logs confirm the right stack.  To
build against the opt-in Cray MPICH stack, prefix with
`LORRAX_PHDF5_MPI_STACK=mpich`.

## Smoke tests

```bash
# cusolvermp — 4 ranks × 4 GPUs
LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \
    CUSOLVERMP_FORCE_NCCL=1 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    python3 -u -m common.cusolvermp_eigh_test --grid 2 2

# phdf5 round-trip (write + parallel read, exact equality)
LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async HDF5_USE_FILE_LOCKING=FALSE \
    python3 -u -m common.phdf5_write_test

# phdf5 bench @ 16 GPUs / 4 nodes (write or read; pick one)
LORRAX_NNODES=4 LORRAX_NGPU=4 LORRAX_NTASKS=16 \
    src/ffi/common/cpp/run_shifter.sh env \
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
  `H5Pset_fapl_mpio` + NERSC I/O tuning (`cb_nodes=world_size`,
  `cb_buffer_size=64M`, `striping_factor=16`, 4 MiB alignment + stripe
  unit, `H5D_FILL_TIME_NEVER`).  Handler: D2H-to-pinned on a private
  CUDA stream, then blocking `H5Dwrite`/`H5Dread` — host thread parks
  for the IO, XLA's device stream stays free for queued compute.  One
  write/read in flight at a time (shared pinned buffer).

## Adding a new target

1. `src/ffi/<lib>/cpp/<feature>_ffi.cc`: `XLA_FFI_DEFINE_HANDLER_SYMBOL`.
   Templates on dtype: follow `phdf5/cpp/write_ffi.cc`.  Stateful
   context: follow `cusolvermp/cpp/eigh_ffi.cc`.
2. Append the `.cc` to `LORRAX_FFI_SOURCES` in `common/cpp/CMakeLists.txt`;
   add any `-l<lib>` to `target_link_libraries`.
3. Register the target in `_FFI_TARGET_SYMBOLS` in `common/ffi_loader.py`;
   add ctypes decls for any lifecycle C entry points.
4. Communicator bootstrap: reuse `cusolvermp/context.py` (NCCL / CAL) or
   `phdf5/cpp/context.cc`'s `ensure_mpi_initialized` (MPI).
5. Host-file deps (SDK, module install): `src/ffi/<lib>/scripts/stage_*.sh`
   following `stage_nvhpc.sh` / `stage_openmpi.sh`.  Extend
   `run_shifter.sh` to bind-mount the stage to a stable container path.
6. Python wrapper `src/ffi/<lib>/<feature>.py`: `shard_map` →
   `jax.ffi.ffi_call`.  Re-export from `src/ffi/<lib>/__init__.py`.
7. Rebuild.

See [TEMPLATE.md](TEMPLATE.md) for a fuller walkthrough.
