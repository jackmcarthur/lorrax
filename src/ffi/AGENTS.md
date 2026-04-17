# `src/ffi/` — JAX Foreign Function Interface bridge

Scaffolding for calling compiled (C/C++/CUDA) libraries from JAX via the
XLA FFI.  Current targets — all **working** on 1–4 nodes × 4×A100
(NERSC Perlmutter):

| Subpackage | Library | Process model | Status |
|---|---|---|---|
| `ffi.cusolvermp` | **cuSOLVERMp** (multi-process, multi-GPU / multi-node) | 1 Python proc per GPU | F64 + C128 validated at n=128 |
| `ffi.cusolvermg` | **cuSOLVERMg** (single-process, multi-GPU, in-container) | 1 Python proc × N GPUs | F64 validated at n∈{128, 2048} |
| `ffi.phdf5`      | **parallel HDF5** (MPI-IO, multi-process) | 1 Python proc per GPU | Read + write sharded slabs, 0.000e+00 round-trip, ~4–9 GB/s |

Multi-process via NCCL (cuSOLVERMp) or MPI-IO (phdf5) is the real
scaffold — the same build + loader + bootstrap pattern is how a future
ELPA or GDS target would plug in.  The single-process Mg path is kept
as the simplest callable entry point for small jobs where distributed
setup would be overkill.

---

## Cold-start sequence (fresh clone on Perlmutter)

```bash
cd sources/lorrax

# 1. Stage NVHPC subset (cuSOLVERMp + libcal + headers) to /pscratch.
src/ffi/cusolvermp/scripts/stage_nvhpc.sh

# 2. Stage parallel-HDF5 + OpenMPI headers/libs to /pscratch.
src/ffi/phdf5/scripts/stage_openmpi.sh

# 3. Grab a GPU alloc and point run_shifter.sh at it.
lxalloc                               # 1 node × 4 GPUs
export SLURM_JOBID=<from lxalloc>

# 4. Build liblorrax_ffi.so.  Runs inside the shifter container.
src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh

# 5. Verify: 4-GPU round-trip for each target.
LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \
    CUSOLVERMP_FORCE_NCCL=1 \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    python3 -u -m common.cusolvermp_eigh_test --grid 2 2

LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    HDF5_USE_FILE_LOCKING=FALSE \
    python3 -u -m common.phdf5_write_test
```

All staging steps are idempotent and re-runnable.  Cache of downloaded
binaries sits next to each stage script; artifacts go to `/pscratch`
(only path Shifter is willing to bind-mount for `--volume`).

If any step fails, see the per-target sections below.  For non-NERSC
clusters, see [PORTING.md](PORTING.md).

---

## Layout

```
ffi/
├── AGENTS.md                  ← you are here
├── PORTING.md                 ← per-cluster notes, MPI stack choice, known issues
├── TEMPLATE.md                ← template for new FFI targets
├── common/                    shared build + loader
│   ├── ffi_loader.py          ctypes-loads liblorrax_ffi.so, registers handlers
│   ├── broadcast.py           JAX KV-store broadcast helpers
│   └── cpp/
│       ├── CMakeLists.txt     one build producing liblorrax_ffi.so
│       ├── build.sh           thin wrapper (run inside Shifter)
│       ├── run_shifter.sh     launches Shifter w/ bind-mounts + MPI stack switch
│       ├── ffi_helpers.h      error-propagation macros (FFI_RETURN_IF_ERROR, LORRAX_*_CHECK)
│       └── api.cc             extern "C" ABI (context create/destroy, etc.)
├── cusolvermp/
│   ├── __init__.py            public: distributed_eigh
│   ├── context.py             Python singleton (NCCL bootstrap via KV store)
│   ├── eigh.py                shard_map wrapper
│   ├── scripts/
│   │   └── stage_nvhpc.sh     copy NVHPC subset to /pscratch
│   └── cpp/
│       ├── ctx.h              Ctx + CalNcclShim
│       ├── context.cc         cal_comm_create with NCCL-backed allgather
│       └── eigh_ffi.cc        XLA_FFI_DEFINE_HANDLER_SYMBOL(Eigh{F64,C128})
├── cusolvermg/
│   ├── __init__.py            public: eigh_mg
│   ├── eigh.py                single-process jit wrapper
│   └── cpp/
│       └── eigh_mg_ffi.cc     XLA_FFI_DEFINE_HANDLER_SYMBOL(EighMgF64)
└── phdf5/
    ├── __init__.py            public: open_file, close_file, write_sharded_slab, read_sharded_slab
    ├── context.py             Python-side file-handle cache + mesh validation
    ├── write.py               shard_map wrapper for writes
    ├── read.py                shard_map wrapper for reads
    ├── scripts/
    │   ├── stage_openmpi.sh   stage conda-forge HDF5 (libmpi.so.40) — DEFAULT
    │   └── stage_cray.sh      stage cray-hdf5-parallel (libmpi.so.12) — opt-in, unstable
    └── cpp/
        ├── ctx.h              PhdfCtx (MPI_Comm + HDF5 plists + pinned buf)
        ├── context.cc         MPI_Init_thread + H5Fcreate/H5Fopen + NERSC I/O tuning hints
        ├── phdf5_interface.h  dtype → hid_t template trait + runtime tag dispatcher
        ├── write_ffi.cc       XLA_FFI_DEFINE_HANDLER_SYMBOL(PhdfWriteFfi)
        └── read_ffi.cc        XLA_FFI_DEFINE_HANDLER_SYMBOL(PhdfReadFfi)
```

---

## Dependencies

All three targets share one compiled shared library
`build/liblorrax_ffi.so`, one loader (`common/ffi_loader.py`), and one
launcher (`common/cpp/run_shifter.sh`).

- **Container**: `nvcr.io/nvidia/jax:25.04-py3` (CUDA 12.9, JAX 0.5.3.dev,
  NCCL 2.26, `libcusolver*`, `libcusolverMg*`, `libucc`, HPC-X OpenMPI
  all in-container).
- **NVIDIA HPC SDK** (needed for `ffi.cusolvermp` only): staged to
  `/pscratch/.../lorrax_nvhpc` via `cusolvermp/scripts/stage_nvhpc.sh`
  and bind-mounted into Shifter at `/lorrax_nvhpc`.
- **Parallel HDF5** (needed for `ffi.phdf5`): staged to
  `/pscratch/.../lorrax_phdf5_openmpi` (or `.../lorrax_phdf5_cray`) via
  the corresponding `phdf5/scripts/stage_*.sh` and bind-mounted at
  `/lorrax_phdf5`.  Two stacks are supported; see
  [PORTING.md](PORTING.md) for the choice.
- **No Python build-system deps**: no pybind/nanobind/scikit-build.
  The `.so` is plain C ABI loaded at runtime via `ctypes.CDLL`.

---

## Staging

Shifter on Perlmutter forbids `/opt/*` as a `--volume` source, and
only accepts `--volume` from `/pscratch` (or `/global/cfs`) — _not_
`$HOME`.  So anything we need inside the container that isn't shipped
by the JAX image has to be copied to `/pscratch` first.  The stage
scripts do this:

```bash
src/ffi/cusolvermp/scripts/stage_nvhpc.sh       # ~100 MB, one-time
src/ffi/phdf5/scripts/stage_openmpi.sh          # ~40 MB, one-time
src/ffi/phdf5/scripts/stage_cray.sh             # ~12 MB, only if using MPICH stack
```

Each is idempotent and caches downloaded artifacts next to itself
(in the repo, not on `/pscratch`) so re-runs are fast.

---

## Build

```bash
lxalloc                                                # 1-node × 4-GPU
export SLURM_JOBID=<from lxalloc output>
src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
```

Output: `src/ffi/common/cpp/build/liblorrax_ffi.so`.

`build.sh` inherits the MPI stack from `run_shifter.sh` — it
auto-picks HDF5 headers / libmpi paths based on
`LORRAX_PHDF5_MPI_STACK={openmpi,mpich}` (default `openmpi`).  To
rebuild against the opt-in Cray MPICH stack:

```bash
LORRAX_PHDF5_MPI_STACK=mpich \
    src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
```

---

## Run the cuSOLVERMp test (multi-process)

Recommended — uses CUDA's async mempool so JAX and libcal share VRAM:

```bash
LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \
    CUSOLVERMP_FORCE_NCCL=1 \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    python3 -u -m common.cusolvermp_eigh_test --grid 2 2
```

Required env:

| Var | Why |
|---|---|
| `CUSOLVERMP_FORCE_NCCL=1` | Route libcal's runtime collectives through NCCL instead of UCC.  Without it cuSOLVERMp tries UCC → InfiniBand → fails on many sites. |
| `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` | Use CUDA's async mempool so JAX and libcal's internal `cudaMalloc` share VRAM.  Without it JAX's default BFC pool is invisible to `cudaMalloc` and libcal hits OOM at `MEM_FRACTION=0.95` even though most of VRAM is actually free. |

Validated (n=128, 1 node × 4×A100, 2×2 grid):
- F64 symmetric : max |evals−ref| = 9.1e-13
- C128 Hermitian: max |evals−ref| = 5.7e-13

## Run the cuSOLVERMg test (single-process)

```bash
LORRAX_NGPU=4 LORRAX_NTASKS=1 \
    src/ffi/common/cpp/run_shifter.sh \
    python3 -u -m common.cusolvermg_eigh_test
```

Validated (F64):
- n=128,  tile=32  → max |evals-ref| = 9.1e-13, 57 ms (post-warmup)
- n=2048, tile=256 → max |evals-ref| = 2.2e-11, 509 ms

## Run the phdf5 tests (multi-process)

4-GPU round-trip (write + parallel read, exact-equality check):

```bash
LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    HDF5_USE_FILE_LOCKING=FALSE \
    python3 -u -m common.phdf5_write_test
```

16-GPU (4-node) write benchmark vs the gather-to-rank-0 baseline:

```bash
LORRAX_NNODES=4 LORRAX_NGPU=4 LORRAX_NTASKS=16 \
    src/ffi/common/cpp/run_shifter.sh env \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    HDF5_USE_FILE_LOCKING=FALSE \
    python3 -u -m common.phdf5_vs_gather_bench -n 16384 --iters 3
```

Same for reads:

```bash
LORRAX_NNODES=4 LORRAX_NGPU=4 LORRAX_NTASKS=16 \
    src/ffi/common/cpp/run_shifter.sh env \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    HDF5_USE_FILE_LOCKING=FALSE \
    python3 -u -m common.phdf5_read_bench -n 16384 --iters 3
```

Validated on OpenMPI stack (default):
- 4-GPU round-trip : exact equality (0.000e+00 max error, serial + parallel read)
- 16-GPU write     : 4.08 GB/s (8.02× over gather-to-rank-0)
- 16-GPU read      : 9.27 GB/s (4.06× over every-rank full h5py read at warm cache)

---

## How the cuSOLVERMp NCCL bootstrap works (summary)

1. Python: rank 0 fills a 128-byte `ncclUniqueId` via our `lrx_*` C API.
2. Python: **all ranks** broadcast the bytes via
   `jax.distributed.global_state.client.key_value_set/blocking_key_value_get`
   (which uses JAX's already-live KV store — avoids the
   `multihost_utils.broadcast_one_to_all` gotcha that promotes `uint8`
   → `uint64` under `jax_enable_x64=True`).
3. Python: calls our `lrx_create_cusolvermp_context(...)` ctypes
   function.  In C++:
   - `ncclCommInitRank` → full NCCL communicator.
   - `cal_comm_create(params, &cal_comm)` with callbacks
     (`allgather`, `req_test`, `req_free`) that route via
     `ncclAllGather` on our private CUDA stream.  This is NVIDIA's
     documented non-MPI CAL bootstrap.
   - `cusolverMpCreate` → handle tied to our stream.
   - `cusolverMpCreateDeviceGrid(handle, cal_comm, p, q, …)` → grid.
4. FFI handler per solve: creates matrix descriptors, runs
   `cusolverMpSyevd_bufferSize` + `cusolverMpSyevd`, done.

## How the phdf5 MPI-IO bootstrap works (summary)

1. Python: `open_file(path, mesh=mesh, mode='w'|'a'|'r')` calls
   `lrx_phdf5_open(path, p, q, rank, world_size, mode)`.
2. C++: `MPI_Init_thread(MPI_THREAD_MULTIPLE)` on first open (one-shot).
   Duplicates `MPI_COMM_WORLD`, creates FAPL with `H5Pset_fapl_mpio` +
   NERSC I/O tuning hints (`romio_cb_write=enable`, `cb_buffer_size=64M`,
   `cb_nodes=world_size`, `striping_factor=16`, `striping_unit=4M`,
   `H5Pset_alignment(4MiB)`), then `H5Fcreate` or `H5Fopen`.
3. Python: `write_sharded_slab(fh, ds_name, A)` first calls
   `lrx_phdf5_ensure_dataset(...)` to collectively create/open the
   dataset and get an `hid_t`, then enters a `shard_map` with
   `in_specs=P('x','y'), out_specs=P()`.
4. FFI handler per shard: D2H-memcpy on a private CUDA stream →
   blocking `H5Dwrite` on the host thread (CUDA stream free to run
   queued device work) → sync event back to XLA's stream.
5. Reads mirror writes: `read_sharded_slab` calls
   `lrx_phdf5_open_dataset_ro`, then `shard_map` with `in_specs=(),
   out_specs=P('x','y')` → FFI does `H5Dread` → H2D-memcpy.

See `phdf5/cpp/write_ffi.cc` and `phdf5/cpp/read_ffi.cc` for the
exact event / pinned-buffer ordering.

---

## Adding a new FFI target (e.g. ELPA)

1. Create `src/ffi/<lib>/cpp/<feature>_ffi.cc` with
   `XLA_FFI_DEFINE_HANDLER_SYMBOL(<Name>, <Host>, <Bind>)`.  Look at
   `phdf5/cpp/write_ffi.cc` for the dtype-template-dispatch pattern,
   or `cusolvermp/cpp/eigh_ffi.cc` for the NCCL-context pattern.
2. Append the `.cc` to `LORRAX_FFI_SOURCES` in
   `common/cpp/CMakeLists.txt`; add `-l<lib>` to `target_link_libraries`.
3. Add `"lorrax_<lib>_<feature>": "<Name>"` to
   `_FFI_TARGET_SYMBOLS` in `common/ffi_loader.py`.
4. If the library needs its own communicator:
   - For NCCL / CAL bootstrap: reuse `cusolvermp/context.py`'s
     `ncclUniqueId` KV-store pattern.
   - For MPI bootstrap: reuse `phdf5/cpp/context.cc`'s
     `ensure_mpi_initialized()` singleton.
5. If the library requires bind-mounted host files (SDKs, module
   installs), add a `stage_*.sh` script under
   `src/ffi/<lib>/scripts/` following the phdf5/NVHPC pattern, and
   extend `run_shifter.sh` to bind-mount the stage dir to a stable
   path inside the container.
6. Write a Python wrapper `src/ffi/<lib>/<feature>.py` that calls
   `get_lib()` → gets the context → wraps `jax.ffi.ffi_call` in a
   `shard_map` with the appropriate `PartitionSpec`.
7. Re-export from `src/ffi/<lib>/__init__.py`.
8. Rebuild: `src/ffi/common/cpp/run_shifter.sh bash
   src/ffi/common/cpp/build.sh`.

See [TEMPLATE.md](TEMPLATE.md) for a skeleton walkthrough.

---

## Non-goals (this iteration)

- Autodiff / custom_vjp.
- GDS (`cuFile`) VFD — no stable MPI-IO path in April 2026.
- Async VOL connector for phdf5 — single-write-in-flight is fine for
  the current zeta / wfn workloads; revisit when pipelined throughput
  becomes a bottleneck.
- `input_output_aliases` into/out of `ffi_call` — would donate the
  input buffer to the output and eliminate a copy.
- Multi-node cuSOLVERMp — NCCL bootstrap is multi-node-ready but
  untested there.  On Perlmutter this needs `NCCL_IB_*` / OFI config.
