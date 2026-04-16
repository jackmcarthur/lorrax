# `src/ffi/` — JAX Foreign Function Interface bridge

Scaffolding for calling compiled (C/C++/CUDA) libraries from JAX via the
XLA FFI.  Current targets — both **working** on 1 node × 4×A100 (NERSC
Perlmutter):

| Subpackage | Library | Process model | Status |
|---|---|---|---|
| `ffi.cusolvermp` | **cuSOLVERMp** (multi-process, multi-GPU / multi-node) | 1 Python proc per GPU | F64 + C128 validated at n=128 |
| `ffi.cusolvermg` | **cuSOLVERMg** (single-process, multi-GPU, in-container) | 1 Python proc × N GPUs | F64 validated at n∈{128, 2048} |

Multi-process via NCCL is the real scaffold — the same build + loader +
bootstrap pattern is how ELPA (or any MPI-or-NCCL distributed solver)
would plug in.  The single-process Mg path is kept around as the
simplest callable entry point for small jobs where distributed setup
would be overkill.

## Layout

```
ffi/
├── AGENTS.md                  ← you are here
├── common/                    shared build + loader
│   ├── ffi_loader.py          ctypes-loads liblorrax_ffi.so, registers handlers
│   └── cpp/
│       ├── CMakeLists.txt     one build producing liblorrax_ffi.so
│       ├── build.sh           thin wrapper (run inside Shifter)
│       ├── run_shifter.sh     launches Shifter w/ bind-mounts the build needs
│       ├── xla_ffi_glue.h     CUDA / CUSOLVER error-checking macros
│       └── api.cc             extern "C" ABI (context create/destroy, etc.)
├── cusolvermp/
│   ├── __init__.py            public: distributed_eigh
│   ├── context.py             Python singleton (NCCL bootstrap via KV store)
│   ├── eigh.py                shard_map wrapper
│   └── cpp/
│       ├── ctx.h              Ctx + CalNcclShim
│       ├── context.cc         cal_comm_create with NCCL-backed allgather
│       └── eigh_ffi.cc        XLA_FFI_DEFINE_HANDLER_SYMBOL(Eigh{F64,C128})
└── cusolvermg/
    ├── __init__.py            public: eigh_mg
    ├── eigh.py                single-process jit wrapper
    └── cpp/
        └── eigh_mg_ffi.cc     XLA_FFI_DEFINE_HANDLER_SYMBOL(EighMgF64)
```

## Dependencies

- **Container**: `nvcr.io/nvidia/jax:25.04-py3` (CUDA 12.9, JAX 0.5.3.dev,
  NCCL 2.26, `libcusolver*`, `libcusolverMg*`, `libucc` all in-container).
- **NVIDIA HPC SDK** (for cuSOLVERMp; the Mg path needs nothing
  outside the container): `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/` on
  NERSC, staged to `/pscratch/.../lorrax_nvhpc` and bind-mounted into
  Shifter at `/lorrax_nvhpc`.
- **No Python build-system deps**: no pybind/nanobind/scikit-build.
  The `.so` is plain C ABI loaded at runtime via `ctypes.CDLL`.

## Bind-mount of HPC SDK (only for the cuSOLVERMp path)

Shifter forbids `/opt/nvidia` as a bind source.  Stage the minimum
subset:

```bash
STAGE=/pscratch/sd/j/jackm/lorrax_nvhpc/25.5_cuda12.9
mkdir -p $STAGE/math_libs/12.9/{lib64,targets/x86_64-linux/include} \
         $STAGE/comm_libs/12.9/nccl/include
cp -a /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/include/*.h \
      $STAGE/math_libs/12.9/targets/x86_64-linux/include/
cp -a /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64/{libcusolverMp*,libcal*} \
      $STAGE/math_libs/12.9/lib64/
cp -a /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/comm_libs/12.9/nccl/include/nccl.h \
      $STAGE/comm_libs/12.9/nccl/include/
```

`run_shifter.sh` bind-mounts `$LORRAX_FFI_NVHPC_DIR` (default
`/pscratch/sd/j/jackm/lorrax_nvhpc`) to `/lorrax_nvhpc` and bakes an
rpath so `libcusolverMp.so` / `libcal.so` resolve automatically.

## Build

```bash
lxalloc                                                # 1-node × 4-GPU
export SLURM_JOBID=<from lxalloc output>
src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
```

Output: `src/ffi/common/cpp/build/liblorrax_ffi.so`.

## Run the cuSOLVERMp test (multi-process)

```bash
LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \
    CUSOLVERMP_FORCE_NCCL=1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 -u -m common.cusolvermp_eigh_test --grid 2 2
```

Required env vars:

| Var | Why |
|---|---|
| `CUSOLVERMP_FORCE_NCCL=1` | Route libcal's runtime collectives through NCCL instead of UCC.  Without it cuSOLVERMp tries UCC → InfiniBand → fails on many sites. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` | Leave headroom for cuSOLVERMp + libcal workspace.  LORRAX's module default of 0.95 starves the solver. |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | Allocate on demand so the above reservation isn't burned up front. |

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

## Adding a new FFI target (e.g. ELPA)

1. Create `src/ffi/<lib>/cpp/<feature>_ffi.cc` with
   `XLA_FFI_DEFINE_HANDLER_SYMBOL(<Name>, <Host>, <Bind>)`.
2. Append the `.cc` to `LORRAX_FFI_SOURCES` in
   `common/cpp/CMakeLists.txt`; add `-l<lib>` to target_link_libraries.
3. Add `"lorrax_<lib>_<feature>": "<Name>"` to
   `_FFI_TARGET_SYMBOLS` in `common/ffi_loader.py`.
4. If the library needs its own communicator (ELPA needs an MPI
   communicator, say), reuse the `ncclUniqueId` bootstrap pattern in
   `cusolvermp/context.py` with the library's own unique-id broadcast
   primitive.
5. Write a Python wrapper `src/ffi/<lib>/<feature>.py` that calls
   `get_lib()` → gets the context → wraps `jax.ffi.ffi_call` in a
   `shard_map` with the appropriate `PartitionSpec`.
6. Rebuild: `bash src/ffi/common/cpp/build.sh`.

## Non-goals (this iteration)

- Autodiff / custom_vjp.
- Multi-node (the NCCL bootstrap is multi-node-ready but untested
  there; on Perlmutter this needs NCCL_IB / OFI config).
- `input_output_aliases` into/out of `ffi_call` — would donate the
  input buffer to the output and eliminate a copy.
