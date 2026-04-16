# `src/ffi/` — JAX Foreign Function Interface bridge

Scaffolding for calling compiled (C/C++/CUDA) libraries from JAX via the
XLA FFI.  Currently targets NVIDIA multi-GPU linear algebra.

Two variants are wired in:

| Subpackage | Library | Process model | Status |
|---|---|---|---|
| `ffi.cusolvermg` | **cuSOLVERMg** (single-process, multi-GPU) | 1 Python proc × N GPUs | **Working** — F64 eigh validated at n=128 and n=2048 on 4×A100 |
| `ffi.cusolvermp` | **cuSOLVERMp** (multi-process, multi-GPU/multi-node) | 1 Python proc per GPU | **WIP** — builds, NCCL bootstrap works, but `cusolverMpSyevd_bufferSize` traps (see "Open issue" below) |

The cuSOLVERMg path is the recommended entry point for new callers on a
single node.  cuSOLVERMp will be unblocked once the ncclComm ↔ cal_comm
plumbing is sorted out.

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
├── cusolvermg/                ← the working path
│   ├── __init__.py            public: eigh_mg
│   ├── eigh.py                Python wrapper (single-process, jit)
│   └── cpp/
│       └── eigh_mg_ffi.cc     XLA_FFI_DEFINE_HANDLER_SYMBOL(EighMgF64)
└── cusolvermp/                ← WIP (see Open issue below)
    ├── __init__.py            public: distributed_eigh
    ├── context.py             Python singleton (NCCL bootstrap via KV store)
    ├── eigh.py                shard_map wrapper
    └── cpp/
        ├── ctx.h
        ├── context.cc         cusolverMpCreate / Grid / ncclCommInitRank
        └── eigh_ffi.cc        XLA_FFI_DEFINE_HANDLER_SYMBOL(Eigh{F64,C128})
```

## Dependencies

- **Container**: `nvcr.io/nvidia/jax:25.04-py3` (CUDA 12.9, JAX 0.5.3.dev,
  NCCL 2.26, `libcusolver*`, `libcusolverMg*`, `libucc` all in-container).
- **NVIDIA HPC SDK** (for cuSOLVERMp ONLY — the Mg path needs nothing
  outside the container): `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/`
  on NERSC, staged to `/pscratch/.../lorrax_nvhpc` and bind-mounted
  into Shifter at `/lorrax_nvhpc`.
- **No Python build-system deps**: no pybind/nanobind/scikit-build.
  The `.so` is plain C ABI, loaded at runtime via `ctypes.CDLL`.

## Bind-mount of HPC SDK (only needed for the cuSOLVERMp path)

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
`/pscratch/sd/j/jackm/lorrax_nvhpc`) to `/lorrax_nvhpc` inside the
container and sets `LD_LIBRARY_PATH` so `libcusolverMp.so` and
`libcal.so` resolve at load time.

## Build

```bash
lxalloc                                                # 1-node × 4-GPU
export SLURM_JOBID=<from lxalloc output>
src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
```

Output: `src/ffi/common/cpp/build/liblorrax_ffi.so`.

## Run the cusolverMg test (the working path)

```bash
# single process, 4 GPUs visible
LORRAX_NGPU=4 LORRAX_NTASKS=1 \
    src/ffi/common/cpp/run_shifter.sh \
    python3 -u -m common.cusolvermg_eigh_test
```

Validated results on 1 node × 4 A100:
- n=128 , tile=32  → max |evals-ref| = 9.1e-13, wall = 57 ms
- n=2048, tile=256 → max |evals-ref| = 2.2e-11, wall = 509 ms

Eigenvector residuals `‖A q_i − λ_i q_i‖∞` ≈ 7e-14 (F64) in the
row-vector view (cuSOLVERMg writes column-major; JAX reads row-major;
for A = A^T they match by transposition).

## Run the cusolverMp test (currently WIP)

See the 2026-04-16 report for context; this path builds fine but
`cusolverMpSyevd_bufferSize` traps SIGFPE (divide-by-zero) deep inside
the library regardless of `mb ∈ {32, 64}`, `compz ∈ {N, V}`, or
`CUSOLVERMP_FORCE_NCCL=1`.

## Open issue — cuSOLVERMp

Suspected cause: the NVIDIA `mp_syevd.c` sample passes `ncclComm_t`
directly to `cusolverMpCreateDeviceGrid` (which expects `cal_comm_t`);
C's lax pointer conversion makes this compile silently.  The sample runs
**under MPI**, so libcal's internal MPI-based initialization path is
live and presumably intercepts the ncclComm correctly.  Our LORRAX FFI
process uses `jax.distributed` + direct NCCL (no MPI), so the CAL layer
never sees MPI and misreads the comm — `cal_comm_get_size` returns 0 and
bufferSize divides by zero.

Follow-up options:
1. Link + call `MPI_Init` from the FFI handle-creation path (the
   container has OpenMPI at `/opt/hpcx/ompi/`).
2. Use `cal_comm_create` with an NCCL-backed allgather callback, which
   is the documented CAL-without-MPI path.
3. Upgrade to NVHPC 25.9 (cuSOLVERMp 0.7.0.0) in a CUDA 13 container
   (`nvcr.io/nvidia/jax:25.08-py3`) to pick up any NCCL-direct fixes.

## Adding a new FFI target

1. Create `src/ffi/<lib>/cpp/<feature>_ffi.cc` with
   `XLA_FFI_DEFINE_HANDLER_SYMBOL(<Name>, <Host>, <Bind>)`.
2. Append the `.cc` to `LORRAX_FFI_SOURCES` in
   `common/cpp/CMakeLists.txt`.
3. Add an entry to `_FFI_TARGET_SYMBOLS` in `common/ffi_loader.py`.
4. Write a Python wrapper `src/ffi/<lib>/<feature>.py` that calls
   `get_lib()` then `jax.ffi.ffi_call(<target_name>, …)`.
5. Rebuild: `bash src/ffi/common/cpp/build.sh`.

## Non-goals (this iteration)

- Autodiff / custom_vjp.
- Complex Hermitian in the Mg path (wired but not validated; the type
  dispatch in `eigh_mg_ffi.cc` currently only implements F64).
- Multi-node (cuSOLVERMg is single-node; cuSOLVERMp can do it once the
  bufferSize issue is resolved).
- Automatic row-major ↔ column-major layout conversion.  Users of
  `eigh_mg` read the eigenvectors in row-major view (see test).
