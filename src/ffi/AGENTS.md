# `src/ffi/` — JAX Foreign Function Interface bridge

Scaffolding for calling compiled (C/C++/CUDA) libraries from JAX via the
XLA FFI. The first concrete target is **cuSOLVERMp** distributed
Hermitian eigensolve on one node × N GPUs.

## Layout

```
ffi/
├── AGENTS.md                  ← you are here
├── common/                    platform-agnostic glue
│   ├── ffi_loader.py          ctypes-loads the .so, registers FFI targets
│   └── cpp/
│       ├── CMakeLists.txt     single build; produces liblorrax_ffi.so
│       ├── build.sh           thin wrapper (run inside Shifter)
│       ├── run_shifter.sh     launches Shifter w/ the bind-mount this .so needs
│       ├── xla_ffi_glue.h     small helpers (dtype dispatch, error chks)
│       └── api.cc             extern "C" ABI for ctypes consumers
└── cusolvermp/                one subpackage per parallel-LA library
    ├── __init__.py            re-exports distributed_eigh
    ├── context.py             Python singleton (NCCL bootstrap + handle)
    ├── eigh.py                distributed_eigh wrapped in shard_map
    └── cpp/
        ├── ctx.h              shared struct
        ├── context.cc         cusolverMpCreate / Grid / ncclCommInitRank
        └── eigh_ffi.cc        XLA_FFI_DEFINE_HANDLER_SYMBOL Eigh{F64,C128}
```

## Dependencies

- **Container**: `nvcr.io/nvidia/jax:25.04-py3` (CUDA 12.9, JAX 0.5.3.dev,
  NCCL 2.26, CUDA runtime/cuBLAS/cuSOLVER/libucc all included).
- **NVIDIA HPC SDK** (for cuSOLVERMp and libcal — not in the container):
  `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/` on NERSC.
- **No Python compile-time deps**: no pybind/nanobind/scikit-build. The
  `.so` is loaded at runtime by `ctypes.CDLL`.

## Bind-mount the HPC SDK subset into Shifter

Shifter doesn't allow `/opt/nvidia` as a bind source.  Stage the required
subset to a bindable location and mount it at `/lorrax_nvhpc`:

```bash
STAGE=/pscratch/sd/j/jackm/lorrax_nvhpc/25.5_cuda12.9
mkdir -p $STAGE/math_libs/12.9/{lib64,targets/x86_64-linux/include} \
         $STAGE/comm_libs/12.9/nccl/include
cp -a /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/include/cusolverMp*.h \
      $STAGE/math_libs/12.9/targets/x86_64-linux/include/
cp -a /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64/{libcusolverMp*,libcal*} \
      $STAGE/math_libs/12.9/lib64/
cp -a /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/comm_libs/12.9/nccl/include/nccl.h \
      $STAGE/comm_libs/12.9/nccl/include/
```

The `run_shifter.sh` wrapper bind-mounts `$LORRAX_FFI_NVHPC_DIR`
(default `/pscratch/sd/j/jackm/lorrax_nvhpc`) to `/lorrax_nvhpc`
inside the container.

## Build

```bash
# from the lorrax root, after `lxalloc` + export SLURM_JOBID
src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
```

Output: `src/ffi/common/cpp/build/liblorrax_ffi.so`.

The shared library contains:
- **XLA FFI handlers** (exported C symbols `EighF64`, `EighC128`, …) —
  passed to `jax.ffi.register_ffi_target` via `jax.ffi.pycapsule`.
- **extern "C" `lrx_*` wrappers** for context lifecycle (NCCL bootstrap,
  cusolverMp handle, grid, workspace) — called from Python through
  `ctypes`.

## Run tests

```bash
# from the lorrax root, interactive alloc + SLURM_JOBID set
LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh \
    python3 -u -m common.cusolvermp_eigh_test
```

## Adding a new FFI target

1. Create `src/ffi/<library>/cpp/<feature>_ffi.cc` with an
   `XLA_FFI_DEFINE_HANDLER_SYMBOL(<CppName>, …)`.
2. Add the symbol to `_FFI_TARGET_SYMBOLS` in
   `common/ffi_loader.py`.
3. Add the `.cc` to `common/cpp/CMakeLists.txt`'s
   `LORRAX_FFI_SOURCES`.
4. Add a Python wrapper `src/ffi/<library>/<feature>.py` that
   (a) imports `ffi_loader.get_lib()`, (b) exposes a
   `@partial(shard_map, …)`-wrapped entry point.
5. Rebuild: `bash src/ffi/common/cpp/build.sh`.

## Non-goals (this iteration)

- Autodiff / custom_vjp.
- Multi-node (one node, ≤8 GPUs).
- Automatic row-major ↔ column-major layout conversion.
