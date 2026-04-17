# Porting the cuSOLVERMp FFI to a new cluster

This doc is the "what do I need and where do I find it" checklist for
running the LORRAX `ffi.cusolvermp` subpackage on a cluster other than
NERSC Perlmutter.

## Hard requirements

1. **NVIDIA GPU** with compute capability ≥ 7.0 (tested on A100 / H100).
2. **CUDA Toolkit ≥ 12.0** (cuSOLVERMp currently requires double-precision
   GPU compute; CUDA 12.9 is what we use, but 12.X ≥ 12.4 should work).
3. **NCCL ≥ 2.18** (ships with CUDA 12.4+; any recent JAX install has it).
4. **A JAX build** with `jax.ffi` — any `jax[cuda12] ≥ 0.5` is fine.  JAX
   must match the CUDA toolkit major version.
5. **cuSOLVERMp + libcal** from NVIDIA HPC SDK 22.7+ (the part that's not
   in the plain CUDA toolkit).  HPC SDK 25.5 is what we validated on.

That's everything.  No MPI requirement — our bootstrap goes through JAX's
distributed KV store + NCCL, not MPI.

## Where cuSOLVERMp lives on common clusters

Search these paths in order — the build system (see below) will try them
automatically.

| Source | Typical path | How to enable |
|---|---|---|
| NVIDIA HPC SDK module | `$NVHPC_ROOT/math_libs/<cuda>/targets/x86_64-linux/` | `module load nvhpc` (or `nvhpc-sdk`, `nvidia-hpc`, site-specific) |
| HPC SDK default install | `/opt/nvidia/hpc_sdk/Linux_x86_64/<ver>/math_libs/<cuda>/targets/x86_64-linux/` | just be on a node with HPC SDK installed |
| Standalone apt / rpm | `/usr/local/cuda-12.X/targets/x86_64-linux/` | `apt install cuda-cusolvermp-12-X` (CUDA 12.8+) |
| NVIDIA HPC container | `/opt/nvidia/hpc_sdk/Linux_x86_64/.../` inside `nvcr.io/nvidia/nvhpc:<tag>` | `docker pull nvcr.io/nvidia/nvhpc:23.11-devel-cuda_multi-ubuntu22.04` or similar |
| Cray (OLCF) | `$CPE_ROOT/hpc_sdk/` | `module load nvhpc-sdk` |

Concretely on NERSC Perlmutter: `module load nvhpc` sets
`$NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/<ver>`.  Inside the
`nvcr.io/nvidia/jax:25.04-py3` Shifter container (which does NOT include
HPC SDK) we bind-mount a staged subset — see "Shifter-specific" below.

## Build system hooks

[`src/ffi/common/cpp/CMakeLists.txt`](common/cpp/CMakeLists.txt) accepts:

- `-DNVHPC_ROOT=<path>` — directory containing `math_libs/<cuda>/`.
- `-DNVHPC_CUDA_SUBDIR=<cuda>` — e.g. `12.9`.  Omit to use the newest
  CUDA found under `math_libs/`.
- `-DCUSOLVERMP_INCLUDE_DIR=<path>` / `-DCUSOLVERMP_LIB_DIR=<path>` —
  override autodiscovery completely (useful for non-standard installs
  or when libcusolverMp and libcal live in different dirs).

If you pass nothing, CMake tries in order:

1. `$NVHPC_ROOT` env var (set by `module load nvhpc`).
2. `$HPCSDK_ROOT`, `$NVHPC_SDK_PATH` env vars.
3. `/opt/nvidia/hpc_sdk/Linux_x86_64/*/` (newest first).
4. `/lorrax_nvhpc/*` (the Shifter bind-mount convention; harmless
   elsewhere).
5. `/usr/local/nvhpc/Linux_x86_64/*/`.

The NCCL include/lib is looked up in the container's standard locations
(`/usr/include/nccl.h` + ld cache) first, and falls back to
`$NVHPC_ROOT/comm_libs/<cuda>/nccl/` if the container doesn't ship NCCL.

## Runtime setup

Three env vars matter regardless of cluster:

```bash
CUSOLVERMP_FORCE_NCCL=1                     # libcal runtime collectives → NCCL
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async      # share VRAM pool with libcal
# (optional) LD_LIBRARY_PATH must include the cusolverMp lib directory
```

The `cuda_async` setting is what lets JAX keep `MEM_FRACTION` at the
default 0.95 while libcal's internal `cudaMalloc` calls still succeed.
See `src/ffi/AGENTS.md` for details.

## Native-JAX vs container workflows

- **Native JAX**: `pip install jax[cuda12]` into a virtualenv / conda env
  that also has access to `libcusolverMp.so` on `LD_LIBRARY_PATH`.
  Build the .so from within that env and you're done — no container
  required.  Python's `jax.ffi.include_dir()` points at whatever JAX
  version the venv has, and our CMake picks that up automatically.
- **Singularity / Apptainer (DOE standard on most sites)**: similar to
  native — use `--bind /opt/nvidia` etc. freely, no restrictions.  The
  `run_shifter.sh` recipe below maps to `singularity exec --bind
  ... --nv image.sif ...`.
- **Shifter (NERSC only)**: restricts bind-mount sources.  See below.

## Shifter-specific: staging the HPC SDK subset

NERSC's Shifter forbids mounting `/opt/nvidia` directly.  Stage a minimal
copy to a bindable location (NERSC: `/pscratch/sd/...` or `$HOME` works),
then bind-mount it.  The stage is ~80 MB total.

```bash
STAGE=/pscratch/sd/<u>/<user>/lorrax_nvhpc/25.5_cuda12.9
mkdir -p $STAGE/math_libs/12.9/{lib64,targets/x86_64-linux/include} \
         $STAGE/comm_libs/12.9/nccl/include
cp -a /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/include/*.h \
      $STAGE/math_libs/12.9/targets/x86_64-linux/include/
cp -a /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64/{libcusolverMp*,libcal*} \
      $STAGE/math_libs/12.9/lib64/
cp -a /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/comm_libs/12.9/nccl/include/nccl.h \
      $STAGE/comm_libs/12.9/nccl/include/
```

Then run everything via [`src/ffi/common/cpp/run_shifter.sh`](common/cpp/run_shifter.sh),
which bind-mounts `$LORRAX_FFI_NVHPC_DIR` (default `/pscratch/.../lorrax_nvhpc`)
to `/lorrax_nvhpc` inside the container.

## Checklist for a new cluster

1. **Find HPC SDK**: `module spider nvhpc` → note the `$NVHPC_ROOT` value.
   If unavailable, file a ticket with the cluster admins (it's a standard
   NVIDIA install); or install the HPC SDK standalone into `$HOME`.
2. **Pick a JAX environment**:
   - a venv with `pip install jax[cuda12]`, OR
   - the `nvcr.io/nvidia/jax:25.04-py3` container (or newer — we pinned
     25.04 on NERSC for the CUDA 12.9 match).
3. **Build**: `bash src/ffi/common/cpp/build.sh`.  With `$NVHPC_ROOT` in
   the environment, autodetection should just work.  Otherwise pass
   `-DNVHPC_ROOT=...` explicitly.
4. **Run**: set the two env vars above (`CUSOLVERMP_FORCE_NCCL`,
   `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`) and launch with one process
   per GPU (`srun -n <gpus>` or `mpirun -n <gpus>` — our Python
   bootstrap uses `jax.distributed.initialize()` which handles SLURM,
   PMI, and TCP).
5. **Verify**: `python -m common.cusolvermp_eigh_test --grid 2 2` on
   4 GPUs.  Expect `max |evals - ref|` ≈ 1e-10.

## Known gotchas

- **libcal's IB complaint** (`Failed to parse ib device list`): harmless
  warning if `CUSOLVERMP_FORCE_NCCL=1` is set.  libcal is probing for
  InfiniBand transports it won't use anyway.
- **CUDA driver vs toolkit mismatch**: cuSOLVERMp needs a driver that
  supports the toolkit it was built against.  `nvidia-smi` column
  "CUDA Version" is the driver's *maximum* supported toolkit; must be
  ≥ the toolkit we linked.
- **Multi-node**: the NCCL bootstrap works across nodes but needs
  cluster-specific NCCL env (e.g. `NCCL_IB_HCA`, `NCCL_NET_GDR_LEVEL`,
  or `NCCL_NET_PLUGIN=ofi` on Perlmutter).  Not validated in this
  iteration.

## Reference

- NVIDIA cuSOLVERMp install: https://docs.nvidia.com/cuda/cusolvermp/
- NVIDIA HPC SDK download: https://developer.nvidia.com/hpc-sdk
- JAX FFI: https://jax.readthedocs.io/en/latest/ffi.html
