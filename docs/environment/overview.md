# Environment: the runtime stack

What a LORRAX process runs on, and how the runtime configures JAX, XLA's GPU
memory pool and the process topology before the first physics `jit`.

| page | owns |
|---|---|
| this page | the platform routes, the Frontera CPU layer stack, the startup report, JAX and the GPU memory pool, troubleshooting |
| [Perlmutter](machines/perlmutter.md) | the GPU lane (`lorrax_A` module, sealed FFI bundle, `lx`), task geometry, CPU-MPI runs |
| [Frontera](machines/frontera.md) | machine facts, cold start, build recipes |
| [Collective transports](transports.md) | gloo vs `impl=mpi` vs NCCL |

Owned elsewhere, linked rather than restated:
[`docs/dev/env_vars.md`](../dev/env_vars.md) is the registry of every
environment variable (spelling, default, grammar);
[`docs/architecture/ffi_layout.md`](../architecture/ffi_layout.md) owns how a
native library is built and reached.

---

## 1. Platforms, one route each

| platform | stack | launched by |
|---|---|---|
| Perlmutter GPU (A100 40/80 GB) | bare-host CUDA 13.2, a JAX/JAXLIB 0.9.1 venv, the sealed FFI bundle; all selected by the `lorrax_A` module | `lx run` ([Perlmutter §1](machines/perlmutter.md#1-entry-point-lx)) |
| Perlmutter CPU (Milan) | the same venv, CPU platform, MPI collectives | [Perlmutter §5](machines/perlmutter.md) |
| Frontera CPU (CLX) | apptainer image + staged runtime bundle + Intel MPI (layers below) | `config/frontera/templates/gw_dev.sbatch` |
| another SLURM cluster | `config/<cluster>/` | [`config/README.md`](../../config/README.md) §Porting |

The FFI layer is required on every platform ([Installation](../installation/index.md)).

### The Frontera CPU layer stack {#layer-stack}

Bottom to top; every script is in `config/frontera/`.

| # | layer | built / staged by | depends on | failure signature |
|---|---|---|---|---|
| 1 | host OS, SLURM, Intel MPI 2020.4 | the machine | — | — |
| 2 | container image (`py312.sif`) | not vendored ([ledger](machines/frontera.md#not-yet-vendored)) | 1 | outside the container, jax wheels fail with `GLIBC_2.28 not found` (host glibc 2.17) |
| 3 | uv venv (`$WORK/lorrax_env/.venv`, jax 0.9.1) | not vendored; `pyproject.toml` is the dependency authority | 2 | `ModuleNotFoundError` at the first import |
| 4 | mpi4py + parallel-h5py overlay with `sitecustomize.py` | `build_mpi_overlay.sh` (`fetch` on login, `build` in the SIF) | 2, 3, host MPI + parallel HDF5 | a driver that does not end through `runtime.run_main_and_finalize()` exits rc=1 after succeeding ("MPI routine after finalizing MPICH") |
| 5 | MPIwrapper with `lorrax_thread.patch` (`MPI_THREAD_MULTIPLE`) | `build_mpiwrapper.sh --fresh` on a login node; verifies the patch in the disassembly | host gcc/gfortran + Intel MPI | `MPITRAMPOLINE_LIB` unset: MPItrampoline refuses at startup; an unpatched wrapper loads and reintroduces a multi-node segfault/hang class |
| 6 | host FFI `.so` (`liblorrax_ffi_host.so`) | `build_ffi_host.sh` | 2, host MPI, CBLAS/ScaLAPACK, SLATE | refusal at startup, naming the `.so` |
| 7 | env glue (`gpu_env.sh`, `mpi_transport_env.sh`, staged PMI2 lib) | sourced per job; `stage_host_pmi.sh` | 5, 6, SLURM | without the staged PMI2 lib, `srun --mpi=pmi2` binds TACC's PMI-1 `libpmi.so` and `MPIR_pmi_init` fails; without `mpi_transport_env.sh`, the login shell's `FI_PROVIDER`/`I_MPI_PMI_LIBRARY` leak in |
| 8 | staged runtime bundle (`lorrax_cpu_bundle.tar` → node-local `/tmp`) | `build_cpu_runtime_bundle.sh` once per revision, `stage_runtime.sh` per job | 3, 4, `src/` | announced fallback to the Lustre venv on rank 0, at the cold-import cost in [frontera.md §3](machines/frontera.md#3-cold-start) |
| 9 | launch template | `templates/gw_dev.sbatch` | all | edit the `#SBATCH` block and deck variables only |

---

## 2.0 The startup report {#startup-block}

Production drivers print a four-line rank-0 preamble (ranks, devices, mesh,
affinity, JAX/precision/collectives, startup time).
`LORRAX_DEBUG_PRINT=1` renders the full block below from the same measured
facts. After backend init, `os.environ` no longer describes the client;
this block reads the live client, so where it disagrees with a documented
default, the block is what ran. Rank 0 of a P=4 Perlmutter run:

```text
  This is rank 0 of 4, and it addresses 1 of the 4 devices in the job.
  jax.distributed.initialize() took its explicit form with coordinator_address='nid003401:22021', num_processes=4 and local_device_ids=[0].
  The JAX platform resolved to 'gpu' on devices of kind 'NVIDIA A100-SXM4-40GB', from JAX_PLATFORMS='cuda,cpu', under jax 0.9.1 / jaxlib 0.9.1 with 64-bit values enabled.
  Default matmul precision is pinned to 'highest', so f32 and complex64 dots run at fp32 rather than TensorFloat32; f64/c128 is unaffected either way.
  The run's device mesh is 2x2 over axes ('x', 'y'), and its communicator cliques were warmed before the first physics jit.
  Cross-process collectives run on NCCL because this is a GPU platform, so JAX_CPU_COLLECTIVES_IMPLEMENTATION does not apply.
  The XLA memory pool, read from jax.local_devices()[0].memory_stats() and not from os.environ, reports a limit of 37.74 GB with 0.00 GB in use and a peak of 0.00 GB so far.
  XLA_PYTHON_CLIENT_PREALLOCATE resolved to true (raw 'true') and XLA_PYTHON_CLIENT_ALLOCATOR resolved to 'cuda_async' (raw 'cuda_async') — LORRAX's GPU pool policy: cudaMallocAsync with its pool reserved at 0.89 (runtime.set_default_gpu_pool).
  The live client holds the reserved pool: bytes_limit 37.74 GB = 0.89 x 42.40 GB.
  FFI build provenance: …/releases/c52b2c42-bundle-8e3c3650ea2a/lib/liblorrax_ffi.so | sealed bundle 8e3c3650ea2ab71a | rev c52b2c42565d | sha cb25804baded2322
  The distributed backends available for eigh on this mesh are cusolvermp, distributed, native; which one runs is the input-file key, not an environment variable.
  The JAX persistent compile cache is OFF, so every rank compiles every module in this run; set ISDF_JAX_CACHE_DIR …
  The fail-fast excepthook is installed, so an uncaught exception on any rank exits the step non-zero …
```

Read, in order:

1. **JAX generation.** Both `jax` and `jaxlib` must be 0.9.x (§2).
2. **Matmul precision.** `highest` or `float32`; anything else is a warning line (§2).
3. **Mesh.** The axis names every `PartitionSpec` uses; a mesh of the wrong shape is the first suspect for a slow or wrong distributed run.
4. **Memory pool.** The live `bytes_limit`, the allocator pair, and the check `bytes_limit = f × cuDeviceTotalMem`. A `WARNING: the live client does NOT hold LORRAX's pool` means something built the CUDA client before the runtime set the policy (§2.1).
5. **FFI provenance.** Path, sealed-bundle digest, source revision and content hash of the loaded `.so`. The path exposes a launcher that put another checkout on `PYTHONPATH`; `LORRAX_FFI_SO` selects the library.

---

## 2. JAX configuration

**One JAX generation.** `pyproject.toml` and `runtime/jax_support.py`
declare `jax` and `jaxlib` in `[0.9.0, 0.10.0)`; a test fails if the two
drift, and the runtime refuses any other generation of either package before
the first physics `jit`, checking the parsed versions and the private API
shapes the code uses. `tools/require_jax09.py` is the pre-import check for
launch scripts. There is no escape hatch.

**64-bit values.** `JAX_ENABLE_X64=1` is runtime-owned: applied even when jax
was imported first, and a resolved `False` refuses at startup.

**Matmul precision.** XLA:GPU lowers an f32 `dot_general` at DEFAULT
precision to TensorFloat32 (10-bit mantissa), and a complex64 dot decomposes
into real f32 dots. On the BSE ladder matvec that is a relative forward error
of $1.9\times10^{-4}$ against $3.2\times10^{-7}$ at fp32. `runtime.bootstrap()`
pins `jax_default_matmul_precision = 'highest'`; `LORRAX_MATMUL_PRECISION`
overrides it and refuses anything but `highest` or `float32` (`high` is a
3-pass TF32 decomposition on XLA:GPU). complex128, the GW/BSE production
dtype, is unaffected.

**Compile cache.** One owner, `common.jax_compile_cache`; the directory
resolution and controls are in [`env_vars.md` §2e](../dev/env_vars.md#2e-compile-cache). The
persistent key includes every array shape, so a new system size misses.

### 2.1 The GPU memory pool {#gpu-pool}

**Policy.** On a CUDA run with an NVIDIA device present,
`runtime.set_default_gpu_pool()` (called from `set_default_env()`, before
any client exists) sets

$$
\texttt{ALLOCATOR}=\texttt{cuda\_async},\qquad
\texttt{PREALLOCATE}=\texttt{true},\qquad
f=\texttt{XLA\_CLIENT\_MEM\_FRACTION}=0.89\ (\texttt{runtime.GPU\_POOL\_FRACTION}),
$$

one value on 40 and 80 GB cards. CPU runs, GPU-less nodes and ROCm are left
to jaxlib's defaults. No module, launcher or run script sets these variables.
The rule is all or nothing:

| exported by the caller | result |
|---|---|
| none of `ALLOCATOR`, `PREALLOCATE` | the policy (the fraction too, unless one is exported) |
| `ALLOCATOR` (any valid value), with or without `PREALLOCATE` | the caller's pair; the startup report names it as not the policy |
| `PREALLOCATE` on, no `ALLOCATOR` | **refuses**: BFC with $f\cdot M$ pre-grabbed, never released to NCCL or cuSOLVERMp |
| `PREALLOCATE=false`, no `ALLOCATOR` | BFC, unreserved (the test suite's exception, below) |
| `ALLOCATOR=cuda_async` with `PREALLOCATE` off | **refuses**: an unreserved async pool |
| both `XLA_CLIENT_MEM_FRACTION` and `XLA_PYTHON_CLIENT_MEM_FRACTION` | **refuses** (jaxlib refuses the pair inside plugin discovery, where it reads as "Unable to initialize backend 'cuda'") |
| an allocator spelling jaxlib does not accept | **refuses** (`runtime._check_allocator_env`) |

**What XLA builds (jaxlib 0.9.1).** PJRT's `cuda_async` allocator draws from
the device's **default** mempool (`create_new_pool=false`), the pool every FFI
`cudaMallocAsync` (cuBLASMp W-solve, cuSOLVERMp LU) also uses. With $M$ the
device total, it reserves $R = f M$ once and sets the pool's release threshold
to $R$, so idle memory stays mapped. With `PREALLOCATE=false` the threshold is
0: the pool unmaps every idle byte at each stream synchronize and the next
launch maps it again, a 25–110 ms device-idle stall per executable. The
fraction is not a cap (`AllocateRaw` never checks it); it sizes $R$ and the
reported `bytes_limit` $=R$, from which every planner budgets
$B = 0.9R = 0.801M$ (`common.gpu_utils.get_device_memory_gb`).

**Memory outside the pool.** The CUDA context, NCCL communicators, the
cuSOLVERMp context and its grow-only `cudaMalloc` workspace live outside
$R$. Per rank at P=4 on A100-40GB (sandbox claim 2697):

| cumulative | bytes outside the pool |
|---|---|
| CUDA context + modules | 0.46 GB |
| + three XLA NCCL cliques (x, y, xy) | 1.37 GB |
| + cuSOLVERMp context, eigh $n=8192$ | 2.92 GB |
| + cuSOLVERMp potrf $n=8192$ (context workspace) | 4.71 GB |
| + jaxlib local eigh $n=10^4$ (workspace is XLA's) | 4.72 GB |

On 40 GB, $M - R = 0.11 \times 42.4 = 4.66$ GB is less than 4.72 GB, so these
bytes fit only because the driver releases **idle** reserved memory, on
demand, to an unrelated allocation in the same process: raw `cudaMalloc`,
`cuMemCreate`, NCCL communicator init, module loads and launch-time
local-memory growth all take it (sandbox claim 2700). The driver
cannot release XLA's live bytes, a block whose stream-ordered free has not
retired, or idle memory inside a partly used pool chunk (fragmentation). The
out-of-memory condition is therefore

$$
N_\text{XLA live} + N_\text{outside} + N_\text{trapped idle} > M ,
$$

independent of whether the pool is reserved.

**One process per GPU.** The release does not cross processes: a second
process on the same GPU gets only $M - R$ minus the first process's outside
bytes. `tests/conftest.py` therefore pins its workers to BFC with
`PREALLOCATE=false` (they share GPUs with `tests/harness.py`'s mesh-cell child,
which runs `ALLOCATOR=platform`).

**Accounting.** `cuda_async` and BFC keep `memory_stats()` populated.
`platform` is plain `cudaMalloc` and reports `bytes_limit = peak_bytes_in_use =
0`, which blinds `gw_init`, `gw_output` and `runtime.aot_memory`; memory figures
then come from an `nvidia-smi` sample of the whole GPU.

### 2.2 The CPU-run plugin skip

On a run that resolves to CPU, jax 0.9.1 still dlopens the CUDA library stack
during plugin discovery (its cost on a cold Frontera node:
[frontera.md §3](machines/frontera.md#3-cold-start)).
`runtime.skip_gpu_plugin_discovery()`, armed by `bootstrap()` /
`set_default_env()` when `JAX_PLATFORMS=cpu` or no NVIDIA device node is
visible, answers the discovery with a stub module; the same venv still runs
GPU jobs. `LORRAX_CPU_SKIP_GPU_PLUGINS=0` disables it, announced.

### 2.3 Device selection and multi-host

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m gw.gw_jax -i cohsex.in    # restrict GPUs
export XLA_FLAGS="--xla_force_host_platform_device_count=4"  # CPU mock mesh
```

`runtime.initialize_communicator_stack()` owns multi-process bring-up
([services](../architecture/services.md#runtime)); every rank calls it. Under
SLURM with `SLURM_NTASKS > 1` it calls `jax.distributed.initialize()` with
`local_device_ids` derived from `CUDA_VISIBLE_DEVICES`; off SLURM set
`JAX_COORDINATOR_ADDRESS`, `JAX_NUM_PROCESSES` and `JAX_PROCESS_INDEX`. One GPU
per rank is pinned by `src/ffi/cpp/select_gpu.sh` through
`CUDA_VISIBLE_DEVICES`, not by `--gpus-per-task=1`, which breaks JAX's
topology exchange. `jax.distributed` bring-up costs about 1 s, flat to P=64; a
slow "distributed init" is the CUDA plugin load inside the first
`jax.devices()`.

---

## 3. Troubleshooting

| symptom | cause and fix |
|---|---|
| `Unable to initialize backend 'cuda'` on a GPU node | `nvidia-smi` and `CUDA_VISIBLE_DEVICES`; an allocator or fraction variable exported after `set_default_env()` ran escapes its refusals (§2.1) |
| `WARNING: the live client does NOT hold LORRAX's pool` | the CUDA client was built before `set_default_env()`; import the driver (or call `runtime.bootstrap()`) before anything touches a jax device |
| `RESOURCE_EXHAUSTED: Out of memory` | read the memory-planner report in `gwjax.out` against [memory-model](../architecture/memory-model.md); lower `memory_per_device_gb` if outside-pool libraries are unusually large |
| `cusolverMpSyevd: status=7` with NCCL error 1 | memory outside XLA's pool could not be had: XLA's live bytes plus the outside bytes exceed the card, or a caller's BFC arena is holding it (the startup report names the pair) |
| a CPU/MPI run exits rc=1 after succeeding | the driver did not end through `runtime.run_main_and_finalize()` ([transports](transports.md)) |
| HDF5 "file is already open" on Lustre | `HDF5_USE_FILE_LOCKING=FALSE` (runtime default) |
| wrong data from `psum_scatter` on CPU, rc=0 | the gloo transport; CPU collectives must run `impl=mpi` ([transports](transports.md)) |
| compile-cache `KeyError` warnings | clear the directory `common.jax_compile_cache` names at startup |

Debug flags: `JAX_DEBUG_NANS=1`, `JAX_DISABLE_JIT=1`, `JAX_LOG_COMPILES=1`,
`TF_CPP_MIN_LOG_LEVEL=0`; profiling through `common.jax_profile`.

---

## 4. Dependencies

[`pyproject.toml`](../../pyproject.toml) is the dependency authority (runtime
dependencies; groups `dev`, `jax`, `build`, `profile`; extras `cuda12`,
`cuda13`).

**NVIDIA GPUs require `nvidia-mathdx`** (header-only cuFFTDx, pinned in the
CUDA extras). Every k-axis convolution (ζ fit, Σ, COHSEX, BSE) runs kernels
that NVRTC compiles at run time from that wheel's headers; a CUDA run without
it refuses with `GATE mathdx-headers`. Compiled images are cached in
`$SCRATCH/.cache/lorrax/kconv_mathdx` (else `~/.cache/lorrax/kconv_mathdx`).
