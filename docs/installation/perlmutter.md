# Perlmutter (NERSC)

This page installs LORRAX on NERSC Perlmutter, the reference platform (4×A100
GPU nodes, CUDA 13.2, JAX 0.9.1). There are two routes. A **clone** works on
any NERSC account: you build the native pair yourself and run with plain
`srun`; use it for development. A **module** is one prebuilt venv, source
snapshot and sealed FFI bundle that users load with `module load`; it is the
default runtime, and §2 also shows maintainers how to publish one. Read
[Installation](index.md) first; launch geometry and machine behaviour are
owned by [the Perlmutter machine page](../environment/machines/perlmutter.md).

## 1. A clone {#clone}

A clone needs no file outside itself and your own directories. Everything it
writes goes to one of these places:

| what | default | move it with |
|---|---|---|
| the Python venv | `<clone>/.venv` | (the clone location) |
| SLATE and the cuSOLVERMp/cuBLASMp stage | `<clone>/.build` | `LORRAX_BUILD_PREFIX` (to share one SLATE between clones) |
| the two FFI libraries | `<clone>/src/ffi/cpp/build{,_host}/` | nothing; the loader looks there |
| JAX compile cache, mathdx kernel cache | `$SCRATCH/.cache/lorrax/` | [`ISDF_JAX_CACHE_DIR`](../reference/env_vars.md) (compile cache only) |
| the CUDA driver's JIT cache | `$SCRATCH/.nv/ComputeCache` once `gpu_env.sh` is sourced (CUDA's default is `~/.nv`) | `CUDA_CACHE_PATH` |
| uv's download cache and Python | `~/.cache/uv`, `~/.local/share/uv` | `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` (uv's own variables) |

The home quota is small, so the commands below point uv and the CUDA JIT
cache at `$SCRATCH`. `$SCRATCH` is purged: keep the clone on CFS, or rebuild
after a purge.

### 1.1 Environment

Use a login shell with the site's default modules (`PrgEnv-gnu`,
`cudatoolkit/13.2`). Each build script loads and pins the modules it needs;
run scripts need none.

```bash
export UV_CACHE_DIR=$SCRATCH/.cache/uv UV_PYTHON_INSTALL_DIR=$SCRATCH/.local/uv-python
command -v uv || {
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=$SCRATCH/.local/bin UV_NO_MODIFY_PATH=1 sh
  export PATH=$SCRATCH/.local/bin:$PATH
}
git clone https://github.com/jackmcarthur/lorrax.git $SCRATCH/lorrax
cd $SCRATCH/lorrax
uv sync --extra cuda13
```

`--extra cuda13` installs the CUDA-13 JAX 0.9 wheels and `nvidia-mathdx`,
which every k-axis convolution needs on an NVIDIA GPU. Plain `uv sync`
installs CPU JAX only. uv downloads its own Python 3.12; the sync takes about
30 s.

### 1.2 Build the native pair (once per clone) {#build}

Build both legs on a compute node. The build needs no GPU (nvcc compiles
for the listed architectures and the NVRTC kernels compile at run time), so
the build step asks for none and the node's GPUs stay free for the test step
below. Take one allocation `JOBID` for both, for example
`salloc -N 1 -C gpu -G 4 -q interactive -t 1:00:00 -A <account>`:

```bash
srun --jobid=$JOBID -N 1 -n 1 -c 64 --gres=none bash -c '
  bash src/ffi/cpp/stage/slate_build_perlmutter.sh cpu &&
  bash config/perlmutter/build_ffi_host.sh &&
  bash config/perlmutter/build_ffi_cuda.sh'
```

| step | builds | time (one Perlmutter node) |
|---|---|---|
| `slate_build_perlmutter.sh cpu` | SLATE with `gpu_backend=none` into `.build/slate` | about 2 min |
| `build_ffi_host.sh` | the host leg, `src/ffi/cpp/build_host/liblorrax_ffi_host.so` | about 30 s |
| `build_ffi_cuda.sh` | stages the cuSOLVERMp/cuBLASMp wheels, then the CUDA leg, `src/ffi/cpp/build/liblorrax_ffi.so` | about 1 min |

Every leg ends in `scripts/verify_ffi_build.sh` and prints `VERIFY PASSED`
([the verify contract](ffi-build.md#verify)). Both legs link the one MPI
pinned in `config/perlmutter/ffi_mpi.sh`, because both are loaded into one
process ([Perlmutter build facts](ffi-build.md#perlmutter)). Rebuild both
legs after a pull that changes `src/ffi/cpp/`: a stale leg refuses with
`HANDLER ABI MISMATCH`. The libraries are unsealed, so the first load in a run
prints `LEGACY-UNSEALED` with the file hash; that is expected for a private
build.

### 1.3 Run the test suite {#suite}

```bash
source config/perlmutter/gpu_env.sh
srun --jobid=$JOBID -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh \
  .venv/bin/python -m pytest tests/hsuite -q -p no:cacheprovider
```

The result is `1 passed`, in about 5 min on cold caches.
[Contributing](../contributing.md#the-test-suite) owns what the suite checks.

- `config/perlmutter/gpu_env.sh` holds the machine's run settings:
  `MPICH_GPU_SUPPORT_ENABLED=0`
  ([why](../environment/machines/perlmutter.md#mpich-gpu-support)) and
  `CUDA_CACHE_PATH` on `$SCRATCH`. Source it once per shell before any GPU
  `srun`. Nothing else needs setting.
- Use one rank per GPU, through `select_gpu.sh` and never
  `--gpus-per-task=1`
  ([why](../environment/machines/perlmutter.md#required-gpu-task-geometry)).
- A driver's `srun` line has the same shape: replace the pytest command with
  `.venv/bin/python -u -m gw.gw_jax -i deck.in`, run from the run directory.
  P16 is `-N 4 -n 16 --gpus-per-node=4`; a multi-node step also needs
  `--network=no_vni`
  ([why](../environment/machines/perlmutter.md#network-transport-at-startup)).
- Never run a driver on a login node.

## 2. The module {#module}

A module is one venv, one `git archive` source snapshot and one sealed FFI
bundle under an install prefix, rendered as an Lmod modulefile. It is the
default runtime on Perlmutter because a sealed bundle is attested at load
(every library hashed against its manifest) and every user of the module runs
the same bytes.

### 2.1 Using a module {#using-the-module}

```bash
module use <prefix>/modulefiles
module load lorrax
srun -N 1 -n 4 --gpus-per-node=4 $LORRAX_ROOT/src/ffi/cpp/select_gpu.sh \
  python -u -m gw.gw_jax -i deck.in
```

The module sets:

- `PATH` (the venv and CUDA), `PYTHONPATH` (the snapshot's `src` and
  `services/*/src`) and `LD_LIBRARY_PATH` (cuDNN, the toolkit and HPC SDK
  math and NCCL libraries, the CUDA compat driver, the pinned HDF5 and
  MPICH, libfabric);
- `LORRAX_ROOT` (the snapshot), `LORRAX_FFI_SO` and `LORRAX_FFI_HOST_SO`
  (the two legs of its sealed bundle), `CUDA_HOME`;
- `MPICH_GPU_SUPPORT_ENABLED=0`, `JAX_PLATFORMS=cuda,cpu`,
  `JAX_ENABLE_X64=1`;
- `LORRAX_SHIFTER`, `LORRAX_MPI_TYPE`, `LORRAX_MPICH_GPU_SUPPORT` and
  `LORRAX_FFI_EXPECT_BACKENDS`, which only the m4598 `lx` launcher and
  `src/ffi/cpp/in_container.sh` read; `srun` ignores them.

It sets no allocator, compile-cache, HDF5 or profiling policy: the runtime
owns those ([GPU pool](../environment/overview.md#gpu-pool)), and an export
would override it. `gpu_env.sh` is not needed with a module. Do not add a
checkout to `PYTHONPATH` or pin one FFI leg: a mixed source tree or a
partial bundle refuses at startup ([Installation](index.md)).

### 2.2 Publishing a module (maintainers) {#publish}

The recipe is `config/perlmutter/module/`; it works for any install prefix.

| what | pinned in |
|---|---|
| Python 3.12 (uv-managed, under the prefix), JAX 0.9.1 `cuda13-local`, cuDNN, nvidia-mathdx, `cudatoolkit/13.2` | `module/stack.sh` |
| every other Python package | the checkout's `uv.lock` |
| cuSOLVERMp, cuBLASMp | `build_ffi_cuda.sh` (repeated in `stack.sh`) |
| Cray MPICH, LibSci | `ffi_mpi.sh` |
| parallel HDF5 | `build_ffi_cuda.sh`, `build_ffi_host.sh` |

The JAX plugin uses the toolkit's CUDA and installs no CUDA runtime;
`setup_env.sh` installs cuDNN and nvidia-mathdx without their dependency
closures and refuses a pip CUDA runtime in the venv, because one would shadow
the toolkit's.

| path under `$LORRAX_MODULE_PREFIX` | written by |
|---|---|
| `python/`, `venv/` | `setup_env.sh` (once per prefix) |
| `build/` (SLATE, cuSOLVERMp stage) | `build.sh` (first build only) |
| `releases/source-<rev8>/`, `releases/<rev8>-bundle-<id12>/` | `publish.sh` (read-only, never replaced) |
| `modulefiles/<name>/<version>.lua` | `publish.sh` (the previous file kept as `.before-<rev8>`) |

Run every command from a clean checkout at the revision to publish; every
script refuses a checkout with uncommitted changes, because the snapshot and
the bundle record the commit they came from.

0. **Environment.** The prefix must be durable and group-readable, for
   example `/global/common/software/<project>/lorrax`; `$SCRATCH` is purged.
   Compute nodes mount `/global/common` read-only.

   ```bash
   export LORRAX_MODULE_PREFIX=/abs/prefix     # required, no default
   export LORRAX_MODULE_NAME=lorrax            # default lorrax
   export LORRAX_MODULE_VERSION=0.1.0          # default 0.1.0
   export UV_CACHE_DIR=$SCRATCH/.cache/uv      # uv's own; home quota
   ```

1. **Venv**, once per prefix, on a login node (about 20 s). A pin change
   needs a new prefix: the venv is never refreshed in place, so a published
   module's interpreter never changes under it.

   ```bash
   bash config/perlmutter/module/setup_env.sh $PWD
   ```

2. **Candidate**, once per revision, in a zero-GPU compute step (about
   1 min, plus about 2 min for SLATE on the first build in a prefix).

   ```bash
   srun --jobid=$JOBID -N 1 -n 1 -c 64 --gres=none \
     bash config/perlmutter/module/build.sh $PWD /abs/cand
   ```

   The script clones the revision into `cand/src` and links its `.venv` to
   the module venv; builds both legs with `build_ffi_host.sh` and
   `build_ffi_cuda.sh` (each ends in `VERIFY PASSED`); seals the pair into
   `cand/bundle` with the private closure cuBLASMp, cuSOLVERMp, BLAS++,
   LAPACK++ and SLATE; writes `cand/source` (`git archive` plus
   `SOURCE_COMMIT`); and renders `cand/modulefiles/`. It refuses a candidate
   path that is relative, already exists, or lies inside the checkout or
   inside `releases/`.

3. **Acceptance**, 1 node with 4 GPUs (about 6 min cold). Launch from a
   shell with no LORRAX module loaded and no LORRAX tree on `PYTHONPATH`.
   `cand` must be writable from compute nodes, because the acceptance
   markers are written to `cand/accept/`.

   ```bash
   S=/abs/cand/source/src/ffi/cpp/select_gpu.sh
   srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node=4 $S bash config/perlmutter/module/accept.sh /abs/cand
   srun --jobid=$JOBID -N 1 -n 4 --gpus-per-node=4 $S bash config/perlmutter/module/accept.sh /abs/cand
   ```

   Every process loads the candidate module and runs from `cand/accept/`.
   The step shape selects the checks:
   - `-n 1` runs Gate 10 (`src/ffi/cpp/gate_one_odr.py`: a CUDA process with
     both legs open does host parallel-HDF5 work). It must be alone in its
     MPI world, because its HDF5 file belongs to one process.
   - `-n 4` runs `verify_runtime.py` and `tests/hsuite` (`1 passed`).
     `verify_runtime.py` checks the pinned versions, 4 ranks × 1 GPU, that
     `runtime` is imported from the snapshot, that both legs are mapped from
     the bundle with their manifest hashes, and that no pip CUDA library is
     mapped.

   Passing writes `accept/gate10.ok` and `accept/ok.<rank>`.

4. **Publish**, on a login node:

   ```bash
   bash config/perlmutter/module/publish.sh /abs/cand
   ```

   The script needs `gate10.ok` and `ok.0`–`ok.3`. It writes a fresh
   `git archive` and a copy of the bundle, checks the copy against the
   candidate hash by hash, makes both read-only and renames the modulefile
   into place.

5. **Check**, in a fresh shell on a compute node:
   `module use $LORRAX_MODULE_PREFIX/modulefiles && module load lorrax`, then
   `python -c 'import runtime, jax; print(runtime.__file__, jax.devices())'`.

6. **Rollback.** Render the module against an earlier release pair:
   `bash config/perlmutter/module/install_module.sh <releases/source-…> <releases/…-bundle-…> <modulefile>`.
