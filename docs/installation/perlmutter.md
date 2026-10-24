# Perlmutter (NERSC)

This page installs LORRAX on NERSC Perlmutter, the reference platform (4×A100
GPU nodes, CUDA 13.2, JAX 0.12.0.1). There are two routes. A **clone** works on
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
| the cuSOLVERMp/cuBLASMp link stage | `<clone>/.build/cusolvermp` | (the clone location) |
| the two FFI libraries | `<clone>/src/ffi/cpp/build{,_host}/` | nothing; the loader looks there |
| JAX compile cache, mathdx kernel cache | `$SCRATCH/.cache/lorrax/` | [`ISDF_JAX_CACHE_DIR`](../reference/env_vars.md) (compile cache only) |
| the CUDA driver's JIT cache | `$SCRATCH/.nv/ComputeCache` once `gpu_env.sh` is sourced (CUDA's default is `~/.nv`) | `CUDA_CACHE_PATH` |
| uv's download cache and Python | `~/.cache/uv`, `~/.local/share/uv` | `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` (uv's own variables) |

The home quota is small, so the commands below point uv and the CUDA JIT
cache at `$SCRATCH`. `$SCRATCH` is purged: keep the clone on CFS, or rebuild
after a purge.

### 1.1 Environment

Use a login shell with access to the site's pinned MPI, HDF5 and LibSci
installs. The build presets name those libraries explicitly and use plain
GCC; the CUDA compiler and math libraries come from the venv's locked wheels.

```bash
export UV_CACHE_DIR=$SCRATCH/.cache/uv UV_PYTHON_INSTALL_DIR=$SCRATCH/.local/uv-python
command -v uv || {
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=$SCRATCH/.local/bin UV_NO_MODIFY_PATH=1 sh
  export PATH=$SCRATCH/.local/bin:$PATH
}
git clone --filter=blob:none https://github.com/jackmcarthur/lorrax.git $SCRATCH/lorrax
cd $SCRATCH/lorrax
uv sync --extra cuda13
```

`--extra cuda13` installs the CUDA-13 JAX 0.12 wheels and `nvidia-mathdx`,
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
  bash config/perlmutter/build_ffi_host.sh &&
  bash config/perlmutter/build_ffi_cuda.sh'
```

| step | builds | time (one Perlmutter node) |
|---|---|---|
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

**The build records, and every run enforces, each leg's dependency closure**
([the dependency closure](ffi-build.md#closure)). Each build writes
`<leg>.closure.json` beside its leg, naming every library the leg maps:

- the staged cuSOLVERMp 0.9.1 and cuBLASMp 0.10;
- the cuBLAS, cuSOLVER, cuFFT and CUDA runtime files that the venv's JAX
  plugin loads (the `nvidia-*-cu13` wheels);
- NCCL: the `nvidia-nccl-cu13` 2.32.3 wheel (the `uv.lock` pin), on every
  route. The CUDA leg compiles and links against it, and `build_ffi_cuda.sh`
  refuses another wheel version or a record whose NCCL is not that wheel. The
  HPC SDK's `comm_libs` NCCL (2.29.3) is never used;
- the pinned MPI, HDF5 and LibSci of `config/perlmutter/ffi_mpi.sh`.

At start-up the loader opens exactly those files, then refuses the process if
any other copy of one of them is mapped. So a clone runs on what its build
verified, whatever the shell's `LD_LIBRARY_PATH` holds. `cudatoolkit/13.2`
puts the HPC SDK's cuSOLVERMp 0.8 there under the same SONAME, and with it the
first distributed LU or eigh fails with `status=7`. The site-default LibSci
maps cray-mpich 9.1.0. Neither can be loaded in place of the recorded file.
A refusal reads

```text
GATE native-closure: got <wrong file>; want <recorded file>; fix: <remedy>
```

The remedy is to remove what maps the other copy (an `LD_PRELOAD`, or a
package such as mpi4py built against another MPI), or to rebuild the leg in the
environment you run in. Setting `LD_LIBRARY_PATH` neither causes nor fixes it,
with one exception: on a multi-node step libfabric dlopens `libcudart.so` by
that name, through `LD_LIBRARY_PATH`. The wheel ships only `libcudart.so.13`,
so `cudatoolkit/13.2`'s `libcudart.so` (13.2.75) would be mapped beside the
recorded wheel runtime. `gpu_env.sh` links `libcudart.so` to the venv's
runtime, first on `LD_LIBRARY_PATH`. A multi-node step whose `libcudart.so`
names another file refuses at start-up with a `GATE native-closure` line
naming both files ([launch libraries](ffi-build.md#closure)). Source
`gpu_env.sh` in the shell that runs `srun`.
Moving or purging `.build`, or a `uv sync` that changes the CUDA wheels, makes
the record stale, and the next run refuses until the legs are rebuilt.

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
  `CUDA_CACHE_PATH` on `$SCRATCH`, and `SLURM_NETWORK=no_vni` when unset in
  a launch shell. Source it once per shell before any GPU `srun`; explicit
  network choices are preserved. Nothing else needs setting.
- Use one rank per GPU, through `select_gpu.sh` and never
  `--gpus-per-task=1`
  ([why](../environment/machines/perlmutter.md#required-gpu-task-geometry)).
- A driver's `srun` line has the same shape: replace the pytest command with
  `.venv/bin/python -u -m gw.gw_jax -i deck.in`, run from the run directory.
  P16 is `-N 4 -n 16 --gpus-per-node=4`. The sourced setup supplies the network
  default before the step is created
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

The bundle's manifest carries each leg's dependency closure. Its libraries
are the toolkit's CUDA math files (the module venv has no CUDA math wheels),
the venv's `nvidia-nccl-cu13` 2.32.3 wheel (the same NCCL as a clone), the
sealed cuSOLVERMp/cuBLASMp copies, and the pinned MPI and HDF5. Every start-up enforces that closure exactly as for a clone
([the dependency closure](ffi-build.md#closure)). The module's
`LD_LIBRARY_PATH` serves JAX and the external system libraries; it cannot
change which file a leg uses.

It sets no allocator, compile-cache, HDF5 or profiling policy: the runtime
owns those ([GPU pool](../environment/overview.md#gpu-pool)), and an export
would override it. `gpu_env.sh` is not needed with a module. Do not add a
checkout to `PYTHONPATH` or pin one FFI leg: a mixed source tree or a
partial bundle refuses at startup ([Installation](index.md)).

### 2.2 Publishing a module (maintainers) {#publish}

The recipe is `config/perlmutter/module/`; it works for any install prefix.

| what | pinned in |
|---|---|
| Python 3.12 (uv-managed, under the prefix), `cudatoolkit/13.2` (nvcc; no library at run time) | `module/stack.sh` |
| every Python package, the CUDA wheels included (JAX 0.12.0.1 `cuda13`, cuBLAS, cuFFT, cuSOLVER, cuSPARSE, CUDA runtime, NVRTC, cuDNN, nvidia-mathdx): the clone's set | the checkout's `uv.lock`, extra `cuda13` |
| cuSOLVERMp, cuBLASMp | `build_ffi_cuda.sh` (repeated in `stack.sh`) |
| NCCL: `nvidia-nccl-cu13` 2.32.3 (the clone's `uv.lock` pin) | `build_ffi_cuda.sh`, `stack.sh` |
| Cray MPICH, LibSci | `ffi_mpi.sh` |
| parallel HDF5 | `build_ffi_cuda.sh`, `build_ffi_host.sh` |

The module venv holds a clone's CUDA libraries. `setup_env.sh` installs
`uv export --extra cuda13` of the checkout's lock with `--no-deps`, refuses a
package at any other version (and a lock whose JAX or NCCL disagrees with
`stack.sh`), and links `venv/lorrax-launch/libcudart.so` to the wheel's CUDA
runtime, the name libfabric dlopens. Each leg's closure therefore records the
wheels, as for a clone (the JAX plugin's directories come before the toolkit's
in `build_ffi_cuda.sh`). The modulefile lists no toolkit library directory;
`install_module.sh` refuses one. The toolkit module stays loaded for nvcc and
ptxas, so `PATH` and `CUDA_HOME` are those of a clone shell. `verify_runtime.py`
refuses a mapped toolkit CUDA library, requires the wheels' cudart, cuBLAS and
cuFFT and every `jax*`/`nvidia-*` package at its locked version, requires
exactly the NCCL wheel's `libnccl`, and prints both legs' enforced closure
identities. A module run and a clone run of one tree map the same CUDA, NCCL,
cuSOLVERMp, cuBLASMp, MPI and HDF5 files. Measured on Fe V2 P=4 (2026-10-17):
the module run is bitwise identical to the clone run.

The wheel venv is about 4.3 GB. The project's `/global/common/software` quota
is 10 GiB, so check `showquota --cmn <project>` before `setup_env.sh`: a build
over quota fails in `seal_bundle.py` with `Disk quota exceeded`.

| path under `$LORRAX_MODULE_PREFIX` | written by |
|---|---|
| `python/`, `venv/` (or `LORRAX_MODULE_VENV`, an absolute path: `lorrax_cuda13_runtime` keeps `.venv`) | `setup_env.sh` (once per prefix) |
| `native/` (the cuSOLVERMp stage; `build/` before 2026-10-17, which `seal_bundle.py` refuses as a search-path directory) | `build.sh` (first build only) |
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
   1 min).

   ```bash
   srun --jobid=$JOBID -N 1 -n 1 -c 64 --gres=none \
     bash config/perlmutter/module/build.sh $PWD /abs/cand
   ```

   The script clones the revision into `cand/src` and links its `.venv` to
   the module venv; builds both legs with `build_ffi_host.sh` and
   `build_ffi_cuda.sh` (each ends in `VERIFY PASSED`); seals the pair into
   `cand/bundle` with the private closure cuBLASMp and cuSOLVERMp; writes `cand/source` (`git archive` plus
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
