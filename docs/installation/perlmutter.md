# Perlmutter (NERSC)

Perlmutter (4×A100 GPU nodes) is the reference platform. This page installs
LORRAX from a clone on any NERSC account and runs the test suite with plain
`srun`. It needs no `lx`, no `lorrax_A` module and no file outside the clone
and your own directories. Launch geometry and machine behaviour are owned by
[machines/perlmutter.md](../environment/machines/perlmutter.md).

Members of project m4598 can instead use the maintainers' `lorrax_A` module
and its `lx` launcher, which supply a prebuilt sealed pair
([below](#lorrax-a)).

## Where things go

| what | default | move it with |
|---|---|---|
| the Python venv | `<clone>/.venv` | (the clone location) |
| SLATE and the cuSOLVERMp/cuBLASMp stage | `<clone>/.build` | `LORRAX_BUILD_PREFIX` (share one SLATE between clones) |
| the two FFI libraries | `<clone>/src/ffi/cpp/build{,_host}/` | nothing; the loader looks there |
| JAX compile cache, mathdx kernel cache | `$SCRATCH/.cache/lorrax/` | [`ISDF_JAX_CACHE_DIR`](../dev/env_vars.md) (compile cache only) |
| the CUDA driver's JIT cache | `$SCRATCH/.nv/ComputeCache` once `gpu_env.sh` is sourced (CUDA's default is `~/.nv`) | `CUDA_CACHE_PATH` |
| uv's download cache and Python | `~/.cache/uv`, `~/.local/share/uv` | `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` (uv's own variables) |

By default uv and the CUDA driver's JIT cache write under `$HOME`. The home
quota is small, so the commands below point both at `$SCRATCH`. `$SCRATCH` is purged; keep the clone
on CFS or rebuild after a purge.

## 1. Environment

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
which the k-convolution needs on every NVIDIA GPU. Plain `uv sync` installs
CPU JAX only. `uv sync` takes about 30 s; uv downloads its own Python 3.12.

## 2. Build the native pair (once per clone, about 4 min) {#build}

LORRAX needs both FFI libraries at every process count
([Installation](index.md)). Build them on a compute node; with an allocation
`JOBID` (for example from
`salloc -N 1 -C gpu -q interactive -t 1:00:00 -A <account>`):

```bash
srun --jobid=$JOBID -N 1 -n 1 -c 64 --gres=none bash -c '
  bash src/ffi/cpp/stage/slate_build_perlmutter.sh cpu &&
  bash config/perlmutter/build_ffi_host.sh &&
  bash config/perlmutter/build_ffi_cuda.sh'
```

| step | builds | time |
|---|---|---|
| `slate_build_perlmutter.sh cpu` | SLATE `gpu_backend=none` into `.build/slate` | ~2 min |
| `build_ffi_host.sh` | the host leg, `src/ffi/cpp/build_host/liblorrax_ffi_host.so` | ~30 s |
| `build_ffi_cuda.sh` | stages the cuSOLVERMp/cuBLASMp wheels, then the CUDA leg, `src/ffi/cpp/build/liblorrax_ffi.so` | ~1 min |

Every leg ends in `scripts/verify_ffi_build.sh` and prints `VERIFY PASSED`
([the verify contract](../building_ffi.md#the-verify-contract)). Both legs link
the one MPI pinned in `config/perlmutter/ffi_mpi.sh`. Rebuild both legs after
a pull that changes `src/ffi/cpp/`: a stale leg refuses with
`HANDLER ABI MISMATCH`. The libraries are unsealed, so the first load in a run
prints `LEGACY-UNSEALED` with the file hash; that is expected for a private
build. Sealing a pair for other users is
[Building the FFI libraries](../building_ffi.md#seal-the-deployable-pair).

## 3. Run the test suite {#suite}

```bash
source config/perlmutter/gpu_env.sh
srun --jobid=$JOBID -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh \
  .venv/bin/python -m pytest tests/hsuite -q -p no:cacheprovider
```

The result is `1 passed` in about 5 min on cold caches.
[Contributing](../contributing.md#the-test-suite) owns what the suite checks.

- `config/perlmutter/gpu_env.sh` holds the machine's run settings:
  `MPICH_GPU_SUPPORT_ENABLED=0`
  ([why](../environment/machines/perlmutter.md#2-the-lorrax_a-module-and-the-ffi-bundle))
  and `CUDA_CACHE_PATH` on `$SCRATCH`. Source it once per shell before any
  GPU `srun`. Nothing else needs setting.

- Use one rank per GPU. `select_gpu.sh` pins each rank to one GPU and leaves
  the other three visible, which NCCL needs. `--gpus-per-task=1` hides them,
  and the first collective fails with `invalid device ordinal`.
- Runs and a driver's `srun` line have the same shape. Replace the pytest
  command with `.venv/bin/python -u -m gw.gw_jax -i deck.in` from the run
  directory. P16 is `-N 4 -n 16 --gpus-per-node=4`. A multi-node step also
  needs `--network=no_vni`
  ([why](../environment/machines/perlmutter.md#network-transport-at-startup)).
- Never run a driver on a login node.

## The `lorrax_A` module (m4598) {#lorrax-a}

`lorrax_A` supplies the runtime venv and one sealed pair: `LORRAX_FFI_SO` and
`LORRAX_FFI_HOST_SO` point into a single
[sealed bundle](../architecture/ffi_layout.md#2c-the-deployable-unit-is-one-sealed-pair).
The `lx` launcher (`lx run <cmd>`, `lx test`) loads it on a compute node.
Neither `lx` nor the module is in this repository; both live in the
maintainers' project space. The bundle's CUDA leg is built by
`config/perlmutter/build_ffi_cuda.sh`, and its sealing and publication follow
[machines/perlmutter.md §2](../environment/machines/perlmutter.md#2-the-lorrax_a-module-and-the-ffi-bundle).

A private pair replaces the module's bundle only as a pair: pin both legs to
one sealed bundle. Pinning one leg, or mixing a private leg with the module's,
refuses at startup.

The nvcc translation units of the CUDA leg are compiled for
`CMAKE_CUDA_ARCHITECTURES=80` (A100) unless the build sets another
architecture. The mathdx k-convolution router compiles its kernels per k-grid
with NVRTC at run time.
