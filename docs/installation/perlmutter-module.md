# Perlmutter module (maintainers)

The module stack is Perlmutter's default runtime
([runtime defaults](index.md#defaults)). One module is one venv, one
`git archive` source snapshot and one sealed FFI bundle, under an install
prefix. The recipe is `config/perlmutter/module/`. A clone for development
is [Perlmutter](perlmutter.md).

| what | pinned in |
|---|---|
| Python 3.12 (uv-managed, under the prefix), JAX 0.9.1 `cuda13-local`, cuDNN, nvidia-mathdx, `cudatoolkit/13.2` | `module/stack.sh` |
| every other Python package | the checkout's `uv.lock` |
| cuSOLVERMp, cuBLASMp | `build_ffi_cuda.sh` |
| Cray MPICH, LibSci | `ffi_mpi.sh` |
| parallel HDF5 | `build_ffi_cuda.sh`, `build_ffi_host.sh` |

The JAX plugin uses the toolkit's CUDA and installs no CUDA runtime.
`setup_env.sh` refuses a pip CUDA runtime in the venv.

## Layout

| path under `$LORRAX_MODULE_PREFIX` | written by |
|---|---|
| `python/`, `venv/` | `setup_env.sh` (once per prefix) |
| `build/` (SLATE, cuSOLVERMp stage) | `build.sh` (first build only) |
| `releases/source-<rev8>/`, `releases/<rev8>-bundle-<id12>/` | `publish.sh` (read-only, never replaced) |
| `modulefiles/<name>/<version>.lua` | `publish.sh` (previous file kept as `.before-<rev8>`) |

## Steps

Run every command from a clean checkout at the revision to publish. Every
script refuses a checkout that has uncommitted changes.

0. Set the environment. The prefix is durable and group-readable, for example
   `/global/common/software/<project>/lorrax`. Do not use `$SCRATCH`, which is
   purged. Compute nodes mount `/global/common` read-only.

   ```bash
   export LORRAX_MODULE_PREFIX=/abs/prefix     # required, no default
   export LORRAX_MODULE_NAME=lorrax            # default lorrax
   export LORRAX_MODULE_VERSION=0.1.0          # default 0.1.0
   export UV_CACHE_DIR=$SCRATCH/.cache/uv      # uv's own; home quota
   ```

1. **Venv**, once per prefix, on a login node (about 20 s). A pin change
   needs a new prefix, because the venv is never refreshed in place.

   ```bash
   bash config/perlmutter/module/setup_env.sh $PWD
   ```

2. **Candidate**, once per revision, on a zero-GPU compute step. It takes
   about 1 min, plus about 2 min for SLATE on the first build in a prefix.

   ```bash
   srun --jobid=$JOBID -N 1 -n 1 -c 64 --gres=none \
     bash config/perlmutter/module/build.sh $PWD /abs/cand
   ```

   The script clones the revision into `cand/src` and links its `.venv` to
   the module venv. It builds both legs with `build_ffi_host.sh` and
   `build_ffi_cuda.sh`; each leg ends in `VERIFY PASSED`. It seals the pair
   into `cand/bundle`, with the private closure cuBLASMp, cuSOLVERMp, BLAS++,
   LAPACK++ and SLATE. It writes `cand/source` (`git archive` plus
   `SOURCE_COMMIT`) and renders `cand/modulefiles/`. It refuses a candidate
   path that is relative, that exists, or that lies inside the checkout or
   inside `releases/`.

3. **Acceptance**, 1 node with 4 GPUs (about 6 min cold). Launch it from a
   shell with no LORRAX module loaded and no LORRAX tree on `PYTHONPATH`.
   `cand` must be writable from compute nodes, because `tests/hsuite` writes
   inside the source tree.

   ```bash
   S=/abs/cand/source/src/ffi/cpp/select_gpu.sh
   srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node=4 $S bash config/perlmutter/module/accept.sh /abs/cand
   srun --jobid=$JOBID -N 1 -n 4 --gpus-per-node=4 $S bash config/perlmutter/module/accept.sh /abs/cand
   ```

   Every process loads the candidate module and runs from `cand/accept/`.
   The step shape selects the checks:
   - `-n 1` runs Gate 10 (`gate_one_odr.py`). It must be alone in its MPI
     world, because its HDF5 file belongs to one process.
   - `-n 4` runs `verify_runtime.py` and `tests/hsuite`, which reports
     `1 passed`. `verify_runtime.py` checks the pinned versions, 4 ranks × 1
     GPU, and `runtime` imported from the snapshot. It also checks that both
     legs are mapped from the bundle with their manifest hashes and that no
     pip CUDA library is mapped.

   Passing writes `accept/gate10.ok` and `accept/ok.<rank>`.

4. **Publish**, on a login node:

   ```bash
   bash config/perlmutter/module/publish.sh /abs/cand
   ```

   The script needs `gate10.ok` and `ok.0`–`ok.3`. It writes a fresh `git archive` and a
   copy of the bundle, checking the copy against the candidate hash by hash.
   It makes both read-only and renames the modulefile into place.

5. **Check** in a fresh shell, on a compute node:
   `module use $LORRAX_MODULE_PREFIX/modulefiles && module load lorrax`, then
   `python -c 'import runtime, jax; print(runtime.__file__, jax.devices())'`.

6. **Rollback.** Render the module against an earlier release pair:
   `bash config/perlmutter/module/install_module.sh <releases/source-…> <releases/…-bundle-…> <modulefile>`.

## Using the module

```bash
module use $LORRAX_MODULE_PREFIX/modulefiles
module load lorrax
srun -N 1 -n 4 --gpus-per-node=4 $LORRAX_ROOT/src/ffi/cpp/select_gpu.sh \
  python -u -m gw.gw_jax -i deck.in
```

The module sets the following:
- `PATH` (venv and CUDA);
- `PYTHONPATH` (the snapshot's `src` and `services/*/src`);
- `LD_LIBRARY_PATH` (cuDNN, the toolkit and HPC SDK math and NCCL
  libraries, the CUDA compat driver, the pinned HDF5 and MPICH, libfabric);
- `LORRAX_ROOT`, `LORRAX_FFI_SO`, `LORRAX_FFI_HOST_SO`, `CUDA_HOME`;
- `MPICH_GPU_SUPPORT_ENABLED=0`, `JAX_PLATFORMS=cuda,cpu`,
  `JAX_ENABLE_X64=1`.

It sets no allocator or cache policy. Multi-node steps add `--network=no_vni`
([why](../environment/machines/perlmutter.md#network-transport-at-startup)).
Do not add a checkout to `PYTHONPATH` or pin one FFI leg: a mixed source or
a partial bundle refuses at startup ([Installation](index.md)).

`LORRAX_SHIFTER`, `LORRAX_MPI_TYPE` and `LORRAX_MPICH_GPU_SUPPORT` are read
only by project m4598's `lx` launcher, which is not in this repository.
`srun` ignores them. The m4598 `lorrax_A` module has this contract. Its
current venv predates this recipe: it was built on NERSC's Python module with
an editable install.
