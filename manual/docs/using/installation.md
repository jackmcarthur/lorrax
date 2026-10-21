# Installation

`uv sync --extra cuda13` Python environment, the native FFI pair, and the machine routes for Perlmutter, Frontera and other sites.
{: .page-summary }

A LORRAX installation has three parts: a Python 3.12 environment with the JAX 0.9 series, a pair of native libraries built from `src/ffi/cpp/`, and a launcher that gives each process one device. All three are required at every process count, one process included.

## Requirements {: #requirements }

| Component | Requirement | Note |
|---|---|---|
| Python | ≥ 3.12 | uv installs its own 3.12 |
| JAX, jaxlib | `>= 0.9.0, < 0.10.0` | Any other minor series refuses at startup; there is no override |
| CUDA extra | `cuda13` (driver ≥ 580) or `cuda12` | Installs `jax[cudaNN]` and `nvidia-mathdx==25.6.0` |
| Python packages | `h5py >= 3.14`, `numpy >= 2.3.1`, `scipy >= 1.16`, `matplotlib`, `xmlschema`, `xsdata` | Pinned in `pyproject.toml` and `uv.lock` |
| GPU | NVIDIA sm_80 or newer | The floor of the CUDA FFI leg |
| MPI | the site MPI | Both FFI legs link the same `libmpi` |
| HDF5 | parallel HDF5 ≥ 1.12, built with `HDF5_IS_PARALLEL` | Both legs link the HDF5 the run maps |
| Linear algebra | BLAS + ScaLAPACK (LibSci, MKL or OpenBLAS + netlib); SLATE built with `gpu_backend=none`; cuSOLVERMp ≥ 0.7 and cuBLASMp | cuSOLVERMp 0.6.x returns wrong LU results on a 2-D process grid |
| FFT | any FFTW3-ABI library, found at run time | Never linked into the host leg |
| `jax_xc` | the patched 0.0.11 of `config/xc/` | Only for the [`nscf`](../drivers/nscf/index.md) driver and the dense-Hamiltonian route |

The service packages under `services/` (`distrib_la`, `lxkit`, `minimax`, `symmetry_maps`, `vcoul`, `wfn_loader`, `zeta_loader`) are uv workspace members and install with the project.

## The native FFI pair {: #ffi }

| Leg | File | Contains |
|---|---|---|
| host | `liblorrax_ffi_host.so` | ScaLAPACK, SLATE, a CBLAS GEMM, the FFTW3-ABI FFT, parallel HDF5. CUDA-free. |
| CUDA | `liblorrax_ffi.so` | cuSOLVERMp, cuBLASMp, cuBLAS, cuSOLVER, cuFFT, NCCL, the NVRTC-built mathdx k-convolution kernels, parallel HDF5 |

A CPU run loads the host leg; a GPU run loads both into one process. There is no pure-JAX fallback for the k-convolution, the distributed linear algebra or the slab I/O, so a missing leg refuses at startup. Each build ends in `scripts/verify_ffi_build.sh` and prints `VERIFY PASSED`; each leg records its dependency closure in `<leg>.closure.json`, and every startup refuses a process that maps any other copy of a recorded library. A private build loads with a `LEGACY-UNSEALED` line and its hash; only a sealed pair is a production provider.

## Routes by machine {: #routes }

| Machine | Route | Entry |
|---|---|---|
| NERSC Perlmutter (A100, CUDA 13.2) | a published module, or a clone that builds its own pair | `config/perlmutter/` |
| TACC Frontera (Cascade Lake CPU) | a clone with the host leg, run inside apptainer | `config/frontera/` |
| a single-node GPU box with only a driver | all-wheel CUDA 13 lane | `config/cloud/` |
| any other site | a clone with both legs built by CMake | `src/ffi/cpp/` |

### Perlmutter: a clone {: #perlmutter }

From a login shell with the default modules (`PrgEnv-gnu`, `cudatoolkit/13.2`):

```bash
export UV_CACHE_DIR=$SCRATCH/.cache/uv UV_PYTHON_INSTALL_DIR=$SCRATCH/.local/uv-python
git clone --filter=blob:none https://github.com/jackmcarthur/lorrax.git $SCRATCH/lorrax
cd $SCRATCH/lorrax
uv sync --extra cuda13
```

Build both legs once per clone, on a compute node, with no GPU requested:

```bash
srun --jobid=$JOBID -N 1 -n 1 -c 64 --gres=none bash -c '
  bash src/ffi/cpp/stage/slate_build_perlmutter.sh cpu &&
  bash config/perlmutter/build_ffi_host.sh &&
  bash config/perlmutter/build_ffi_cuda.sh'
```

| Step | Builds |
|---|---|
| `slate_build_perlmutter.sh cpu` | SLATE with `gpu_backend=none` into `.build/slate` |
| `build_ffi_host.sh` | `src/ffi/cpp/build_host/liblorrax_ffi_host.so` |
| `build_ffi_cuda.sh` | stages cuSOLVERMp and cuBLASMp, then `src/ffi/cpp/build/liblorrax_ffi.so` |

Rebuild both legs after a pull that changes `src/ffi/cpp/`, after moving `.build`, or after a `uv sync` that changes the CUDA wheels; a stale leg refuses. Run with `config/perlmutter/gpu_env.sh` sourced in the shell that calls `srun`, one rank per GPU through `src/ffi/cpp/select_gpu.sh`:

```bash
source config/perlmutter/gpu_env.sh
srun -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh .venv/bin/python -u -m gw.gw_jax -i gw.in
```

`gpu_env.sh` sets `MPICH_GPU_SUPPORT_ENABLED=0` and puts `CUDA_CACHE_PATH` on `$SCRATCH`. Never use `--gpus-per-task=1`. A multi-node step adds `--network=no_vni`.

### Perlmutter: the module {: #perlmutter-module }

```bash
module use <prefix>/modulefiles
module load lorrax
srun -N 1 -n 4 --gpus-per-node=4 $LORRAX_ROOT/src/ffi/cpp/select_gpu.sh python -u -m gw.gw_jax -i gw.in
```

The module carries one venv, one source snapshot and one sealed FFI bundle, and sets `LORRAX_ROOT`, `LORRAX_FFI_SO`, `LORRAX_FFI_HOST_SO`, `MPICH_GPU_SUPPORT_ENABLED=0`, `JAX_PLATFORMS=cuda,cpu` and `JAX_ENABLE_X64=1`. Do not add a checkout to `PYTHONPATH` or pin one leg by hand: a mixed source tree or a partial bundle refuses. Maintainers publish a module with the scripts in `config/perlmutter/module/` (`setup_env.sh`, `build.sh`, `accept.sh`, `publish.sh`).

### Frontera {: #frontera }

Frontera runs the host leg only: its RTX 5000 GPUs (sm_75) are below the CUDA leg's floor. JAX runs inside the apptainer image `py312.sif`, because the host glibc is 2.17. Multi-process CPU collectives run on MPI through MPItrampoline and a patched MPIwrapper.

| Artifact | Script | Runs on |
|---|---|---|
| host leg | `config/frontera/build_ffi_host.sh` | login node |
| MPIwrapper | `config/frontera/build_mpiwrapper.sh --fresh` | login node |
| mpi4py + parallel h5py overlay | `config/frontera/build_mpi_overlay.sh fetch`, then `build` | login node, then compute node in the image |
| staged PMI2 library | `config/frontera/stage_host_pmi.sh` | login node |
| node-local runtime bundle | `config/frontera/build_cpu_runtime_bundle.sh` | inside the image |
| launch | `config/frontera/templates/gw_dev.sbatch` | batch job |

The image, the venv `$WORK/lorrax_env/.venv` and the SLATE host install under `$WORK/slate_builds/cpu/install` are not rebuilt by any script in the repository. The launch template refuses when `LORRAX_ROOT` or `LORRAX_RUN_DIR` is unset; edit its `#SBATCH` block (account, nodes, time) before submitting.

### Other sites {: #other }

```bash
uv sync --extra cuda13          # or --extra cuda12; plain `uv sync` for a CPU-only install
cmake -S src/ffi/cpp -B build_cuda -DLORRAX_FFI_PLATFORM=cuda \
    -DCUSOLVERMP_INCLUDE_DIR=<dir> -DCUSOLVERMP_LIB_DIR=<dir> -DLORRAX_FFI_HAVE_CAL=OFF \
    -DHDF5_ROOT=<parallel HDF5 prefix> \
    -DLORRAX_MPI_INCLUDE_DIR=<dir with mpi.h> -DLORRAX_MPICH_LIB_DIR=<dir with libmpi>
cmake --build build_cuda -j
scripts/verify_ffi_build.sh --leg cuda build_cuda/liblorrax_ffi.so
```

The host leg is the same command with `-DLORRAX_FFI_PLATFORM=host` and `-DLORRAX_SLATE_HOST_INSTALL_DIR=<gpu_backend=none prefix>`, or `bash src/ffi/cpp/build_host.sh`. A library outside the in-tree build directories is selected with [`LORRAX_FFI_SO` and `LORRAX_FFI_HOST_SO`](environment.md#ffi). `config/cloud/` builds both legs from wheels on a machine with only an NVIDIA driver ≥ 580.

## Checking an installation {: #check }

The test suite runs every driver on the bundled magnetic H₂⁻ fixture and compares each stage with `tests/hsuite/reference/`:

```bash
srun -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh \
  .venv/bin/python -m pytest tests/hsuite -q -p no:cacheprovider
```

It ends `1 passed`. `python -m tests.hsuite.chain --out <dir>` runs the same chain outside pytest and leaves every deck, report and output in `<dir>/run/`.

## Sizing a job {: #sizing }

- **Process count.** One process per device, and the device count a perfect square: 1, 4, 9, 16, 25, 36, …. Any other count refuses at startup and names the two nearest squares; 32 GPUs (8 Perlmutter nodes) is not a legal job, 36 (9 nodes) is.
- **One GPU model and memory class.** Mixed models refuse (`GATE heterogeneous_gpu_targets`). On Perlmutter a multi-node job requests `-C 'gpu&hbm40g'` or `-C 'gpu&hbm80g'`.
- **Shared run directory.** A multi-node run writes into a directory every node mounts (`$SCRATCH`, CFS). A node-local one (`/tmp`, local NVMe) refuses at the first write (`GATE io_global_commit`, stage `SlabIO.shared_directory`).

Every planner prices its stage against one budget per device, set from the card's memory before the deck is read:

| Card | Budget, `linalg = local` | Budget, `linalg = distributed` |
|---|---|---|
| A100-40GB | 31.6 GB | 28.7 GB |
| A100-80GB | 66.4 GB | 63.4 GB |

[`memory_per_device_gb`](../drivers/gw/keywords.md#memory_per_device_gb) can only lower it; a larger value warns and keeps the rule. No stage refuses on a price. A stage priced over the budget prints `memory over budget at <stage>: needs … GB/rank, budget … GB/rank, over by … GB; continuing (an OOM is possible)`, listed under `WARNINGS` in the report, runs at its smallest tile, and fails at allocation if the memory is really missing. Large objects are tiled over every rank, so the fix is more ranks or larger cards. A compiled program larger than the card refuses before it runs (`GATE xla_rematerialization`), with the same fix.

After a GW run, the `MAJOR-STAGE DEVICE AND HOST MEMORY` block of `gwjax.out` lists each stage's device peak (max and min over ranks), its planner price and γ = peak / price, and the host high-water. A stage with γ > 1 was under-priced; the stage that set the run's peak is the one to size against.

## Startup refusals {: #refusals }

!!! refuses
    - `Could not locate liblorrax_ffi_host.so` or `… liblorrax_ffi.so`: no library for this platform. Build the leg the message names, or load a module.
    - `HANDLER ABI MISMATCH`: the library and the Python tree disagree on the handler ABI. Rebuild both legs from the same tree.
    - `partial sealed-bundle override refused`: only one leg of a sealed pair is pinned. Unset both pins, or pin both legs of one bundle.
    - `mixed native providers`: the two legs come from different bundles. Select both from one bundle.
    - `GATE native-closure: got …; want …; fix: …`: the process maps a library other than the recorded one. Remove what maps it (an `LD_PRELOAD`, a package built against another MPI), or rebuild the leg.
    - `JAX09_ENV_REFUSED`, or the `runtime.jax_support` refusal before the first `jit`: JAX or jaxlib outside the 0.9 series. Install the 0.9 series.
    - `GATE mathdx-headers`: a CUDA run without `nvidia-mathdx`. Run `uv sync --extra cuda13` (or `cuda12`).

