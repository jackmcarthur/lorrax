# Installation

Python environment, the native FFI pair, and the machine routes for Perlmutter, Frontier, Frontera and other sites.
{: .page-summary }

A LORRAX installation has three parts: a Python 3.12 environment with the JAX 0.9 series, a pair of native libraries built from `src/ffi/cpp/`, and a launcher that gives each process one device. All three are required at every process count, one process included.

## Requirements {: #requirements }

### Base requirements {: #req-base }

Always required, on CPU and GPU alike.

| Component | Requirement | What it is for |
|---|---|---|
| Python | ≥ 3.12 | The runtime; uv installs its own 3.12 |
| JAX, jaxlib | `>= 0.9.0, < 0.10.0` | Every array operation. Any other minor series refuses at startup, and there is no override |
| Python packages | `h5py >= 3.14`, `numpy >= 2.3.1`, `scipy >= 1.16`, `matplotlib`, `xmlschema`, `xsdata` | File I/O, the QE schema reader and plots; pinned in `pyproject.toml` and `uv.lock` |
| MPI | the site MPI | Communication between processes; both FFI legs link the same `libmpi` |
| HDF5 | parallel HDF5 ≥ 1.12, built with `HDF5_IS_PARALLEL` | Parallel reads and writes of the large files; both legs link the HDF5 the run maps |
| Host FFI leg | `liblorrax_ffi_host.so`, built from `src/ffi/cpp/` ([below](#ffi)) | Loaded by every run, a GPU run included |
| `jax_xc` | the patched 0.0.11 of `config/xc/` | Only for the [`nscf`](../drivers/nscf/index.md) driver and the dense-Hamiltonian route |

### GPU requirements {: #req-gpu }

A GPU run loads the matching device leg and the host leg.

**NVIDIA GPUs**

| Component | Requirement | What it is for |
|---|---|---|
| GPU | NVIDIA sm_80 or newer | The floor of the CUDA FFI leg |
| CUDA extra | `cuda13` (driver ≥ 580) or `cuda12` | Installs `jax[cudaNN]` and `nvidia-mathdx==25.6.0`, whose kernels do the k-convolutions |
| CUDA FFI leg | `liblorrax_ffi.so` with nvcc, cuBLAS, cuFFT, NVRTC and NCCL | The complete NVIDIA stack; no partial CUDA leg is built |

The GPU linear algebra comes from these libraries:

| Library | Status | What it gives, and when LORRAX uses it |
|---|---|---|
| cuSOLVERMp ≥ 0.7 and cuBLASMp | required, part of the CUDA leg | The default distributed eigensolver and LU on a multi-GPU mesh. cuSOLVERMp 0.6.x returns wrong LU results on a 2-D process grid |
| cuSOLVER, through jaxlib | installed with the CUDA extra | The single-device eigensolver and LU, used when one matrix fits on one GPU |
| SLATE (CUDA build) | optional | Selectable alternative for the distributed Cholesky and eigensolver; never the default |

**AMD GPUs on Frontier**

Use the validated JAX 0.9.1 ROCm environment and the native ROCm/host pair,
not a CUDA extra. The ROCm leg uses the same HIP, rocBLAS, rocSOLVER, RCCL
and rocFFT libraries as the JAX plugin; SLATE-HIP and ELPA supply the
distributed operations. The [Frontier route](#frontier) selects the
environment and explains the separate native builds.

### CPU requirements {: #req-cpu }

For runs on CPUs. A CPU run loads the host leg only.

| Component | Requirement | What it is for |
|---|---|---|
| FFT | any FFTW3-ABI library: FFTW, `cray-fftw`, MKL's FFTW3 interface or AOCL | The lattice FFTs; bound at run time and never linked into the host leg |

The distributed eigensolver and LU on CPU are ScaLAPACK, together with BLACS and a CBLAS GEMM. One of these stacks provides all three:

| Stack | Typical site |
|---|---|
| Cray LibSci | Cray systems |
| Intel MKL (`mkl_scalapack_lp64` with the `mkl_blacs_<mpi>_lp64` that matches the MPI) | Intel systems; a BLACS built for another MPI hangs in the first collective |
| netlib ScaLAPACK with OpenBLAS or BLIS | generic Linux clusters |
| AMD AOCL | AMD CPU systems |

SLATE built with `gpu_backend=none` is optional on CPU. It provides a Cholesky factorization; its CPU eigensolver crashes, so LORRAX never selects it there.

The service packages under `services/` (`distrib_la`, `lxkit`, `minimax`, `symmetry_maps`, `vcoul`, `wfn_loader`, `zeta_loader`) are uv workspace members and install with the project.

## The native FFI pair {: #ffi }

| Leg | File | Contains |
|---|---|---|
| host | `liblorrax_ffi_host.so` | ScaLAPACK, SLATE, a CBLAS GEMM, the FFTW3-ABI FFT, parallel HDF5. CUDA-free. |
| CUDA | `liblorrax_ffi.so` | cuSOLVERMp, cuBLASMp, cuBLAS, cuSOLVER, cuFFT, NCCL, the NVRTC-built mathdx k-convolution kernels, parallel HDF5 |
| ROCm | `liblorrax_ffi_rocm.so` | HIP, rocBLAS/rocSOLVER, SLATE/ELPA, ROCm FFT kernels and parallel HDF5 |

A CPU run loads the host leg; a GPU run loads the host and its matching
device leg into one process. A missing required leg refuses at startup.
Each leg records its dependency closure in `<leg>.closure.json`, and startup
checks that the process maps those libraries. The acceptance contract is
`scripts/verify_ffi_build.sh`; skipped runtime gates on a login node do not
certify a GPU build. A private checkout build prints `LEGACY-UNSEALED` and
its hash, with its closure still enforced; a published module carries a
sealed pair.

## Routes by machine {: #routes }

| Machine | Route | Entry |
|---|---|---|
| NERSC Perlmutter (A100, CUDA 13.2) | a published module, or a clone that builds its own pair | `config/perlmutter/` |
| OLCF Frontier (MI250X, ROCm) | a validated ROCm environment and a bare-metal host/ROCm pair | `config/frontier/` |
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

### Frontier (OLCF) {: #frontier }

Use the existing validated JAX/jaxlib 0.9.1 environment with `rocm-sdk`
7.14.1 wheels. From the checkout, source the setup once:

```bash
source config/frontier/gpu_env.sh
```

It selects `LORRAX_OLCF_VENV`, the active `VIRTUAL_ENV`, or the default
`$HOME/envs/lx-jax091-rocm7141`, in that order. It activates the environment,
checks for the ROCm plugin without starting a GPU client, prepares the
host/ROCm library paths and selects this checkout's source. It does not
create the Python environment or build native libraries. For a fresh Python
3.12 installation, use the [pinned ROCm/JAX recipe](https://github.com/jackmcarthur/lorrax/blob/main/docs/installation/frontier.md#fresh-python-installation):
AMD's multi-architecture wheel index supplies
`rocm[libraries,device-gfx90a]==7.14.1` and
`jax_rocm7_plugin`/`jax_rocm7_pjrt==0.9.1+rocm7.14.1`; PyPI supplies
`jax==0.9.1` and `jaxlib==0.9.1`. That recipe installs the editable workspace
packages from `pyproject.toml`, then the root package, and records the final
package manifest. No CUDA extra is installed.

The normal PyPI `h5py` wheel is supported; no MPI-enabled `h5py` build is
needed. Its private serial HDF5 libraries are separate from the one pinned
parallel HDF5 family shared by both native legs. The tested environment
uses `h5py 3.16.0` with private HDF5 2.0.0, while the native pair uses site
HDF5 1.14.3.7. Runtime owns the load order and file-handle separation.

If the shared dependency installs are already present, build only the two
LORRAX legs for this checkout:

```bash
bash config/frontier/build_ffi_host.sh
bash config/frontier/build_ffi_rocm.sh
```

On a first installation, run `build_openblas_seq.sh`, `build_slate_hip.sh`
and `build_slate_cpu.sh` from `config/frontier/` before those commands.
Their installs under `$HOME/software/slate_builds` are shared across
checkouts. The ROCm builder creates or verifies its content-keyed ELPA
prefix under `$HOME/software/elpa_builds`. Rebuilding LORRAX does not
require deleting or rebuilding these dependencies.

After building both legs, certify them inside an active allocation with
four free GCDs:

```bash
bash config/frontier/verify.sh
```

The command sources `gpu_env.sh`, verifies the recorded host/ROCm closures
and intended host FFTW, then compares native and XLA Fourier transforms at
four MPI processes. It refuses disabled acceptance checks. The host build
cache supplies the FFTW prefix; a copied host leg without that cache needs
the existing `LORRAX_FFTW3_STAGE` set to its verified FFTW prefix. No added
library paths or runtime toolkit module are needed. Skipped runtime gates
on a login node do not certify a build.

Run a calculation inside the allocation:

```bash
source config/frontier/gpu_env.sh
srun -N 1 -n 4 -c 7 --gpus-per-task=1 --gpu-bind=closest \
  python -u -m gw.gw_jax -i /path/to/run/gw.in
```

This is a 2 × 2 mesh on four GCDs, one per MPI process. A Frontier node has
eight GCDs, but eight processes are not a valid square mesh; sixteen form
a 4 × 4 mesh on two complete nodes. The helper and runtime own the ROCm
and RCCL/OFI setup; do not use the NVIDIA `select_gpu.sh` or copy Perlmutter
network settings. No container or runtime ROCm toolkit module is needed.

The records can also be inspected with `python -m lxkit.closure verify <leg.so>`.
Reusing a build from another checkout also requires matching FFI sources,
Frontier build inputs and JAX/jaxlib headers; a matching ABI number or a
closure JSON alone is insufficient. At runtime both closures must be
enforced. Full build pins, paths and reuse rules are in the
[Frontier installation source](https://github.com/jackmcarthur/lorrax/blob/main/docs/installation/frontier.md).

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

It ends `1 passed`. `python -m tests.hsuite.chain --out <dir>` runs the same workflow outside pytest and leaves every input file, report and output in `<dir>/run/`.

## Sizing a job {: #sizing }

Process-count rules, the per-GPU memory budget and how many GPUs a calculation needs are on [Running on clusters](clusters.md).

## Startup refusals {: #refusals }

!!! refuses
    - `Could not locate liblorrax_ffi_host.so` or `… liblorrax_ffi.so`: no library for this platform. Build the leg the message names, or load a module.
    - `HANDLER ABI MISMATCH`: the library and the Python tree disagree on the handler ABI. Rebuild both legs from the same tree.
    - `partial sealed-bundle override refused`: only one leg of a sealed pair is pinned. Unset both pins, or pin both legs of one bundle.
    - `mixed native providers`: the two legs come from different bundles. Select both from one bundle.
    - `GATE native-closure: got …; want …; fix: …`: the process maps a library other than the recorded one. Remove what maps it (an `LD_PRELOAD`, a package built against another MPI), or rebuild the leg.
    - `JAX09_ENV_REFUSED`, or the `runtime.jax_support` refusal before the first `jit`: JAX or jaxlib outside the 0.9 series. Install the 0.9 series.
    - `GATE mathdx-headers`: a CUDA run without `nvidia-mathdx`. Run `uv sync --extra cuda13` (or `cuda12`).

