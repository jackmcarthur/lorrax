# Installation

`uv sync --extra cuda13` Python environment, the native FFI pair, and the machine routes for Perlmutter, Frontera and other sites.
{: .page-summary }

A LORRAX installation has three parts: a Python 3.12 environment with the JAX 0.12 series, a pair of native libraries built from `src/ffi/cpp/`, and a launcher that gives each process one device. All three are required at every process count, one process included.

## Requirements {: #requirements }

### Base requirements {: #req-base }

Always required, on CPU and GPU alike.

| Component | Requirement | What it is for |
|---|---|---|
| Python | ≥ 3.12 | The runtime; uv installs its own 3.12 |
| JAX, jaxlib | `>= 0.12.0, < 0.13.0` | Every array operation. Any other minor series refuses at startup, and there is no override |
| Python packages | `h5py >= 3.14`, `numpy >= 2.3.1`, `scipy >= 1.16`, `matplotlib`, `xmlschema`, `xsdata` | File I/O, the QE schema reader and plots; pinned in `pyproject.toml` and `uv.lock` |
| MPI | the site MPI | Communication between processes; both FFI legs link the same `libmpi` |
| HDF5 | parallel HDF5 ≥ 1.12, built with `HDF5_IS_PARALLEL` | Parallel reads and writes of the large files; both legs link the HDF5 the run maps |
| Host FFI leg | `liblorrax_ffi_host.so`, built from `src/ffi/cpp/` ([below](#ffi)) | Loaded by every run, a GPU run included |
| `jax_xc` | the patched 0.0.11 of `config/xc/` | Only for the [`nscf`](../drivers/nscf/index.md) driver and the dense-Hamiltonian route |

### GPU requirements {: #req-gpu }

For runs on NVIDIA GPUs. A GPU run loads both FFI legs.

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

The Cholesky factorization is LORRAX's own, on CPU and GPU alike.

The service packages under `services/` (`distrib_la`, `lxkit`, `minimax`, `symmetry_maps`, `vcoul`, `wfn_loader`, `zeta_loader`) are uv workspace members and install with the project.

## The native FFI pair {: #ffi }

| Leg | File | Contains |
|---|---|---|
| host | `liblorrax_ffi_host.so` | ScaLAPACK, a CBLAS GEMM, the FFTW3-ABI FFT, parallel HDF5. CUDA-free. |
| CUDA | `liblorrax_ffi.so` | cuSOLVERMp, cuBLASMp, cuBLAS, cuSOLVER, cuFFT, NCCL, the NVRTC-built mathdx k-convolution kernels, parallel HDF5 |

A CPU run loads the host leg; a GPU run loads both into one process. There is no pure-JAX fallback for the k-convolution, the distributed linear algebra or the slab I/O, so a missing leg refuses at startup. Each build ends in `scripts/verify_ffi_build.sh` and prints `VERIFY PASSED`; each leg records its dependency closure in `<leg>.closure.json`, and every startup refuses a process that maps any other copy of a recorded library. A private build loads with a `LEGACY-UNSEALED` line and its hash; only a sealed pair is a production provider.

## Routes by machine {: #routes }

| Machine | Route | Entry |
|---|---|---|
| NERSC Perlmutter (A100, CUDA 13.2) | a published module, or a clone that builds its own pair | `config/perlmutter/` |
| TACC Frontera (Cascade Lake CPU) | a clone with the host leg, run inside apptainer | `config/frontera/` |
| one Ubuntu machine with an NVIDIA driver (workstation, WSL2, a rented GPU box) | everything from pip wheels plus the distro OpenMPI | `config/local/` |
| any other cluster | a clone with both legs built from the Perlmutter scripts, re-pinned to the site ([porting](#porting)) | `src/ffi/cpp/` |

The Perlmutter route is the reference build. The sections below say what it takes from the site, what it installs itself and which settings are not obvious, so that the same choices can be made on another cluster.

### Perlmutter: what the site provides {: #perlmutter-site }

LORRAX links against these Perlmutter modules and system files. Everything in one process must use the same MPI, so the versions are pinned as a set in `config/perlmutter/ffi_mpi.sh`.

| Component | Pin | Role, and why this pin |
|---|---|---|
| Compiler, host leg | `PrgEnv-gnu`, the Cray `CC` wrapper | builds `liblorrax_ffi_host.so` |
| Compiler, CUDA leg | `gcc-native/14`, plain `gcc`/`g++` (not the Cray wrapper) | host compiler for nvcc; the wrapper would inject Cray libraries the CUDA leg must not link |
| CUDA toolkit | `cudatoolkit/13.2` | nvcc, cuFFT, cuSOLVER and NVRTC at **build** time only; at run time the CUDA libraries come from the pip wheels |
| MPI | `cray-mpich/9.0.1` (`libmpi_gnu_123.so.12`) | the one `libmpi` both legs link |
| ScaLAPACK, BLAS | `cray-libsci/25.09.0`, threaded (`_mp`) flavour | the CPU distributed eigensolver and LU, and the host GEMM. The site default (26.03.0) is built against MPICH 9.1.0, which would put a second MPI in the process |
| Parallel HDF5 | `cray-hdf5-parallel/1.14.3.7` | the parallel reads and writes of both legs |
| FFTW | `cray-fftw` | found by `dlopen` at run time, never linked |
| NCCL network plugin | `/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib/libnccl-net.so` | NCCL over the Slingshot network (libfabric) for multi-node runs |

Not used on Perlmutter: the site Python or conda (uv installs its own CPython 3.12), CUDA-aware MPI and its GTL library (device collectives go through NCCL), the site cuSOLVERMp module, mpi4py and mpi4jax, and ELPA (a ROCm-leg backend).

### Perlmutter: what LORRAX installs {: #perlmutter-installs }

| Component | Version | How |
|---|---|---|
| JAX, jaxlib, the CUDA 13 plugin and PJRT | 0.12.0.1 | `uv sync --extra cuda13`, from `uv.lock` |
| CUDA runtime libraries | `nvidia-cublas` 13.6, `nvidia-cufft` 12.3, `nvidia-cusolver` 12.2, `nvidia-cuda-runtime` 13.3, `nvidia-nccl-cu13` 2.30.7 | the same `uv sync`; these, not the toolkit's copies, are what runs |
| cuFFTDx headers | `nvidia-mathdx==25.6.0` | the same `uv sync`; the k-convolution kernels are compiled from them at run time |
| cuSOLVERMp, cuBLASMp | `nvidia-cusolvermp-cu13==0.9.1.9318.post1`, `nvidia-cublasmp-cu13==0.10.0.3695` | `build_ffi_cuda.sh` installs the wheels with `uv pip install --no-deps --target <clone>/.build/cusolvermp-<ver>/` and links against them |
| h5py | 3.16, the PyPI wheel with its own serial HDF5 | small serial files; every parallel read and write goes through the FFI's parallel-HDF5 handlers, so h5py is **not** built against the site HDF5 |
| native FFI pair | `src/ffi/cpp/build_host/liblorrax_ffi_host.so`, `src/ffi/cpp/build/liblorrax_ffi.so` | `build_ffi_host.sh`, `build_ffi_cuda.sh`; each writes a `.closure.json` and a provenance stamp beside the library |
| MPIwrapper | upstream eschnett/MPIwrapper, ABI 2.10.0 | `build_mpiwrapper.sh`; only for multi-process **CPU** runs, which use JAX's MPI collectives through MPItrampoline |

### Perlmutter: a clone {: #perlmutter }

1. Keep uv's cache and interpreters on scratch, then clone and install the Python environment:

    ```bash
    export UV_CACHE_DIR=$SCRATCH/.cache/uv UV_PYTHON_INSTALL_DIR=$SCRATCH/.local/uv-python
    git clone --filter=blob:none https://github.com/jackmcarthur/lorrax.git $SCRATCH/lorrax
    cd $SCRATCH/lorrax
    uv sync --extra cuda13
    ```

2. Take one allocation, for example `salloc -N 1 -C gpu -G 4 -q interactive -t 1:00:00 -A <account>`, and build both legs on its node with no GPU requested. Each build ends with `VERIFY PASSED`.

    ```bash
    srun --jobid=$JOBID -N 1 -n 1 -c 64 --gres=none bash -c '
      bash config/perlmutter/build_ffi_host.sh &&
      bash config/perlmutter/build_ffi_cuda.sh'
    ```

3. Run with `config/perlmutter/gpu_env.sh` sourced in the shell that calls `srun`, one rank per GPU through `select_gpu.sh`:

    ```bash
    source config/perlmutter/gpu_env.sh
    srun -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh .venv/bin/python -u -m gw.gw_jax -i gw.in
    ```

Rebuild both legs after a pull that changes `src/ffi/cpp/`, after moving `.build`, or after a `uv sync` that changes the CUDA wheels; a stale leg refuses.

### Perlmutter: settings that are not obvious {: #perlmutter-settings }

The build scripts and `gpu_env.sh` set the following. Each one fixes a failure that a straightforward build or launch runs into, and each has a counterpart on any Cray or Slurm system.

**Building the host leg** (`build_ffi_host.sh`).

| Setting | Why |
|---|---|
| unload `craype-accel-nvidia80` and `cudatoolkit` first | otherwise the compiler wrapper links `libmpi_gtl_cuda`, and through it `libcuda`, into the CUDA-free leg |
| load `cray-libsci`, record its library path, unload it | otherwise the wrapper also adds the sequential `-lsci_gnu`, and two LibSci copies end up in one library; the recorded path is passed as `LORRAX_CBLAS_DIR` |
| unload `cray-fftw` | otherwise CMake's OpenMP probe adopts `fftw3_omp` as the OpenMP library |
| `-DLORRAX_SCALAPACK_LIBRARIES="-L<libsci> -Wl,--no-as-needed -lsci_gnu_mpi_mp -lsci_gnu_mp -lgomp …"` | the ScaLAPACK probe looks for MKL; on LibSci the link line is given explicitly |
| `-DLORRAX_MPI_INCLUDE_DIR`, `-DLORRAX_MPICH_LIB_DIR`, `-DLORRAX_MPI_LIBRARY`, `-DHDF5_ROOT` | the CMake searches use `NO_DEFAULT_PATH`, so the wrapper's implicit include and library paths are invisible to them |

**Building the CUDA leg** (`build_ffi_cuda.sh`).

| Setting | Why |
|---|---|
| `gcc-native/14` with plain `gcc`/`g++` | see the table above: the Cray wrapper would add site libraries |
| cuSOLVERMp and cuBLASMp from wheels, `-DCUSOLVERMP_INCLUDE_DIR`, `-DCUSOLVERMP_LIB_DIR` | `cudatoolkit/13.2` ships cuSOLVERMp 0.8 under the same SONAME, and cuSOLVERMp 0.6.x returns wrong LU results on a 2-D grid |
| `-DLORRAX_FFI_HAVE_CAL=OFF` | cuSOLVERMp ≥ 0.7 talks to NCCL directly, without the CAL layer |
| `-DNCCL_INCLUDE`, `-DNCCL_LIBRARY` pointing into the venv's `nvidia-nccl-cu13` | the leg must link the NCCL that JAX loads |
| `-DHDF5_PREFER_PARALLEL=ON`, `-DNVHPC_ROOT` | pick the parallel HDF5 and the toolkit root explicitly |
| CUDA architectures `80;86-real;89-real;90-real;100-real;120` (the default) | sm_80 is the floor; `LORRAX_FFI_HAVE_CUBLASMP` and `LORRAX_FFI_HAVE_CUFFT` are always on, and passing `OFF` is a configure error |

**Which libraries a run loads.** Each leg carries `$ORIGIN` as its only `RUNPATH` and exports a fixed symbol list. Because `LD_LIBRARY_PATH` outranks a `RUNPATH`, a module that puts another copy of a library first would win; `cudatoolkit/13.2`, for example, puts its own cuSOLVERMp 0.8 there, which fails with `syevd status=7`. The build therefore records the leg's complete dependency closure in `<leg>.closure.json`, and every startup refuses a process that maps any other copy (`GATE native-closure`). The pinned MPICH, HDF5 and LibSci directories (`LORRAX_PM_RUNTIME_LIB_DIRS`) must precede `/opt/cray/pe/lib64`, whose links point at the site defaults.

**Launching** (`gpu_env.sh` and the runtime).

| Setting | Why |
|---|---|
| `MPICH_GPU_SUPPORT_ENABLED=0` | `craype-accel-nvidia80` sets it to 1, and `MPI_Init` then aborts with `GTL library is not linked`; LORRAX does not use CUDA-aware MPI |
| `libcudart.so` linked to the wheel's runtime and placed first on `LD_LIBRARY_PATH` | libfabric opens the unversioned name; without the link a second CUDA runtime loads and a multi-node step refuses |
| `SLURM_NETWORK=no_vni` (the same as `srun --network=no_vni`) | with a VNI, MPI-IO and NCCL's network initialization fail with `ENOSPC` |
| `NCCL_NET="AWS Libfabric"`, `NCCL_NET_GDR_LEVEL=PHB`, `NCCL_SOCKET_IFNAME=hsn`, `FI_CXI_DISABLE_HOST_REGISTER=1`, `FI_CXI_RDZV_THRESHOLD=0`, `FI_MR_CACHE_MONITOR=kdreg2` | set by the runtime from `config/perlmutter/site.toml` unless already set. The rendezvous threshold prevents a cross-node deadlock, and `kdreg2` replaces the default memory-registration monitor, which deadlocked inside a cuSOLVERMp split |
| `select_gpu.sh`, never `--gpus-per-task=1` | `select_gpu.sh` gives each rank one entry of Slurm's GPU list in `CUDA_VISIBLE_DEVICES`; `--gpus-per-task=1` confines each rank so that the first collective fails with `invalid device ordinal` |
| `CUDA_CACHE_PATH=$SCRATCH/.nv/ComputeCache`; JAX compile cache in `$SCRATCH/.cache/lorrax/jax_compile` | keep the kernel caches off the small home quota |

The runtime also chooses the JAX coordinator (the step's first host, a port derived from the job and step ids; `JAX_COORDINATOR_ADDRESS` overrides it) and merges two `--xla_gpu_*` compilation flags into `XLA_FLAGS`; a value you set wins. Do not set XLA's allocator or memory-fraction variables.

A multi-process CPU run additionally sources `config/perlmutter/cpu_mpi_env.sh`, which sets `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`, `MPITRAMPOLINE_LIB` to the MPIwrapper library, `LD_PRELOAD=/opt/cray/pe/lib64/libpmi.so.0`, `JAX_NUM_CPU_DEVICES=1` and `MPICH_ASYNC_PROGRESS=1`.

### Perlmutter: the module {: #perlmutter-module }

```bash
module use <prefix>/modulefiles
module load lorrax
srun -N 1 -n 4 --gpus-per-node=4 $LORRAX_ROOT/src/ffi/cpp/select_gpu.sh python -u -m gw.gw_jax -i gw.in
```

The module carries one venv, one source snapshot and one sealed FFI bundle, and sets `LORRAX_ROOT`, `LORRAX_FFI_SO`, `LORRAX_FFI_HOST_SO`, `MPICH_GPU_SUPPORT_ENABLED=0`, `SLURM_NETWORK=no_vni`, `JAX_PLATFORMS=cuda,cpu` and `JAX_ENABLE_X64=1`, so `gpu_env.sh` is not needed. Do not add a checkout to `PYTHONPATH` or pin one leg by hand: a mixed source tree or a partial bundle refuses. Maintainers build a module with the scripts in `config/perlmutter/module/`: `setup_env.sh` (the venv, from the same lock), `build.sh` (a fresh clone, both legs, the sealed bundle, the snapshot and the modulefile), `accept.sh` (acceptance runs on 1 and 4 GPUs) and `publish.sh`.

### Porting to another cluster {: #porting }

Copy `config/perlmutter/` and replace the site-specific pieces:

| Perlmutter piece | Replace with |
|---|---|
| `ffi_mpi.sh`: MPICH 9.0.1, LibSci 25.09.0, HDF5 1.14.3.7 | the site's MPI, ScaLAPACK and parallel HDF5 (≥ 1.12), chosen so that all three are built against the **same** MPI. `config/frontier/ffi_mpi.sh` is a second example |
| LibSci | MKL (`mkl_scalapack_lp64` with the `mkl_blacs_<mpi>_lp64` that matches the MPI), netlib ScaLAPACK with OpenBLAS or BLIS, or AOCL; give the link line in `LORRAX_SCALAPACK_LIBRARIES` |
| `cray-hdf5-parallel` | a parallel HDF5 ≥ 1.12; `config/local/` builds 1.14.6 from source when the system has none |
| `cudatoolkit/13.2` | any CUDA 13 toolkit, or the `nvidia/cu13` directory of the `jax[cuda13]` wheels, which includes nvcc (`config/local/` does this) |
| the compiler-wrapper module dance | on a non-Cray system, plain compilers and explicit `-D` paths; on a Cray system, the same unloads |
| `site.toml` (NCCL plugin and libfabric settings) | the site's NCCL network plugin and its recommended variables, or nothing on a single node |
| `gpu_env.sh` (`no_vni`, GTL off, the `libcudart` link) | the site's equivalents; on a single machine, `config/local/launch.sh` with `mpirun` |
| `select_gpu.sh` | keep it; it reads Slurm's variables, or `LORRAX_GPU_DEVICE` elsewhere |
| cuSOLVERMp 0.9.1 and cuBLASMp 0.10.0 wheels | the same wheels, from pypi.org |

Build each leg with a site script that ends the way the Perlmutter and `config/local/` scripts do: CMake with explicit `-D` paths, then `src/ffi/cpp/stage/write_checkout_closure.py` on the library, which writes the `.closure.json` every startup requires (a leg without one refuses with `no closure record`), then `scripts/verify_ffi_build.sh`. On a cluster without Cray compiler wrappers, `config/local/build_ffi.sh` and `build_ffi_host.sh` are the closer template: plain `gcc`/`g++`, nvcc from the wheels, and the system MPI and ScaLAPACK. Check the result with the test suite ([below](#check)).

### Frontera {: #frontera }

Frontera runs the host leg only: its RTX 5000 GPUs (sm_75) are below the CUDA leg's floor. JAX runs inside the apptainer image `py312.sif`, because the host glibc is 2.17. Multi-process CPU collectives run on MPI through MPItrampoline and a patched MPIwrapper.

| Artifact | Script | Runs on |
|---|---|---|
| host leg | `LORRAX_HOST_SCALAPACK=1 config/frontera/build_ffi_host.sh` | login node |
| MPIwrapper | `config/frontera/build_mpiwrapper.sh --fresh` | login node |
| mpi4py + parallel h5py overlay | `config/frontera/build_mpi_overlay.sh fetch`, then `build` | login node, then compute node in the image |
| staged PMI2 library | `config/frontera/stage_host_pmi.sh` | login node |
| node-local runtime bundle | `config/frontera/build_cpu_runtime_bundle.sh` | inside the image |
| launch | `config/frontera/templates/gw_dev.sbatch` | batch job |

The image and the venv `$WORK/lorrax_env/.venv` are not rebuilt by any script in the repository. The launch template refuses when `LORRAX_ROOT` or `LORRAX_RUN_DIR` is unset; edit its `#SBATCH` block (account, nodes, time) before submitting.

### Other sites {: #other }

```bash
uv sync --extra cuda13          # or --extra cuda12; plain `uv sync` for a CPU-only install
```

then build the two legs with a site script modelled on [Porting](#porting). The CMake options a CUDA leg needs are `LORRAX_FFI_PLATFORM=cuda`, `CMAKE_CUDA_COMPILER`, `CUDA_TOOLKIT_ROOT`, `CUSOLVERMP_INCLUDE_DIR` and `CUSOLVERMP_LIB_DIR`, `LORRAX_FFI_HAVE_CAL=OFF`, `NCCL_INCLUDE` and `NCCL_LIBRARY`, `HDF5_ROOT` (a parallel HDF5), and `LORRAX_MPI_INCLUDE_DIR`, `LORRAX_MPICH_LIB_DIR` and `LORRAX_MPI_LIBRARY`; the host leg takes `LORRAX_FFI_PLATFORM=host`, the MPI and HDF5 options and `LORRAX_SCALAPACK_LIBRARIES`. A library outside the in-tree build directories is selected with [`LORRAX_FFI_SO` and `LORRAX_FFI_HOST_SO`](environment.md#ffi). For one machine with only an NVIDIA driver ≥ 580, `config/local/` builds both legs from wheels, and its README walks through it.

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
    - `JAX_ENV_REFUSED`, or the `runtime.jax_support` refusal before the first `jit`: JAX or jaxlib outside the 0.12 series. Install the 0.12 series.
    - `GATE mathdx-headers`: a CUDA run without `nvidia-mathdx`. Run `uv sync --extra cuda13` (or `cuda12`).

