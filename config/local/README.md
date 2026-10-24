# Local CUDA 13 build

This lane builds and runs LORRAX on one Ubuntu machine with an NVIDIA driver: a workstation, a laptop under WSL2, or a rented single-node GPU box. Everything comes from pip wheels except two things the machine provides, the kernel driver and the distro OpenMPI. `jax[cuda13]` installs the whole CUDA 13 userspace, `nvcc` included, into `site-packages/nvidia/cu13`, and that prefix serves as the CUDA toolkit root. Parallel HDF5 is built once from source, because Ubuntu's packaged HDF5 (1.10) is below the 1.12 floor of [Building the FFI libraries](../../docs/installation/ffi-build.md).

The pins match the Perlmutter lane (JAX 0.12.0.1, cuSOLVERMp 0.9.1, CUDA 13). They live in `stack.sh`, which every script sources.

## Requirements

- Ubuntu 22.04 or 24.04, or WSL2 with an Ubuntu distribution.
- An NVIDIA driver of version 580 or newer (the CUDA 13 floor).
- About 15 GB of free disk for the venv and the HDF5 build.
- The distro packages:

  ```bash
  sudo apt-get install -y build-essential gfortran git curl pkg-config python3.12-venv \
      libopenmpi-dev libscalapack-openmpi-dev libopenblas-dev
  ```

## Build

```bash
git clone --filter=blob:none https://github.com/jackmcarthur/lorrax
cd lorrax
bash config/local/setup_env.sh       # venv, CUDA 13 wheels, parallel HDF5 (about 10 min the first time)
bash config/local/build_ffi.sh       # liblorrax_ffi.so, the CUDA leg, with its acceptance gates
bash config/local/build_ffi_host.sh  # liblorrax_ffi_host.so, the host leg
```

Both legs are required at every process count. Each build writes a dependency closure record (`<leg>.so.closure.json`) beside its library, and the runtime refuses a leg without one. The build trees are `src/ffi/cpp/build_local/` and `src/ffi/cpp/build_host_local/`.

To reuse a parallel HDF5 built elsewhere, point `LORRAX_LOCAL_PHDF5` at its prefix before running `setup_env.sh`. The script then skips the HDF5 build.

## Run

`launch.sh` sets the library paths and starts N processes, one per GPU, under `mpirun`.

```bash
bash config/local/launch.sh -n 1 python -u -m gw.gw_jax -i cohsex.in
bash config/local/launch.sh -n 4 python -u -m gw.gw_jax -i cohsex.in   # on a 4-GPU box
```

`mpirun` reports that the process "exited improperly" and returns 1 even after a run that finished correctly. The drivers end with an explicit `os._exit` after their outputs are written, so judge a run by its outputs and its closing log line.

## Test

The driver suite (`tests/hsuite`) runs on one GPU as a smoke run.

```bash
bash config/local/launch.sh -n 1 python -u -m tests.hsuite.chain --out ~/hsuite_runs/run1 \
    --cache-dir ~/hsuite_runs/jax-cache
```

It prints one wall time per stage and ends with `hsuite PASS`. The landing verdict for a driver or physics change is the four-GPU run ([contributing](../../docs/contributing.md#the-test-suite)), so a one-GPU pass is a smoke run only. The static suites and the pre-push checks run with `tools/release_check.sh`.

## Verified

| Date | Machine | main | What ran |
|---|---|---|---|
| 2026-10-20 | WSL2 Ubuntu 24.04, RTX 5070 Laptop GPU (8.5 GB, sm_120), 6 cores, 7 GB RAM in WSL | 7e4999976 | both legs built with closure records, then the full `tests/hsuite` chain at P1: `hsuite PASS` in 18 min with a cold compile cache, 7 min warm |
| 2026-08-26 | the same laptop | 07d7191e1 | both legs built; `scripts/verify_ffi_build.sh --leg cuda` passed every gate; a cuSOLVERMp eigh bench passed; the GN-PPM regression deck ran end to end |

Not yet verified on this lane: more than one GPU, and production-size decks.

## Rented GPU boxes

On a rented box (Vast.ai, RunPod, DataCrunch, Lambda and similar) choose Ubuntu 22.04 or 24.04 with a driver of 580 or newer, at least 100 GB of disk and a generous `/dev/shm` for NCCL. For four GPUs, check that `nvidia-smi topo -m` shows NVLink rather than PCIe. The build steps are the same. A first check on a 4-GPU box is

```bash
bash config/local/launch.sh -n 4 python -u config/perlmutter/module/verify_runtime.py
```

## Quantum ESPRESSO

QE is not part of this lane. Copy `WFN.h5` files from another machine, since they do not depend on where they were made, or install QE and check that `pw2bgw.x` is present.

## How this lane differs from Perlmutter

- **cuBLASMp.** This lane pins 0.9.1.3056, the newest published CUDA 13 wheel. Perlmutter's 0.10.0.3695 is not on pip. To change it, set `LORRAX_LOCAL_CUBLASMP_VERSION`.
- **NVTX** is a separate wheel (`nvidia-nvtx`) that the JAX closure does not pull in, so `setup_env.sh` installs it.
- **NCCL, cuDNN and NVSHMEM** keep their own wheel prefixes beside `nvidia/cu13`. `stack.sh` puts each on the runtime library path, and `build_ffi.sh` names NCCL explicitly.
- **MPI and the distro libraries.** The distro OpenMPI, ScaLAPACK and OpenBLAS live in the system directory that `libmpi` resolves to. `stack.sh` pins that directory, so the dependency closure records them and the run uses the same files.
- **Kernel architectures.** The build keeps `CMAKE_CUDA_ARCHITECTURES=80`, which embeds PTX, so newer GPUs JIT the kernels on first load. For native sm_90 kernels, add `-DLORRAX_FFI_PREBUILD_SM80_KERNELS=OFF -DCMAKE_CUDA_ARCHITECTURES=90` (not measured).
- **WSL2.** The runtime recognizes the driver mapped at `/usr/lib/wsl/lib/libcuda.so.1`, so runs under WSL2 use the GPU.
- **Drivers older than 580.** These are untested. Datacenter GPUs can run the CUDA 13 userspace on r535 and newer drivers through NVIDIA's `cuda-compat-13` package, but a 580 or newer driver is simpler.
