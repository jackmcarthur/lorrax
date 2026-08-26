# Generic-cloud CUDA 13 lane (rented H100/A100 boxes)

Runs LORRAX's device leg on a rented single-node GPU box — Vast.ai, RunPod,
DataCrunch, Lambda, or any Ubuntu machine with an NVIDIA datacenter driver.
Cloud twin of [`config/perlmutter/cuda13_module/`](../perlmutter/cuda13_module/):
same pins (JAX 0.9.1, cuSOLVERMp 0.9.1, CUDA 13), but where Perlmutter takes
the toolchain from Lmod modules and NVHPC, here **everything is pip wheels**
— `jax[cuda13]` pulls the entire CUDA 13 userspace *including nvcc*, and the
CUDA-13-generation wheels share one prefix (`site-packages/nvidia/cu13`)
that serves directly as `CUDA_TOOLKIT_ROOT`. The host contributes exactly
two things: the kernel driver and `/dev/nvidia*`.

Unlike the Perlmutter lane this one builds the **phdf5 handlers too**
(OpenMPI + own-built parallel HDF5 1.14.6; Ubuntu's packaged HDF5 is 1.10,
below the 1.12 floor in `src/ffi/PORTING.md`). SLATE stays out, as on every
CUDA-13 lane.

## Verified (2026-08-26)

On a WSL2 Ubuntu 24.04 box, RTX 5070 Laptop (sm_120), driver 595.71 —
deliberately *not* the deployment target, which is the point: nothing
site-specific survives in the closure.

* `setup_env.sh` + `build_ffi.sh` produce `liblorrax_ffi.so` with backends
  `cusolvermp,cublasmp,cufft,phdf5`, `cal=0`.
* `scripts/verify_ffi_build.sh --leg cuda` under `LORRAX_FFI_VERIFY_ENV=runtime`:
  **every gate ran and passed** (one-MPI, closure resolution, one-HDF5, ABI).
* `launch.sh -n 1 python tests/bench/cusolvermp_eigh_test.py -n 128 --grid 1 1`:
  ALL PASSED, `max |eval − ref| = 5.7e-13` — a live cuSOLVERMp eigh through
  the artifact. (Falsification: the same run refuses loudly on a broken
  closure; that failure mode was observed, then fixed, during bring-up.)
* NOT yet verified: multi-GPU (this box has one), production GW decks, QE.
  First action on a real 4-GPU box:
  `launch.sh -n 4 python -u config/perlmutter/cuda13_module/verify_runtime.py`.

## Renting the box

Requirements: Ubuntu 22.04/24.04, **driver ≥ 580** (CUDA 13 floor — filter
for it, or on a root VM install it yourself), 4×H100 **SXM** (check
`nvidia-smi topo -m` shows NVLink, not PCIe), ≥100 GB disk, generous
`/dev/shm` (NCCL shared-memory transport).

Roughly (2026-08): Vast.ai ~$1.5–2.2/GPU/h (marketplace, filter by driver
version), RunPod ~$2–3.4/GPU/h (container, pick a CUDA 13 template),
DataCrunch ~$2–3/GPU/h (real VM — root, own driver, `docker` if wanted).
Setup below costs ~10 min of box time at datacenter bandwidth; run it on a
cheap 1-GPU instance first if iterating.

```bash
apt-get update && apt-get install -y build-essential gfortran git curl \
    libopenmpi-dev pkg-config python3.12-venv
git clone -b cloud/h100-cuda13-bootstrap https://github.com/jackmcarthur/lorrax
cd lorrax
bash config/cloud/setup_env.sh     # venv + CUDA13 wheels + parallel HDF5 (~10 min)
bash config/cloud/build_ffi.sh     # liblorrax_ffi.so + full acceptance gates
bash config/cloud/launch.sh -n 4 python -u config/perlmutter/cuda13_module/verify_runtime.py
bash config/cloud/launch.sh -n 4 python -u tests/bench/cusolvermp_eigh_test.py -n 512 --grid 2 2
```

Then real work, e.g.:

```bash
bash config/cloud/launch.sh -n 4 python -u -m gw.gw_jax -i cohsex.in
```

## Quantum ESPRESSO

Not part of this lane. For WFN inputs either copy existing `WFN.h5` from
Perlmutter/CFS (cheapest — QE output is deck-dependent, not machine-
dependent), or `apt-get install quantum-espresso` and check `pw2bgw.x`
exists; if the distro package lacks it, build QE 7.4.1 from source with
`make pw pp` (~15 min, CPU-only is fine at these deck sizes).

## The GW driver needs BOTH FFI libraries

`gw.gw_jax` refuses to start without a host-platform library beside the CUDA
one. `build_ffi_host.sh` builds it here: phdf5 + ScaLAPACK (distro netlib
`libscalapack-openmpi` — exports all eleven required symbols including
C-BLACS, measured) + CBLAS (distro OpenBLAS). SLATE off, mklfft
self-disables (DFTI is Intel-only; the XLA FFT lowering stands). `launch.sh`
exports both `LORRAX_FFI_SO` and `LORRAX_FFI_HOST_SO`.

Verified end-to-end 2026-08-26: the tracked GN-PPM regression deck
(`tests/regression/gnppm_debug/gnppm_test.in`, `memory_per_device_gb`
lowered 28→6 for an 8 GB laptop GPU) ran to completion on the RTX 5070 —
170 s wall, sigma table + `sigma_mnk.h5` written. Against the frozen
2026-08-09 reference: sigX, VH, Eo **byte-identical**; sigC differs by up
to 5.5e-2 eV, consistent with the post-freeze head-correction landings on
main (`f83d5ea7`, `d2d6d521`, `a103f1b0`) — the reference predates them.
The dirty `mpirun` exit after "process finalized explicitly" is the
documented deliberate `os._exit`; judge by artifacts.

## Site facts that differ from Perlmutter, and why

* **cuBLASMp pin.** Perlmutter's lane names 0.10.0.3695; that wheel is on
  neither pypi.org nor pypi.nvidia.com (checked 2026-08-26; newest published
  cu13 wheel is 0.9.1.3056, which this lane pins). If 0.10 reappears, bump
  `LORRAX_CLOUD_CUBLASMP_VERSION` — `compat.h` carries the API delta.
* **NVTX** is a separate unsuffixed wheel (`nvidia-nvtx`); the jax closure
  does not pull it. `setup_env.sh` installs it explicitly.
* **NCCL/cuDNN/NVSHMEM** version independently of the toolkit and keep their
  own wheel prefixes beside `nvidia/cu13`; `stack.sh` puts each on the
  runtime `LD_LIBRARY_PATH` and `build_ffi.sh` names NCCL explicitly.
* **Two former shims are now in source** (2026-08-26): the FFI CMakeLists
  searches `<root>/lib` alongside `lib64` (the pip wheel layout), and
  defines `OMPI_SKIP_MPICXX` on both legs (OpenMPI's `mpi.h` otherwise
  emits MPI-C++-binding refs nothing links; measured as a dlopen
  `undefined symbol: _ZN3MPI8Datatype4FreeEv`). Likewise
  `runtime._gpu_is_present` now recognizes WSL2's mapped
  `/usr/lib/wsl/lib/libcuda.so.1`, so GW runs no longer silently demote
  to CPU there. Both legs rebuilt shim-free and re-verified (all gates,
  eigh bench, `Backend: GPU` with no override).
* **Kernel architectures.** The build keeps the default
  `CMAKE_CUDA_ARCHITECTURES=80` (embeds PTX, so sm_90/H100 JITs it on first
  load; the AOT conv cubins fall back to their NVRTC path by design). For
  native sm_90 kernels: `-DLORRAX_FFI_PREBUILD_SM80_KERNELS=OFF
  -DCMAKE_CUDA_ARCHITECTURES=90` — unmeasured, PTX JIT has been fine.
* **Driver < 580 fallback (untested).** Datacenter GPUs can run CUDA 13
  userspace on older r535+ drivers via the `cuda-compat-13-*` forward-compat
  package. Prefer simply renting a ≥580 host.
