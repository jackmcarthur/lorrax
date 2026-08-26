# Perlmutter bare-host CUDA 13 lane

This is an experimental alternative to LORRAX's CUDA 12 Shifter lane. It
keeps the large CUDA toolchain in NERSC's familiar module system while using
a private Python environment for JAX and the native libraries NERSC does not
yet provide at the needed versions.

The division is deliberate:

| Component | Provider |
|---|---|
| CUDA compiler/runtime | `cudatoolkit/13.2` |
| cuBLAS, cuSOLVER, cuFFT | NVHPC 26.5 `math_libs/13.2` |
| NCCL | NVHPC 26.5 `comm_libs/13.2` |
| JAX/JAXlib CUDA plugin | `jax[cuda13-local]==0.9.1` |
| cuDNN | `nvidia-cudnn-cu13==9.12.0.46`, installed without dependencies |
| cuSOLVERMp/cuBLASMp | newest pinned NVIDIA wheels, installed without dependencies |

Installing NVIDIA packages with `--no-deps` is important: otherwise pip also
installs a second CUDA runtime and silently defeats the module-based design.
The NERSC cuDNN modules currently target CUDA 12, hence the one private cuDNN
DSO wheel.

From a fresh scratch clone on Perlmutter:

```bash
bash config/perlmutter/cuda13_module/setup_env.sh
bash config/perlmutter/cuda13_module/build_ffi.sh
bash config/perlmutter/cuda13_module/install_module.sh
module load lorrax_C13
LX_BASE_MODULE=lorrax_C13 lx doctor
LX_BASE_MODULE=lorrax_C13 lx run -G 4 -n 4 \
  python -u tests/bench/cusolvermp_eigh_test.py -n128 --grid 2 2
# Combined landing/acceptance leg (all native backends plus a real consumer):
LX_BASE_MODULE=lorrax_C13 lx run -G 4 -n 4 \
  python -u config/perlmutter/cuda13_module/verify_runtime.py
```

The default environment, native stage, and build tree live inside the clone
(`.venv`, `.native`, and `src/ffi/cpp/build_cuda13_module`). Override
`LORRAX_CUDA13_ENV`, `LORRAX_CUDA13_NATIVE`, or `LORRAX_CUDA13_BUILD` before
running all three scripts to put them elsewhere.

This first lane builds `cusolvermp`, `cublasmp`, and `cufft`. It intentionally
does not claim device SLATE or parallel HDF5. It also sets
`LORRAX_MPICH_GPU_SUPPORT=0`: Perlmutter's current Cray MPI GPU transport is
linked to CUDA 12, while this cuSOLVERMp configuration communicates with NCCL.
