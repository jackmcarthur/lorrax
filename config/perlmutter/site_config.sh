#!/usr/bin/env bash
# ============================================================================
# LORRAX site configuration — Perlmutter (NERSC)
#
# Sourced by install.sh to patch the modulefile template with site-specific
# paths and SLURM / Shifter settings.  Everything the modulefile hard-codes
# for a given cluster is set here.  Port to a new cluster by copying this
# file under config/<cluster>/ and editing the values.
# ============================================================================

# ----------------------------------------------------------------------------
# LORRAX install layout
# ----------------------------------------------------------------------------

# Where LORRAX source lives (dir containing src/, pyproject.toml).  Empty =
# auto-detect from repo location at install time.
LORRAX_INSTALL_ROOT=""

# Supplemental Python packages for the Shifter container.  The NVIDIA JAX
# container includes JAX / CUDA / NumPy but lacks h5py, scipy, matplotlib.
# To build from scratch:
#   pip install --target="$LORRAX_SITE_PACKAGES" \
#       h5py scipy matplotlib contourpy cycler fonttools kiwisolver \
#       packaging pillow pyparsing python-dateutil six
LORRAX_SITE_PACKAGES="$HOME/software/lorrax_site"

# Shifter container image.  No default is intentional: the former default was
# the retired CUDA-12/JAX-0.7 image, while current LORRAX requires JAX/JAXLIB
# 0.9 and Perlmutter production uses the bare-host ``lorrax_A`` CUDA-13 lane.
# A site that still installs this legacy Shifter module must name a verified
# 0.9-series image explicitly; runtime/jax_support.py checks it again in the
# driver before the first physics jit.
LORRAX_IMAGE="${LORRAX_IMAGE:-}"

# Modulefile install location.
#   Personal: $HOME/modulefiles
#   Shared:   /global/common/software/<project>/modulefiles
LORRAX_MODULEFILE_DIR="$HOME/modulefiles"

# Module name (override to install multiple LORRAX variants side-by-side):
#   LORRAX_MODULE_NAME=lorrax_A bash .../config/perlmutter/install.sh
: "${LORRAX_MODULE_NAME:=lorrax}"

# ----------------------------------------------------------------------------
# Shifter + bind-mount layout (Cray MPICH stack)
# ----------------------------------------------------------------------------

# Shifter module list (gpu + Cray MPICH bind-mounts).
LORRAX_SHIFTER_MODULES="gpu,mpich"

# cuSolverMp lib subdir under /lorrax_nvhpc/.  Must contain
# math_libs/<cuda>/lib64/libcusolverMp.so + the matching
# math_libs/<cuda>/targets/x86_64-linux/include/cusolverMp.h.
#
# 0.9.1_cuda12.9: released nvidia-cusolvermp-cu12==0.9.1.9318.post1,
# paired with cuBLASMp 0.10.0.3695. Stage both and NCCL >= 2.29.2.
# Older cuSOLVERMp 0.4.2--0.9.0 can produce silently wrong eigenvectors
# at large N; the native provider refuses versions below 0.9.1 on every grid.
# This is the legacy container stage; the current bare-host installer uses
# the exact CUDA-13 wheel pair from pyproject.toml instead.
LORRAX_NVHPC_SUBPATH="0.9.1_cuda12.9/math_libs/12.9/lib64"

# Where Shifter bind-mounts Cray MPICH libs inside the container when
# --module=mpich is active.  Standard NERSC layout.
LORRAX_MPICH_CONTAINER_DIR="/opt/udiImage/modules/mpich"

# Optional Darshan I/O profiling library path (empty to skip).  NERSC
# side-mounts it via siteFs; other clusters likely don't have it.
LORRAX_DARSHAN_LIB_DIR="/global/common/software/nersc9/darshan/default/lib"

# Default host-side paths for the 3 FFI bind-mount stages.  Users can
# override with LORRAX_FFI_{NVHPC,PHDF5,CRAY}_DIR before `module load`.
# Relocated off $SCRATCH to $HOME/software 2026-06-24: scratch is purged on
# inactivity, and these are small read-once libs that belong on non-purged $HOME.
LORRAX_FFI_NVHPC_DIR_DEFAULT="$HOME/software/lorrax_nvhpc"
# 2026-08-06: moved off lorrax_phdf5_cray/stage, which held HDF5 **1.12**
# while config/perlmutter/build_ffi_host.sh has loaded
# cray-hdf5-parallel/1.14.3.7 since it was written.  The host FFI leg
# therefore asked the container for libhdf5_parallel_gnu.so.310 and got
# `not found`, which took the WHOLE library down at dlopen (CLAIMS 89).
# The version is IN THE PATH on purpose: which HDF5 is in the stage is the
# fact that drifted, and a name that cannot express it is how it drifted
# unnoticed for months.  GATE 7 (src/ffi/cpp/gate_one_hdf5.sh) now compares
# this tree against what each leg links.
#
# CHANGING THIS INVALIDATES BOTH FFI .so FILES.  The device leg links
# whatever is at /lorrax_phdf5 (src/ffi/cpp/CMakeLists.txt:356-360), so a
# stage swap without a matching `src/ffi/cpp/build.sh` is the same skew
# again with the legs reversed.  Move both together.
LORRAX_FFI_PHDF5_DIR_DEFAULT="$HOME/software/lorrax_phdf5_cray_1.14.3.7/stage"
# Cray runtime libs (libsci, libmpi_gtl_cuda, ...): src/ffi/cpp/stage/cray_runtime_stage.sh.
LORRAX_FFI_CRAY_DIR_DEFAULT="$HOME/software/lorrax_cray_runtime/stage"

# FFTW3 engine for the container.  The image ships none (measured
# 2026-08-06, in-container on a compute node), and the flat-k FFT handler
# resolves FFTW3 at RUN time by dlopen -- so without this mount the FFT
# handlers refuse and every cell of tests/test_fft_flat_k_numerics.py fails
# with "mklfft: no FFTW3 engine in this process".  Populate with
# src/ffi/cpp/stage/fftw_stage_cray.sh after `module load cray-fftw`.
LORRAX_FFI_FFTW_DIR_DEFAULT="$HOME/software/lorrax_fftw_cray/stage"

# Installation shares the CPU defaults; its builder and rank prelude source
# this smaller owner directly so GPU/site edits do not invalidate an adapter.
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/cpu_mpi_config.sh"
