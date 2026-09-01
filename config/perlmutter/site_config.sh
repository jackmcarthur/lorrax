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

# Extra PYTHONPATH entries for the Shifter container (colon-separated).  Use
# for deps outside LORRAX src/ and site-packages.  Leave empty if not needed.
LORRAX_DEPS=""

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
# SLURM defaults (used by lxalloc)
# ----------------------------------------------------------------------------

# Charge account.  Perlmutter: AY-yearly allocation ID (e.g. m2651).
LORRAX_SLURM_ACCOUNT="m2651"

# QOS.  NERSC interactive GPU QOS.
LORRAX_SLURM_QOS="interactive"

# Node constraint.  Perlmutter: "gpu" selects A100 nodes.
LORRAX_SLURM_CONSTRAINT="gpu"

# GPUs per node.  Perlmutter GPU nodes have 4 A100s.
LORRAX_GPUS_PER_NODE="4"

# ----------------------------------------------------------------------------
# Shifter + bind-mount layout (Cray MPICH stack)
# ----------------------------------------------------------------------------

# Shifter module list (gpu + Cray MPICH bind-mounts).
LORRAX_SHIFTER_MODULES="gpu,mpich"

# srun --mpi= PMI flavour.  Cray MPICH speaks cray_shasta; pmi2 / pmix will
# silently give singleton MPI_COMM_WORLD with shifter-mpich.
LORRAX_MPI_TYPE_DEFAULT="cray_shasta"

# cuSolverMp lib subdir under /lorrax_nvhpc/.  Must contain
# math_libs/<cuda>/lib64/libcusolverMp.so + the matching
# math_libs/<cuda>/targets/x86_64-linux/include/cusolverMp.h.
#
# 0.7.2_cuda12.9: standalone PyPI wheel ``nvidia-cusolvermp-cu12==0.7.2.888``
# extracted into $HOME/software/lorrax_nvhpc/0.7.2_cuda12.9/...  This
# is the working baseline: it includes the CAL→NCCL ABI fix (release 0.7.0)
# AND the race-condition follow-up (release 0.7.2) so distributed LU on a
# 2D process grid actually returns correct answers.  Validated on Px=Py=2
# at machine precision for getrf/getrs across multiple (N, NRHS) configs.
#
# 25.5_cuda12.9: the older NVHPC-25.5 install, cuSolverMp 0.6.0.  Kept
# alongside 0.7.2 for fallback; load with LORRAX_NVHPC_SUBPATH override
# in a shell.  0.6.0 silently returns wrong answers for getrf/getrs on
# any Px>1 AND Py>1 mesh — the patched FFI prints a runtime warning
# (see src/ffi/cpp/cusolvermp/context.cc) but the warning can't fix the
# bug.  Use 1xN / Nx1 mesh if you must run 0.6.0.
LORRAX_NVHPC_SUBPATH="0.7.2_cuda12.9/math_libs/12.9/lib64"

# Where Shifter bind-mounts Cray MPICH libs inside the container when
# --module=mpich is active.  Standard NERSC layout.
LORRAX_MPICH_CONTAINER_DIR="/opt/udiImage/modules/mpich"

# Optional Darshan I/O profiling library path (empty to skip).  NERSC
# side-mounts it via siteFs; other clusters likely don't have it.
LORRAX_DARSHAN_LIB_DIR="/global/common/software/nersc9/darshan/default/lib"

# Default host-side paths for the 3 FFI bind-mount stages.  Users can
# override with LORRAX_FFI_{NVHPC,PHDF5,SLATE}_DIR before `module load`.
# Relocated off $SCRATCH to $HOME/software 2026-06-24: scratch is purged on
# inactivity, and these are small read-once libs that belong on non-purged $HOME
# (matches LORRAX_SLATE_INSTALL_DIR_DEFAULT below).
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
LORRAX_FFI_SLATE_DIR_DEFAULT="$HOME/software/lorrax_slate_cray/stage"

# FFTW3 engine for the container.  The image ships none (measured
# 2026-08-06, in-container on a compute node), and the flat-k FFT handler
# resolves FFTW3 at RUN time by dlopen -- so without this mount the FFT
# handlers refuse and every cell of tests/test_fft_flat_k_numerics.py fails
# with "mklfft: no FFTW3 engine in this process".  Populate with
# src/ffi/cpp/stage/fftw_stage_cray.sh after `module load cray-fftw`.
LORRAX_FFI_FFTW_DIR_DEFAULT="$HOME/software/lorrax_fftw_cray/stage"

# Host SLATE install (bundled blaspp/lapackpp alongside).  Override with
# LORRAX_SLATE_INSTALL_DIR before `module load`.
LORRAX_SLATE_INSTALL_DIR_DEFAULT="$HOME/software/slate/install"

# CPU MPI adapter for JAX's MPItrampoline backend.  This is an unmodified,
# pinned upstream MPIwrapper build against Cray MPICH; the wrapper is a small
# ABI adapter, not an MPI implementation.  The builder installs immutable,
# content-addressed releases and atomically advances ``current`` only after
# all gates pass, so a failed rebuild cannot remove the live adapter.
LORRAX_MPIWRAPPER_ROOT_DEFAULT="$HOME/software/lorrax_mpiwrapper_cray"
LORRAX_MPIWRAPPER_PREFIX_DEFAULT="$LORRAX_MPIWRAPPER_ROOT_DEFAULT/current"
LORRAX_MPIWRAPPER_COMMIT_DEFAULT="966f4231c96153a08295fc7d0bcbd65e916a73fd"
LORRAX_MPIWRAPPER_ABI_DEFAULT="2.10.0"

# Reproducible host toolchain for that adapter.  Cray MPICH is still the MPI
# implementation; MPIwrapper only translates MPItrampoline's stable ABI.
LORRAX_PM_PRGENV_DEFAULT="PrgEnv-gnu/8.7.0"
LORRAX_PM_MPICH_DEFAULT="cray-mpich/9.1.0"
LORRAX_PM_CMAKE_DEFAULT="cmake/3.30.2"
