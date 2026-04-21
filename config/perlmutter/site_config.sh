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
LORRAX_SITE_PACKAGES="$HOME/scratchperl/.isdf/isdf_venvs/isdf_site"

# Extra PYTHONPATH entries for the Shifter container (colon-separated).  Use
# for deps outside LORRAX src/ and site-packages.  Leave empty if not needed.
LORRAX_DEPS="/pscratch/sd/j/jackm/lorrax_sandbox/sources"

# Shifter container image.
LORRAX_IMAGE="nvcr.io/nvidia/jax:25.04-py3"

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

# NVHPC subdir under /lorrax_nvhpc/ that contains math_libs/<cuda>/lib64
# with libcusolverMp + libcal (must match the stage_nvhpc.sh layout).
LORRAX_NVHPC_SUBPATH="25.5_cuda12.9/math_libs/12.9/lib64"

# Where Shifter bind-mounts Cray MPICH libs inside the container when
# --module=mpich is active.  Standard NERSC layout.
LORRAX_MPICH_CONTAINER_DIR="/opt/udiImage/modules/mpich"

# Optional Darshan I/O profiling library path (empty to skip).  NERSC
# side-mounts it via siteFs; other clusters likely don't have it.
LORRAX_DARSHAN_LIB_DIR="/global/common/software/nersc9/darshan/default/lib"

# Default host-side paths for the 3 FFI bind-mount stages.  Users can
# override with LORRAX_FFI_{NVHPC,PHDF5,SLATE}_DIR before `module load`.
# NERSC exports $SCRATCH for every user — use it as the default root.
LORRAX_FFI_NVHPC_DIR_DEFAULT="$SCRATCH/lorrax_nvhpc"
LORRAX_FFI_PHDF5_DIR_DEFAULT="$SCRATCH/lorrax_phdf5_cray/stage"
LORRAX_FFI_SLATE_DIR_DEFAULT="$SCRATCH/lorrax_slate_cray/stage"

# Host SLATE install (bundled blaspp/lapackpp alongside).  Override with
# LORRAX_SLATE_INSTALL_DIR before `module load`.
LORRAX_SLATE_INSTALL_DIR_DEFAULT="$HOME/software/slate/install"
