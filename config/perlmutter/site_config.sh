#!/usr/bin/env bash
# ============================================================================
# LORRAX site configuration — Perlmutter (NERSC)
#
# This file is sourced by install.sh. Edit these paths for your installation.
# For a shared group installation, point LORRAX_INSTALL_ROOT to a location
# under /global/common/software/<project>/.
# ============================================================================

# Where LORRAX source is installed (the directory containing src/, pyproject.toml)
# install.sh auto-detects this from the repo if left empty.
LORRAX_INSTALL_ROOT=""

# Supplemental Python packages for the Shifter container.
# The NVIDIA JAX container includes JAX/CUDA/NumPy but lacks h5py, scipy,
# matplotlib, etc. This directory provides them.
#
# To build this site directory from scratch:
#   pip install --target=$LORRAX_SITE_PACKAGES h5py scipy matplotlib \
#       contourpy cycler fonttools kiwisolver packaging pillow pyparsing \
#       python-dateutil six
#
LORRAX_SITE_PACKAGES="$HOME/scratchperl/.isdf/isdf_venvs/isdf_site"

# Extra PYTHONPATH entries for the Shifter container (colon-separated).
# Use this for dependencies that live outside of LORRAX src/ and site-packages.
# Example: jax_xc_local lives at <sandbox>/sources/jax_xc_local/ (a symlink),
# so we add <sandbox>/sources/ to make `import jax_xc_local` work.
# Leave empty if no extra deps are needed.
LORRAX_DEPS="/pscratch/sd/j/jackm/lorrax_sandbox/sources"

# Shifter container image (NVIDIA JAX with CUDA 12, Python 3.12)
LORRAX_IMAGE="nvcr.io/nvidia/jax:25.04-py3"

# Where to install the modulefile.
# Personal: $HOME/modulefiles    (only you can `module load lorrax`)
# Shared:   /global/common/software/<project>/modulefiles
LORRAX_MODULEFILE_DIR="$HOME/modulefiles"
