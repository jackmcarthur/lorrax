#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --job-name=isdf_uv_jax
#SBATCH --output=out_%j.txt

set -euo pipefail

# Paths
: "${PM_SOFTWARE_PREFIX:=${SCRATCH:-$HOME}}"
VENV_DIR="${PM_SOFTWARE_PREFIX%/}/isdf-venv"

# Load recommended runtime env
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HDF5_USE_FILE_LOCKING=FALSE

# Activate venv
source "$VENV_DIR/bin/activate"

# Example run (adjust input path)
srun python3 ${SLURM_SUBMIT_DIR}/src/isdf/gw_isdf/cohsex_jax.py -i ${SLURM_SUBMIT_DIR}/examples/cohsex_test/cohsex_test.in 