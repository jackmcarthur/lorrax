#!/bin/bash -l
#
# Single-node smoke test for isdf_cohsex under Shifter on Perlmutter.
# - 1 process per GPU (4 tasks on a standard GPU node)
# - uses stock NVIDIA JAX image
# - installs Python deps (incl. normal h5py) into a persistent venv on /workspace/output
#
# Before using:
# - set --account
# - set --volume mounts to point at your repo + pscratch output
#
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --job-name=isdf_shifter_1n
#SBATCH --output=out_%j.txt
#
# NOTE: account/image/module/volume are intentionally NOT hard-coded here.
# Use `cluster_shifter/submit_singlenode.sh` which reads cluster_shifter/config
# and passes those via `sbatch --account/--image/--module/--volume ...`.

set -euo pipefail

export ISDF_CODE_DIR=/workspace/ISDF
export ISDF_VENV_ROOT_DIR=/workspace/venvroot
export ISDF_RUN_DIR=/workspace/run

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HDF5_USE_FILE_LOCKING=FALSE

# (Optional) Offline wheelhouse mode:
# export WHEELHOUSE=/workspace/wheelhouse

# Ensure the venv exists (run once, reused across jobs).
srun -n 1 shifter bash -lc "bash ${ISDF_CODE_DIR}/cluster_shifter/bootstrap_venv.sh"

# Activate venv and run. (Your code resolves relative paths against the input file directory.)
INPUT_IN_CONTAINER="${ISDF_INPUT_IN_CONTAINER:-${ISDF_CODE_DIR}/examples/cohsex_test/cohsex_test.in}"
srun -n "${SLURM_NTASKS}" shifter bash -lc "
  source ${ISDF_VENV_ROOT_DIR}/isdf_cohsex_py311/bin/activate
  python3 ${ISDF_CODE_DIR}/src/isdf/gw_isdf/cohsex_jax.py -i ${INPUT_IN_CONTAINER}
"


