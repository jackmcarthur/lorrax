#!/bin/bash -l
#
# Multi-node JAX run for isdf_cohsex under Shifter on Perlmutter.
# - 1 process per GPU (4 tasks per node)
# - uses stock NVIDIA JAX image
# - installs Python deps (incl. normal h5py) into a persistent venv on /workspace/output
#
# Key point: we set ONE coordinator address for all ranks so jax.distributed.initialize works reliably.
#
# Before using:
# - set --account
# - set --volume mounts to point at your repo + pscratch output
#
#SBATCH --time=02:00:00
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --job-name=isdf_shifter_mn
#SBATCH --output=out_%j.txt
#
# NOTE: account/image/module/volume are intentionally NOT hard-coded here.
# Use `cluster_shifter/submit_multinode.sh` which reads cluster_shifter/config
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

# JAX multi-process coordinator.
COORD_HOST="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"
export JAX_COORDINATOR_ADDRESS="${COORD_HOST}:12355"
export JAX_PROCESS_COUNT="${SLURM_NTASKS}"
export JAX_PROCESS_INDEX="${SLURM_PROCID}"

echo "[slurm] nodes=${SLURM_JOB_NUM_NODES} ntasks=${SLURM_NTASKS} ntasks-per-node=${SLURM_NTASKS_PER_NODE}"
echo "[jax] coordinator=${JAX_COORDINATOR_ADDRESS} count=${JAX_PROCESS_COUNT}"

# Ensure the venv exists (run once, reused across jobs).
srun -n 1 shifter bash -lc "bash ${ISDF_CODE_DIR}/cluster_shifter/bootstrap_venv.sh"

# Activate venv and run.
INPUT_IN_CONTAINER="${ISDF_INPUT_IN_CONTAINER:-${ISDF_CODE_DIR}/examples/cohsex_test/cohsex_test.in}"
srun -n "${SLURM_NTASKS}" shifter bash -lc "
  source ${ISDF_VENV_ROOT_DIR}/isdf_cohsex_py311/bin/activate
  python3 ${ISDF_CODE_DIR}/src/isdf/gw_isdf/cohsex_jax.py -i ${INPUT_IN_CONTAINER}
"


