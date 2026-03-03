#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --job-name=isdf_jax_shifter
#SBATCH --output=out_%j.txt
#SBATCH --image=nvcr.io/nvidia/jax:25.04-py3
#SBATCH --module=gpu,nccl-plugin

# Mount project dir (adjust) and scratch output
# SBATCH --volume="/global/cfs/cdirs/<proj>/ISDF:/workspace/ISDF" \
# SBATCH --volume="/pscratch/sd/<u>/<user>/isdf:/workspace/output"

set -euo pipefail

# Example run: update paths for input/output
srun shifter python3 /workspace/ISDF/src/gw_isdf/gw_jax.py -i /workspace/ISDF/examples/cohsex_test/cohsex_test.in 