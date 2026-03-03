#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --job-name=cohsex_jax
#SBATCH --output=cohsex_%j.out
#SBATCH --account=m2651
#SBATCH --image=nvcr.io/nvidia/jax:25.04-py3

# ============================================================================
# COHSEX-JAX batch job script for NERSC Perlmutter
# ============================================================================
# Usage: Copy this script to your run directory and submit with sbatch:
#   cp cluster_shifter/run_cohsex.slurm myrun/
#   cd myrun && sbatch run_cohsex.slurm
#
# Change RUN_DIR below or just run from the directory containing gw.inp
# ============================================================================

CODE_DIR="$HOME/software/isdf_cohsex"
RUN_DIR="."
INPUT_FILE="gw.inp"

cd "$RUN_DIR"
srun shifter --module=gpu --env=PYTHONPATH=${CODE_DIR}/src \
    python3 -m gw_isdf.gw_jax -i "$INPUT_FILE"
