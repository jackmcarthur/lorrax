#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --job-name=lorrax_gw
#SBATCH --output=gw_%j.out
#SBATCH --account=m2651

# ============================================================================
# LORRAX GW batch job — Perlmutter
#
# Usage:
#   cd /path/to/run_dir   # must contain cohsex.in, WFN.h5, etc.
#   sbatch /path/to/run_gw.slurm
#
# Or with a custom input file:
#   sbatch --export=LORRAX_INPUT=my_cohsex.in /path/to/run_gw.slurm
# ============================================================================

module load lorrax

INPUT="${LORRAX_INPUT:-cohsex.in}"

echo "Running LORRAX GW: $(pwd)/$INPUT"
echo "Nodes: $SLURM_NNODES  GPUs: $((SLURM_NNODES * 4))"

lxrun python3 -u -m gw.gw_jax -i "$(pwd)/$INPUT" 2>&1 | tee gw.out
