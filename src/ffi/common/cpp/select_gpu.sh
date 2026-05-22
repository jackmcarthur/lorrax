#!/bin/sh
# Pin each rank to its node-local GPU.
# SLURM (Perlmutter, Frontier) sets SLURM_LOCALID.
# PBS+pmix (Polaris) sets PMIX_LOCAL_RANK.
# PBS+pmi2 sets PMI_LOCAL_RANK.
# bare OpenMPI sets OMPI_COMM_WORLD_LOCAL_RANK.
#
# Used by run_shifter.sh between srun and the user command.  Unlike an
# inline `bash -c 'export ...; exec "$@"'`, writing this as a script file
# means srun invokes it as a single process image (so PMI_* env vars, which
# mpich's libmpi reads at MPI_Init_thread, aren't stripped by an intermediate
# subshell).
#
# Per-rank GPU isolation via CUDA_VISIBLE_DEVICES — NERSC "Method 3":
# https://docs.nersc.gov/jobs/affinity/
# Needed so SLATE's blas::get_device_count() returns 1 per rank (matching
# JAX's 1-process-per-GPU model).
LOCAL_RANK="${SLURM_LOCALID:-${PMIX_LOCAL_RANK:-${PMI_LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-}}}}"
if [ -z "$LOCAL_RANK" ]; then
    NTASKS="${SLURM_NTASKS:-1}"
    if [ "$NTASKS" -gt 1 ]; then
        echo "select_gpu.sh: no local-rank env var set (tried SLURM_LOCALID, PMIX_LOCAL_RANK, PMI_LOCAL_RANK, OMPI_COMM_WORLD_LOCAL_RANK); refusing to silently coalesce all ranks on GPU 0" >&2
        exit 64
    fi
    LOCAL_RANK=0
fi
export CUDA_VISIBLE_DEVICES="$LOCAL_RANK"
exec "$@"
