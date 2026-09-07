#!/bin/bash
# Per-rank wrapper: pin PYTHONPATH to the LANE worktree (perf/psi-irr-zeta-orbit-tiles-2026-09-05), cold compile cache,
# unique JAX coordinator per step (skills/execute_workflow/SKILL.md).
W=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_main_de8dcfbc_fixed
export PYTHONPATH="$W/src:$W/services/wfn_loader/src:$W/services/distrib_la/src:$W/services/symmetry_maps/src:$W/services/vcoul/src:$W/services/minimax/src:$W/services/zeta_loader/src:$W/services/lxkit/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${SLURM_STEP_NODELIST:-}" ]; then
  coordinator_host=$(scontrol show hostnames "${SLURM_STEP_NODELIST}" | sed -n '1p')
  case ${SLURM_STEP_ID:?} in (*[!0-9]*) exit 86;; esac
  export JAX_COORDINATOR_ADDRESS="${coordinator_host}:$((22000 + SLURM_STEP_ID % 30000))"
fi
export LORRAX_CHECKOUT="$W"
export ISDF_JAX_CACHE_DIR=""
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 XLA_PYTHON_CLIENT_ALLOCATOR=bfc
echo "[rankwrap] rank=${SLURM_PROCID:-?} host=$(hostname) src=$W"
exec "$@"
