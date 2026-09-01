#!/usr/bin/env bash
set -euo pipefail

checkout=/pscratch/sd/j/jackm/wt_sigma_exec_2026-08-31
evidence="$checkout/tmp/sigma_exec_p4_20260831"
variant=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829
python=/pscratch/sd/j/jackm/wt_hybrid_wiring_2026-08-29/.venv/bin/python

run_arm() {
    local arm=$1
    shift
    cd "$evidence/$arm"
    env \
        LORRAX_RUN_RECEIPT="$evidence/job_receipt.txt" \
        LORRAX_SIGMA_PLAN=delivered \
        LORRAX_DELIVERED_TAU_GRID=free \
        LORRAX_DELIVERED_PLAN_CACHE="$evidence/$arm/tmp/delivered_sigma_plan_v1.pkl" \
        LORRAX_CERTIFIED_MPA_FIT="$evidence/$arm/input_mpa_fit.h5" \
        "$@" \
        "$python" -u "$variant/run_existing_fit.py"
}

run_arm stage_p4 \
    ISDF_JAX_CACHE_DIR= \
    LORRAX_SIGMA_TAU_TIMING=1 \
    XLA_FLAGS="--xla_dump_to=$evidence/stage_p4/xla_dump_rank${SLURM_PROCID:-0} --xla_dump_hlo_as_text"

run_arm batch8_p4 \
    LORRAX_SIGMA_TAU_TIMING=0
