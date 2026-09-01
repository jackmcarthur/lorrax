#!/usr/bin/env bash
set -euo pipefail

checkout=/pscratch/sd/j/jackm/wt_sigma_exec_2026-08-31
arm="$checkout/tmp/sigma_exec_p4_20260831/batch6_p4"
variant=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829
python=/pscratch/sd/j/jackm/wt_hybrid_wiring_2026-08-29/.venv/bin/python

cd "$arm"
exec env \
    LORRAX_SIGMA_PLAN=delivered \
    LORRAX_DELIVERED_TAU_GRID=free \
    LORRAX_DELIVERED_PLAN_CACHE="$arm/tmp/delivered_sigma_plan_v1.pkl" \
    LORRAX_CERTIFIED_MPA_FIT="$arm/input_mpa_fit.h5" \
    LORRAX_SIGMA_TAU_TIMING=0 \
    "$python" -u "$variant/run_existing_fit.py"
