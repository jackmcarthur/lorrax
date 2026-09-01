#!/usr/bin/env bash
set -euo pipefail

git -C /pscratch/sd/j/jackm/wt_resfloor_2026-08-31 rev-parse HEAD
git -C /pscratch/sd/j/jackm/wt_resfloor_2026-08-31 status --short --branch
exec /bin/bash \
  /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829/run_rank.sh \
  /pscratch/sd/j/jackm/wt_hybrid_wiring_2026-08-29/.venv/bin/python -u \
  /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829/run_existing_fit.py
