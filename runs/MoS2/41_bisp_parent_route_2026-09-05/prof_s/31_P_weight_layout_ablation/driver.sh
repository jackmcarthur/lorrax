#!/bin/bash
set -euo pipefail
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906 rev-parse HEAD
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906 diff --exit-code 9f569c4bf75bad40e4f5895946874b4c503e4410 -- src services > source.diff
export LORRAX_DEBUG_PRINT=1
exec python3 -u ablation.py -i cohsex.in > driver.rank${SLURM_PROCID}.log 2>&1
