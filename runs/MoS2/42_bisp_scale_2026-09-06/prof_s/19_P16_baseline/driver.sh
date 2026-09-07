#!/bin/bash
set -euo pipefail
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906 rev-parse HEAD > source_head.txt
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906 diff HEAD -- src services tests > source.diff
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906 diff --exit-code 71ae0bde -- src services > source_pin.diff
export LORRAX_DEBUG_PRINT=1
exec python3 -u -m gw.gw_jax -i cohsex.in > driver.rank${SLURM_PROCID}.log 2>&1
