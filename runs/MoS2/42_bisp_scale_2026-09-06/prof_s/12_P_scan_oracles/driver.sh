#!/bin/bash
set -euo pipefail
W=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906
git -C "$W" rev-parse HEAD > source_head.txt
git -C "$W" diff HEAD -- src services tests > source.diff
export LORRAX_DEBUG_PRINT=1
exec python3 -u gate.py > driver.rank${SLURM_PROCID}.log 2>&1
