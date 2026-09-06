#!/bin/bash
set -euo pipefail
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906 rev-parse HEAD > source_head.txt
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906 diff HEAD -- src services tests > source.diff
export LORRAX_DEBUG_PRINT=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$PWD/xla_dump_rank${SLURM_PROCID}"
exec python3 -u run_driver.py -i cohsex.in > driver.rank${SLURM_PROCID}.log 2>&1
