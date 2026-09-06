#!/bin/bash
set -euo pipefail
test "$(git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_zw_codex_20260906 rev-parse HEAD:src)" = dced6947bb1f6334102226224408bfee40d0d4bb
test "$(git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_zw_codex_20260906 rev-parse HEAD:services)" = be5ab668aaf6927aeae00e721b34dcc1f4493452
test -z "$(git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_zw_codex_20260906 diff -- src services)"
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_zw_codex_20260906 rev-parse HEAD > source.rank${SLURM_PROCID}.txt
export LORRAX_DEBUG_PRINT=1
python3 -u -m gw.gw_jax -i cohsex.in > driver.rank${SLURM_PROCID}.log 2>&1
test -s eqp0.dat
test -s eqp1.dat
