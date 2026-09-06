#!/bin/bash
set -euo pipefail
test "$(git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_main_de8dcfbc_fixed rev-parse HEAD:src)" = 5c73799af715138a4d032a1b4b9a5fcba4d98870
test "$(git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_main_de8dcfbc_fixed rev-parse HEAD:services)" = ef6ed79ace84026b6f973c59f5c9e88a2bc03484
test -z "$(git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_main_de8dcfbc_fixed diff -- src services)"
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_main_de8dcfbc_fixed rev-parse HEAD > source.rank${SLURM_PROCID}.txt
export LORRAX_DEBUG_PRINT=1
python3 -u -m gw.gw_jax -i cohsex.in > driver.rank${SLURM_PROCID}.log 2>&1
test -s eqp0.dat
test -s eqp1.dat
