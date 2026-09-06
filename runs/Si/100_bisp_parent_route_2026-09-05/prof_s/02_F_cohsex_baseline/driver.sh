#!/bin/bash
set -euo pipefail
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_main_de8dcfbc_fixed rev-parse HEAD
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_main_de8dcfbc_fixed diff --exit-code e1559a071e244b4f049c924781b668d9e1560739 -- src services > source.diff
export LORRAX_DEBUG_PRINT=1
exec python3 -u -m gw.gw_jax -i cohsex.in > driver.rank${SLURM_PROCID}.log 2>&1
