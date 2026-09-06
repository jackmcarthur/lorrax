#!/bin/bash
set -uo pipefail
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export XLA_FLAGS=--xla_force_host_platform_device_count=4
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906 rev-parse HEAD > source_head.txt
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tools/gate0.sh /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906 > gate0.log 2>&1
rc=$?
echo "$rc" > gate0.exit
exit "$rc"
