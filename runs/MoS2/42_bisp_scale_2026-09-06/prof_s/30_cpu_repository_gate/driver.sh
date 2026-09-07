#!/bin/bash
set -euo pipefail
W=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export XLA_FLAGS=--xla_force_host_platform_device_count=4
git -C "$W" rev-parse HEAD > source_head.txt
git -C "$W" diff HEAD -- src services tests > source.diff
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tools/gate0.sh "$W" > gate0.log 2>&1
printf 'REPOSITORY_GATE_PASS\n' > result.txt
