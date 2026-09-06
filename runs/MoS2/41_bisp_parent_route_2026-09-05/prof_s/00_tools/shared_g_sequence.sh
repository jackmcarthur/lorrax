#!/bin/bash
set -uo pipefail
cd /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906
while ! grep -q 'RESULT runs/Si/100_bisp_parent_route_2026-09-05/prof_s/17_P_spin_elementwise_profile' runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/00_tools/last_ablations.log; do
 sleep 10
done
bash runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/30_P_shared_g_ablation/run.sh 57966610
