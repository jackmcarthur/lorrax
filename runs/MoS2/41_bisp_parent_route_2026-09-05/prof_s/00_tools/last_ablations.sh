#!/bin/bash
set -uo pipefail
cd /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906
while ! grep -q 'RESULT runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/21_F_dynamic_profile' runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/00_tools/dynamic_sequence.log; do
 sleep 10
done
for arm in \
 runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/22_P_combined_signature_ablation \
 runs/Si/100_bisp_parent_route_2026-09-05/prof_s/17_P_spin_elementwise_profile; do
 echo "START $arm $(date -u +%FT%TZ)"
 bash "$arm/run.sh" 57966610
 rc=$?
 echo "RESULT $arm $rc $(date -u +%FT%TZ)"
 if [ "$rc" -ge 90 ] && [ "$rc" -le 99 ]; then exit "$rc"; fi
done
