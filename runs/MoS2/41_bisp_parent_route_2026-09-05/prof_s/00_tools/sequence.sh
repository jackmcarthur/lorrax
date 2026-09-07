#!/bin/bash
# Launch exactly one P4 arm at a time on the explicitly authorized pool.
set -uo pipefail
W=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906
cd "$W"
for arm in \
 runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/11_P_full_static_profile \
 runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/12_F_full_static_profile \
 runs/Si/100_bisp_parent_route_2026-09-05/prof_s/01_P_cohsex_baseline \
 runs/Si/100_bisp_parent_route_2026-09-05/prof_s/02_F_cohsex_baseline \
 runs/Si/100_bisp_parent_route_2026-09-05/prof_s/03_P_gn_baseline \
 runs/Si/100_bisp_parent_route_2026-09-05/prof_s/04_F_gn_baseline \
 runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/07_P_packed_bare_baseline \
 runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/08_F_packed_bare_baseline \
 runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/09_P_dynamic_eps5_baseline \
 runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/10_F_dynamic_eps5_baseline
 do
 echo "START $arm $(date -u +%FT%TZ)"
 bash "$arm/run.sh" 57966610
 rc=$?
 echo "RESULT $arm $rc $(date -u +%FT%TZ)"
 # Preserve later independent arms when a payload or profiler fails.
 if [ "$rc" -ge 90 ] && [ "$rc" -le 99 ]; then exit "$rc"; fi
 done
