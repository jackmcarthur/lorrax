#!/bin/bash
set -uo pipefail
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/Si/100_bisp_parent_route_2026-09-05/prof_s/20_P_gn_final_profile/run.sh 57966610
rc=$?
echo "20_P_gn_final_profile exit $rc"
[ "$rc" = 0 ] || exit "$rc"
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/35_P_full_static_final/run.sh 57966610
rc=$?
echo "35_P_full_static_final exit $rc"
[ "$rc" = 0 ] || exit "$rc"
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/Si/100_bisp_parent_route_2026-09-05/prof_s/15_P_cohsex_final/run.sh 57966610
rc=$?
echo "15_P_cohsex_final exit $rc"
[ "$rc" = 0 ] || exit "$rc"
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/25_P_packed_bare_final/run.sh 57966610
rc=$?
echo "25_P_packed_bare_final exit $rc"
[ "$rc" = 0 ] || exit "$rc"
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/26_P_dynamic_final/run.sh 57966610
rc=$?
echo "26_P_dynamic_final exit $rc"
[ "$rc" = 0 ] || exit "$rc"
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/29_combined_regression/run.sh 57966610
rc=$?
echo "29_combined_regression exit $rc"
[ "$rc" = 0 ] || exit "$rc"
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/36_P_static_final_profile/run.sh 57966610
rc=$?
echo "36_P_static_final_profile exit $rc"
[ "$rc" = 0 ] || exit "$rc"
