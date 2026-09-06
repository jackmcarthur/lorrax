#!/bin/bash
set -uo pipefail
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/38_P_dynamic_final_profile/run.sh 57966610
rc=$?
echo "38_P_dynamic_final_profile exit $rc"
[ "$rc" = 0 ] || exit "$rc"
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/Si/100_bisp_parent_route_2026-09-05/prof_s/21_P_gn_fixed_rules/run.sh 57966610
rc=$?
echo "21_P_gn_fixed_rules exit $rc"
[ "$rc" = 0 ] || exit "$rc"
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/Si/100_bisp_parent_route_2026-09-05/prof_s/22_F_gn_fixed_rules/run.sh 57966610
rc=$?
echo "22_F_gn_fixed_rules exit $rc"
[ "$rc" = 0 ] || exit "$rc"
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/43_P_dynamic_fixed_rules/run.sh 57966610
rc=$?
echo "43_P_dynamic_fixed_rules exit $rc"
[ "$rc" = 0 ] || exit "$rc"
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/44_F_dynamic_fixed_rules/run.sh 57966610
rc=$?
echo "44_F_dynamic_fixed_rules exit $rc"
[ "$rc" = 0 ] || exit "$rc"
bash /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906/runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/41_ast_gate/run.sh 57966610
rc=$?
echo "41_ast_gate exit $rc"
[ "$rc" = 0 ] || exit "$rc"
