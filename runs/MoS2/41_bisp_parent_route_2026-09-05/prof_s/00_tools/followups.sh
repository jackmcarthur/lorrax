#!/bin/bash
set -uo pipefail
W=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906
cd "$W"
base=runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s
export LX_BASE_MODULE=lorrax_A
lx run --jid 57966610 --wait 1800 -N 1 -G 0 -n 1 -- "$base/05_P_full_static_baseline/rankwrap.sh" env JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 XLA_FLAGS=--xla_force_host_platform_device_count=4 python3 "$base/00_tools/prepare_dynamic.py" > "$base/00_tools/prepare_dynamic.log" 2>&1
rc=$?
echo "PREP_DYNAMIC $rc"
for name in 14_P_host_boundary 15_F_host_boundary 13_P_restore_lifetime_ablation 16_P_weight_shape_ablation 17_P_plan_reuse_ablation 18_P_restore_shape_ablation; do
 echo "START $name $(date -u +%FT%TZ)"
 bash "$base/$name/run.sh" 57966610
 rc=$?
 echo "RESULT $name $rc $(date -u +%FT%TZ)"
 if [ "$rc" -ge 90 ] && [ "$rc" -le 99 ]; then exit "$rc"; fi
 done
if [ -d runs/Si/100_bisp_parent_route_2026-09-05/prof_s/11_P_gn_profile ] && [ -d "$base/21_F_dynamic_profile" ]; then
 for arm in \
 runs/Si/100_bisp_parent_route_2026-09-05/prof_s/11_P_gn_profile \
 runs/Si/100_bisp_parent_route_2026-09-05/prof_s/12_F_gn_profile \
 "$base/20_P_dynamic_profile" "$base/21_F_dynamic_profile"; do
  echo "START $arm $(date -u +%FT%TZ)"
  bash "$arm/run.sh" 57966610
  rc=$?
  echo "RESULT $arm $rc $(date -u +%FT%TZ)"
  if [ "$rc" -ge 90 ] && [ "$rc" -le 99 ]; then exit "$rc"; fi
 done
fi
