#!/bin/bash
set -uo pipefail
W=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906
S=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export XLA_FLAGS=--xla_force_host_platform_device_count=4
git -C "$W" diff --exit-code 71ae0bde -- src services > source_pin.diff || exit 2
python3 "$S/tools/rules_gate.py" --src "$W/src" --allowlist "$S/tools/rules_gate_allowlist.json" > rules.log 2>&1
rc=$?
printf '%s\n' "$rc" > rules_exit.txt
exit "$rc"
