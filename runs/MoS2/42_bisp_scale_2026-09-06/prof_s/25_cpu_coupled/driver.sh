#!/bin/bash
set -euo pipefail
W=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export XLA_FLAGS=--xla_force_host_platform_device_count=4
git -C "$W" rev-parse HEAD > source_head.txt
git -C "$W" diff HEAD -- src services tests > source.diff
python3 -m pytest -q "$W/tests/multi_device/bispinor_physics_oracles.py" --basetemp "$PWD/pytest_tmp" > pytest.log 2>&1
echo CPU_SCAN_PASS > result.txt
