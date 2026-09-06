#!/bin/bash
set -euo pipefail
W=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export XLA_FLAGS=--xla_force_host_platform_device_count=4
git -C "$W" rev-parse HEAD > source_head.txt
git -C "$W" diff HEAD -- src services tests > source.diff
python3 -m pytest -q "$W/services/symmetry_maps/tests/test_symmetry_maps_emulated_mesh.py" -k 'operator_rotation or four_spin_block_rotation' --basetemp "$PWD/spin_tmp" > spin_pytest.log 2>&1
python3 -m pytest -q "$W/tests/test_bispinor_dynamic_packed_route.py" --basetemp "$PWD/config_tmp" > config_pytest.log 2>&1
echo CPU_PYTEST_PASS > result.txt
