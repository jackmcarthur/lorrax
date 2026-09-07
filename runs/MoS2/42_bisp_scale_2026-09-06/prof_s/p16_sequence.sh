#!/bin/bash
set -euo pipefail
W=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906
R=$W/runs/MoS2/42_bisp_scale_2026-09-06/prof_s
cd "$W"
git diff --exit-code HEAD -- src services tests
candidate=$(git rev-parse HEAD)
printf '%s\n' "$candidate" > "$R/p16_candidate_pin.txt"
trap 'git -C "$W" restore --source="$candidate" -- src/gw/photon_sigma.py src/gw/cohsex_sigma.py src/gw/w_isdf.py' EXIT
git restore --source=71ae0bde -- src/gw/photon_sigma.py src/gw/cohsex_sigma.py src/gw/w_isdf.py
git diff --exit-code 71ae0bde -- src services > "$R/p16_baseline_source_check.txt"
bash "$R/19_P16_baseline/run.sh" 57982945
bash "$R/20_P16_baseline_profile/run.sh" 57982945
git restore --source="$candidate" -- src/gw/photon_sigma.py src/gw/cohsex_sigma.py src/gw/w_isdf.py
git diff --exit-code HEAD -- src services > "$R/p16_candidate_source_check.txt"
bash "$R/21_P16_candidate/run.sh" 57982945
bash "$R/22_P16_candidate_profile/run.sh" 57982945
python3 "$R/00_tools/gate_candidate.py" "$R/19_P16_baseline" "$R/20_P16_baseline_profile"
python3 "$R/00_tools/gate_candidate.py" "$R/19_P16_baseline" "$R/21_P16_candidate"
python3 "$R/00_tools/gate_candidate.py" "$R/19_P16_baseline" "$R/22_P16_candidate_profile"
trap - EXIT
