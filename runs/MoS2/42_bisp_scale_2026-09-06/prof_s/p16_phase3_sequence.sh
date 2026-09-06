#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
bash 51_P_static_P16/run.sh 57988457
python3 00_tools/gate_candidate.py 35_P3_static_baseline 51_P_static_P16 > 51_P_static_P16/gate.log 2>&1
bash 52_F_static_P16/run.sh 57988457
python3 00_tools/gate_candidate.py 36_F3_static_receipts 52_F_static_P16 > 52_F_static_P16/gate.log 2>&1
bash 53_P_dynamic_P16/run.sh 57988457
python3 00_tools/gate_candidate.py 32_P3_dynamic_baseline 53_P_dynamic_P16 > 53_P_dynamic_P16/gate.log 2>&1
bash 54_F_dynamic_P16/run.sh 57988457
python3 00_tools/gate_candidate.py /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/MoS2/42_bisp_scale_2026-09-06/107_F_dynamic_P4_baseline 54_F_dynamic_P16 > 54_F_dynamic_P16/gate.log 2>&1
