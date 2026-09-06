#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
while ! grep -Eq '\[lx\] step .* exit ' 66_P4_parent_band_seam/driver.1.log; do sleep 5; done
python3 00_tools/gate_candidate.py 60_P4_mapped_restore 66_P4_parent_band_seam > 66_P4_parent_band_seam/gate.log 2>&1 || exit 1
bash 62_P_static_P16_final/run.sh 57988457
python3 00_tools/gate_candidate.py 51_P_static_P16 62_P_static_P16_final > 62_P_static_P16_final/gate.log 2>&1
bash 63_F_static_P16_final/run.sh 57988457
python3 00_tools/gate_candidate.py 52_F_static_P16 63_F_static_P16_final > 63_F_static_P16_final/gate.log 2>&1
bash 64_P_dynamic_P16_final/run.sh 57988457
python3 00_tools/gate_candidate.py 53_P_dynamic_P16 64_P_dynamic_P16_final > 64_P_dynamic_P16_final/gate.log 2>&1
bash 65_F_dynamic_P16_final/run.sh 57988457
python3 00_tools/gate_candidate.py 54_F_dynamic_P16 65_F_dynamic_P16_final > 65_F_dynamic_P16_final/gate.log 2>&1
