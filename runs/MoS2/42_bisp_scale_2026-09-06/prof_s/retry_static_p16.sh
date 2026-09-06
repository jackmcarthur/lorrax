#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
while ! grep -Eq '\[lx\] step .* exit ' 65_F_dynamic_P16_final/driver.1.log 2>/dev/null; do sleep 5; done
bash 67_F_static_P16_timing_fixed/run.sh 57988457
python3 00_tools/gate_candidate.py 52_F_static_P16 67_F_static_P16_timing_fixed > 67_F_static_P16_timing_fixed/gate.log 2>&1
