#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
bash 33_P3_dynamic_profile/run.sh 57988457
bash ../../../Si/100_bisp_parent_route_2026-09-05/prof_s/23_P3_gn_baseline/run.sh 57988457
bash ../../../Si/100_bisp_parent_route_2026-09-05/prof_s/24_P3_gn_profile/run.sh 57988457
bash 35_P3_static_baseline/run.sh 57988457
bash 36_F3_static_receipts/run.sh 57988457
bash 37_F3_dynamic_receipts/run.sh 57988457
