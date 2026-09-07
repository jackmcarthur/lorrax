#!/bin/bash
set -euo pipefail
printf '%s\n' c2f69987 > source_head.txt
export LORRAX_DEBUG_PRINT=1
exec python3 -u run_driver.py -i cohsex.in > driver.rank${SLURM_PROCID}.log 2>&1
