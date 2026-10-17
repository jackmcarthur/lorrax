#!/bin/bash
# usage: leg_bench.sh OUTJSON CASES TAG1 TAG2 ...   (inside lx run -G 1 -n 1; libs from /pscratch/sd/j/jackm/cs_claude/sigma_depth2/kc_land/libs/<tag>/libkc.so)
set -u
O=$1; CASES=$2; shift 2
cd /pscratch/sd/j/jackm/cs_claude/sigma_depth2/wt_kc_land
source config/perlmutter/gpu_env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_THREAD_TIMEOUT=1
# the standalone libkc.so links /opt nvrtc 13.2; the runtime venv (redeployed 2026-10-16 21:24) now ships builtins 13.3 only
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64
L=""; for t in "$@"; do L="$L --lib $t=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/kc_land/libs/$t/libkc.so"; done
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader
python3 -u /pscratch/sd/j/jackm/cs_claude/sigma_depth2/kc_land/kconv_land_bench.py $L --ev /pscratch/sd/j/jackm/cs_claude/sigma_depth2/kc_gather/libkcev.so --cases "$CASES" \
   --tables /pscratch/sd/j/jackm/cs_claude/sigma_depth2/kc_occ_build/fe8_tables_w0.npz --modes ${KC_MODES:-9,7} --reps ${KC_REPS:-20} --rounds ${KC_ROUNDS:-3} --out $O
echo LEG_RC=$?
