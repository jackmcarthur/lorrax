#!/bin/bash
# one rank.  usage: m11_run.sh VAR OUTDIR CASES...   VAR = tag[@lib]; lib base -> wt_kc_occ tree, else wt_kc_m11
set -u
VAR=$1; OUT=$2; shift 2
TAG=${VAR%@*}; LIB=${VAR#*@}
E=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/kc_m11
if [ "$LIB" = base ]; then TREE=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/wt_kc_occ; else TREE=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/wt_kc_m11; fi
cd $TREE
source $TREE/config/perlmutter/gpu_env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_THREAD_TIMEOUT=1
export LORRAX_FFI_SO=$E/$LIB/lib/liblorrax_ffi.so LORRAX_FFI_HOST_SO=$E/$LIB/lib/liblorrax_ffi_host.so
for d in $TREE/services/*/src; do export PYTHONPATH=$PYTHONPATH:$d; done
export PYTHONPATH=$TREE/src:$PYTHONPATH
mkdir -p $OUT
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader > $OUT/gpu.$TAG.txt 2>&1
${M11_PREFIX:-} python3 -u $E/tools/m11_bench.py $OUT $TAG "$@"
echo "RUN_RC=$?"
