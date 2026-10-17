#!/bin/bash
# one rank. usage: m11_ncu.sh VAR OUT CASE   (VAR tag@lib)
set -u
VAR=$1; OUT=$2; CASE=$3
TAG=${VAR%@*}; LIB=${VAR#*@}
E=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/kc_m11
if [ "$LIB" = base ]; then TREE=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/wt_kc_occ; else TREE=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/wt_kc_m11; fi
cd $TREE
source $TREE/config/perlmutter/gpu_env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_THREAD_TIMEOUT=1 M11_REPS=1 M11_NOSAVE=1
export LORRAX_FFI_SO=$E/$LIB/lib/liblorrax_ffi.so LORRAX_FFI_HOST_SO=$E/$LIB/lib/liblorrax_ffi_host.so
for d in $TREE/services/*/src; do export PYTHONPATH=$PYTHONPATH:$d; done
export PYTHONPATH=$TREE/src:$PYTHONPATH
mkdir -p $OUT
NCU=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Compute/ncu
resume() { dcgmi profile --resume >> $OUT/dcgmi.log 2>&1; echo "resume rc=$? $(date -u +%T)" >> $OUT/dcgmi.log; }
trap resume EXIT TERM INT HUP
dcgmi profile --pause >> $OUT/dcgmi.log 2>&1; echo "pause rc=$? $(date -u +%T)" >> $OUT/dcgmi.log
timeout 600 $NCU --kernel-name regex:lrx_kconv --launch-skip 2 --launch-count 1 --set full --import-source yes --clock-control none -f -o $OUT/ncu_$TAG python3 -u $E/tools/m11_bench.py $OUT/scratch $TAG "$CASE" > $OUT/ncu_$TAG.log 2>&1
echo "NCU_RC=$?" >> $OUT/ncu_$TAG.log
resume
trap - EXIT TERM INT HUP
$NCU --import $OUT/ncu_$TAG.ncu-rep --page details --csv > $OUT/details_$TAG.csv 2>&1
$NCU --import $OUT/ncu_$TAG.ncu-rep --page source --csv --print-source sass > $OUT/sass_$TAG.csv 2>&1
echo done
