#!/bin/bash
# one rank, cwd = checkout.  usage: leg_ncu_bench.sh VAR OUTDIR TABLES [extra bench args]
set -u
VAR=$1; OUT=$2; TAB=$3; shift 3
KB=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/kc_occ_build
NCU=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Compute/ncu
source $PWD/config/perlmutter/gpu_env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_THREAD_TIMEOUT=1
export LORRAX_FFI_SO=$KB/$VAR/lib/liblorrax_ffi.so LORRAX_FFI_HOST_SO=$KB/$VAR/lib/liblorrax_ffi_host.so
for d in $PWD/services/*/src; do export PYTHONPATH=$PYTHONPATH:$d; done
mkdir -p $OUT
resume() { dcgmi profile --resume >> $OUT/dcgmi.log 2>&1; echo "resume rc=$? $(date -u +%T)" >> $OUT/dcgmi.log; }
trap resume EXIT TERM INT HUP
dcgmi profile --pause >> $OUT/dcgmi.log 2>&1; echo "pause rc=$? $(date -u +%T)" >> $OUT/dcgmi.log
# launch 0, 1: compile/warm of mode 7 (bench warms twice); profile the 3rd mode-7 launch
timeout 900 $NCU --kernel-name regex:lrx_kconv --launch-skip 2 --launch-count 1 --set full \
   --import-source yes --clock-control none -f -o $OUT/ncu_$VAR \
   python3 -u $KB/tools/kconv_bench.py --tables $TAB --reps 1 --mode9 0 "$@" > $OUT/ncu_$VAR.log 2>&1
echo "NCU_RC=$?" >> $OUT/ncu_$VAR.log
resume
trap - EXIT TERM INT HUP
$NCU --import $OUT/ncu_$VAR.ncu-rep --page details --csv > $OUT/details_$VAR.csv 2>&1
$NCU --import $OUT/ncu_$VAR.ncu-rep --page source --csv --print-source sass > $OUT/sass_$VAR.csv 2>&1
echo done
