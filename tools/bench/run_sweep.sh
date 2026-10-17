#!/bin/bash
# one rank, cwd = checkout.  usage: run_sweep.sh VAR OUTDIR CASES...   (case = name:nx,ny,nz:mx:spin[:tables])
set -u
VAR=$1; OUT=$2; shift 2
KB=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/kc_occ_build
source $PWD/config/perlmutter/gpu_env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_THREAD_TIMEOUT=1
export LORRAX_FFI_SO=$KB/$VAR/lib/liblorrax_ffi.so LORRAX_FFI_HOST_SO=$KB/$VAR/lib/liblorrax_ffi_host.so
for d in $PWD/services/*/src; do export PYTHONPATH=$PYTHONPATH:$d; done
mkdir -p $OUT
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv > $OUT/gpu.$VAR.txt 2>&1
for cs in "$@"; do
  IFS=: read -r name grid mx spin tab nsv <<< "$cs"
  g=$(echo $grid | tr , ' ')
  extra=""; [ -n "${tab:-}" ] && extra="--tables $tab"; [ -n "${nsv:-}" ] && extra="$extra --ns $nsv"
  SAVE="--save $OUT/$VAR.$name.npz"; [ "${KC_NOSAVE:-0}" = 1 ] && SAVE=""
  ${KC_PREFIX:-} python3 -u $KB/tools/kconv_bench.py --grid $g --mx $mx --my 640 --spin $spin $extra \
      --reps ${KC_REPS:-30} --mode9 ${KC_M9:-1} $SAVE --json $OUT/$VAR.jsonl > $OUT/$VAR.$name.log 2>&1
  echo "$VAR $name rc=$? $(grep -o 'KCB.*' $OUT/$VAR.$name.log | head -c 400)"
done
