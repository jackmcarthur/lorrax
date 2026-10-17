#!/bin/bash
# every rank, cwd = checkout.  usage: leg_wresp.sh LIB LEGDIR DECKDIR   (stops after SC map 0's W response)
set -u
LIB=$1; L=$2; DECK=$3
E=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/kc_m11
TREE=$PWD; RK=$(printf %04d ${SLURM_PROCID:-0})
if [ "$RK" = 0000 ]; then
  mkdir -p "$L"; cp -a "$DECK/." "$L/"
  sed -i "s/^sc_max_iter = .*/sc_max_iter = 1/" "$L/cohsex.in"
  echo "LEG tree=$TREE head=$(git -C $TREE rev-parse HEAD) dirty=$(git -C $TREE status --porcelain -uno | wc -l) lib=$LIB begin=$(date -u +%FT%TZ)" > "$L/leg.launch.txt"
  touch "$L/.ready"
fi
n=0; while [ ! -f "$L/.ready" ]; do sleep 1; n=$((n+1)); [ $n -gt 300 ] && exit 92; done
source $TREE/config/perlmutter/gpu_env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_THREAD_TIMEOUT=1
export LORRAX_FFI_SO=$E/$LIB/lib/liblorrax_ffi.so LORRAX_FFI_HOST_SO=$E/$LIB/lib/liblorrax_ffi_host.so
cd "$L" || exit 90
env | grep -E "^(LORRAX|JAX_)" | sort > env.rank$RK.txt
python3 -u -m gw.gw_jax -i cohsex.in > gw.rank$RK.log 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  if grep -q "<- SC W response" gw.rank0000.log 2>/dev/null; then sleep 3; kill $PID 2>/dev/null; break; fi
  sleep 2
done
wait $PID; rc=$?
[ "$RK" = 0000 ] && echo "LEG_END rc=$rc end=$(date -u +%FT%TZ)" >> "$L/leg.launch.txt"
exit 0
