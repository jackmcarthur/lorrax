#!/bin/bash
# usage: rebuild_at.sh SHA -- move lorrax_main_1027 to SHA, uv sync, build the FFI pair on an
# interactive m4598 CPU node (never m2651_g), release it, then record lorrax_head.txt.
set -uo pipefail
SHA=$1; C=/global/cfs/cdirs/m4598/jackm/lorrax_main_1027; R=/global/cfs/cdirs/m4598/jackm/mos2_hbn/4-gw60k
export UV_CACHE_DIR=/global/cfs/cdirs/m4598/jackm/.cache/uv UV_PYTHON_INSTALL_DIR=/global/cfs/cdirs/m4598/jackm/.local/uv-python PATH=$HOME/.local/bin:$PATH
cd $C && git fetch -q origin && git checkout -q $SHA && [ "$(git rev-parse HEAD)" = "$(git rev-parse $SHA)" ] || { echo CHECKOUT_FAILED; exit 1; }
git log --oneline -1; uv sync --extra cuda13 2>&1 | tail -1
salloc --no-shell -q interactive -A m4598 -C cpu -N 1 -t 00:30:00 -J mos2hbn_ffi 2>&1 | tail -1
J=$(squeue -u $USER -n mos2hbn_ffi -h -o %i | head -1); [ -n "$J" ] || { echo NO_ALLOC; exit 1; }
trap 'scancel $J; echo RELEASED $J' EXIT
s=$(date +%s)
srun --jobid=$J -N 1 -n 1 -c 128 bash -c "cd $C && bash config/perlmutter/build_ffi_host.sh && bash config/perlmutter/build_ffi_cuda.sh" > $R/ffi_build_$(git -C $C rev-parse --short HEAD).out 2>&1
rc=$?; echo "ffi_build rc=$rc wall=$(( $(date +%s)-s ))s"; grep -a -E "VERIFY PASSED|GATE 11" $R/ffi_build_*$(git -C $C rev-parse --short HEAD).out | tail -2
[ $rc = 0 ] && git -C $C rev-parse HEAD > $R/lorrax_head.txt && echo "HEAD_RECORDED $(cat $R/lorrax_head.txt)"
