#!/bin/bash
# every rank, cwd = checkout.  usage: leg_hsuite.sh LIB OUTDIR
set -u
LIB=$1; OUT=$2
E=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/kc_m11
source $PWD/config/perlmutter/gpu_env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_THREAD_TIMEOUT=1
export LORRAX_FFI_SO=$E/$LIB/lib/liblorrax_ffi.so LORRAX_FFI_HOST_SO=$E/$LIB/lib/liblorrax_ffi_host.so
python3 -u -m tests.hsuite.chain --out $OUT
