#!/bin/bash
# MPA FIT-EFFICIENCY lane (2026-08-10), perf/mpa-fit-efficiency-2026-08-10.
#
# ALLOCATOR REGIME: BFC@0.85 on every timing this lane reports, the house
# default.  The fleet modulefile exports XLA_PYTHON_CLIENT_ALLOCATOR=platform
# in the srun --env list, so `env -u` inside the container is the only place
# it can be unset -- this is copied from scripts/perf_mpa16/env.sh so the
# fit walls this lane reports are comparable to the ones it is re-measuring.
#
# PLACEMENT: -G=1 one-GPU legs, four to a node (fleet-wide placement is on).
set -u
export BASE=/pscratch/sd/j/jackm/mpa_fitperf_0810
export WT=$BASE/wt
export LORRAX_CHECKOUT=$WT
export LX_BASE_MODULE=lorrax_J070
export FFI_DEV=/pscratch/sd/j/jackm/merge_ckpt_2026-08-08/build_dev/liblorrax_ffi.so
export FFI_HOST=/pscratch/sd/j/jackm/merge_ckpt_2026-08-08/build_host/liblorrax_ffi_host.so

# READ-ONLY, the production lane's products (same paths perf_mpa16 uses).
export PROD=/pscratch/sd/j/jackm/mpa_wcprod_0809
export WC_STORE=$PROD/stores/W_omega_full_wc.h5    # W - v samples, 'W_c'
export WC_NAME=W_qmunu_omega
export FIT_NP8=$PROD/stores/mpa_fit_np8_wc.h5      # the shipped n_p=8 field

export REPORTS=$BASE/_reports
export STORES=$BASE/stores
mkdir -p "$REPORTS" "$STORES" "$BASE/runs"

bfc_env() {
  echo "env -u XLA_PYTHON_CLIENT_ALLOCATOR -u TF_GPU_ALLOCATOR XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 JAX_ENABLE_X64=1 LORRAX_FFI_SO=$FFI_DEV LORRAX_FFI_HOST_SO=$FFI_HOST"
}
