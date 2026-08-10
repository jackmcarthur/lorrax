#!/bin/bash
# MPA 16-GPU FARM MEASUREMENT lane (2026-08-10) -- measurement only.
#
# ALLOCATOR REGIME: BFC@0.85 on every timing this lane reports.  The fleet
# modulefile exports XLA_PYTHON_CLIENT_ALLOCATOR=platform in the srun --env
# list, so `env -u` inside the container is the only place it can be unset.
#
# PLACEMENT: -G=1 ONE-GPU LEGS, which is the fleet-wide one-leg placement
# (`lx doctor` -> "one-GPU placement: on").  This is the difference from the
# 2026-08-09 production scripts, which predate that fix and reserved a WHOLE
# node per leg (`-G=4 -n=1` + CUDA_VISIBLE_DEVICES=0) to dodge the
# SLURM_LOCALID collision.  Four one-GPU legs per node is the packing this
# lane exists to measure, so CUDA_VISIBLE_DEVICES is NOT set here: lx passes
# LORRAX_GPU_DEVICE and select_gpu.sh honours it.
set -u
export BASE=/pscratch/sd/j/jackm/mpa_farm16_0810
export WT=$BASE/wt
export LORRAX_CHECKOUT=$WT
export LX_BASE_MODULE=lorrax_J070
export FFI_DEV=/pscratch/sd/j/jackm/merge_ckpt_2026-08-08/build_dev/liblorrax_ffi.so
export FFI_HOST=/pscratch/sd/j/jackm/merge_ckpt_2026-08-08/build_host/liblorrax_ffi_host.so

# READ-ONLY, other lane's products.
export PROD=/pscratch/sd/j/jackm/mpa_wcprod_0809
export FIT_NP8=$PROD/stores/mpa_fit_np8_wc.h5      # the n_p=8 pole field
export WC_STORE=$PROD/stores/W_omega_full_wc.h5    # W - v samples, 'W_c'
export WC_NAME=W_qmunu_omega
export SEED=$PROD/seed/zeta_q.h5

export PARTIALS=$BASE/partials
export REPORTS=$BASE/_reports
export STORES=$BASE/stores
mkdir -p "$PARTIALS" "$REPORTS" "$STORES" "$BASE/runs"

bfc_env() {
  echo "env -u XLA_PYTHON_CLIENT_ALLOCATOR -u TF_GPU_ALLOCATOR XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 JAX_ENABLE_X64=1 LORRAX_FFI_SO=$FFI_DEV LORRAX_FFI_HOST_SO=$FFI_HOST"
}

# A run directory with the one ABSOLUTE WFN path and the frozen zeta seed --
# a per-run WFN symlink makes the zeta-reuse string differ per leg, every leg
# re-fits, and the re-fit's 16.81 GiB allocation OOMs its co-tenants.
mkrun() {
  local RUN=$1
  local SRC=/pscratch/sd/j/jackm/si_bigcond_prep
  mkdir -p "$RUN/tmp"; cd "$RUN"
  [ -e centroids_frac_1128.txt ] || ln -s $SRC/centroids_frac_1128.txt centroids_frac_1128.txt
  [ -e eps0mat.h5 ] || ln -s /pscratch/sd/j/jackm/si_gnppm_0809/bgw_gn/eps0mat.h5 eps0mat.h5
  [ -e kin_ion.h5 ] || cp /pscratch/sd/j/jackm/mpa_closer_0809/gnppm/kin_ion.h5 kin_ion.h5
  [ -e tmp/zeta_q.h5 ] || cp $SEED tmp/zeta_q.h5
  echo "[mkrun] $RUN ready"
}
