#!/bin/bash
# MPA WINDOW-GROUP FARM lane (2026-08-10) -- the L1 implementation.
#
# ALLOCATOR REGIME: BFC@0.85 on every timing this lane reports, which is
# the campaign default and what the 26.4-minute pole-farm baseline was
# taken under.  The fleet modulefile exports
# XLA_PYTHON_CLIENT_ALLOCATOR=platform in the srun --env list, so `env -u`
# inside the container is the only place it can be unset.
#
# PLACEMENT: -G=1 one-GPU legs, FOUR PER NODE.  That is the four-GPU rule
# as it applies to a process farm: the node is filled, and the 2026-08-10
# measurement established that packing four one-GPU legs onto a node costs
# nothing measurable on this path (pass legs reproduced the whole-node
# walls to within 2 %).  CUDA_VISIBLE_DEVICES is NOT set; lx passes
# LORRAX_GPU_DEVICE and select_gpu.sh honours it.
set -u
export BASE=/pscratch/sd/j/jackm/mpa_winfarm_0810
export WT=$BASE/wt
export LORRAX_CHECKOUT=$WT
export LX_BASE_MODULE=lorrax_J070
export FFI_DEV=/pscratch/sd/j/jackm/merge_ckpt_2026-08-08/build_dev/liblorrax_ffi.so
export FFI_HOST=/pscratch/sd/j/jackm/merge_ckpt_2026-08-08/build_host/liblorrax_ffi_host.so

# READ-ONLY, other lanes' products.
export PROD=/pscratch/sd/j/jackm/mpa_wcprod_0809
export FIT_NP8=$PROD/stores/mpa_fit_np8_wc.h5      # the n_p=8 pole field
export DECK_TMPL=$PROD/scripts/deck_np8.in
export SEED=$PROD/seed/zeta_q.h5

export CENSUS=$BASE/census
export PARTIALS=$BASE/partials
export POLEPARTS=$BASE/partials_pole
export REPORTS=$BASE/_reports
mkdir -p "$CENSUS" "$PARTIALS" "$POLEPARTS" "$REPORTS" "$BASE/runs"

bfc_env() {
  echo "env -u XLA_PYTHON_CLIENT_ALLOCATOR -u TF_GPU_ALLOCATOR XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 JAX_ENABLE_X64=1 LORRAX_FFI_SO=$FFI_DEV LORRAX_FFI_HOST_SO=$FFI_HOST"
}

# A run directory with the one ABSOLUTE WFN path and the frozen zeta seed.
# A per-run WFN symlink makes the zeta-reuse string differ per leg, every
# leg re-fits, and the re-fit's 16.81 GiB allocation OOMs its co-tenants.
mkrun() {
  local RUN=$1
  local SRC=/pscratch/sd/j/jackm/si_bigcond_prep
  mkdir -p "$RUN/tmp"; cd "$RUN"
  [ -e centroids_frac_1128.txt ] || ln -s $SRC/centroids_frac_1128.txt centroids_frac_1128.txt
  [ -e eps0mat.h5 ] || ln -s /pscratch/sd/j/jackm/si_gnppm_0809/bgw_gn/eps0mat.h5 eps0mat.h5
  [ -e kin_ion.h5 ] || cp /pscratch/sd/j/jackm/mpa_closer_0809/gnppm/kin_ion.h5 kin_ion.h5
  [ -e tmp/zeta_q.h5 ] || cp $SEED tmp/zeta_q.h5
}

# The deck, with the farm keys appended rather than substituted -- the
# production template predates them, and appending keeps this lane's deck
# byte-identical to the production one everywhere else.
mkdeck() {                      # mkdeck <tag> <pole_subset> <partial_out>
  local TAG=$1 POLES=$2 POUT=$3
  sed -e "s|MPA_FIT_FILE|$FIT_NP8|" \
      -e "s|MPA_HEAD_LABEL|as_shipped|" \
      -e "s|MPA_POLE_SUBSET|$POLES|" \
      -e "s|MPA_PASS_PARTIAL_OUT|$POUT|" \
      -e "s|MPA_PASS_PARTIAL_IN||" \
      -e "s|SIGMA_DIAG_FILE|PARTIAL_NOT_A_SELF_ENERGY_${TAG}.dat|" \
      $DECK_TMPL > deck.in
}
