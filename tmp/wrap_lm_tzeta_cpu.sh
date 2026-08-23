#!/bin/bash
# CPU-only sibling of wrap_lm_tzeta.sh -- for -G 0 steps where there is no
# CUDA plugin to select at all (KNOWN_SANDBOX_ERRORS.md: a -G0 step forcing
# JAX_PLATFORMS=cuda,cpu dies "Backend 'cuda' is not in the list of known
# backends" since no GPU means no CUDA PJRT plugin, not just no device).
set -euo pipefail
WT=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/wt_lm_tzeta
export LORRAX_FFI_SO=/pscratch/sd/j/jackm/lorrax_cuda13_module_20260814/src/ffi/cpp/build_cuda13_phdf5_streamfix/liblorrax_ffi.so
export PYTHONPATH="$WT/src:$WT/services/distrib_la/src:$WT/services/lxkit/src:$WT/services/minimax/src:$WT/services/symmetry_maps/src:$WT/services/vcoul/src:$WT/services/wfn_loader/src:$WT/services/zeta_loader/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS="cpu"
exec python3 -u "$@"
