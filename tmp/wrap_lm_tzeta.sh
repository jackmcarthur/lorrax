#!/bin/bash
# Wrapper for `lx run`: pins PYTHONPATH to THIS worktree's src/ + services/*/src
# (never the sibling lorrax_A checkout) and the FFI .so, before exec'ing python3.
# See KNOWN_SANDBOX_ERRORS.md "retarget_pythonpath" row.
set -euo pipefail
WT=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/wt_lm_tzeta
export LORRAX_FFI_SO=/pscratch/sd/j/jackm/lorrax_cuda13_module_20260814/src/ffi/cpp/build_cuda13_phdf5_streamfix/liblorrax_ffi.so
export PYTHONPATH="$WT/src:$WT/services/distrib_la/src:$WT/services/lxkit/src:$WT/services/minimax/src:$WT/services/symmetry_maps/src:$WT/services/vcoul/src:$WT/services/wfn_loader/src:$WT/services/zeta_loader/src${PYTHONPATH:+:$PYTHONPATH}"
# KNOWN_SANDBOX_ERRORS.md: the CUDA FFI .so needs libfabric.so.1, missing
# from the container's default LD_LIBRARY_PATH (single-node only -- do not
# reuse this wrapper for -N>1, see the 2026-08-23 cross-node-hang row).
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/opt/cray/libfabric/1.22.0/lib64"
# FORCE, not default-if-unset: lx run's own container env already sets
# JAX_PLATFORMS=cuda, which would win over a ${VAR:-...} default and starve
# jax.io_callback sites (isdf/core.py) of a CPU device (KNOWN_SANDBOX_ERRORS.md).
export JAX_PLATFORMS="cuda,cpu"
exec python3 -u "$@"
