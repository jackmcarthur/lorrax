#!/bin/bash
set -e
WT=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/wt_pt_metalgate
export PYTHONPATH="$WT/src:$WT/services/distrib_la/src:$WT/services/lxkit/src:$WT/services/minimax/src:$WT/services/symmetry_maps/src:$WT/services/vcoul/src:$WT/services/wfn_loader/src:$WT/services/zeta_loader/src"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/cray/libfabric/1.22.0/lib64"
export LORRAX_FFI_SO=/pscratch/sd/j/jackm/lorrax_cuda13_module_20260814/src/ffi/cpp/build_cuda13_phdf5_streamfix/liblorrax_ffi.so
export JAX_PLATFORMS=cuda,cpu
exec python3 -u "$@"
