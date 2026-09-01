#!/usr/bin/env bash
set -euo pipefail

arm=${1:?arm}
base=/pscratch/sd/j/jackm/wt_regress2_2026-08-31
checkout="$base/candidates/$arm"
probe="$base/evidence/regress2/$arm"
variant=/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829
runtime=/pscratch/sd/j/jackm/wt_hybrid_wiring_2026-08-29
python="$runtime/.venv/bin/python"
site="$runtime/.venv/lib/python3.12/site-packages"
native_ld="$runtime/.native/cusolvermp-0.9.1_cuda13.2/lib:$site/nvidia/cu13/lib:$site/nvidia/nccl/lib:$site/nvidia/cudnn/lib:$site/nvidia/nvshmem/lib:$site/nvidia/cublasmp/cu13/lib:/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.3/lib:/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/lib:/opt/cray/libfabric/1.22.0/lib64:/opt/cray/pe/lib64"

cd "$probe"
exec env -u XLA_PYTHON_CLIENT_ALLOCATOR \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    LORRAX_BAND_DEGENERACY=snap \
    LORRAX_CHECKOUT="$checkout" \
    PYTHONPATH="$checkout/src:$checkout/services/distrib_la/src:$checkout/services/minimax/src:$checkout/services/symmetry_maps/src:$checkout/services/wfn_loader/src" \
    LD_LIBRARY_PATH="$native_ld:${LD_LIBRARY_PATH:-}" \
    LORRAX_FFI_SO="$runtime/src/ffi/cpp/build_cuda13_executor_20260829/liblorrax_ffi.so" \
    LORRAX_SIGMA_PLAN=delivered \
    LORRAX_DELIVERED_TAU_GRID=free \
    LORRAX_DELIVERED_PLAN_CACHE="$probe/tmp/delivered_sigma_plan_v1.pkl" \
    LORRAX_CERTIFIED_MPA_FIT="$probe/input_mpa_fit.h5" \
    LORRAX_RUN_RECEIPT="$probe/job_receipt.txt" \
    LORRAX_ATTEMPT_WALL_SECONDS=900 \
    /bin/bash "$variant/run_rank.sh" \
    "$python" -u "$variant/run_existing_fit.py"
