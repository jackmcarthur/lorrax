#!/usr/bin/env bash
# Certify the built pair in a Frontier allocation with four available GCDs.
#   bash config/frontier/verify.sh
# Artifact gates, host FFTW identity, then native-vs-XLA GPU transforms.
set -euo pipefail
LORRAX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -ne 0 ]]; then
    echo "usage: bash config/frontier/verify.sh" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "[frontier verify] Run inside a Frontier allocation with four available GCDs." >&2
    exit 1
fi
for _control in LORRAX_FFI_VERIFY LORRAX_GATE_ONE_MPI LORRAX_GATE_ONE_HDF5 LORRAX_GATE_ONE_FFTW; do
    if [[ "${!_control:-on}" == off ]]; then
        echo "[frontier verify] $_control=off disables a required acceptance check." >&2
        exit 1
    fi
done
source "$LORRAX_ROOT/config/frontier/gpu_env.sh"
cd "$LORRAX_ROOT"

# The build's FFTW hint names the intended engine; no ROCm module or additional
# search path is needed. An existing explicit stage also supports copied legs
# whose original build cache is no longer present.
if [[ -z "${LORRAX_FFTW3_STAGE:-}" ]]; then
    _cache="$(dirname "$LORRAX_FFI_HOST_SO")/CMakeCache.txt"
    _fftw="$(sed -n 's/^LORRAX_FFTW3_LIBRARY:[^=]*=//p' "$_cache" 2>/dev/null || true)"
    if [[ ! -f "$_fftw" ]]; then
        echo "[frontier verify] No compiled FFTW hint in $_cache; set the existing LORRAX_FFTW3_STAGE to the intended FFTW prefix." >&2
        exit 1
    fi
    export LORRAX_FFTW3_STAGE="$(dirname "$(dirname "$_fftw")")"
fi
if [[ ! -f "$LORRAX_FFTW3_STAGE/lib/libfftw3.so" ]]; then
    echo "[frontier verify] $LORRAX_FFTW3_STAGE/lib/libfftw3.so is missing." >&2
    exit 1
fi
export LORRAX_GATE_FFTW_PY="$VIRTUAL_ENV/bin/python"
export LORRAX_FFI_VERIFY_STRICT=1
export LORRAX_FFI_EXPECT_MPI="$LORRAX_OLCF_MPI_SONAME"
export LORRAX_FFI_EXPECT_HDF5_SOVERSION=310
export LORRAX_PHDF5_STAGE="$LORRAX_OLCF_HDF5_ROOT"
_rocm_so="${LORRAX_FFI_ROCM_SO:-$LORRAX_ROOT/src/ffi/cpp/build_rocm/liblorrax_ffi_rocm.so}"

# Check each live closure independently: the generic verifier's optional peer
# argument deliberately disables its mapped-HDF5 check for cross-environment
# pairs. The explicit shared SOVERSION above still gates both legs here.
srun -N1 -n1 -c7 --threads-per-core=1 --cpu-bind=cores --exclusive \
    env -u LORRAX_FFI_EXPECT_PEER_SO JAX_PLATFORMS=cpu \
    LORRAX_FFI_EXPECT_BACKENDS=scalapack,gemm,phdf5,fft \
    bash "$LORRAX_ROOT/scripts/verify_ffi_build.sh" --leg host "$LORRAX_FFI_HOST_SO"
srun -N1 -n1 -c7 --threads-per-core=1 --cpu-bind=cores --exclusive \
    env -u LORRAX_FFI_EXPECT_PEER_SO LORRAX_FFI_EXPECT_BACKENDS=phdf5,elpa,kdft \
    bash "$LORRAX_ROOT/scripts/verify_ffi_build.sh" --leg rocm "$_rocm_so"
# ELPA: closure in an empty environment and a 64x64 eigensolve vs LAPACK (scripts/dep_check.py).
_elpa="$(sed -n 's/^LORRAX_ELPA_ROCM_DIR:[^=]*=//p' "$(dirname "$_rocm_so")/CMakeCache.txt" 2>/dev/null || true)"
if [[ -n "$_elpa" && -d "$_elpa" ]]; then
    srun -N1 -n1 -c7 --threads-per-core=1 --cpu-bind=cores --exclusive \
        "$VIRTUAL_ENV/bin/python" "$LORRAX_ROOT/scripts/dep_check.py" elpa "$_elpa" \
        --blas "$_elpa/lib/libopenblas.so.0" --solve
else
    echo "[frontier verify] FAILED ELPA: no LORRAX_ELPA_ROCM_DIR in $(dirname "$_rocm_so")/CMakeCache.txt -- fix: rebuild the rocm leg (config/frontier/build_ffi_rocm.sh)" >&2
    exit 1
fi
# The GPU request supplies Slurm's per_task binding, one GCD per rank.
srun -N1 -n4 -c7 --threads-per-core=1 --cpu-bind=cores --exclusive \
    --gpus-per-task=1 env JAX_PLATFORMS=rocm \
    "$VIRTUAL_ENV/bin/python" -u "$LORRAX_ROOT/tests/test_fourier_plan_xla_gate.py"
echo "[frontier verify] PASS: artifact checks, pinned host FFTW and P4 GPU transforms."
