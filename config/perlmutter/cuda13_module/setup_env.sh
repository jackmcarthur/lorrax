#!/usr/bin/env bash
# Create the Python half of the Perlmutter CUDA 13.2 module lane and stage
# the newest cuSOLVERMp/cuBLASMp headers and DSOs into one CMake prefix.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/stack.sh"
lorrax_cuda13_load_runtime_modules

for path in "$LORRAX_CUDA13_CUDA/bin/nvcc" \
            "$LORRAX_CUDA13_MATH/lib64/libcusolver.so" \
            "$LORRAX_CUDA13_COMM/nccl/lib/libnccl.so"; do
    if [[ ! -e "$path" ]]; then
        echo "[cuda13 setup] REFUSED: required module file is absent: $path" >&2
        exit 2
    fi
done

UV="${UV:-$(command -v uv || true)}"
if [[ -z "$UV" ]]; then
    echo "[cuda13 setup] REFUSED: uv is not on PATH." >&2
    exit 2
fi

if [[ ! -x "$LORRAX_CUDA13_ENV/bin/python" ]]; then
    "$UV" venv --python "$(command -v python)" "$LORRAX_CUDA13_ENV"
fi

echo "[cuda13 setup] installing LORRAX and JAX $LORRAX_CUDA13_JAX_VERSION (local CUDA)"
"$UV" pip install --python "$LORRAX_CUDA13_ENV/bin/python" \
    -e "$LORRAX_CUDA13_ROOT" \
    "pytest>=8.0.0" \
    "jax[cuda13-local]==$LORRAX_CUDA13_JAX_VERSION"

# JAX's local-CUDA extra deliberately does not install a CUDA runtime.  NERSC
# supplies that through cudatoolkit/13.2.  The one missing compatible piece is
# cuDNN: the NERSC cuDNN modules currently target CUDA 12, so install only its
# DSO wheel, never its dependency closure (which would duplicate the toolkit).
"$UV" pip install --python "$LORRAX_CUDA13_ENV/bin/python" --no-deps \
    "nvidia-cudnn-cu13==$LORRAX_CUDA13_CUDNN_VERSION" \
    "nvidia-cusolvermp-cu13==$LORRAX_CUDA13_CUSOLVERMP_VERSION" \
    "nvidia-cublasmp-cu13==$LORRAX_CUDA13_CUBLASMP_VERSION"

SITE="$(lorrax_cuda13_site_packages)"
CUSOLVERMP_ROOT="$SITE/nvidia/cu13"
CUBLASMP_ROOT="$SITE/nvidia/cublasmp/cu13"
for path in "$CUSOLVERMP_ROOT/include/cusolverMp.h" \
            "$CUSOLVERMP_ROOT/lib/libcusolverMp.so.0" \
            "$CUBLASMP_ROOT/include/cublasmp.h" \
            "$CUBLASMP_ROOT/lib/libcublasmp.so.0"; do
    if [[ ! -e "$path" ]]; then
        echo "[cuda13 setup] REFUSED: vendor wheel did not supply $path" >&2
        exit 2
    fi
done

mkdir -p "$LORRAX_CUDA13_STAGE/include" "$LORRAX_CUDA13_STAGE/lib"
ln -sfn "$CUSOLVERMP_ROOT/include/cusolverMp.h" "$LORRAX_CUDA13_STAGE/include/cusolverMp.h"
ln -sfn "$CUBLASMP_ROOT/include/cublasmp.h" "$LORRAX_CUDA13_STAGE/include/cublasmp.h"
ln -sfn "$CUSOLVERMP_ROOT/lib/libcusolverMp.so.0" "$LORRAX_CUDA13_STAGE/lib/libcusolverMp.so.0"
ln -sfn "$LORRAX_CUDA13_STAGE/lib/libcusolverMp.so.0" "$LORRAX_CUDA13_STAGE/lib/libcusolverMp.so"
ln -sfn "$CUBLASMP_ROOT/lib/libcublasmp.so.0" "$LORRAX_CUDA13_STAGE/lib/libcublasmp.so.0"
ln -sfn "$LORRAX_CUDA13_STAGE/lib/libcublasmp.so.0" "$LORRAX_CUDA13_STAGE/lib/libcublasmp.so"

echo "[cuda13 setup] environment: $LORRAX_CUDA13_ENV"
echo "[cuda13 setup] native stage: $LORRAX_CUDA13_STAGE"
echo "[cuda13 setup] toolkit:      $LORRAX_CUDA13_CUDA"
"$LORRAX_CUDA13_ENV/bin/python" - <<'PY'
import jax
import jaxlib
print(f"[cuda13 setup] jax={jax.__version__} jaxlib={jaxlib.__version__}")
PY
