#!/usr/bin/env bash
# Create the Python half of the local CUDA 13 lane, stage the
# cuSOLVERMp/cuBLASMp headers and DSOs into one CMake prefix, and build
# parallel HDF5 against the distro OpenMPI.
#
# Prerequisites (one apt line, run it yourself — this script refuses rather
# than sudo-ing):
#   apt-get install -y build-essential gfortran git curl libopenmpi-dev \
#                      pkg-config python3.12-venv
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/stack.sh"

for tool in gcc g++ mpicc curl; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "[local setup] REFUSED: $tool is not on PATH (see apt line in header)." >&2
        exit 2
    fi
done

UV="${UV:-$(command -v uv || true)}"
if [[ -z "$UV" ]]; then
    echo "[local setup] uv not found; installing to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$HOME/.local/bin/uv"
fi

if [[ ! -x "$LORRAX_LOCAL_ENV/bin/python" ]]; then
    "$UV" venv --python 3.12 "$LORRAX_LOCAL_ENV"
fi

echo "[local setup] installing LORRAX and JAX $LORRAX_LOCAL_JAX_VERSION (full pip CUDA closure)"
"$UV" pip install --python "$LORRAX_LOCAL_ENV/bin/python" \
    -e "$LORRAX_LOCAL_ROOT" \
    "pytest>=8.0.0" cmake ninja \
    "jax[cuda13]==$LORRAX_LOCAL_JAX_VERSION" \
    nvidia-nvtx

# The Mp libraries (and the header-only nvidia-mathdx) version independently of
# the toolkit; install exactly the pinned set and nothing else (their dependency closures are already
# satisfied by the jax[cuda13] install above).
"$UV" pip install --python "$LORRAX_LOCAL_ENV/bin/python" --no-deps \
    "nvidia-cusolvermp-cu13==$LORRAX_LOCAL_CUSOLVERMP_VERSION" \
    "nvidia-cublasmp-cu13==$LORRAX_LOCAL_CUBLASMP_VERSION" \
    "nvidia-mathdx==$LORRAX_LOCAL_MATHDX_VERSION"

SITE="$(lorrax_local_site_packages)"
CU13="$(lorrax_local_cuda_root)"
CUSOLVERMP_ROOT="$SITE/nvidia/cu13"
CUBLASMP_ROOT="$SITE/nvidia/cublasmp/cu13"
for path in "$CU13/bin/nvcc" \
            "$CU13/include/cuda_runtime.h" \
            "$SITE/nvidia/mathdx/include/cufftdx.hpp" \
            "$CUSOLVERMP_ROOT/include/cusolverMp.h" \
            "$CUSOLVERMP_ROOT/lib/libcusolverMp.so.0" \
            "$CUBLASMP_ROOT/include/cublasmp.h" \
            "$CUBLASMP_ROOT/lib/libcublasmp.so.0"; do
    if [[ ! -e "$path" ]]; then
        echo "[local setup] REFUSED: vendor wheel did not supply $path" >&2
        exit 2
    fi
done

mkdir -p "$LORRAX_LOCAL_STAGE/include" "$LORRAX_LOCAL_STAGE/lib"
ln -sfn "$CUSOLVERMP_ROOT/include/cusolverMp.h" "$LORRAX_LOCAL_STAGE/include/cusolverMp.h"
ln -sfn "$CUBLASMP_ROOT/include/cublasmp.h" "$LORRAX_LOCAL_STAGE/include/cublasmp.h"
ln -sfn "$CUSOLVERMP_ROOT/lib/libcusolverMp.so.0" "$LORRAX_LOCAL_STAGE/lib/libcusolverMp.so.0"
ln -sfn "$LORRAX_LOCAL_STAGE/lib/libcusolverMp.so.0" "$LORRAX_LOCAL_STAGE/lib/libcusolverMp.so"
ln -sfn "$CUBLASMP_ROOT/lib/libcublasmp.so.0" "$LORRAX_LOCAL_STAGE/lib/libcublasmp.so.0"
ln -sfn "$LORRAX_LOCAL_STAGE/lib/libcublasmp.so.0" "$LORRAX_LOCAL_STAGE/lib/libcublasmp.so"

# The wheels ship only versioned .so.N; CMake's find_library needs bare .so
# aliases inside the toolkit root (same trick as config/frontera/
# stage_ffi_deps.sh, one directory instead of nine).
for so in "$CU13"/lib/*.so.*; do
    [[ -e "$so" ]] || continue
    base="$(basename "$so")"
    bare="${base%%.so*}.so"
    [[ -e "$CU13/lib/$bare" ]] || ln -sfn "$so" "$CU13/lib/$bare"
done

# ---------------------------------------------------------------------------
# Parallel HDF5 from source, against the distro OpenMPI.  Ubuntu's packaged
# libhdf5-openmpi-dev is 1.10.x — below the 1.12 floor (docs/installation/ffi-build.md).
# ---------------------------------------------------------------------------
if [[ ! -e "$LORRAX_LOCAL_PHDF5/lib/libhdf5.so" ]]; then
    echo "[local setup] building parallel HDF5 $LORRAX_LOCAL_PHDF5_VERSION (once, ~5 min)"
    mkdir -p "$LORRAX_LOCAL_NATIVE"
    tarball="$LORRAX_LOCAL_NATIVE/hdf5-$LORRAX_LOCAL_PHDF5_VERSION.tar.gz"
    [[ -e "$tarball" ]] || curl -sL -o "$tarball" \
        "https://github.com/HDFGroup/hdf5/releases/download/hdf5_$LORRAX_LOCAL_PHDF5_VERSION/hdf5-$LORRAX_LOCAL_PHDF5_VERSION.tar.gz"
    tar xzf "$tarball" -C "$LORRAX_LOCAL_NATIVE"
    (
        cd "$LORRAX_LOCAL_NATIVE/hdf5-$LORRAX_LOCAL_PHDF5_VERSION"
        CC=mpicc ./configure --enable-parallel --disable-fortran --disable-cpp \
            --prefix="$LORRAX_LOCAL_PHDF5" > configure.log 2>&1
        make -j"$(nproc)" > build.log 2>&1
        make install > install.log 2>&1
    )
fi
[[ -e "$LORRAX_LOCAL_PHDF5/lib/libhdf5.so" ]] || {
    echo "[local setup] REFUSED: parallel HDF5 build produced no libhdf5.so" >&2
    exit 2
}

echo "[local setup] environment:  $LORRAX_LOCAL_ENV"
echo "[local setup] cuda root:    $CU13"
echo "[local setup] native stage: $LORRAX_LOCAL_STAGE"
echo "[local setup] phdf5:        $LORRAX_LOCAL_PHDF5"
"$LORRAX_LOCAL_ENV/bin/python" - <<'PY'
import jax
import jaxlib
print(f"[local setup] jax={jax.__version__} jaxlib={jaxlib.__version__}")
PY
