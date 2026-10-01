#!/usr/bin/env bash
# ============================================================================
# Build one module candidate: both FFI legs, the sealed bundle, a source
# snapshot and a candidate modulefile, all beneath <candidate dir>.
#
#   LORRAX_MODULE_PREFIX=/abs/prefix bash build.sh <clean checkout> <candidate dir>
#
# Run on a zero-GPU compute step (the legs' verify gates load the libraries).
# Refusals: a checkout that is not a clean Git root at a full revision, a
# candidate path that is relative, exists, lies inside the checkout or inside
# $PREFIX/releases, and a missing venv.  Nothing under releases/ or
# modulefiles/ is touched: publish.sh does that after accept.sh.
#
# The legs are built by the clone route's scripts (../build_ffi_host.sh,
# ../build_ffi_cuda.sh) in a fresh clone of the revision whose .venv is the
# module venv, so they compile against the jaxlib they will run with.  SLATE
# and the cuSOLVERMp/cuBLASMp stage live in $PREFIX/build, built once.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/stack.sh"

SRC="$(realpath -e "${1:?usage: build.sh <clean checkout> <candidate dir>}")"
CAND="${2:?usage: build.sh <clean checkout> <candidate dir>}"
REV="$(lorrax_module_clean_rev "$SRC")"
[[ "$CAND" == /* ]] || lorrax_module_die "candidate dir must be absolute: $CAND"
CAND="$(realpath -m "$CAND")"
[[ -d "$(dirname "$CAND")" ]] || lorrax_module_die "candidate parent does not exist: $(dirname "$CAND")"
[[ ! -e "$CAND" && ! -L "$CAND" ]] || lorrax_module_die "candidate dir exists: $CAND (choose a fresh path)"
case "$CAND/" in
    "$SRC/"*) lorrax_module_die "candidate dir must be outside the checkout" ;;
    "$RELEASES/"*) lorrax_module_die "candidate dir must be outside $RELEASES; publish.sh publishes" ;;
esac
[[ -x "$VENV/bin/python" ]] || lorrax_module_die "no venv at $VENV; run setup_env.sh first"

echo "[build] revision=$REV candidate=$CAND"
mkdir -p "$CAND"
git clone --quiet --no-checkout --shared "$SRC" "$CAND/src"
git -C "$CAND/src" checkout --quiet --detach "$REV"
ln -s "$VENV" "$CAND/src/.venv"
export LORRAX_BUILD_PREFIX="$PREFIX/build"
# The module venv does not install lorrax; the legs' dynamic gates (GATE 8)
# import it from the clone being built.
export PYTHONPATH="$CAND/src/src"
SLATE_LIB="$LORRAX_BUILD_PREFIX/slate/cpu/install/lib64"
STAGE_LIB="$LORRAX_BUILD_PREFIX/cusolvermp-$CUSOLVERMP_VERSION/lib"

[[ -e "$SLATE_LIB/libslate.so.2" ]] \
    || bash "$CAND/src/src/ffi/cpp/stage/slate_build_perlmutter.sh" cpu
# The SLATE closure is preloaded by absolute path, so a stage built before its
# RPATH pinned LibSci maps a second MPI at run time (ffi_mpi.sh).
for lib in libblaspp.so.2 liblapackpp.so.2 libslate.so.2; do
    lorrax_pm_gate_private_lib "$SLATE_LIB/$lib" \
        || lorrax_module_die "$SLATE_LIB/$lib fails GATE 1; rebuild it: slate_build_perlmutter.sh cpu --fresh"
done
bash "$CAND/src/config/perlmutter/build_ffi_host.sh" --fresh
bash "$CAND/src/config/perlmutter/build_ffi_cuda.sh" --fresh

# Private closure, dependency-first (MPI, HDF5, CUDA and NCCL are the machine's).
"$VENV/bin/python" "$CAND/src/src/ffi/cpp/stage/seal_bundle.py" \
    --cuda "$CAND/src/src/ffi/cpp/build/liblorrax_ffi.so" \
    --host "$CAND/src/src/ffi/cpp/build_host/liblorrax_ffi_host.so" \
    --private-lib "$STAGE_LIB/libcublasmp.so.0" \
    --private-lib "$STAGE_LIB/libcusolverMp.so.0" \
    --private-lib "$SLATE_LIB/libblaspp.so.2" \
    --private-lib "$SLATE_LIB/liblapackpp.so.2" \
    --private-lib "$SLATE_LIB/libslate.so.2" \
    --output "$CAND/bundle"

mkdir -p "$CAND/source"
git -C "$CAND/src" archive --format=tar "$REV" | tar -x -C "$CAND/source"
echo "$REV" > "$CAND/source/SOURCE_COMMIT"

BUNDLE_ID="$("$VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundle_id"])' \
    "$CAND/bundle/lorrax_ffi_bundle.json")"
cat > "$CAND/CANDIDATE" <<EOF
REV=$REV
BUNDLE_ID=$BUNDLE_ID
MODULE_NAME=$MODULE_NAME
MODULE_VERSION=$MODULE_VERSION
EXPECT='{"jax": "$JAX_VERSION", "jaxlib": "$JAX_VERSION", "jax-cuda13-plugin": "$JAX_VERSION", "jax-cuda13-pjrt": "$JAX_VERSION", "nvidia-cudnn-cu13": "$CUDNN_VERSION", "nvidia-mathdx": "$MATHDX_VERSION"}'
EOF
bash "$HERE/install_module.sh" "$CAND/source" "$CAND/bundle" \
    "$CAND/modulefiles/$MODULE_NAME/$MODULE_VERSION.lua"
echo "[build] candidate ready: $CAND (bundle ${BUNDLE_ID:0:12}); next: accept.sh"
