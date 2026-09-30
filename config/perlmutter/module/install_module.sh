#!/usr/bin/env bash
# ============================================================================
# Render module.lua.in for one source snapshot and one sealed bundle.
#
#   LORRAX_MODULE_PREFIX=/abs/prefix \
#     bash install_module.sh <source snapshot> <sealed bundle> <modulefile .lua>
#
# build.sh renders the candidate module; publish.sh renders the published
# one.  The file is written beside its target and renamed into place, and an
# existing file is kept as <file>.before-<rev8>.  Library paths are resolved
# here, on the machine, from the toolkit module, ../ffi_mpi.sh and the MPI
# library's own libfabric dependency; nothing is copied from another module.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/stack.sh"

usage="usage: install_module.sh <source snapshot> <sealed bundle> <modulefile .lua>"
ROOT="$(realpath -e "${1:?$usage}")"
BUNDLE="$(realpath -e "${2:?$usage}")"
OUT="${3:?$usage}"
[[ "$OUT" == /*.lua ]] || lorrax_module_die "modulefile must be an absolute .lua path: $OUT"
[[ -r "$ROOT/SOURCE_COMMIT" && -r "$ROOT/src/runtime/__init__.py" ]] \
    || lorrax_module_die "$ROOT is not a source snapshot (SOURCE_COMMIT, src/runtime)"
[[ ! -e "$ROOT/.git" ]] || lorrax_module_die "$ROOT is a checkout; the module runs a git-archive snapshot"
for f in lorrax_ffi_bundle.json lib/liblorrax_ffi.so lib/liblorrax_ffi_host.so; do
    [[ -e "$BUNDLE/$f" ]] || lorrax_module_die "$BUNDLE is not a sealed bundle (no $f)"
done
REV="$(cat "$ROOT/SOURCE_COMMIT")"
BUNDLE_ID="$("$VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundle_id"])' \
    "$BUNDLE/lorrax_ffi_bundle.json")"

lorrax_module_load_toolkit
SITE="$(lorrax_module_site_packages)"
FABRIC_LIB="$(ldd "$LORRAX_PM_MPI_LIBRARY" | awk '/libfabric/ {print $3; exit}')"
[[ -n "$FABRIC_LIB" ]] || lorrax_module_die "cannot resolve libfabric from $LORRAX_PM_MPI_LIBRARY"
COMPAT="/usr/local/cuda-$CUDA_VER/compat"
LD_DIRS=(
    "$SITE/nvidia/cudnn/lib"
    "$CUDA/lib64" "$CUDA/nvvm/lib64" "$CUDA/extras/CUPTI/lib64"
    "$SDK/math_libs/$CUDA_VER/lib64" "$SDK/comm_libs/$CUDA_VER/nccl/lib"
)
[[ -d "$COMPAT" ]] && LD_DIRS+=("$COMPAT")
LD_DIRS+=("$PHDF5_ROOT/lib" "$LORRAX_PM_MPICH_ROOT/lib" "$(dirname "$FABRIC_LIB")" /opt/cray/pe/lib64)
for d in "${LD_DIRS[@]}"; do [[ -d "$d" ]] || lorrax_module_die "missing library dir $d"; done
LD_PATH="$(IFS=:; echo "${LD_DIRS[*]}")"
PY_DIRS=("$ROOT/src")
for d in "$ROOT"/services/*/src; do [[ -d "$d" ]] && PY_DIRS+=("$d"); done
PYTHONPATH_LIST="$(IFS=:; echo "${PY_DIRS[*]}")"

mkdir -p "$(dirname "$OUT")"
TMP="$OUT.tmp.$$"
sed \
    -e "s|@REV8@|${REV:0:8}|g" \
    -e "s|@BUNDLE12@|${BUNDLE_ID:0:12}|g" \
    -e "s|@JAX@|$JAX_VERSION|g" \
    -e "s|@TOOLKIT_MODULE@|$TOOLKIT_MODULE|g" \
    -e "s|@MPICH_MODULE@|$LORRAX_PM_MPICH_MODULE|g" \
    -e "s|@ROOT@|$ROOT|g" \
    -e "s|@VENV@|$VENV|g" \
    -e "s|@CUDA@|$CUDA|g" \
    -e "s|@FFI_CUDA@|$BUNDLE/lib/liblorrax_ffi.so|g" \
    -e "s|@FFI_HOST@|$BUNDLE/lib/liblorrax_ffi_host.so|g" \
    -e "s|@PYTHONPATH@|$PYTHONPATH_LIST|g" \
    -e "s|@LD_PATH@|$LD_PATH|g" \
    "$HERE/module.lua.in" > "$TMP"
if grep -n '@[A-Z0-9_]*@' "$TMP"; then rm -f "$TMP"; lorrax_module_die "unrendered placeholder"; fi
[[ -e "$OUT" ]] && cp -p "$OUT" "$OUT.before-${REV:0:8}"
mv -f "$TMP" "$OUT"
echo "[install_module] $OUT -> source ${REV:0:8}, bundle ${BUNDLE_ID:0:12}"
