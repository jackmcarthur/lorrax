#!/usr/bin/env bash
# Install the CUDA 13 bare-host lane as a personal Lmod module.  This script
# deliberately does not edit ~/.bashrc; Perlmutter already searches
# $HOME/modulefiles, and an unusual installation can use `module use`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/stack.sh"

MODULE_NAME="${LORRAX_CUDA13_MODULE_NAME:-lorrax_C13}"
MODULE_VERSION="${LORRAX_CUDA13_MODULE_VERSION:-0.1.0}"
MODULE_ROOT="${LORRAX_MODULEFILE_DIR:-$HOME/modulefiles}"
MODULE_FILE="$MODULE_ROOT/$MODULE_NAME/$MODULE_VERSION.lua"
SO="$LORRAX_CUDA13_BUILD/liblorrax_ffi.so"

for path in "$LORRAX_CUDA13_ENV/bin/python" "$SO"; do
    if [[ ! -e "$path" ]]; then
        echo "[cuda13 module] REFUSED: required build output is absent: $path" >&2
        exit 2
    fi
done

mkdir -p "$(dirname "$MODULE_FILE")"
sed \
    -e "s|@MODULE_NAME@|$MODULE_NAME|g" \
    -e "s|@PYTHON_MODULE@|$LORRAX_CUDA13_PYTHON_MODULE|g" \
    -e "s|@TOOLKIT_MODULE@|$LORRAX_CUDA13_TOOLKIT_MODULE|g" \
    -e "s|@ROOT@|$LORRAX_CUDA13_ROOT|g" \
    -e "s|@VENV@|$LORRAX_CUDA13_ENV|g" \
    -e "s|@CUDA@|$LORRAX_CUDA13_CUDA|g" \
    -e "s|@FFI@|$SO|g" \
    -e "s|@RUNTIME_LD@|$(lorrax_cuda13_runtime_ld_library_path)|g" \
    -e "s|@PYTHONPATH@|$(lorrax_cuda13_pythonpath)|g" \
    "$HERE/module.lua.in" > "$MODULE_FILE"

echo "[cuda13 module] installed $MODULE_FILE"
echo "[cuda13 module] validate with:"
echo "  module load $MODULE_NAME"
echo "  LX_BASE_MODULE=$MODULE_NAME lx doctor"
