#!/usr/bin/env bash
# ============================================================================
# Render image_module.lua.in for one install_deps.sh prefix (the node-local
# runtime image route, docs/installation/perlmutter.md section 1.4).
#
#   bash install_image_module.sh <install_deps.sh PREFIX> <modulefile .lua>
#
# The prefix must hold a packed image (current -> images/rt-<id>.tar.zst) and bin/lorrax-run; the
# image's revision names the module.  An existing modulefile is kept as
# <file>.before-<rev8> and the new one is renamed into place, so a rollback is
# one mv.  Stage under a candidate name first; install as the default only
# after lorrax-postinstall prints INSTALL VERIFIED and the hsuite passes
# through lorrax-run.
#
# lx (the sandbox launcher) reads LORRAX_ROOT and LORRAX_SHIFTER from the base
# module and needs <LORRAX_ROOT>/src/ffi/cpp/{select_gpu,in_container}.sh and
# src/runtime/__init__.py on the login node.  The image's tree exists only on
# nodes that restored it, so this also writes a small lx site directory,
# <prefix>/lx/<image> (LX_STUB_DIR overrides it, e.g. for a candidate), whose
# two helpers are lx_shim.sh.in: they enter the image through lorrax-run.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage="usage: install_image_module.sh <install prefix> <modulefile .lua>"
PREFIX="$(realpath -e "${1:?$usage}")"
OUT="${2:?$usage}"
die() { echo "[install_image_module] REFUSED: $*" >&2; exit 2; }
[[ "$OUT" == /*.lua ]] || die "modulefile must be an absolute .lua path: $OUT"
[[ -x "$PREFIX/bin/lorrax-run" && -x "$PREFIX/bin/lorrax-postinstall" ]] \
    || die "$PREFIX has no bin/lorrax-run and bin/lorrax-postinstall (install_deps.sh)"
grep -q 'LORRAX_RUNTIME=image' "$PREFIX/lorrax-env.sh" 2>/dev/null \
    || die "$PREFIX/lorrax-env.sh is not an image install (install_deps.sh without --in-place)"
IMAGE="$(readlink -f "$PREFIX/current" 2>/dev/null || true)"
[[ -f "$IMAGE" ]] || die "no packed image at $PREFIX/current"
REF8="$(basename "$IMAGE" | sed -E 's/^rt-([0-9a-f]{8,}).*/\1/' | cut -c1-8)"
[[ "$REF8" =~ ^[0-9a-f]{8}$ ]] || die "cannot read the revision from $(basename "$IMAGE")"
TREE="$(sed -n 's/^tree=//p' "$IMAGE.meta" 2>/dev/null | head -n 1)"
[[ "$TREE" == /* ]] || die "no tree= in $IMAGE.meta"
STUB="${LX_STUB_DIR:-$PREFIX/lx/$(basename "$IMAGE" .tar.zst)}"
render() {
    sed -e "s|@PREFIX@|$PREFIX|g" -e "s|@REF8@|$REF8|g" -e "s|@TREE@|$TREE|g" \
        -e "s|@STUB@|$STUB|g" "$1" > "$2"
    if grep -n '@[A-Z0-9_]*@' "$2"; then rm -f "$2"; die "unrendered placeholder in $2"; fi
}

# The lx site directory: written once per image, never rewritten in place.
if [[ ! -f "$STUB/src/runtime/__init__.py" ]]; then
    mkdir -p "$STUB.tmp.$$/src/ffi/cpp" "$STUB.tmp.$$/src/runtime"
    render "$HERE/lx_shim.sh.in" "$STUB.tmp.$$/src/ffi/cpp/select_gpu.sh"
    cp "$STUB.tmp.$$/src/ffi/cpp/select_gpu.sh" "$STUB.tmp.$$/src/ffi/cpp/in_container.sh"
    chmod 755 "$STUB.tmp.$$"/src/ffi/cpp/*.sh
    printf '# lx site marker, never imported: the runtime is %s/lorrax/src/runtime.\n' \
        "$TREE" > "$STUB.tmp.$$/src/runtime/__init__.py"
    mkdir -p "$(dirname "$STUB")"
    mv -T "$STUB.tmp.$$" "$STUB"
fi

mkdir -p "$(dirname "$OUT")"
TMP="$OUT.tmp.$$"
render "$HERE/image_module.lua.in" "$TMP"
[[ -e "$OUT" ]] && cp -p "$OUT" "$OUT.before-$REF8"
mv -f "$TMP" "$OUT"
echo "[install_image_module] $OUT -> $PREFIX (image $(basename "$IMAGE"); lx site $STUB)"
