#!/usr/bin/env bash
# ============================================================================
# Publish an accepted candidate into $LORRAX_MODULE_PREFIX:
#   releases/source-<rev8>            a fresh git archive of the revision
#   releases/<rev8>-bundle-<id12>     a copy of the sealed bundle, read-only
#   modulefiles/<name>/<version>.lua  re-rendered, renamed into place; the
#                                     previous file is kept as .before-<rev8>
#
#   LORRAX_MODULE_PREFIX=/abs/prefix bash publish.sh <candidate dir>
#
# Refuses a candidate without accept/{gate10.ok,ok.0,…,ok.3} and an existing release of
# either name: releases are never replaced, and earlier ones stay for rollback
# (re-render the module against them with install_module.sh).
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/stack.sh"
CAND="$(realpath -e "${1:?usage: publish.sh <candidate dir>}")"
# shellcheck disable=SC1091
source "$CAND/CANDIDATE"
for f in gate10.ok ok.0 ok.1 ok.2 ok.3; do
    [[ -e "$CAND/accept/$f" ]] || lorrax_module_die "no accept/$f in $CAND (run accept.sh at -n 1 and -n 4)"
done
SRC_REL="$RELEASES/source-${REV:0:8}"
BUN_REL="$RELEASES/${REV:0:8}-bundle-${BUNDLE_ID:0:12}"
for d in "$SRC_REL" "$BUN_REL"; do
    [[ ! -e "$d" ]] || lorrax_module_die "release exists: $d (releases are immutable)"
done
mkdir -p "$RELEASES"

# The candidate's source tree holds acceptance output; publish a clean archive.
mkdir "$SRC_REL.tmp.$$"
git -C "$CAND/src" archive --format=tar "$REV" | tar -x -C "$SRC_REL.tmp.$$"
echo "$REV" > "$SRC_REL.tmp.$$/SOURCE_COMMIT"
cp -a "$CAND/bundle" "$BUN_REL.tmp.$$"
"$VENV/bin/python" - "$CAND/bundle" "$BUN_REL.tmp.$$" <<'PY'
import hashlib, sys
from pathlib import Path
a, b = (Path(p) for p in sys.argv[1:])
digest = lambda root: {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(root.rglob("*")) if p.is_file()}
assert digest(a) == digest(b), "bundle copy differs from the accepted candidate"
PY
chmod -R a-w "$SRC_REL.tmp.$$" "$BUN_REL.tmp.$$"
mv "$SRC_REL.tmp.$$" "$SRC_REL"
mv "$BUN_REL.tmp.$$" "$BUN_REL"
bash "$HERE/install_module.sh" "$SRC_REL" "$BUN_REL" \
    "$PREFIX/modulefiles/$MODULE_NAME/$MODULE_VERSION.lua"
echo "[publish] $SRC_REL"
echo "[publish] $BUN_REL"
echo "[publish] check: module use $PREFIX/modulefiles && module load $MODULE_NAME/$MODULE_VERSION"
