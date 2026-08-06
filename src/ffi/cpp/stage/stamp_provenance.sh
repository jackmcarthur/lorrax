#!/usr/bin/env bash
# stamp_provenance.sh — write a PROVENANCE file beside a freshly built FFI .so.
#
#   usage: stamp_provenance.sh <path/to/lib.so> [key=value ...]
#
# WHY THIS EXISTS.  ffi_loader.build_provenance() prints this file in every
# run's startup report, so a job log records the build it loaded.  Without it
# the loader falls back to "NO PROVENANCE FILE (pre-stamp build) | <bytes> |
# sha <...>", which locates the artifact but cannot date it.
#
# That gap cost real time twice.  2026-08-02: two 32-node legs were debugged
# against a lib whose revision nobody had checked.  2026-08-05: a 4-node GPU
# log was analysed by inspecting the on-disk liblorrax_ffi.so, which had been
# REBUILT seven minutes after that log's last write — so the ELF being read
# was provably not the ELF that produced the numbers, and nothing on disk said
# so.  A build artifact that cannot answer "which source produced you" makes
# every later forensic question unanswerable.
#
# Format is flat key=value because build_provenance() parses it that way.
# Extra `key=value` arguments are appended verbatim, for leg-specific facts
# (vendor stage paths, CUDA subdir, ...).
set -euo pipefail

SO="${1:?usage: stamp_provenance.sh <lib.so> [key=value ...]}"
shift || true

if [ ! -f "$SO" ]; then
    echo "[provenance] no such file: $SO" >&2
    exit 1
fi

# Repo root: this script lives at src/ffi/cpp/stage/, so ../../../.. is it.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LORRAX_ROOT:-$(cd "$_here/../../../.." && pwd)}"

_rev=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
_branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
_dirty=no
git -C "$ROOT" diff --quiet 2>/dev/null || _dirty=yes

DEST="$(dirname "$SO")/PROVENANCE"
{
    echo "so=$SO"
    echo "sha256=$(sha256sum "$SO" | cut -d' ' -f1)"
    echo "bytes=$(stat -c %s "$SO")"
    echo "git_rev=$_rev"
    echo "git_branch=$_branch"
    echo "git_dirty=$_dirty"
    echo "source_tree=$ROOT"
    echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "built_on=$(hostname)"
    echo "exported_symbols=$(nm -D --defined-only "$SO" 2>/dev/null | wc -l)"
    for kv in "$@"; do
        echo "$kv"
    done
} > "$DEST"

echo "[provenance] stamped $DEST"
sed 's/^/[provenance]   /' "$DEST"
if [ "$_dirty" = yes ]; then
    echo "[provenance] WARNING: source tree had uncommitted changes; git_rev" >&2
    echo "[provenance]   alone does NOT identify this build." >&2
fi
