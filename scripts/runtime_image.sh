#!/usr/bin/env bash
# ============================================================================
# Node-local runtime image: run LORRAX without reading the install filesystem.
#
#   runtime_image.sh install <tree> <install-dir>            (login node, once:
#                    pack into <install-dir>/images, write <install-dir>/bin/lorrax-run)
#   <install-dir>/bin/lorrax-run <command...>                (srun's command)
#   runtime_image.sh pack    <tree> <archive.tar.zst|.tar>   (login node, once)
#   runtime_image.sh restore <archive>                       (each node, idempotent)
#   runtime_image.sh run     <archive> -- <command...>       (srun's command)
#   runtime_image.sh clean   [<archive>]                     (drop unused images)
#
# WHY.  A run reads thousands of small files (the interpreter, the venv with its
# nvidia/* wheels, ptxas/libdevice, the MathDx headers, the source tree, both FFI
# legs and their recorded closure).  On a metadata-slow shared filesystem
# (Perlmutter /global/common over DVS) that cost a 3.5x slowdown against the
# same tree on $SCRATCH.  The image turns the whole runtime into ONE sequential
# read per node.
#
# HOW.  The installer builds the whole runtime AT a node-local canonical path,
# <tree> = /tmp/lorrax-$USER/<release-id> (the uv-managed interpreter inside it:
# UV_PYTHON_INSTALL_DIR=<tree>/python), and `pack` stores it as one archive on
# the persistent filesystem plus <archive>.meta.  `restore` puts it back at the
# SAME path on every node, once per node (flock + completion marker), so every
# absolute path the venv, the editable install and the closure records name is
# identical: nothing is relocated, and the closure gate's (size, mtime_ns) fast
# path holds because the archive is POSIX/pax (nanosecond mtimes).
#
# Where /tmp cannot hold the tree, the tree goes to the first root of
# LORRAX_IMAGE_ROOTS (default: $TMPDIR, /dev/shm, /local/scratch, /lscratch)
# with room, and <tree> becomes a symlink to it.  Every path is still valid; the
# closure gate compares realpaths, so it still does not rehash.  With no
# node-local root large enough, `restore` refuses with the sizes it saw
# (LORRAX_IMAGE_FALLBACK=<dir> names a persistent extracted copy to link to
# instead: correct, but it reads the shared filesystem like a clone does).
#
# Concurrency.  Every rank of a step runs `restore`.  Each first takes a SHARED
# "use" lock on the image and holds it for its lifetime (the open fd is
# inherited across exec), so `clean` -- run after every restore for the other
# images of this user on this node -- removes only images no process is using.
# Extraction is serialised by a separate EXCLUSIVE "install" lock: the first
# rank extracts into a temporary name, renamed into place after the marker is
# written; the others wait on the install lock and find the marker.
#
# Requirements on the node: bash, coreutils, GNU tar, flock(1); zstd for a
# .tar.zst archive (a plain .tar needs nothing more).
# ============================================================================
set -euo pipefail

_say() { [ "${SLURM_LOCALID:-0}" = 0 ] && echo "[lorrax image] $*" >&2 || true; }
_die() { echo "[lorrax image] ERROR: $*" >&2; exit 97; }
_meta() { sed -n "s/^$2=//p" "$1.meta" | head -n 1; }
_free_bytes() { df -PB1 "$1" 2>/dev/null | awk 'NR==2 {print $4}'; }
_now() { date +%s.%N; }
_since() { awk -v a="$1" -v b="$(_now)" 'BEGIN {printf "%.1f", b - a}'; }

cmd_pack() {
    local tree=${1:?tree} archive=${2:?archive}
    [ "$tree" = "$(cd "$tree" && pwd -P)" ] || _die "pack: $tree must be an absolute path without symlinks"
    [ -e "$archive" ] && _die "pack: $archive exists; a release id is never repacked"
    local parent base t0 bytes
    parent=$(dirname "$tree"); base=$(basename "$tree"); t0=$(_now)
    bytes=$(du -sb "$tree" | cut -f1)
    case "$archive" in
        *.tar.zst) tar --format=posix -C "$parent" -cf - "$base" | zstd -T0 -3 -q -o "$archive.tmp" ;;
        *.tar)     tar --format=posix -C "$parent" -cf "$archive.tmp" "$base" ;;
        *) _die "pack: archive must end in .tar.zst or .tar" ;;
    esac
    local id
    id="$(stat -c %s "$archive.tmp")-$(sha256sum "$archive.tmp" | cut -c1-16)"
    {
        echo "tree=$tree"
        echo "bytes=$bytes"
        echo "id=$id"
        echo "packed=$(date -u +%FT%TZ) on $(hostname)"
    } > "$archive.meta.tmp"
    mv "$archive.tmp" "$archive"; mv "$archive.meta.tmp" "$archive.meta"
    _say "packed $tree ($bytes bytes) -> $archive ($(stat -c %s "$archive") bytes) in $(_since "$t0") s"
}

# Sets IMAGE_TREE; leaves fd 9 holding a SHARED lock on <parent>/.<base>.use.lock
# for the caller's lifetime (it is inherited across exec).  Extraction is
# serialised by an EXCLUSIVE lock on .<base>.install.lock, held only while one
# process extracts: a waiting rank must never ask for the lock that running
# ranks hold shared, or it waits for their exit (a deadlock seen at P4).
cmd_restore() {
    local archive=${1:?archive}
    [ -f "$archive" ] && [ -f "$archive.meta" ] || _die "no image $archive (+ .meta)"
    local tree bytes id parent base
    tree=$(_meta "$archive" tree); bytes=$(_meta "$archive" bytes); id=$(_meta "$archive" id)
    parent=$(dirname "$tree"); base=$(basename "$tree")
    IMAGE_TREE=$tree
    mkdir -p -m 700 "$parent" 2>/dev/null || _die "cannot create $parent"
    exec 9>>"$parent/.$base.use.lock"
    flock -s 9                                                # in use from here: clean skips it
    [ "$(cat "$tree/.lorrax-image" 2>/dev/null)" = "$id" ] && return 0   # restored before
    exec 8>>"$parent/.$base.install.lock"
    flock -x 8
    local have
    have=$(cat "$tree/.lorrax-image" 2>/dev/null || true)
    if [ "$have" = "$id" ]; then
        exec 8>&-; return 0                                   # another rank just did it
    fi
    [ -z "$have" ] || _die "$tree holds image $have, not $id: a release id names one archive; pack under a new id"
    local t0 need place="" root free
    t0=$(_now)
    need=$(( bytes + $(stat -c %s "$archive") + bytes / 10 + (1 << 30) ))   # tree + fetched archive
    for root in "$parent" ${LORRAX_IMAGE_ROOTS:-${TMPDIR:-} /dev/shm /local/scratch /lscratch}; do
        [ -n "$root" ] && [ -d "$root" ] && [ -w "$root" ] || continue
        free=$(_free_bytes "$root"); [ -n "$free" ] || continue
        if [ "$free" -ge "$need" ]; then place=$root; break; fi
        _say "$root: $free bytes free < $need needed"
    done
    rm -rf "$tree" "$parent/.$base.tmp."* 2>/dev/null || true   # a torn restore of a killed job
    if [ -z "$place" ]; then
        [ -n "${LORRAX_IMAGE_FALLBACK:-}" ] && [ -d "$LORRAX_IMAGE_FALLBACK/$base" ] \
            || _die "no node-local directory holds $need bytes (tried $parent ${LORRAX_IMAGE_ROOTS:-${TMPDIR:-} /dev/shm /local/scratch /lscratch}); set LORRAX_IMAGE_ROOTS to a node-local directory with room, or LORRAX_IMAGE_FALLBACK to a directory holding an extracted copy of $base"
        ln -sfn "$LORRAX_IMAGE_FALLBACK/$base" "$tree"
        echo "$id" > "$tree/.lorrax-image" 2>/dev/null || true
        _say "WARNING: no node-local room; $tree -> $LORRAX_IMAGE_FALLBACK/$base (shared filesystem)"
        exec 8>&-; return 0
    fi
    local dest="$place"
    [ "$place" = "$parent" ] || { dest="$place/lorrax-${USER:-$(id -un)}"; mkdir -p -m 700 "$dest"; }
    local tmp="$dest/.$base.tmp.$$" t1
    mkdir -p "$tmp"
    # Fetch the archive into node-local memory with parallel 16 MiB reads, then
    # unpack locally.  On a DVS mount (Perlmutter CFS, $HOME, /global/common) every
    # read request is a server round trip: zstd's/tar's own 128 KiB reads ran at
    # 9-12 MB/s, one 16 MiB stream at ~95 MB/s, 16 streams at ~415 MB/s (ri_image.md).
    local local_copy="$tmp/archive" size n chunk k
    size=$(stat -c %s "$archive"); n=${LORRAX_IMAGE_STREAMS:-16}
    chunk=$(( ((size + n - 1) / n + (16 << 20) - 1) / (16 << 20) * (16 << 20) ))
    for (( k = 0; k * chunk < size; k++ )); do
        dd if="$archive" of="$local_copy" bs=16M conv=notrunc status=none \
           iflag=skip_bytes,count_bytes oflag=seek_bytes \
           skip=$(( k * chunk )) seek=$(( k * chunk )) count="$chunk" &
    done
    wait
    [ "$(stat -c %s "$local_copy")" = "$size" ] || _die "fetching $archive: got $(stat -c %s "$local_copy") of $size bytes"
    t1=$(_since "$t0")
    case "$archive" in
        *.zst) zstd -dc -q "$local_copy" | tar -C "$tmp" -xf - ;;
        *)     tar -C "$tmp" -xf "$local_copy" ;;
    esac
    rm -f "$local_copy"
    [ -d "$tmp/$base" ] || _die "$archive does not hold $base/"
    echo "$id" > "$tmp/$base/.lorrax-image"
    [ "$dest" = "$parent" ] || rm -rf "$dest/$base"
    mv "$tmp/$base" "$dest/$base"; rmdir "$tmp"
    [ "$dest" = "$parent" ] || ln -sfn "$dest/$base" "$tree"
    echo "[lorrax image] $(hostname) rank ${SLURM_PROCID:-0}: restored $base ($bytes bytes) to $dest in $(_since "$t0") s (fetch $t1 s, $n streams) from $archive" >&2
    exec 8>&-                                                 # waiting ranks proceed
    cmd_clean "$archive" || true
}

# Remove this user's other images on this node that no process holds.
cmd_clean() {
    local keep="" parent=/tmp/lorrax-${USER:-$(id -un)}
    if [ -n "${1:-}" ] && [ -f "$1.meta" ]; then
        keep=$(_meta "$1" tree); parent=$(dirname "$keep")
    fi
    local lock base tree
    for lock in "$parent"/.*.use.lock; do
        [ -e "$lock" ] || continue
        base=${lock##*/.}; base=${base%.use.lock}; tree="$parent/$base"
        [ "$tree" = "$keep" ] && continue
        [ -e "$tree/.lorrax-image" ] || continue                     # not an image
        (
            exec 7>>"$lock"
            flock -xn 7 || exit 0                                   # in use
            [ -L "$tree" ] && rm -rf "$(readlink "$tree")"
            rm -rf "$tree" "$parent/.$base.tmp."*                   # lock files stay (no inode races)
            _say "removed unused image $tree"
        )
    done
}

cmd_run() {
    local archive=${1:?archive}; shift
    [ "${1:-}" = "--" ] && shift
    [ $# -gt 0 ] || _die "run: no command"
    cmd_restore "$archive"
    local tree=$IMAGE_TREE src venv cudart
    src="$tree/lorrax"; venv="$src/.venv"
    export LORRAX_IMAGE_ROOT="$tree" VIRTUAL_ENV="$venv"
    export PATH="$venv/bin:$PATH"
    # libfabric dlopens the unversioned libcudart.so (gpu_env.sh); point it at the
    # venv's runtime from inside the image instead of a link under $HOME.
    for cudart in "$venv"/lib/python3*/site-packages/nvidia/cu*/lib/libcudart.so.[0-9]*; do
        [ -e "$cudart" ] || continue
        [ -e "$tree/launch/libcudart.so" ] || { mkdir -p "$tree/launch"; ln -sfn "$cudart" "$tree/launch/libcudart.so" 2>/dev/null || true; }
        export LD_LIBRARY_PATH="$tree/launch${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        break
    done
    [ "${LORRAX_IMAGE_CHDIR:-}" = source ] && cd "$src"
    exec "$src/src/ffi/cpp/select_gpu.sh" "$@"
}

# <install>/images/<id>.tar.zst(.meta), <install>/current -> images/<id>.tar.zst and
# <install>/bin/lorrax-run (this script): a user's srun line is
#   srun ... <install>/bin/lorrax-run python -u -m gw.gw_jax -i deck.in
cmd_install() {
    local tree=${1:?tree} install=${2:?install dir} base
    base=$(basename "$tree")
    mkdir -p "$install/images" "$install/bin"
    cmd_pack "$tree" "$install/images/$base.tar.zst"
    cp "${BASH_SOURCE[0]}" "$install/bin/lorrax-run.tmp" && chmod 755 "$install/bin/lorrax-run.tmp"
    mv "$install/bin/lorrax-run.tmp" "$install/bin/lorrax-run"
    ln -sfn "images/$base.tar.zst" "$install/current.tmp" && mv -T "$install/current.tmp" "$install/current"
    _say "installed: srun ... $install/bin/lorrax-run <command>"
}

if [ "$(basename "$0")" = lorrax-run ]; then
    _here=$(cd "$(dirname "$0")" && pwd)
    cmd_run "${LORRAX_IMAGE:-$(readlink -f "$_here/../current")}" "$@"
fi
case "${1:-}" in
    pack)    shift; cmd_pack "$@" ;;
    install) shift; cmd_install "$@" ;;
    restore) shift; cmd_restore "$@"; echo "$IMAGE_TREE" ;;
    run)     shift; cmd_run "$@" ;;
    clean)   shift; cmd_clean "$@" ;;
    *) sed -n '3,12p' "$0" >&2; exit 2 ;;
esac
