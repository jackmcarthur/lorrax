#!/usr/bin/env bash
# ============================================================================
# Node-local runtime image: run LORRAX without reading the install filesystem.
#
#   <prefix>/bin/lorrax-run [--all-gpus] <command...>        (srun's command)
#   runtime_image.sh install <tree> <prefix>                 (install_deps.sh: pack into
#                    <prefix>/images, point <prefix>/current at it, write <prefix>/bin/lorrax-run)
#   runtime_image.sh pack    <tree> <archive.tar.zst|.tar>
#   runtime_image.sh restore <archive>                       (each node, idempotent)
#   runtime_image.sh run     <archive> [--all-gpus] [--] <command...>
#   runtime_image.sh clean   [<archive>]                     (drop unused images)
#
# WHY.  A run reads thousands of small files (the interpreter, the venv with its
# nvidia/* wheels, ptxas/libdevice, the MathDx headers, the source tree, both FFI
# legs and their recorded closure).  On a metadata-slow shared filesystem
# (Perlmutter /global/common, CFS and $HOME over DVS) every open is a server
# round trip, 3.5-30 ms; that cost a 3.5x slowdown against the same tree on
# $SCRATCH.  The image turns the whole runtime into ONE sequential read per node.
#
# HOW.  install_deps.sh builds the whole runtime AT a node-local canonical path,
# <tree> = /tmp/lorrax-$USER/<release-id> (the uv-managed interpreter inside it),
# and `pack` stores it as one archive on the persistent filesystem plus
# <archive>.meta.  `restore` puts it back at the SAME path on every node, once
# per node (flock + completion marker), so every absolute path the venv, the
# editable install and the closure records name is identical: nothing is
# relocated, and the closure gate's (size, mtime_ns) fast path holds because the
# archive is POSIX/pax (nanosecond mtimes).
#
# Where the canonical parent cannot hold the tree, the tree goes to the first of
# LORRAX_IMAGE_ROOTS (default: /dev/shm, $TMPDIR, /local/scratch, /lscratch)
# with room, and <tree> becomes a symlink to it; the closure gate compares
# realpaths, so it still does not rehash.  With no node-local directory large
# enough, the node runs from ONE extracted copy on the install filesystem
# (LORRAX_IMAGE_FALLBACK, default <prefix>/unpacked/<release-id>, made once by
# the first node that needs it) and says so in one line: correct, but it reads
# the shared filesystem per file like a clone does, and lorrax-postinstall's
# self-check judges whether that is fast enough.
#
# Concurrency.  Every rank of a step runs `restore`.  Each first takes a SHARED
# "use" lock on the image and holds it for its lifetime (the open fd is
# inherited across exec), so `clean` -- run after every restore for the other
# images of this user on this node -- removes only images no process is using.
# Extraction is serialised by a separate EXCLUSIVE "install" lock: the first
# rank extracts into a temporary name, renamed into place after the marker is
# written; the others wait on the install lock and find the marker.  Two jobs of
# one user on one node share the image the same way.  Multi-node: every node
# restores from the install filesystem itself (about 7 s each on Perlmutter).
#
# Requirements on the node: bash, coreutils (GNU dd), GNU tar, flock(1); zstd
# for a .tar.zst archive (a plain .tar needs nothing more).
# ============================================================================
set -euo pipefail

_say() { [ "${SLURM_LOCALID:-0}" = 0 ] && echo "[lorrax image] $*" >&2 || true; }
_die() { echo "[lorrax image] ERROR: $*" >&2; exit 97; }
_meta() { sed -n "s/^$2=//p" "$1.meta" | head -n 1; }
_free_bytes() { df -PB1 "$1" 2>/dev/null | awk 'NR==2 {print $4}'; }
_gb() { awk -v b="$1" 'BEGIN {printf "%.1f GB", b / 1e9}'; }
_now() { date +%s.%N; }
_since() { awk -v a="$1" -v b="$(_now)" 'BEGIN {printf "%.1f", b - a}'; }
_user() { echo "${USER:-$(id -un)}"; }
_roots() { echo "${LORRAX_IMAGE_ROOTS-/dev/shm ${TMPDIR:-} /local/scratch /lscratch}"; }

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
    _say "packed $tree ($(_gb "$bytes")) -> $archive ($(_gb "$(stat -c %s "$archive")")) in $(_since "$t0") s"
}

# Remove an image tree: the link target too when it is one of OUR node-local
# copies (<root>/lorrax-$USER/<base>), never a shared extracted copy.
_drop_tree() {
    local tree=$1 base t
    base=$(basename "$tree")
    if [ -L "$tree" ]; then
        t=$(readlink "$tree")
        case "$t" in */lorrax-"$(_user)"/"$base") rm -rf "$t" ;; esac
    fi
    rm -rf "$tree"
}

# The node has no room: one extracted copy on the install filesystem, shared by
# every such node (first node extracts under a unique name, rename publishes).
_restore_shared() {
    local archive=$1 tree=$2 id=$3 need=$4 tried=$5 base shared tmp t0
    base=$(basename "$tree")
    shared=${LORRAX_IMAGE_FALLBACK:-$(dirname "$(dirname "$archive")")/unpacked}
    t0=$(_now)
    if [ "$(cat "$shared/$base/.lorrax-image" 2>/dev/null)" != "$id" ]; then
        mkdir -p "$shared" 2>/dev/null && [ -w "$shared" ] \
            || _die "no node-local directory holds $(_gb "$need") (free:$tried) and $shared is not writable; set LORRAX_IMAGE_ROOTS to a node-local directory with room"
        tmp="$shared/.$base.tmp.$(hostname).$$"
        rm -rf "$tmp"; mkdir -p "$tmp"
        case "$archive" in
            *.zst) dd if="$archive" bs=16M status=none | zstd -dc -q | tar -C "$tmp" -xf - ;;
            *)     dd if="$archive" bs=16M status=none | tar -C "$tmp" -xf - ;;
        esac
        echo "$id" > "$tmp/$base/.lorrax-image"
        mv -T "$tmp/$base" "$shared/$base" 2>/dev/null || true    # another node published first
        rm -rf "$tmp"
        [ "$(cat "$shared/$base/.lorrax-image" 2>/dev/null)" = "$id" ] \
            || _die "$shared/$base does not hold image $id; remove it and retry"
    fi
    ln -sfn "$shared/$base" "$tree"
    echo "[lorrax image] $(hostname): no node-local directory holds $(_gb "$need") (free:$tried); running from the shared copy $shared/$base ($(_since "$t0") s), which reads the install filesystem per file: the self-check (lorrax-postinstall) judges whether that is fast enough" >&2
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
    mkdir -p -m 700 "$parent" 2>/dev/null \
        || _die "cannot create $parent on $(hostname), the node-local path this image was built at; reinstall with install_deps.sh --node-dir <a directory every compute node has>"
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
    local t0 need place="" root free tried=""
    t0=$(_now)
    need=$(( bytes + $(stat -c %s "$archive") + bytes / 10 + (1 << 30) ))   # tree + fetched archive
    for root in "$parent" $(_roots); do
        [ -n "$root" ] && [ -d "$root" ] && [ -w "$root" ] || continue
        free=$(_free_bytes "$root"); [ -n "$free" ] || continue
        if [ "$free" -ge "$need" ]; then place=$root; break; fi
        tried+=" $root $(_gb "$free")"
    done
    # A torn restore of a killed job: its tree, link target and temporaries.
    _drop_tree "$tree"
    for root in "$parent" $(_roots); do
        [ -n "$root" ] && rm -rf "$root/.$base.tmp."* "$root/lorrax-$(_user)/.$base.tmp."* 2>/dev/null || true
    done
    if [ -z "$place" ]; then
        _restore_shared "$archive" "$tree" "$id" "$need" "$tried"
        exec 8>&-; return 0
    fi
    local dest="$place"
    [ "$place" = "$parent" ] || { dest="$place/lorrax-$(_user)"; mkdir -p -m 700 "$dest"; }
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
    echo "[lorrax image] $(hostname) rank ${SLURM_PROCID:-0}: restored $base ($(_gb "$bytes")) to $dest in $(_since "$t0") s (fetch $t1 s, $n streams) from $archive" >&2
    exec 8>&-                                                 # waiting ranks proceed
    cmd_clean "$archive" || true
}

# Remove this user's other images on this node that no process holds.
cmd_clean() {
    local keep="" parent=/tmp/lorrax-$(_user)
    if [ -n "${1:-}" ] && [ -f "$1.meta" ]; then
        keep=$(_meta "$1" tree); parent=$(dirname "$keep")
    fi
    local lock base tree
    for lock in "$parent"/.*.use.lock; do
        [ -e "$lock" ] || continue
        base=${lock##*/.}; base=${base%.use.lock}; tree="$parent/$base"
        [ "$tree" = "$keep" ] && continue
        [ -e "$tree/.lorrax-image" ] || [ -L "$tree" ] || continue  # not an image
        (
            exec 7>>"$lock"
            flock -xn 7 || exit 0                                   # in use
            _drop_tree "$tree"
            rm -rf "$parent/.$base.tmp."*                           # lock files stay (no inode races)
            _say "removed unused image $tree"
        )
    done
}

cmd_run() {
    local archive=${1:?archive}; shift
    local select=1
    while [ $# -gt 0 ]; do
        case "$1" in
            --all-gpus) select=0; shift ;;   # one process that drives every GPU of the step
            --) shift; break ;;
            *) break ;;
        esac
    done
    [ $# -gt 0 ] || _die "run: no command"
    cmd_restore "$archive"
    local tree=$IMAGE_TREE src venv cudart
    src="$tree/lorrax"; venv="$src/.venv"
    export LORRAX_IMAGE_ROOT="$tree" LORRAX_ROOT="$src" VIRTUAL_ENV="$venv"
    # The install's bin directory (on the install filesystem) leaves PATH: every
    # bare command name a rank's shell or tools look up would probe it first.
    if [ -n "${_LORRAX_RUN_BIN:-}" ]; then
        local p=":$PATH:"
        p=${p//":$_LORRAX_RUN_BIN:"/:}; p=${p#:}; PATH=${p%:}
    fi
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
    ((select)) || exec "$@"
    exec "$src/src/ffi/cpp/select_gpu.sh" "$@"
}

# <prefix>/images/<id>.tar.zst(.meta), <prefix>/current -> images/<id>.tar.zst,
# <prefix>/bin/lorrax-run (this script); other images of the prefix (and their
# shared extracted copies) are removed: a job reads the archive only while it
# restores, and the image of a running job lives on its nodes.
cmd_install() {
    local tree=${1:?tree} install=${2:?install dir} base old
    base=$(basename "$tree")
    mkdir -p "$install/images" "$install/bin"
    cmd_pack "$tree" "$install/images/$base.tar.zst"
    cp "${BASH_SOURCE[0]}" "$install/bin/lorrax-run.tmp" && chmod 755 "$install/bin/lorrax-run.tmp"
    mv "$install/bin/lorrax-run.tmp" "$install/bin/lorrax-run"
    ln -sfn "images/$base.tar.zst" "$install/current.tmp" && mv -T "$install/current.tmp" "$install/current"
    for old in "$install"/images/*.tar.zst "$install"/images/*.tar "$install"/unpacked/*; do
        [ -e "$old" ] || continue
        case "$(basename "$old")" in "$base"|"$base".tar.zst|"$base".tar) continue ;; esac
        rm -rf "$old" "$old.meta"
        _say "removed the previous image $old"
    done
    _say "installed: srun ... $install/bin/lorrax-run <command>"
}

if [ "$(basename "$0")" = lorrax-run ]; then
    _LORRAX_RUN_BIN=$(cd "$(dirname "$0")" && pwd)
    cmd_run "${LORRAX_IMAGE:-$(readlink -f "$_LORRAX_RUN_BIN/../current")}" "$@"
fi
case "${1:-}" in
    pack)    shift; cmd_pack "$@" ;;
    install) shift; cmd_install "$@" ;;
    restore) shift; cmd_restore "$@"; echo "$IMAGE_TREE" ;;
    run)     shift; cmd_run "$@" ;;
    clean)   shift; cmd_clean "$@" ;;
    *) sed -n '3,12p' "$0" >&2; exit 2 ;;
esac
