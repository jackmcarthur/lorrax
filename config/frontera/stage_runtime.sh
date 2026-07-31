#!/bin/bash
# ===========================================================================
# stage_runtime.sh -- unroll the CPU run-time bundle onto this node's local SSD
#
# SOURCE it (do not exec it) from the container-side runner, before python:
#
#     . $LORRAX_ROOT/config/frontera/stage_runtime.sh
#     export PYTHONPATH=$LORRAX_OVERLAY_DIR:$LORRAX_SRC_DIR
#     $LORRAX_PY -u -m gw.kin_ion_io ...
#
# It sets LORRAX_PY, LORRAX_VENV_DIR, LORRAX_OVERLAY_DIR, LORRAX_SRC_DIR to
# node-local paths, and LORRAX_STAGE_S to the seconds it spent staging.
#
# ONE EXTRACT PER NODE, NOT PER RANK.  Every rank runs this; flock(1) makes
# exactly one of them extract and the rest wait on the lock and then find the
# sentinel.  Skipping that would have all ranks-per-node write the same 13k
# files over each other.
#
# WHY THIS IS WORTH DOING (job 7882047, fresh Frontera compute node):
#   /work2 streams at 70 MB/s; /scratch2 at 560 MB/s; /tmp is a local 224 GB
#   SSD with 144 GB free and is writable inside apptainer.  The bundle is one
#   contiguous tar on the fast filesystem, so staging is a streaming read plus
#   a local untar, and every subsequent import is a local-disk page fault
#   instead of a Lustre RPC.
#
# FALLBACK IS LOUD.  If the bundle is missing or the extract fails, this falls
# back to the Lustre venv and SAYS SO on rank 0 -- a silent fallback would
# turn "the startup fix regressed" into "the startup fix does nothing", which
# is the failure mode that is impossible to notice from a timing table.
#
# ENV IN
#   LORRAX_BUNDLE       path to lorrax_cpu_bundle.tar  (required to stage)
#   LORRAX_STAGE_ROOT   node-local dir (default /tmp/lorrax_stage.$UID)
#   LORRAX_STAGE=0      disable staging entirely (use the Lustre venv)
#   LORRAX_VENV_FALLBACK / LORRAX_OVERLAY_FALLBACK / LORRAX_SRC_FALLBACK
#                       what to use when not staging
# ===========================================================================

_lorrax_stage_rank0 () { [ "${SLURM_PROCID:-0}" = "0" ]; }
_lorrax_stage_say () { _lorrax_stage_rank0 && echo "[stage] $*"; return 0; }

LORRAX_STAGE_S=0
_ls_t0=$(date +%s.%N)

_ls_use_fallback () {
  LORRAX_VENV_DIR=${LORRAX_VENV_FALLBACK:-$WORK/lorrax_env/.venv}
  LORRAX_OVERLAY_DIR=${LORRAX_OVERLAY_FALLBACK:-$WORK/lorrax_env_mpi_overlay/site}
  LORRAX_SRC_DIR=${LORRAX_SRC_FALLBACK:-$LORRAX_ROOT/src}
  LORRAX_PY=$LORRAX_VENV_DIR/bin/python
  export LORRAX_VENV_DIR LORRAX_OVERLAY_DIR LORRAX_SRC_DIR LORRAX_PY
  export LORRAX_STAGE_S=0
  export LORRAX_STAGED=0
}

if [ "${LORRAX_STAGE:-1}" = "0" ]; then
  _lorrax_stage_say "DISABLED by LORRAX_STAGE=0 -- running from the shared"
  _lorrax_stage_say "filesystem; expect the full cold-start import cost."
  _ls_use_fallback
elif [ -z "${LORRAX_BUNDLE:-}" ] || [ ! -f "${LORRAX_BUNDLE:-}" ]; then
  _lorrax_stage_say "WARNING: LORRAX_BUNDLE='${LORRAX_BUNDLE:-<unset>}' is not a"
  _lorrax_stage_say "file -- NOT staging.  Falling back to the shared-filesystem"
  _lorrax_stage_say "venv; the node-local startup saving is NOT in force."
  _ls_use_fallback
else
  _ls_root=${LORRAX_STAGE_ROOT:-/tmp/lorrax_stage.$(id -u)}
  _ls_key=$(stat -c '%s-%Y' "$LORRAX_BUNDLE" 2>/dev/null || echo unknown)
  _ls_dir=$_ls_root/$_ls_key
  mkdir -p "$_ls_root" 2>/dev/null
  exec 9>"$_ls_root/.lock" 2>/dev/null
  if flock 9 2>/dev/null; then :; else _lorrax_stage_say "WARNING: no flock"; fi
  if [ ! -f "$_ls_dir/.READY" ]; then
    rm -rf "$_ls_dir.part"
    mkdir -p "$_ls_dir.part"
    if tar -C "$_ls_dir.part" -xf "$LORRAX_BUNDLE" 2>/dev/null; then
      touch "$_ls_dir.part/.READY"
      rm -rf "$_ls_dir"
      mv "$_ls_dir.part" "$_ls_dir"
    else
      _lorrax_stage_say "WARNING: extract of $LORRAX_BUNDLE FAILED"
      rm -rf "$_ls_dir.part"
    fi
  fi
  # Evict stale bundle extracts (keep the newest 2 by mtime — the one just
  # staged plus one predecessor for any still-running job on this node).
  # Each extract is ~800 MB on a 144-224 GB shared /tmp SSD, and every
  # bundle rebuild changes the size-mtime key, so without this the old
  # trees accumulate until the node's SSD fills.  Done while still holding
  # the lock, so no rank races the extractor.  .lock is a file and .part
  # dirs are transient; both are excluded by the dirs-only listing + the
  # newest-2 window.
  # The CURRENT key is never evicted, whatever its mtime (an old extract of
  # the current bundle can be older than two extracts of a newer bundle a
  # sibling job staged — evicting the tree this very run is about to use
  # would be self-sabotage).
  _ls_stale=$(cd "$_ls_root" 2>/dev/null && ls -1td -- */ 2>/dev/null | tail -n +3)
  for _ls_old in $_ls_stale; do
    _ls_old=${_ls_old%/}
    case "$_ls_old" in .|..|""|"$_ls_key") continue ;; esac
    _lorrax_stage_say "evicting stale extract $_ls_root/$_ls_old"
    rm -rf "${_ls_root:?}/$_ls_old"
  done
  unset _ls_stale _ls_old
  flock -u 9 2>/dev/null || true
  exec 9>&- 2>/dev/null

  if [ -x "$_ls_dir/venv/bin/python" ]; then
    LORRAX_VENV_DIR=$_ls_dir/venv
    LORRAX_OVERLAY_DIR=$_ls_dir/overlay
    LORRAX_SRC_DIR=$_ls_dir/src
    LORRAX_PY=$LORRAX_VENV_DIR/bin/python
    export LORRAX_VENV_DIR LORRAX_OVERLAY_DIR LORRAX_SRC_DIR LORRAX_PY
    export LORRAX_STAGED=1
    _lorrax_stage_say "node-local runtime at $_ls_dir"
  else
    _lorrax_stage_say "WARNING: staged tree has no venv/bin/python -- falling"
    _lorrax_stage_say "back to the shared filesystem."
    _ls_use_fallback
  fi
fi

LORRAX_STAGE_S=$(awk -v a="$_ls_t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')
export LORRAX_STAGE_S
_lorrax_stage_say "staged=${LORRAX_STAGED} in ${LORRAX_STAGE_S}s  py=$LORRAX_PY"
unset _ls_t0 _ls_root _ls_key _ls_dir
