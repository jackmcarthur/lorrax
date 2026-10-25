#!/usr/bin/env bash
# ============================================================================
# stage_host_pmi.sh — stage the host SLURM PMI2 client library where the
# in-container Intel MPI bootstrap can dlopen it (env audit P0-3).
#
#   config/frontera/stage_host_pmi.sh [source-lib]
#
# WHY: under `srun --mpi=pmi2`, Intel MPI bootstraps through the PMI2 client
# library named by I_MPI_PMI_LIBRARY.  TACC's login env points that at
# /usr/lib64/libpmi.so — a PMI-1 library (wrong protocol for --mpi=pmi2) —
# and NEITHER host lib is visible inside the py312 container, whose binds
# cover /home1,/work2,/scratch*,/opt/intel but not /usr/lib64 as a library
# dir.  So the host PMI2 lib is COPIED once to $WORK (bind-mounted
# everywhere) and every harness sets
#     I_MPI_PMI_LIBRARY=$WORK/host_pmi/libpmi2.so.0
# (mpi_transport_env.sh does this, honouring LORRAX_PMI2_LIB).
#
# PROVENANCE of the existing staged copy (verified 2026-07-31, login node):
#     sha256(/work2/.../host_pmi/libpmi2.so.0.0.0)
#   = sha256(/usr/lib64/libpmi2.so.0.0.0)
#   = f1fc368a43039a0d1dff5abac54bfc3d95395495a0647431e7a24e1cc4270be4
# i.e. it is a byte-for-byte copy of the host SLURM PMI2 client lib from
# /usr/lib64 (240136 bytes), not a hand-built artifact.  After a TACC SLURM
# upgrade the checksum WILL move; that is expected — re-run this script and
# the announcement below records the new hash.
# ============================================================================
set -euo pipefail

SRC="${1:-/usr/lib64/libpmi2.so.0.0.0}"
# Destination is derived from the same contract the harnesses read:
# LORRAX_PMI2_LIB (default $WORK/host_pmi/libpmi2.so.0).
DEST_LINK="${LORRAX_PMI2_LIB:-${WORK:?WORK is unset — run on a TACC host or set LORRAX_PMI2_LIB}/host_pmi/libpmi2.so.0}"
DEST_DIR=$(dirname "$DEST_LINK")
DEST_REAL="$DEST_DIR/libpmi2.so.0.0.0"

# The known-good hash of the copy every 2026-07 certified run used.
CERTIFIED_SHA=f1fc368a43039a0d1dff5abac54bfc3d95395495a0647431e7a24e1cc4270be4

[ -f "$SRC" ] || {
    echo "[stage_pmi] REFUSING: source PMI2 lib not found: $SRC" >&2
    echo "[stage_pmi] Expected the host SLURM PMI2 client library." >&2
    echo "[stage_pmi] Look for it with:  ls /usr/lib64/libpmi2* /usr/lib64/slurm*/libpmi2* " >&2
    echo "[stage_pmi] then re-run:  $0 /path/to/libpmi2.so.0.0.0" >&2
    exit 2
}

mkdir -p "$DEST_DIR"
cp -f "$SRC" "$DEST_REAL"
ln -sfn "$(basename "$DEST_REAL")" "$DEST_LINK"

GOT_SHA=$(sha256sum "$DEST_REAL" | awk '{print $1}')
echo "[stage_pmi] staged: $SRC -> $DEST_REAL"
echo "[stage_pmi] symlink: $DEST_LINK -> $(basename "$DEST_REAL")"
echo "[stage_pmi] sha256: $GOT_SHA"
if [ "$GOT_SHA" = "$CERTIFIED_SHA" ]; then
    echo "[stage_pmi] matches the 2026-07 certified copy."
else
    echo "[stage_pmi] NOTE: differs from the 2026-07 certified hash" \
         "($CERTIFIED_SHA) — the host SLURM was likely upgraded.  Fine, but" \
         "re-verify one srun --mpi=pmi2 job before a campaign."
fi
echo "[stage_pmi] harness contract: I_MPI_PMI_LIBRARY=$DEST_LINK"
