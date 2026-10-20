#!/bin/bash
# One deck's traced run (lxkit.memory LORRAX_MEMORY_TRACE) and its residual table.
#
#   tests/replays/traced_run.sh <name> <out dir> -- <the deck's launch command ...>
#
# The launch command is the deck's own (hsuite P4, Fe V2 P4, a small Cr Gamma deck at
# N_mu 4-6k on P = 4, one node), unchanged; srun passes the environment to every rank.
# Writes <out>/<name>.trace.json (process 0; .p<i> for process i), <name>.log and
# <name>.residuals.md: per section, measured peak - (inherited live + compiled + vendor).
set -euo pipefail
name=$1 out=$2
shift 2
[ "${1:-}" = "--" ] && shift
mkdir -p "$out"
export LORRAX_MEMORY_TRACE="$out/$name.trace.json" LORRAX_MEMORY_RECEIPTS=calibrate
rc=0
"$@" > "$out/$name.log" 2>&1 || rc=$?
python3 -m lxkit.memory_residuals --all "$out/$name".trace.json* > "$out/$name.residuals.md"
echo "$name: rc=$rc; $(tail -1 "$out/$name.residuals.md")"
exit $rc
