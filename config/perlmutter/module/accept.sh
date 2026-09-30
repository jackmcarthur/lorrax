#!/usr/bin/env bash
# ============================================================================
# Accept one module candidate through the candidate module, from a directory
# that is not a checkout (the candidate's accept/).  The step shape selects:
#   -n 1  Gate 10 (src/ffi/cpp/gate_one_odr.py), which must be alone in its
#         MPI world: its HDF5 file is per process;
#   -n 4  verify_runtime.py and tests/hsuite at P4.
#
#   S=<cand>/source/src/ffi/cpp/select_gpu.sh
#   srun -N 1 -n 1 --gpus-per-node=4 $S bash config/perlmutter/module/accept.sh <cand>
#   srun -N 1 -n 4 --gpus-per-node=4 $S bash config/perlmutter/module/accept.sh <cand>
#
# Launch from a login shell with no LORRAX module loaded and no LORRAX tree on PYTHONPATH.
# Passing writes accept/gate10.ok (-n 1) or accept/ok.<rank> (-n 4);
# publish.sh needs all five.
# The candidate must be writable from compute nodes: tests/hsuite writes its
# runs and compile cache inside the source tree, which is why acceptance runs
# on the candidate and not on a published release under /global/common.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAND="$(realpath -e "${1:?usage: accept.sh <candidate dir>}")"
# shellcheck disable=SC1091
source "$CAND/CANDIDATE"
RANK="${SLURM_PROCID:?run under srun, one rank per GPU}"
# NERSC's login profile puts /opt/nersc/pymon on PYTHONPATH; a LORRAX tree
# there would shadow the snapshot, and verify_runtime.py would refuse later.
IFS=: read -r -a _pp <<< "${PYTHONPATH:-}"
for d in "${_pp[@]}"; do
    [[ ! -e "$d/runtime/__init__.py" ]] || {
        echo "[accept] REFUSED: a LORRAX tree is on PYTHONPATH ($d); launch from a clean shell" >&2
        exit 2
    }
done
type module >/dev/null 2>&1 || source /opt/cray/pe/lmod/lmod/init/bash
module use "$CAND/modulefiles"
module load "$MODULE_NAME/$MODULE_VERSION"
# shellcheck disable=SC1091
source "$LORRAX_ROOT/config/perlmutter/gpu_env.sh"
[[ "$LORRAX_ROOT" == "$CAND/source" ]] || { echo "[accept] REFUSED: LORRAX_ROOT=$LORRAX_ROOT" >&2; exit 2; }

OUT="$CAND/accept"
mkdir -p "$OUT"
cd "$OUT"
say() { if [[ "$RANK" == 0 ]]; then echo "[accept] $*"; fi; }
say "module $MODULE_NAME/$MODULE_VERSION: source ${REV:0:12}, bundle ${BUNDLE_ID:0:12}"

case "${SLURM_NTASKS:?}" in
1)
    mkdir -p "$OUT/gate10"
    PROBE_DIR="$OUT/gate10" python -u "$LORRAX_ROOT/src/ffi/cpp/gate_one_odr.py"
    touch "$OUT/gate10.ok"
    say "Gate 10 PASS"
    ;;
4)
    LORRAX_MODULE_EXPECT="$EXPECT" python -u "$HERE/verify_runtime.py"
    python -m pytest "$LORRAX_ROOT/tests/hsuite" -q -p no:cacheprovider
    touch "$OUT/ok.$RANK"
    say "P4 PASS on rank 0 (publish.sh checks every rank)"
    ;;
*)
    echo "[accept] REFUSED: run with -n 1 (Gate 10) or -n 4 (P4), not $SLURM_NTASKS" >&2
    exit 2
    ;;
esac
