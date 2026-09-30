#!/usr/bin/env bash
# ============================================================================
# Accept one module candidate at P4, per rank, through the candidate module,
# from a directory that is not a checkout (the candidate's accept/):
#   Gate 10 (src/ffi/cpp/gate_one_odr.py), verify_runtime.py, tests/hsuite.
#
#   srun -N 1 -n 4 --gpus-per-node=4 <cand>/source/src/ffi/cpp/select_gpu.sh \
#     bash config/perlmutter/module/accept.sh <cand>
#
# Launch from a login shell with no LORRAX module loaded and no PYTHONPATH.
# Each rank that passes writes accept/ok.<rank>; publish.sh needs all four.
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
[[ -z "${PYTHONPATH:-}" && -z "${LORRAX_FFI_SO:-}${LORRAX_FFI_HOST_SO:-}" ]] || {
    echo "[accept] REFUSED: PYTHONPATH or an FFI pin is set; launch from a clean shell" >&2
    exit 2
}
type module >/dev/null 2>&1 || source /opt/cray/pe/lmod/lmod/init/bash
module use "$CAND/modulefiles"
module load "$MODULE_NAME/$MODULE_VERSION"
# shellcheck disable=SC1091
source "$LORRAX_ROOT/config/perlmutter/gpu_env.sh"
[[ "$LORRAX_ROOT" == "$CAND/source" ]] || { echo "[accept] REFUSED: LORRAX_ROOT=$LORRAX_ROOT" >&2; exit 2; }

OUT="$CAND/accept"
mkdir -p "$OUT/gate10.$RANK"
cd "$OUT"
say() { if [[ "$RANK" == 0 ]]; then echo "[accept] $*"; fi; }

say "module $MODULE_NAME/$MODULE_VERSION: source ${REV:0:12}, bundle ${BUNDLE_ID:0:12}"
PROBE_DIR="$OUT/gate10.$RANK" python -u "$LORRAX_ROOT/src/ffi/cpp/gate_one_odr.py" \
    > "$OUT/gate10.$RANK.log" 2>&1 \
    || { echo "[accept] rank $RANK: Gate 10 FAILED ($OUT/gate10.$RANK.log)" >&2; exit 1; }
say "Gate 10 PASS"
LORRAX_MODULE_EXPECT="$EXPECT" python -u "$HERE/verify_runtime.py"
python -m pytest "$LORRAX_ROOT/tests/hsuite" -q -p no:cacheprovider
touch "$OUT/ok.$RANK"
say "ACCEPTED (rank 0; publish.sh checks all four ranks)"
