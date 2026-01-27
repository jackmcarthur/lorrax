#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${here}/_lib.sh"
load_config

sbatch_script="${here}/perlmutter_shifter_singlenode.sbatch"

input_host_path="${1:-}"
run_vol=()
export_args=()
if [[ -n "${input_host_path}" ]]; then
  mapfile -t run_vol < <(make_volume_run_dir "${input_host_path}")
  input_base="$(basename "${input_host_path}")"
  export_args+=(--export="ALL,ISDF_INPUT_IN_CONTAINER=/workspace/run/${input_base}")
else
  mapfile -t run_vol < <(make_volume_pwd_run_dir)
  export_args+=(--export="ALL,ISDF_INPUT_IN_CONTAINER=/workspace/run/${ISDF_DEFAULT_INPUT_BASENAME}")
fi

mapfile -t base_vols < <(make_volumes_base)

echo "[submit] account:  ${ISDF_SLURM_ACCOUNT}"
echo "[submit] image:    ${ISDF_SHIFTER_IMAGE}"
echo "[submit] modules:  ${ISDF_SHIFTER_MODULES}"
echo "[submit] code:     ${ISDF_CODE_HOST_PATH}"
echo "[submit] venvroot: ${ISDF_VENV_HOST_PATH}"
if [[ -n "${input_host_path}" ]]; then
  echo "[submit] input:    ${input_host_path}"
  echo "[submit] run dir:  $(cd "$(dirname "${input_host_path}")" && pwd)"
else
  echo "[submit] input:    (default) ${ISDF_DEFAULT_INPUT_BASENAME}"
  echo "[submit] run dir:  $(pwd)"
fi

sbatch \
  --account="${ISDF_SLURM_ACCOUNT}" \
  --image="${ISDF_SHIFTER_IMAGE}" \
  --module="${ISDF_SHIFTER_MODULES}" \
  "${export_args[@]}" \
  "${base_vols[@]}" \
  "${run_vol[@]}" \
  "${sbatch_script}"


