#!/usr/bin/env bash
set -euo pipefail

_die() {
  echo "ERROR: $*" >&2
  exit 1
}

_require_file() {
  [[ -f "$1" ]] || _die "Missing required file: $1"
}

load_config() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local cfg="${here}/config"
  _require_file "${cfg}"

  # shellcheck disable=SC1090
  source "${cfg}"

  : "${ISDF_SLURM_ACCOUNT:?Set ISDF_SLURM_ACCOUNT in cluster_shifter/config}"
  : "${ISDF_CODE_HOST_PATH:?Set ISDF_CODE_HOST_PATH in cluster_shifter/config}"
  : "${ISDF_VENV_HOST_PATH:?Set ISDF_VENV_HOST_PATH in cluster_shifter/config}"
  : "${ISDF_DEFAULT_INPUT_BASENAME:=cohsex_test.in}"

  : "${ISDF_SHIFTER_IMAGE:=nvcr.io/nvidia/jax:25.04-py3}"
  : "${ISDF_SHIFTER_MODULES:=gpu,nccl-plugin}"
  : "${ISDF_WHEELHOUSE_HOST_PATH:=}"

  [[ -d "${ISDF_CODE_HOST_PATH}" ]] || _die "ISDF_CODE_HOST_PATH does not exist: ${ISDF_CODE_HOST_PATH}"
  [[ -d "${ISDF_VENV_HOST_PATH}" ]] || _die "ISDF_VENV_HOST_PATH does not exist: ${ISDF_VENV_HOST_PATH}"
}

make_volumes_base() {
  # Print a newline-delimited list of --volume=... args.
  echo "--volume=${ISDF_CODE_HOST_PATH}:/workspace/ISDF"
  echo "--volume=${ISDF_VENV_HOST_PATH}:/workspace/venvroot"
  if [[ -n "${ISDF_WHEELHOUSE_HOST_PATH}" ]]; then
    [[ -d "${ISDF_WHEELHOUSE_HOST_PATH}" ]] || _die "ISDF_WHEELHOUSE_HOST_PATH does not exist: ${ISDF_WHEELHOUSE_HOST_PATH}"
    # Mount wheelhouse to a stable path; job scripts will reference it.
    echo "--volume=${ISDF_WHEELHOUSE_HOST_PATH}:/workspace/wheelhouse"
  fi
}

make_volume_run_dir() {
  # Given a host input file path, mount its directory to /workspace/run.
  local input_host_path="$1"
  [[ -f "${input_host_path}" ]] || _die "Input file does not exist: ${input_host_path}"
  local run_dir
  run_dir="$(cd "$(dirname "${input_host_path}")" && pwd)"
  echo "--volume=${run_dir}:/workspace/run"
}

make_volume_pwd_run_dir() {
  # Mount the current working directory as /workspace/run.
  local run_dir
  run_dir="$(pwd)"
  [[ -d "${run_dir}" ]] || _die "PWD is not a directory: ${run_dir}"
  echo "--volume=${run_dir}:/workspace/run"
}


