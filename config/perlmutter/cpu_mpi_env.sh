#!/usr/bin/env bash
# Source inside every Perlmutter CPU rank shell before importing Python/JAX:
#
#   . "$LORRAX_ROOT/config/perlmutter/cpu_mpi_env.sh"
#
# This selects JAX's MPI CPU collectives while retaining Cray MPICH as the MPI
# implementation.  The prelude fails closed on stale Frontera overlays,
# multiple CPU devices per process, and untracked/modified adapters.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "[cpu_mpi_env.pm] ERROR: source this file; do not execute it" >&2
    exit 2
fi

_lorrax_pm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
for _lorrax_pm_source_pin in "${LORRAX_ROOT:-}" "${LORRAX_CHECKOUT:-}"; do
    if [[ -n "$_lorrax_pm_source_pin" && \
          "$(realpath -m "$_lorrax_pm_source_pin")" != "$_lorrax_pm_root" ]]; then
        echo "[cpu_mpi_env.pm] ERROR: inherited source pin disagrees with the sourced prelude" >&2
        echo "[cpu_mpi_env.pm]   prelude=$_lorrax_pm_root pin=$(realpath -m "$_lorrax_pm_source_pin")" >&2
        return 2
    fi
done
_lorrax_pm_site="$_lorrax_pm_root/config/perlmutter/site_config.sh"
_lorrax_pm_one_mpi_gate="$_lorrax_pm_root/src/ffi/cpp/gate_one_mpi.sh"
if [[ ! -r "$_lorrax_pm_site" ]]; then
    echo "[cpu_mpi_env.pm] ERROR: cannot read $_lorrax_pm_site" >&2
    return 2
fi
# shellcheck disable=SC1090
. "$_lorrax_pm_site"
: "${LORRAX_MPIWRAPPER_PREFIX:=$LORRAX_MPIWRAPPER_PREFIX_DEFAULT}"
: "${LORRAX_MPIWRAPPER_SO:=$LORRAX_MPIWRAPPER_PREFIX/lib64/libmpiwrapper.so}"

if [[ "$LORRAX_MPIWRAPPER_SO" != /* || ! -f "$LORRAX_MPIWRAPPER_SO" ]]; then
    echo "[cpu_mpi_env.pm] ERROR: LORRAX_MPIWRAPPER_SO must name an absolute file:" >&2
    echo "[cpu_mpi_env.pm]   $LORRAX_MPIWRAPPER_SO" >&2
    echo "[cpu_mpi_env.pm] Build it with config/perlmutter/build_mpiwrapper.sh." >&2
    return 2
fi
_lorrax_pm_so="$(readlink -f "$LORRAX_MPIWRAPPER_SO")"
_lorrax_pm_prefix="$(dirname "$(dirname "$_lorrax_pm_so")")"
_lorrax_pm_manifest="$_lorrax_pm_prefix/build-manifest.txt"
if [[ ! -r "$_lorrax_pm_manifest" ]]; then
    echo "[cpu_mpi_env.pm] ERROR: adapter has no tracked build manifest: $_lorrax_pm_manifest" >&2
    return 2
fi
_lorrax_pm_commit="$(sed -n 's/^source_commit=//p' "$_lorrax_pm_manifest")"
_lorrax_pm_abi="$(sed -n 's/^mpiabi_version=//p' "$_lorrax_pm_manifest")"
_lorrax_pm_expected_sha="$(sed -n 's/^artifact_sha256=//p' "$_lorrax_pm_manifest")"
_lorrax_pm_source_modified="$(sed -n 's/^source_modified=//p' "$_lorrax_pm_manifest")"
_lorrax_pm_recipe_dirty="$(sed -n 's/^lorrax_recipe_dirty=//p' "$_lorrax_pm_manifest")"
_lorrax_pm_mpich="$(sed -n 's/^cray_mpich_version=//p' "$_lorrax_pm_manifest")"
_lorrax_pm_builder_sha="$(sed -n 's/^builder_sha256=//p' "$_lorrax_pm_manifest")"
_lorrax_pm_prelude_sha="$(sed -n 's/^prelude_sha256=//p' "$_lorrax_pm_manifest")"
_lorrax_pm_site_sha="$(sed -n 's/^site_config_sha256=//p' "$_lorrax_pm_manifest")"
_lorrax_pm_one_mpi_gate_sha="$(sed -n 's/^one_mpi_gate_sha256=//p' "$_lorrax_pm_manifest")"
_lorrax_pm_actual_sha="$(sha256sum "$_lorrax_pm_so" | awk '{print $1}')"
_lorrax_pm_want_builder_sha="$(sha256sum "$_lorrax_pm_root/config/perlmutter/build_mpiwrapper.sh" | awk '{print $1}')"
_lorrax_pm_want_prelude_sha="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
_lorrax_pm_want_site_sha="$(sha256sum "$_lorrax_pm_site" | awk '{print $1}')"
_lorrax_pm_want_one_mpi_gate_sha="$(sha256sum "$_lorrax_pm_one_mpi_gate" | awk '{print $1}')"
if [[ "$_lorrax_pm_commit" != "$LORRAX_MPIWRAPPER_COMMIT_DEFAULT" || \
      "$_lorrax_pm_abi" != "$LORRAX_MPIWRAPPER_ABI_DEFAULT" || \
      -z "$_lorrax_pm_expected_sha" || \
      "$_lorrax_pm_actual_sha" != "$_lorrax_pm_expected_sha" || \
      "$_lorrax_pm_source_modified" != false || \
      "$_lorrax_pm_recipe_dirty" != false || \
      "$_lorrax_pm_mpich" != "${LORRAX_PM_MPICH_DEFAULT##*/}" || \
      "$_lorrax_pm_builder_sha" != "$_lorrax_pm_want_builder_sha" || \
      "$_lorrax_pm_prelude_sha" != "$_lorrax_pm_want_prelude_sha" || \
      "$_lorrax_pm_site_sha" != "$_lorrax_pm_want_site_sha" || \
      "$_lorrax_pm_one_mpi_gate_sha" != "$_lorrax_pm_want_one_mpi_gate_sha" ]]; then
    echo "[cpu_mpi_env.pm] ERROR: adapter provenance/toolchain/recipe check failed" >&2
    echo "[cpu_mpi_env.pm]   commit=$_lorrax_pm_commit ABI=$_lorrax_pm_abi MPICH=$_lorrax_pm_mpich" >&2
    echo "[cpu_mpi_env.pm]   upstream_modified=$_lorrax_pm_source_modified recipe_dirty=$_lorrax_pm_recipe_dirty" >&2
    return 2
fi
if find "$_lorrax_pm_prefix" -perm /222 -print -quit | grep -q .; then
    echo "[cpu_mpi_env.pm] ERROR: adapter release is writable: $_lorrax_pm_prefix" >&2
    return 2
fi

# Refuse the Frontera mpi4py/parallel-h5py/sitecustomize overlay and all ways
# of injecting a second MPI implementation into this Cray-MPICH process.
if [[ "${PYTHONPATH:-}" == *lorrax_env_mpi_overlay* || \
      -n "${LORRAX_MPI_INIT_FIRST:-}" || \
      -n "${LORRAX_MPI_FINALIZE_FIX:-}" ]]; then
    echo "[cpu_mpi_env.pm] ERROR: stale Frontera MPI overlay/lifecycle controls are present" >&2
    return 2
fi
_lorrax_pm_pmi=/opt/cray/pe/lib64/libpmi.so.0
if [[ ! -f "$_lorrax_pm_pmi" || "$(readlink -f "$_lorrax_pm_pmi")" != /opt/cray/pe/* ]]; then
    echo "[cpu_mpi_env.pm] ERROR: the Cray PMI library is absent: $_lorrax_pm_pmi" >&2
    return 2
fi
_lorrax_pm_preloads="${LD_PRELOAD:-}"
for _lorrax_pm_preload in ${_lorrax_pm_preloads//:/ }; do
    case "$(basename "$_lorrax_pm_preload")" in
        libmpi.so*|libmpi_*|libmpitrampoline*|libmpiwrapper*|libpmi2.so*)
            echo "[cpu_mpi_env.pm] ERROR: conflicting MPI/PMI preload: $_lorrax_pm_preload" >&2
            return 2
            ;;
        libpmi.so.0)
            if [[ "$(readlink -f "$_lorrax_pm_preload" 2>/dev/null || true)" != \
                  "$(readlink -f "$_lorrax_pm_pmi")" ]]; then
                echo "[cpu_mpi_env.pm] ERROR: non-Cray PMI preload: $_lorrax_pm_preload" >&2
                return 2
            fi
            ;;
    esac
done

if [[ -n "${JAX_PLATFORMS:-}" && "$JAX_PLATFORMS" != cpu ]]; then
    echo "[cpu_mpi_env.pm] ERROR: JAX_PLATFORMS must be cpu, got '$JAX_PLATFORMS'" >&2
    return 2
fi
if [[ -n "${JAX_NUM_CPU_DEVICES:-}" && "$JAX_NUM_CPU_DEVICES" != 1 ]]; then
    echo "[cpu_mpi_env.pm] ERROR: JAX_NUM_CPU_DEVICES must be 1 for MPI collectives" >&2
    return 2
fi
if [[ -n "${JAX_CPU_COLLECTIVES_IMPLEMENTATION:-}" && \
      "$JAX_CPU_COLLECTIVES_IMPLEMENTATION" != mpi ]]; then
    echo "[cpu_mpi_env.pm] ERROR: CPU collectives implementation must be mpi" >&2
    return 2
fi

# Cray's PMI library must be resident before jax.distributed starts its
# coordination threads.  Late dlopen otherwise crashes in _pmi_spawn_init.
case ":${LD_PRELOAD:-}:" in
    *"$_lorrax_pm_pmi"*) ;;
    *) export LD_PRELOAD="$_lorrax_pm_pmi${LD_PRELOAD:+:$LD_PRELOAD}" ;;
esac

export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export JAX_NUM_CPU_DEVICES=1
export JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi
export MPITRAMPOLINE_LIB="$_lorrax_pm_so"
# HPE's supported public setting promotes XLA's explicit FUNNELED request to
# MULTIPLE and creates one Cray progress thread.  Never mix it with the
# internal MPIR_CVAR spelling used during early diagnosis.
export MPICH_ASYNC_PROGRESS=1
unset MPIR_CVAR_ASYNC_PROGRESS
export MPICH_GPU_SUPPORT_ENABLED=0

# Clique construction is owned by common.collectives.warm_mesh_cliques().
unset LORRAX_MPI_FORCE_THREAD_MAIN

if [[ "${SLURM_PROCID:-0}" == 0 ]]; then
    echo "[cpu_mpi_env.pm] JAX CPU collectives: MPItrampoline -> $MPITRAMPOLINE_LIB"
    echo "[cpu_mpi_env.pm] adapter: MPI ABI $_lorrax_pm_abi, sha256 ${_lorrax_pm_actual_sha:0:12}"
    echo "[cpu_mpi_env.pm] Cray MPI: libpmi.so.0 preloaded; MPICH_ASYNC_PROGRESS=1"
    echo "[cpu_mpi_env.pm] Async progress needs affinity headroom; placement is launcher-owned."
fi

unset _lorrax_pm_root _lorrax_pm_site _lorrax_pm_so _lorrax_pm_prefix
unset _lorrax_pm_manifest _lorrax_pm_commit _lorrax_pm_abi
unset _lorrax_pm_expected_sha _lorrax_pm_actual_sha _lorrax_pm_preload
unset _lorrax_pm_pmi _lorrax_pm_preloads
unset _lorrax_pm_source_modified _lorrax_pm_recipe_dirty _lorrax_pm_mpich
unset _lorrax_pm_builder_sha _lorrax_pm_prelude_sha _lorrax_pm_site_sha
unset _lorrax_pm_want_builder_sha _lorrax_pm_want_prelude_sha
unset _lorrax_pm_want_site_sha _lorrax_pm_source_pin
unset _lorrax_pm_one_mpi_gate _lorrax_pm_one_mpi_gate_sha
unset _lorrax_pm_want_one_mpi_gate_sha
