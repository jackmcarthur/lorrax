# shellcheck shell=bash
# ============================================================================
# Perlmutter machine settings for a GPU step launched from a clone with srun
# (docs/installation/perlmutter.md).  Source it in the shell that runs srun;
# srun passes the environment to every rank.  The CPU twin is cpu_mpi_env.sh.
#
#   source config/perlmutter/gpu_env.sh
#
# MPICH_GPU_SUPPORT_ENABLED=0.  The site's default craype-accel-nvidia80
# module exports 1, which makes Cray MPICH's MPI_Init demand the GTL library
# ("GPU_SUPPORT_ENABLED is requested, but GTL library is not linked") and
# abort.  Neither FFI leg links GTL: the host leg is CUDA-free and the CUDA
# leg's cuSOLVERMp/cuBLASMp communicate through NCCL.
#
# LD_LIBRARY_PATH  prepended with ~/.cache/lorrax/cudart-hmem/<id> (<id>: the
#                  md5 of the target's path, so two venvs never share a link),
#                  holding libcudart.so -> the CUDA runtime the run python's
#                  JAX plugin resolves (nvidia/cu*/lib/libcudart.so.<major>).
#                  libfabric's CUDA memory interface, used by the NCCL OFI
#                  plugin of a multi-node step, dlopens that unversioned name;
#                  the wheel ships only the versioned file, so without the link
#                  the name resolves to the toolkit module's runtime and the
#                  process maps a second CUDA runtime, which the dependency-
#                  closure gate refuses (site.toml [nccl] launch_libraries).
#                  A python without a CUDA runtime wheel (the module route,
#                  whose record names the toolkit runtime) leaves it unchanged.
#                  The run python: $VIRTUAL_ENV's, else this checkout's .venv,
#                  else python3.  Idempotent.  Skipped under a lorrax module
#                  that stages nvidia/ (LORRAX_STAGE_KEY): its lorrax-launch link
#                  names the staged runtime.
# ============================================================================
# Slurm consumes this when creating a future step. Never claim to repair an
# already-running numeric step; preserve every explicit value, including empty.
if [[ ! ${SLURM_STEP_ID-} =~ ^[0-9]+$ && ! ${SLURM_STEPID-} =~ ^[0-9]+$ ]]; then
    export SLURM_NETWORK="${SLURM_NETWORK-no_vni}"
fi

export MPICH_GPU_SUPPORT_ENABLED=0

# CUDA_CACHE_PATH.  The CUDA driver's JIT cache defaults to ~/.nv/ComputeCache;
# one fresh test suite plus the P1 chain put 208 MB there, and NERSC home
# quotas are small.
# A value the caller set is kept.
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${SCRATCH:?SCRATCH is not set}/.nv/ComputeCache}"

_lorrax_py="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
_lorrax_here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "$_lorrax_py" ] || { [ -x "$_lorrax_here/.venv/bin/python" ] && _lorrax_py="$_lorrax_here/.venv/bin/python"; }
_lorrax_cudart="$("${_lorrax_py:-python3}" - <<'PY' 2>/dev/null
import glob, os, sysconfig
hits = sorted(glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu*", "lib", "libcudart.so.[0-9]*")))
print(os.path.realpath(hits[0]) if hits else "")
PY
)"
# The lorrax module that stages nvidia/ node-local (LORRAX_STAGE_KEY) already lists
# its lorrax-launch/libcudart.so -> the staged runtime first; a shim to the venv's
# copy would map a second runtime beside the staged one.
if [ -n "$_lorrax_cudart" ] && [ -z "${LORRAX_STAGE_KEY:-}" ]; then
    _lorrax_shim="${XDG_CACHE_HOME:-$HOME/.cache}/lorrax/cudart-hmem/$(printf %s "$_lorrax_cudart" | md5sum | cut -c1-12)"
    mkdir -p "$_lorrax_shim"
    ln -sfn "$_lorrax_cudart" "$_lorrax_shim/libcudart.so"
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$_lorrax_shim:"*) ;;
        *) export LD_LIBRARY_PATH="$_lorrax_shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi
unset _lorrax_py _lorrax_here _lorrax_cudart _lorrax_shim
