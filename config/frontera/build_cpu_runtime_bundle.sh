#!/bin/bash
# ===========================================================================
# build_cpu_runtime_bundle.sh -- pack the CPU run-time into ONE streamable file
#
# WHY THIS EXISTS
# ---------------
# A LORRAX CPU run on Frontera pays its startup cost reading the venv off
# Lustre, through the container, one mmap page fault at a time.  Measured on
# a fresh compute node (job 7882047):
#
#     /work2  streaming read  ................  70 MB/s
#     /scratch2 streaming read ...............  560 MB/s
#     node-local /tmp (XFS on the 224 GB SSD)    144 GB free, writable
#                                                 inside the container
#
# and the venv that gets read is 5.6 GB, of which 4.9 GB is the CUDA stack
# (nvidia/*, jax_plugins/, jax_cuda12_plugin/) that a CPU run can never use
# -- but DOES load: jax calls xla_bridge.discover_pjrt_plugins() from
# backends() BEFORE it filters on JAX_PLATFORMS (jax 0.9.1
# jax/_src/xla_bridge.py:797 vs :808), and jax_plugins.xla_cuda12.initialize()
# opens with _load_nvidia_libraries() -- ctypes.cdll.LoadLibrary on libcudart,
# libnvrtc, libcublas, libcublasLt, libnccl, libcupti, libcusparse,
# libcusolver, libcufft, libnvshmem_host, libcudnn.  JAX_PLATFORMS=cpu does
# not prevent one byte of that.
#
# So this script builds the CPU bundle: the venv WITHOUT the CUDA stack, the
# MPI overlay, and the LORRAX sources, byte-compiled, in a single tar on the
# fast filesystem.  stage_runtime.sh then unrolls it onto each node's local
# SSD once per node, and the run reads a local disk instead of Lustre.
#
# NOT A PATCHED DEPENDENCY.  Every file in the bundle is a byte-for-byte copy
# of what pip/uv installed.  Removing a package that the run does not use is a
# deployment choice, not a source modification: the GPU venv is untouched and
# GPU runs keep using it.
#
# USAGE (login node is fine; only tar+cp, no container needed for --no-compile)
#   config/frontera/build_cpu_runtime_bundle.sh                    # default
#   config/frontera/build_cpu_runtime_bundle.sh --out DIR --no-compile
#   # byte-compiling needs the container's python 3.12:
#   apptainer exec --bind /home1,/work2,/scratch1,/scratch2 $LORRAX_SIF \
#       config/frontera/build_cpu_runtime_bundle.sh
#
# ENV
#   LORRAX_VENV     source venv        (default $WORK/lorrax_env/.venv)
#   LORRAX_OVERLAY  MPI overlay site   (default $WORK/lorrax_env_mpi_overlay/site)
#   LORRAX_SRC      LORRAX src tree    (default <repo>/src)
#   LORRAX_BUNDLE_OUT  output dir      (default $SCRATCH/lorrax_bundle)
# ===========================================================================
set -euo pipefail

VENV=${LORRAX_VENV:-$WORK/lorrax_env/.venv}
OVERLAY=${LORRAX_OVERLAY:-$WORK/lorrax_env_mpi_overlay/site}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${LORRAX_SRC:-$(cd "$HERE/../.." && pwd)/src}
OUT=${LORRAX_BUNDLE_OUT:-$SCRATCH/lorrax_bundle}
COMPILE=1
NAME=lorrax_cpu_bundle

while [ $# -gt 0 ]; do
  case "$1" in
    --out)        OUT=$2; shift 2 ;;
    --venv)       VENV=$2; shift 2 ;;
    --src)        SRC=$2; shift 2 ;;
    --overlay)    OVERLAY=$2; shift 2 ;;
    --name)       NAME=$2; shift 2 ;;
    --no-compile) COMPILE=0; shift ;;
    -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -d "$VENV" ]    || { echo "*** no venv at $VENV" >&2; exit 3; }
[ -d "$SRC" ]     || { echo "*** no src at $SRC" >&2; exit 3; }
[ -d "$OVERLAY" ] || { echo "*** no overlay at $OVERLAY" >&2; exit 3; }

mkdir -p "$OUT"
STAGE=$(mktemp -d "$OUT/.build.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT

echo "[bundle] venv    = $VENV"
echo "[bundle] overlay = $OVERLAY"
echo "[bundle] src     = $SRC"
echo "[bundle] out     = $OUT/$NAME.tar"

# --- what a CPU run can never use -------------------------------------------
# nvidia/*                 the CUDA runtime libraries (4.4 GB)
# jax_plugins/*            the PJRT CUDA plugin + its 416 MB .so
# jax_cuda12_plugin/*      the plugin's python extensions
# jax_cuda12_pjrt-*.dist-info  carries entry_points.txt for the "jax_plugins"
#                          group; dropping it is what stops jax's entry-point
#                          scan from advertising a plugin that is not there
# cmake/, ninja/           build-time only
#
# __pycache__ is deliberately NOT excluded: the bundle must be a faithful copy
# of the tree it replaces, so that --no-compile measures "same files, local
# disk" and the difference to the compiled bundle is the bytecode effect ALONE.
EXCL=(
  --exclude='*/site-packages/nvidia'
  --exclude='*/site-packages/nvidia_*'
  --exclude='*/site-packages/jax_plugins'
  --exclude='*/site-packages/jax_cuda12_plugin'
  --exclude='*/site-packages/jax_cuda12_plugin-*'
  --exclude='*/site-packages/jax_cuda12_pjrt-*'
  --exclude='*/site-packages/cmake'
  --exclude='*/site-packages/cmake-*'
  --exclude='*/site-packages/ninja'
  --exclude='*/site-packages/ninja-*'
)

echo "[bundle] copying trimmed venv ..."
mkdir -p "$STAGE/venv"
tar -C "$VENV" "${EXCL[@]}" -cf - . | tar -C "$STAGE/venv" -xf -
for gone in nvidia jax_plugins jax_cuda12_plugin; do
  if [ -e "$STAGE/venv/lib/python3.12/site-packages/$gone" ]; then
    echo "*** FATAL: exclusion of $gone did not take effect -- the bundle"
    echo "    would still carry the CUDA stack and measure nothing." >&2
    exit 4
  fi
done

echo "[bundle] copying overlay + src ..."
mkdir -p "$STAGE/overlay" "$STAGE/src"
tar -C "$OVERLAY" -cf - . | tar -C "$STAGE/overlay" -xf -
tar -C "$SRC"     -cf - . | tar -C "$STAGE/src" -xf -

if [ "$COMPILE" = 1 ]; then
  # Byte-compile everything.  The venv as uv installs it is only partially
  # compiled (measured: 1290 .pyc for 3760 .py; jax itself 226/610), and the
  # run-time env sets PYTHONDONTWRITEBYTECODE=1 because the tree is shared and
  # read-only -- so without this step every process re-compiles those modules
  # from source on every run, forever.
  PY=$("$STAGE/venv/bin/python" -c 'import sys; print(sys.executable)' 2>/dev/null) || PY=""
  if [ -z "$PY" ]; then
    echo "[bundle] WARNING: the staged venv's python does not run here "
    echo "[bundle]          (byte-compiling needs the container's python 3.12)."
    echo "[bundle]          Re-run inside the SIF, or pass --no-compile and accept"
    echo "[bundle]          that every run re-compiles ~2500 modules from source."
  else
    echo "[bundle] byte-compiling (quiet; failures are per-file and tolerated) ..."
    "$STAGE/venv/bin/python" -m compileall -q -j 0 \
        "$STAGE/venv/lib/python3.12/site-packages" \
        "$STAGE/overlay" "$STAGE/src" >/dev/null 2>&1 || true
    NPYC=$(find "$STAGE" -name '*.pyc' | wc -l)
    NPY=$(find "$STAGE" -name '*.py' | wc -l)
    echo "[bundle] byte-compiled: $NPYC .pyc for $NPY .py"
    if [ "$NPYC" -lt $(( NPY / 2 )) ]; then
      echo "[bundle] *** WARNING: fewer than half the modules compiled; the"
      echo "[bundle]     staged bundle will still pay source compilation at run"
      echo "[bundle]     time.  Check that compileall ran under python 3.12."
    fi
  fi
fi

# A manifest so a staged node can prove WHAT it staged.
( cd "$STAGE" && find . -type f | sort | xargs sha256sum 2>/dev/null | sha256sum \
    | awk '{print $1}' ) > "$STAGE/BUNDLE_SHA256" || true
{
  echo "built     $(date -Is)"
  echo "host      $(hostname)"
  echo "venv_src  $VENV"
  echo "src_src   $SRC"
  echo "overlay   $OVERLAY"
  echo "compiled  $COMPILE"
  echo "size_mb   $(du -sm "$STAGE" | awk '{print $1}')"
} > "$STAGE/BUNDLE_INFO"

echo "[bundle] taring ..."
# Stripe the tar wide BEFORE writing it (Lustre stripe is fixed at create):
# staging reads this one file from every node of the job at once, and a
# single-OST file serialises that.  Best effort -- lfs may be absent.
STRIPE=${LORRAX_BUNDLE_STRIPE:-12}
rm -f "$OUT/$NAME.tar.part"
HAVE_LFS=0
command -v lfs >/dev/null 2>&1 && HAVE_LFS=1
if [ "$HAVE_LFS" = 1 ]; then
  lfs setstripe -c "$STRIPE" "$OUT/$NAME.tar.part" 2>/dev/null || HAVE_LFS=0
fi
tar -C "$STAGE" -cf "$OUT/$NAME.tar.part" .
mv "$OUT/$NAME.tar.part" "$OUT/$NAME.tar"
cp "$STAGE/BUNDLE_INFO" "$OUT/$NAME.info"
if [ "$HAVE_LFS" = 1 ]; then
  echo "[bundle] tar stripe_count = $(lfs getstripe -c "$OUT/$NAME.tar" 2>/dev/null)"
else
  # The SIF is python:3.12-bookworm and carries no Lustre client tools, so a
  # bundle built inside the container lands on ONE OST.  That is fine for a
  # handful of nodes and a bottleneck for a large job, and it is silent unless
  # said out loud (observed: job 7882121 read a stripe_count=1 bundle).
  cat <<EOF
[bundle] WARNING: 'lfs' is not available here (expected when building inside
[bundle]          the SIF), so the tar is on the filesystem default stripe --
[bundle]          probably ONE OST, which serialises a large job's staging.
[bundle]          Fix it from a LOGIN node, once:
[bundle]              cd $OUT
[bundle]              lfs setstripe -c $STRIPE $NAME.striped.tar
[bundle]              cp $NAME.tar $NAME.striped.tar && mv $NAME.striped.tar $NAME.tar
[bundle]          Do NOT reach for sbcast instead: measured on Frontera it took
[bundle]          54.0 s to push this tar to 8 nodes, against 1.26 s for the 8
[bundle]          nodes reading it off Lustre themselves (job 7882128).
EOF
fi

echo "[bundle] DONE  $(du -h "$OUT/$NAME.tar" | awk '{print $1}')  $OUT/$NAME.tar"
cat "$OUT/$NAME.info"
