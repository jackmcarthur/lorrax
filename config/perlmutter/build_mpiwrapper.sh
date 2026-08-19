#!/usr/bin/env bash
# Build and atomically activate the unmodified MPIwrapper adapter used by
# JAX CPU MPI collectives on Perlmutter.  Cray MPICH remains the MPI/network
# implementation; this product is only MPItrampoline's ABI adapter.
#
# Usage (on a CPU compute node):
#   config/perlmutter/build_mpiwrapper.sh [--fresh]
#
# Every invocation is a fresh candidate build.  ``--fresh`` is retained as a
# harmless, explicit spelling for the documented recipe; it never removes the
# active release.  A candidate becomes ``$ROOT/current`` only after all ABI,
# dependency, and one-MPI-runtime gates pass.
set -euo pipefail

MPIW_REPO="${LORRAX_MPIWRAPPER_REPO:-https://github.com/eschnett/MPIwrapper.git}"
LORRAX_ROOT="${LORRAX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SITE_CONFIG="$LORRAX_ROOT/config/perlmutter/site_config.sh"
if [[ ! -r "$SITE_CONFIG" ]]; then
    echo "[build_mpiw.pm] ERROR: cannot read $SITE_CONFIG" >&2
    exit 2
fi
# shellcheck disable=SC1090
. "$SITE_CONFIG"
MPIW_COMMIT="$LORRAX_MPIWRAPPER_COMMIT_DEFAULT"
if [[ -n "${LORRAX_MPIWRAPPER_COMMIT:-}" && \
      "$LORRAX_MPIWRAPPER_COMMIT" != "$MPIW_COMMIT" ]]; then
    echo "[build_mpiw.pm] ERROR: Perlmutter production builds pin MPIwrapper $MPIW_COMMIT" >&2
    exit 2
fi
MPIW_ROOT="${LORRAX_MPIWRAPPER_ROOT:-$LORRAX_MPIWRAPPER_ROOT_DEFAULT}"

if [[ "${1:-}" != "" && "${1:-}" != "--fresh" ]]; then
    echo "[build_mpiw.pm] ERROR: unknown argument: $1" >&2
    exit 2
fi

# This script owns the layout below MPIW_ROOT.  Do not accept independently
# redirected source/build/install paths: that was how --fresh could target an
# unrelated directory or delete the active adapter before a replacement was
# known good.
MPIW_ROOT="$(realpath -m "$MPIW_ROOT")"
case "$MPIW_ROOT" in
    ""|/|"$HOME"|"$HOME/software"|"$LORRAX_ROOT"|"$LORRAX_ROOT"/*)
        echo "[build_mpiw.pm] ERROR: unsafe adapter root: $MPIW_ROOT" >&2
        exit 2
        ;;
esac
if [[ "$MPIW_ROOT" != /* || -L "$MPIW_ROOT" ]]; then
    echo "[build_mpiw.pm] ERROR: adapter root must be an absolute, non-symlink directory" >&2
    exit 2
fi
SENTINEL="$MPIW_ROOT/.lorrax-mpiwrapper-root"
if [[ ! -e "$MPIW_ROOT" ]]; then
    mkdir -p "$MPIW_ROOT"
fi
if [[ ! -f "$SENTINEL" ]]; then
    if find "$MPIW_ROOT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        echo "[build_mpiw.pm] ERROR: refusing an existing non-empty root without $SENTINEL" >&2
        exit 2
    fi
    : >"$SENTINEL"
fi

# Serialise the whole publication recipe.  The root is shared across compute
# nodes; without a filesystem lock two candidates can both observe a missing
# release and one ``mv`` becomes a nested directory inside the other release.
if ! command -v flock >/dev/null 2>&1; then
    echo "[build_mpiw.pm] ERROR: flock is required for atomic publication" >&2
    exit 2
fi
exec 9>"$MPIW_ROOT/.build.lock"
if ! flock -n 9; then
    echo "[build_mpiw.pm] ERROR: another adapter build owns $MPIW_ROOT" >&2
    exit 2
fi

STAGE="$MPIW_ROOT/stage"
RELEASES="$MPIW_ROOT/releases"
ACTIVE="$MPIW_ROOT/current"
mkdir -p "$STAGE" "$RELEASES"
WORK_DIR="$(mktemp -d "$STAGE/candidate.XXXXXXXX")"
cleanup() { rm -rf -- "$WORK_DIR"; }
trap cleanup EXIT
SRC="$WORK_DIR/src"
BUILD="$WORK_DIR/build"
CANDIDATE="$WORK_DIR/install"

if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
fi
if [[ -n "${LORRAX_PM_PRGENV:-}" && \
      "$LORRAX_PM_PRGENV" != "$LORRAX_PM_PRGENV_DEFAULT" ]] || \
   [[ -n "${LORRAX_PM_MPICH:-}" && \
      "$LORRAX_PM_MPICH" != "$LORRAX_PM_MPICH_DEFAULT" ]] || \
   [[ -n "${LORRAX_PM_CMAKE:-}" && \
      "$LORRAX_PM_CMAKE" != "$LORRAX_PM_CMAKE_DEFAULT" ]] || \
   [[ -n "${LORRAX_CMAKE:-}" ]]; then
    echo "[build_mpiw.pm] ERROR: production adapter toolchain overrides are not certified" >&2
    echo "[build_mpiw.pm] Use the versioned defaults in config/perlmutter/site_config.sh." >&2
    exit 2
fi
module load \
    "$LORRAX_PM_PRGENV_DEFAULT" \
    "$LORRAX_PM_MPICH_DEFAULT" \
    "$LORRAX_PM_CMAKE_DEFAULT"
# This is a CPU artifact.  CUDA GTL and Darshan in its DT_NEEDED closure make
# it unusable in a plain CPU process and are removed before CMake.
for unwanted_module in gpu craype-accel-nvidia80 cudatoolkit darshan; do
    module unload "$unwanted_module" 2>/dev/null || true
done
CMAKE="$(command -v cmake || true)"
if [[ -z "$CMAKE" ]]; then
    echo "[build_mpiw.pm] ERROR: cmake is not on PATH after module setup" >&2
    exit 2
fi
for compiler in cc CC ftn; do
    if ! command -v "$compiler" >/dev/null 2>&1; then
        echo "[build_mpiw.pm] ERROR: Cray compiler wrapper '$compiler' is absent" >&2
        exit 2
    fi
done
if [[ -z "${CRAY_MPICH_DIR:-}" || -z "${CRAY_MPICH_VERSION:-}" ]]; then
    echo "[build_mpiw.pm] ERROR: the versioned Cray MPICH module did not resolve" >&2
    exit 2
fi
for required_module in "$LORRAX_PM_PRGENV_DEFAULT" \
        "$LORRAX_PM_MPICH_DEFAULT" "$LORRAX_PM_CMAKE_DEFAULT"; do
    if ! module is-loaded "$required_module"; then
        echo "[build_mpiw.pm] ERROR: required module is not active: $required_module" >&2
        module -t list >&2
        exit 2
    fi
done
if [[ "$CRAY_MPICH_VERSION" != "${LORRAX_PM_MPICH_DEFAULT##*/}" ]]; then
    echo "[build_mpiw.pm] ERROR: active Cray MPICH $CRAY_MPICH_VERSION does not match $LORRAX_PM_MPICH_DEFAULT" >&2
    exit 2
fi

git clone "$MPIW_REPO" "$SRC"
if ! git -C "$SRC" cat-file -e "$MPIW_COMMIT^{commit}" 2>/dev/null; then
    git -C "$SRC" fetch origin "$MPIW_COMMIT"
fi
git -C "$SRC" checkout --quiet --detach "$MPIW_COMMIT"
HAVE="$(git -C "$SRC" rev-parse HEAD)"
if [[ "$HAVE" != "$MPIW_COMMIT" ]]; then
    echo "[build_mpiw.pm] ERROR: source is $HAVE, expected $MPIW_COMMIT" >&2
    exit 1
fi
if [[ -n "$(git -C "$SRC" status --porcelain)" ]]; then
    echo "[build_mpiw.pm] ERROR: upstream checkout is modified; refusing a patched adapter" >&2
    git -C "$SRC" status --short >&2
    exit 1
fi

"$CMAKE" -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$CANDIDATE" \
    -DCMAKE_INSTALL_LIBDIR=lib64 \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_Fortran_COMPILER=ftn \
    -DCMAKE_MODULE_LINKER_FLAGS=-Wl,--as-needed
"$CMAKE" --build "$BUILD" --parallel
"$CMAKE" --install "$BUILD"

SO="$CANDIDATE/lib64/libmpiwrapper.so"
if [[ ! -f "$SO" ]]; then
    echo "[build_mpiw.pm] ERROR: install did not produce $SO" >&2
    exit 1
fi
DYNSYMS="$(nm -D "$SO")"
for symbol in mpiwrapper_version_major mpiabi_version_major \
        MPIABI_Init_thread MPIABI_Query_thread MPIABI_Is_thread_main \
        MPIABI_Comm_split MPIABI_Finalize; do
    if ! grep -qE " [TBDR] ${symbol}$" <<<"$DYNSYMS"; then
        echo "[build_mpiw.pm] ERROR: required MPItrampoline ABI symbol missing: $symbol" >&2
        exit 1
    fi
done

read -r ABI_MAJOR ABI_MINOR ABI_PATCH < <(python3 - "$SO" <<'PY'
import ctypes
import sys
lib = ctypes.CDLL(sys.argv[1])
print(*(ctypes.c_int.in_dll(lib, f"mpiabi_version_{part}").value
        for part in ("major", "minor", "patch")))
PY
)
if (( ABI_MAJOR != 2 || ABI_MINOR < 9 )); then
    echo "[build_mpiw.pm] ERROR: MPI ABI $ABI_MAJOR.$ABI_MINOR.$ABI_PATCH is incompatible with JAX 0.9.1 (needs 2.x, x >= 9)" >&2
    exit 1
fi
if [[ "$ABI_MAJOR.$ABI_MINOR.$ABI_PATCH" != "$LORRAX_MPIWRAPPER_ABI_DEFAULT" ]]; then
    echo "[build_mpiw.pm] ERROR: pinned source exported unexpected MPI ABI $ABI_MAJOR.$ABI_MINOR.$ABI_PATCH (expected $LORRAX_MPIWRAPPER_ABI_DEFAULT)" >&2
    exit 1
fi

NEEDED="$(readelf -d "$SO")"
if grep -qE 'libmpi_gtl_cuda|libcuda|libcudart|libdarshan' <<<"$NEEDED"; then
    echo "[build_mpiw.pm] ERROR: CPU adapter acquired a GPU or Darshan dependency:" >&2
    grep -E 'NEEDED.*(libmpi_gtl_cuda|libcuda|libcudart|libdarshan)' <<<"$NEEDED" >&2
    exit 1
fi
GATE_TAG=build_mpiw.pm "$LORRAX_ROOT/src/ffi/cpp/gate_one_mpi.sh" \
    "$SO" 'libmpi_gnu_'

ARTIFACT_SHA="$(sha256sum "$SO" | awk '{print $1}')"
BUILDER_SHA="$(sha256sum "$LORRAX_ROOT/config/perlmutter/build_mpiwrapper.sh" | awk '{print $1}')"
PRELUDE_SHA="$(sha256sum "$LORRAX_ROOT/config/perlmutter/cpu_mpi_env.sh" | awk '{print $1}')"
SITE_SHA="$(sha256sum "$SITE_CONFIG" | awk '{print $1}')"
REPO_HEAD="$(git -C "$LORRAX_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
if [[ -n "$(git -C "$LORRAX_ROOT" status --porcelain -- \
        config/perlmutter/build_mpiwrapper.sh \
        config/perlmutter/cpu_mpi_env.sh \
        config/perlmutter/site_config.sh 2>/dev/null || true)" ]]; then
    RECIPE_DIRTY=true
else
    RECIPE_DIRTY=false
fi
{
    echo "source_repo=$MPIW_REPO"
    echo "source_commit=$MPIW_COMMIT"
    echo "source_modified=false"
    echo "mpiabi_version=$ABI_MAJOR.$ABI_MINOR.$ABI_PATCH"
    echo "cray_mpich_version=$CRAY_MPICH_VERSION"
    echo "cmake=$CMAKE"
    echo "c_compiler=$(command -v cc)"
    echo "cxx_compiler=$(command -v CC)"
    echo "fortran_compiler=$(command -v ftn)"
    echo "artifact=lib64/libmpiwrapper.so"
    echo "artifact_sha256=$ARTIFACT_SHA"
    echo "lorrax_git_head=$REPO_HEAD"
    echo "lorrax_recipe_dirty=$RECIPE_DIRTY"
    echo "builder_sha256=$BUILDER_SHA"
    echo "prelude_sha256=$PRELUDE_SHA"
    echo "site_config_sha256=$SITE_SHA"
    module -t list 2>&1 | sed 's/^/module=/'
} >"$CANDIDATE/build-manifest.txt"

# Content-addressed immutable release, followed by an atomic symlink switch.
# A failed candidate never touches ACTIVE.  A repeated identical build simply
# reuses the already-gated release.
RECIPE_ID="${BUILDER_SHA:0:8}-${PRELUDE_SHA:0:8}-${SITE_SHA:0:8}"
RELEASE="$RELEASES/${MPIW_COMMIT:0:12}-mpich-${CRAY_MPICH_VERSION}-${ARTIFACT_SHA:0:12}-${RECIPE_ID}"
if [[ -e "$RELEASE" ]]; then
    EXISTING_SHA="$(sha256sum "$RELEASE/lib64/libmpiwrapper.so" | awk '{print $1}')"
    if [[ "$EXISTING_SHA" != "$ARTIFACT_SHA" ]]; then
        echo "[build_mpiw.pm] ERROR: immutable release collision at $RELEASE" >&2
        exit 1
    fi
    if ! cmp -s "$CANDIDATE/build-manifest.txt" "$RELEASE/build-manifest.txt"; then
        echo "[build_mpiw.pm] ERROR: existing release manifest differs at $RELEASE" >&2
        exit 1
    fi
    if find "$RELEASE" -perm /222 -print -quit | grep -q .; then
        echo "[build_mpiw.pm] ERROR: existing release is writable: $RELEASE" >&2
        exit 1
    fi
else
    mv "$CANDIDATE" "$RELEASE"
    chmod -R a-w "$RELEASE"
fi
if find "$RELEASE" -perm /222 -print -quit | grep -q .; then
    echo "[build_mpiw.pm] ERROR: release publication did not become read-only" >&2
    exit 1
fi
if [[ -e "$ACTIVE" && ! -L "$ACTIVE" ]]; then
    echo "[build_mpiw.pm] ERROR: active path exists but is not a symlink: $ACTIVE" >&2
    exit 2
fi
NEXT_LINK="$MPIW_ROOT/.current.$$"
ln -s "$RELEASE" "$NEXT_LINK"
mv -Tf "$NEXT_LINK" "$ACTIVE"

echo "[build_mpiw.pm] unmodified Cray-MPICH adapter is active:"
echo "[build_mpiw.pm]   $ACTIVE/lib64/libmpiwrapper.so"
echo "[build_mpiw.pm]   MPI ABI $ABI_MAJOR.$ABI_MINOR.$ABI_PATCH"
echo "$ARTIFACT_SHA  $ACTIVE/lib64/libmpiwrapper.so"
echo "[build_mpiw.pm] Source config/perlmutter/cpu_mpi_env.sh before Python."
