#!/usr/bin/env bash
# ============================================================================
# build_mpi_overlay.sh — build the mpi4py + parallel-h5py PYTHONPATH overlay
# for Frontera (env audit P0-2: the overlay every impl=mpi / PHDF5_HOST run
# depends on was an untracked artifact; this is the vendored recipe).
#
#   config/frontera/build_mpi_overlay.sh fetch     # LOGIN node: download+verify sdists
#   config/frontera/build_mpi_overlay.sh build     # INSIDE the py312 container,
#                                                  # on a COMPUTE node
#
# WHAT IT PRODUCES:  $LORRAX_OVERLAY_PREFIX/site (default
# $WORK/lorrax_env_mpi_overlay/site) containing exactly three things:
#   * mpi4py 4.1.2   — built with Intel MPI 2020.4 wrappers
#   * h5py  3.16.0   — built HDF5_MPI=ON against the host parallel HDF5
#                      1.14.6 (the SAME install the phdf5 FFI links)
#   * sitecustomize.py — the LORRAX_MPI_FINALIZE_FIX / LORRAX_MPI_INIT_FIRST
#                      teardown fix (vendored: config/frontera/sitecustomize.py)
# Activated by a PYTHONPATH PREPEND (site/ first, then src/); the shared uv
# venv is never modified.  Rollback = drop the prepend.
#
# PROVENANCE.  This reproduces the wk_AB build (scorecard AB.0, job 7875713,
# 2026-07-26; original script /scratch2/.../campaigns/wk_AB/build_overlay.sbatch)
# plus the wk_AS sitecustomize.  Five non-obvious facts it depends on:
#   1. The venv has NO pip/setuptools (uv-created).  pip runs out of its own
#      wheel:  `python pip-*.whl/pip ...` (a wheel is a zip with __main__).
#   2. Compute nodes have no outbound network; login nodes have no
#      apptainer.  Hence the two-phase fetch/build split, with sha256
#      verification of every download (pins below).
#   3. CC=mpicc fixes the LINK as well as the compile (setuptools rewrites
#      LDSHARED when CC is overridden).
#   4. I_MPI_CC=gcc — the container has gcc 12 but no icc (same deviation
#      build_ffi_host.sh and build_ffi.sh document).
#   5. Build on node-local /tmp, publish at the end (atomic-ish swap) —
#      never build against a Lustre prefix a flagship may be reading.
# Build-time tools (setuptools/wheel/Cython/pkgconfig) go to a throwaway
# dir on PYTHONPATH and never enter the shipped site/.
#
# RESIDUAL UNCERTAINTY: the original job also ran a 4-rank srun --mpi=pmi2
# collective MPI-IO write+readback smoke (wk_AB/mpiio_smoke.py, not
# vendored).  The in-process verification below (get_config().mpi, hdf5
# version, ldd of the extension) catches every build-time failure class we
# have seen, but a multi-rank MPI-IO smoke on Lustre is still the honest
# end-to-end gate — run one job with the phdf5 writer before a campaign.
# ============================================================================
set -euo pipefail

MODE="${1:-}"
[ "$MODE" = "fetch" ] || [ "$MODE" = "build" ] || {
    echo "usage: $0 fetch|build   (fetch on a login node, build inside the py312 container)" >&2
    exit 2; }

: "${LORRAX_ROOT:?set LORRAX_ROOT to the worktree/repo root (contains config/frontera)}"
: "${LORRAX_OVERLAY_PREFIX:=${WORK:?}/lorrax_env_mpi_overlay}"
: "${LORRAX_VENV:=$WORK/lorrax_env/.venv}"
: "${LORRAX_IMPI_ROOT:=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64}"
: "${LORRAX_HDF5_ROOT:=/home1/apps/intel19/impi19_0/phdf5/1.14.6}"
: "${LORRAX_ICC_RUNTIME:=/opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin}"

PKGS="$LORRAX_OVERLAY_PREFIX/pkgs"

# --- pinned sources (sha256 = the wk_AB MANIFEST, verified against PyPI) ----
# name|filename|sha256|url
PINS='
pip|pip-26.1.2-py3-none-any.whl|382ff9f685ee3bc25864f820aa50505825f10f5458ffff07e30a6d96e5715cab|https://files.pythonhosted.org/packages/5d/95/6b5cb3461ea5673ba0995989746db58eb18b91b54dbf331e72f569540946/pip-26.1.2-py3-none-any.whl
setuptools|setuptools-83.0.0-py3-none-any.whl|29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3|https://files.pythonhosted.org/packages/5d/40/e1e72872c6354b306daef1703549e8e83b4d43cfea356311bf722a043752/setuptools-83.0.0-py3-none-any.whl
wheel|wheel-0.47.0-py3-none-any.whl|212281cab4dff978f6cedd499cd893e1f620791ca6ff7107cf270781e587eced|https://files.pythonhosted.org/packages/87/1b/9e33c09813d65e248f7f773119148a612516a4bea93e9c6f545f78455b7c/wheel-0.47.0-py3-none-any.whl
pkgconfig|pkgconfig-1.6.0-py3-none-any.whl|98e71754855e9563838d952a160eb577edabb57782e49853edb5381927e6bea1|https://files.pythonhosted.org/packages/75/6f/f7ec07fba48f07c555cc4099481df644fbbc12067879072c17ac229f6556/pkgconfig-1.6.0-py3-none-any.whl
cython|cython-3.2.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl|23e80bc885c599e72072e18d0746df82d394b73100c1e153cda7359e6e59fe09|https://files.pythonhosted.org/packages/56/1b/c04520ac7f3157aa12a69b632c16261170dac9fab6c48608cc004b8f1b17/cython-3.2.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
mpi4py|mpi4py-4.1.2.tar.gz|56860286dc45f20e8821e93cb06669e30462348bf866f685553fa4b712d58d02|https://files.pythonhosted.org/packages/75/83/231445bbcf7ef10864746c244ff2d82000011449b79275642c5d4ed8c8f4/mpi4py-4.1.2.tar.gz
h5py|h5py-3.16.0.tar.gz|a0dbaad796840ccaa67a4c144a0d0c8080073c34c76d5a6941d6818678ef2738|https://files.pythonhosted.org/packages/db/33/acd0ce6863b6c0d7735007df01815403f5589a21ff8c2e1ee2587a38f548/h5py-3.16.0.tar.gz
'

pins_each () {  # callback: $1=name $2=file $3=sha $4=url
    local cb=$1 line
    while IFS='|' read -r name file sha url; do
        [ -n "$name" ] || continue
        "$cb" "$name" "$file" "$sha" "$url"
    done <<< "$PINS"
}

verify_one () {
    local f="$PKGS/$2" want=$3 got
    [ -f "$f" ] || { echo "[overlay] missing: $f (run '$0 fetch' on a login node)" >&2; return 1; }
    got=$(sha256sum "$f" | awk '{print $1}')
    [ "$got" = "$want" ] || { echo "[overlay] SHA MISMATCH: $2  got=$got want=$want" >&2; return 1; }
    echo "[overlay]   ok $2"
}

# ===========================================================================
if [ "$MODE" = "fetch" ]; then
    mkdir -p "$PKGS"
    fetch_one () {
        if [ -f "$PKGS/$2" ] && [ "$(sha256sum "$PKGS/$2" | awk '{print $1}')" = "$3" ]; then
            echo "[overlay]   have $2"; return 0
        fi
        echo "[overlay]   curl $2"
        curl -fsSL -o "$PKGS/$2.part" "$4"
        mv "$PKGS/$2.part" "$PKGS/$2"
        [ "$(sha256sum "$PKGS/$2" | awk '{print $1}')" = "$3" ] || {
            echo "[overlay] FAILED: downloaded $2 does not match pinned sha256" >&2; exit 1; }
    }
    echo "[overlay] fetching pinned sources -> $PKGS"
    pins_each fetch_one
    echo "[overlay] verifying"
    pins_each verify_one
    echo "[overlay] fetch done.  Next: '$0 build' inside the py312 container on a"
    echo "[overlay] compute node (binds must include /opt/intel and /home1)."
    exit 0
fi

# ===========================================================================
# MODE = build.  Must run INSIDE the py312 container (glibc >= 2.28, python
# 3.12 — the exact interpreter the venv targets) on a COMPUTE node
# (apptainer is blocked on login nodes; login glibc 2.17 cannot run the venv).
# ===========================================================================
PY="$LORRAX_VENV/bin/python"
SITECUST="$LORRAX_ROOT/config/frontera/sitecustomize.py"

for p in "$PY" "$LORRAX_HDF5_ROOT/include/H5pubconf.h" \
         "$LORRAX_IMPI_ROOT/bin/mpicc" "$SITECUST"; do
    [ -e "$p" ] || { echo "[overlay] missing prerequisite: $p (bind /opt/intel,/home1?)" >&2; exit 2; }
done
echo "[overlay] verifying pinned sources"
pins_each verify_one

B=${LORRAX_OVERLAY_BUILD_DIR:-/tmp/lorrax_overlay_build.$$}
rm -rf "$B"; mkdir -p "$B/tools" "$B/site" "$B/wheels" "$B/tmp"
PIPWHL="$PKGS/pip-26.1.2-py3-none-any.whl"
PIP="$PY $PIPWHL/pip"

# Intel MPI wrapper env (build_ffi_host.sh's recipe)
export I_MPI_ROOT="${LORRAX_IMPI_ROOT%/intel64}"
export I_MPI_CC=gcc I_MPI_CXX=g++
export PATH="$LORRAX_IMPI_ROOT/bin:$PATH"
export LIBRARY_PATH="$LORRAX_IMPI_ROOT/libfabric/lib:$LORRAX_IMPI_ROOT/lib/release:$LORRAX_IMPI_ROOT/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LORRAX_HDF5_ROOT/lib:$LORRAX_IMPI_ROOT/lib/release:$LORRAX_IMPI_ROOT/lib:$LORRAX_IMPI_ROOT/libfabric/lib:$LORRAX_ICC_RUNTIME:${LD_LIBRARY_PATH:-}"
export TMPDIR="$B/tmp"
# never byte-compile into the shared base venv
export PYTHONDONTWRITEBYTECODE=1

echo "[overlay] toolchain:"; "$PY" -VV; mpicc -show || true
grep -m2 -E "H5_HAVE_PARALLEL |H5_VERSION " "$LORRAX_HDF5_ROOT/include/H5pubconf.h"

echo "[overlay] [1/6] build tools -> $B/tools (build-time only, NOT shipped)"
$PIP install -q --no-index --find-links="$PKGS" --no-deps \
    --target="$B/tools" setuptools wheel Cython pkgconfig

echo "[overlay] [2/6] mpi4py wheel"
export PYTHONPATH="$B/tools"
export MPICC=mpicc CC=mpicc
$PIP wheel --no-index --no-build-isolation --no-deps -w "$B/wheels" \
    "$PKGS/mpi4py-4.1.2.tar.gz"

echo "[overlay] [3/6] install mpi4py -> $B/site (h5py's build imports it)"
$PIP install -q --no-index --no-deps --target="$B/site" "$B"/wheels/mpi4py-*.whl

echo "[overlay] [4/6] h5py wheel, HDF5_MPI=ON"
export PYTHONPATH="$B/tools:$B/site"
export HDF5_MPI=ON HDF5_DIR="$LORRAX_HDF5_ROOT" CC=mpicc
$PIP wheel --no-index --no-build-isolation --no-deps -w "$B/wheels" \
    "$PKGS/h5py-3.16.0.tar.gz"

echo "[overlay] [5/6] install h5py -> $B/site + the vendored sitecustomize"
$PIP install -q --no-index --no-deps --target="$B/site" "$B"/wheels/h5py-*.whl
install -m 0644 "$SITECUST" "$B/site/sitecustomize.py"

echo "[overlay] [6/6] verification (single rank, no MPI_Init)"
env -u HDF5_MPI -u HDF5_DIR -u CC -u MPICC PYTHONPATH="$B/site" "$PY" - <<'PYEOF'
import mpi4py, h5py, numpy
print("mpi4py :", mpi4py.__version__, mpi4py.__file__)
print("h5py   :", h5py.__version__, h5py.__file__)
print("hdf5   :", h5py.version.hdf5_version)
print("numpy  :", numpy.__version__)
assert mpi4py.__version__ == "4.1.2", "wrong mpi4py version"
assert h5py.__version__ == "3.16.0", "wrong h5py version"
assert h5py.version.hdf5_version == "1.14.6", "h5py not built against the host phdf5"
assert h5py.get_config().mpi is True, "h5py is NOT parallel (HDF5_MPI=ON did not take)"
print("h5py.get_config().mpi = True")
PYEOF
echo "[overlay] --- ldd of the h5py extension (must resolve libmpi + the phdf5 libhdf5) ---"
LDD_OUT=$(ldd "$B"/site/h5py/*.so 2>/dev/null | grep -E "hdf5|mpi|fabric" | sort -u)
echo "$LDD_OUT"
grep -q "libmpi" <<< "$LDD_OUT" || { echo "[overlay] FAILED: h5py does not link libmpi" >&2; exit 1; }
grep -q "$LORRAX_HDF5_ROOT" <<< "$LDD_OUT" || {
    echo "[overlay] FAILED: h5py's libhdf5 does not resolve to $LORRAX_HDF5_ROOT" >&2; exit 1; }

echo "[overlay] publish -> $LORRAX_OVERLAY_PREFIX/site (atomic-ish swap)"
mkdir -p "$LORRAX_OVERLAY_PREFIX/wheels"
cp -a "$B"/wheels/*.whl "$LORRAX_OVERLAY_PREFIX/wheels/"
rm -rf "$LORRAX_OVERLAY_PREFIX/site.new"
cp -a "$B/site" "$LORRAX_OVERLAY_PREFIX/site.new"
rm -rf "$LORRAX_OVERLAY_PREFIX/site.old"
[ -d "$LORRAX_OVERLAY_PREFIX/site" ] && mv "$LORRAX_OVERLAY_PREFIX/site" "$LORRAX_OVERLAY_PREFIX/site.old"
mv "$LORRAX_OVERLAY_PREFIX/site.new" "$LORRAX_OVERLAY_PREFIX/site"
rm -rf "$LORRAX_OVERLAY_PREFIX/site.old" "$B"
du -sh "$LORRAX_OVERLAY_PREFIX/site"

cat <<EOF
[overlay] done.  Activate with a PYTHONPATH PREPEND (overlay first):
    export PYTHONPATH=$LORRAX_OVERLAY_PREFIX/site:\$LORRAX_ROOT/src
Reminder: the honest end-to-end gate is one srun --mpi=pmi2 multi-rank job
exercising the phdf5 writer (see the header's RESIDUAL UNCERTAINTY note).
EOF
