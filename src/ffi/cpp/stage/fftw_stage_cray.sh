#!/usr/bin/env bash
# fftw_stage_cray.sh — populate a staging dir (default
# $HOME/software/lorrax_fftw_cray/stage, see LORRAX_FFI_FFTW_DIR below) with
# the DOUBLE-PRECISION SERIAL cray-fftw engine, so that the flat-k FFT
# handler's run-time dlopen ladder can find an FFTW3 inside Shifter.
#
# WHY THIS EXISTS
# ---------------
# `src/ffi/cpp/mklfft/fft_flat_k_ffi.cc` resolves the FFTW3 advanced
# interface at RUN time: stage 1 dlsym against an already-loaded provider
# (that is the MKL site — Frontera), stage 2 dlopen a candidate ladder, stage
# 3 refuse loudly.  Nothing reaches DT_NEEDED, which is the point (GATE 5 in
# config/perlmutter/build_ffi_host.sh: zero fftw in DT_NEEDED, because a
# version- and MPI-flavour-stamped SONAME in DT_NEEDED made the WHOLE library
# unloadable in the container and took 19 unrelated linear-algebra cells with
# it).
#
# But run-time resolution still needs a file to resolve TO.  Measured
# 2026-08-06, in-container on a compute node under
# `nvcr.io/nvidia/jax:25.04-py3`:
#
#   * `find /usr /usr/local /opt /lib /lib64 /lorrax_slate /lorrax_phdf5
#      /lorrax_nvhpc -name 'libfftw3*'`            -> EMPTY.
#   * every ladder candidate (libfftw3.so.3, libfftw3.so.mpi31.3,
#     libmkl_rt.so, libfftw3.so) -> "cannot open shared object file".
#   * `/opt/cray/pe` does not exist in the image at all, so the host .so's
#     RPATH (which is what makes the BARE-HOST leg work) resolves nothing.
#
# So the container ships no FFTW3 and one must be staged in, the same way
# /lorrax_phdf5, /lorrax_slate and /lorrax_nvhpc already are.  Shifter at
# NERSC refuses --volume from /opt/ system paths, so bind-mounting
# /opt/cray/pe/fftw directly is NOT available; $HOME/software IS a valid
# siteFs --volume source, which is why every other stage lives there.
#
# Run on a NERSC login node, after `module load cray-fftw` (so $FFTW_ROOT /
# $FFTW_DIR are set).  No container, no GPU, ~4 MB, seconds:
#
#   module load PrgEnv-gnu cray-fftw
#   src/ffi/cpp/stage/fftw_stage_cray.sh
#
# The modulefile then bind-mounts it at /lorrax_fftw and puts
# /lorrax_fftw/lib on the container LD_LIBRARY_PATH; the ladder's
# `libfftw3.so.mpi31.3` candidate finds it there.  Verify with
# src/ffi/cpp/gate_one_fftw.sh (GATE 8) — staging is not the claim, ONE
# mapped engine is.
#
# WHAT IS DELIBERATELY *NOT* STAGED, AND WHY
# ------------------------------------------
# cray-fftw's lib/ carries eight shared libraries: libfftw3{,f}{,_omp,
# _threads,_mpi}.  This stage copies exactly ONE of them (plus its two
# symlink spellings):
#
#   libfftw3.so.mpi31.3   double precision, serial, NEEDED = libm + libc only
#
# because the handler binds exactly three entry points — fftw_plan_many_dft,
# fftw_execute_dft, fftw_destroy_plan — and:
#
#   * libfftw3f_* define fftwf_* (single precision).  The handler is
#     complex128-only and refuses anything else, so a float engine can only
#     ever be dead weight or a wrong answer.
#   * libfftw3_omp / libfftw3_threads exist to serve fftw_plan_with_nthreads,
#     which this handler never calls: it runs its own OpenMP chunk loop and
#     serialises planning behind plan_mutex().  Staging them adds objects
#     that ALSO define fftw_ symbols, i.e. more ways for two engines to end
#     up mapped in one process — the exact hazard GATE 8 exists to catch.
#   * libfftw3_mpi NEEDs a Cray libmpi SONAME.  The serial engine's closure
#     is libm + libc and NOTHING else, so it loads in the container with no
#     shim, no LD_PRELOAD and no MPI-ABI argument.  Staging the MPI variant
#     would throw that away for a library nothing dlopens.
#
# "Stage the whole lib/ because it is only 23 MB" is the reasoning that put
# two HDF5 majors under one mount (see phdf5_stage_cray.sh).  Copy what is
# used; name what is not.

set -euo pipefail

: "${LORRAX_FFI_FFTW_DIR:=$HOME/software/lorrax_fftw_cray/stage}"

# How `nm -D` may spell the entry point.  The version suffix is not optional
# decoration: CUDA's FFTW3 shim exports `fftw_plan_many_dft@@libcufftw.so.11`
# and a `$`-anchored match silently skips it (measured 2026-08-06).
SYM_RE='[[:space:]]fftw_plan_many_dft(@@?[^[:space:]]+)?$'

# --- where the engine comes from.  REFUSES WHEN UNSET. ---------------------
#
# Same rule as phdf5_stage_cray.sh: a substituted default is a guess about an
# environment fact, and the artifact it produces does not fail at stage time —
# it fails much later, inside a container, as a missing engine or (worse) a
# different FFTW than the one anybody named.  There is no fallback path here
# on purpose.
if [[ -z "${CRAY_FFTW_PATH:-}" && -z "${FFTW_ROOT:-}" && -z "${FFTW_DIR:-}" ]]; then
    echo "fftw_stage_cray.sh: REFUSED — FFTW_ROOT/FFTW_DIR are not set." >&2
    echo "  rule   the staged engine is what the container's dlopen ladder" >&2
    echo "         will bind at run time; a guessed path stages silently and" >&2
    echo "         is only discovered at the first transform." >&2
    echo "  got    none of CRAY_FFTW_PATH, FFTW_ROOT, FFTW_DIR." >&2
    echo "  wanted FFTW_ROOT, as exported by the module." >&2
    echo "  fix    module load PrgEnv-gnu cray-fftw" >&2
    echo "         (or set CRAY_FFTW_PATH=<fftw install root> explicitly)." >&2
    exit 2
fi
if [[ -z "${CRAY_FFTW_PATH:-}" ]]; then
    if [[ -n "${FFTW_ROOT:-}" ]]; then
        CRAY_FFTW_PATH="${FFTW_ROOT}"
    else
        CRAY_FFTW_PATH="$(dirname "${FFTW_DIR}")"
    fi
fi
SRC_LIB="${CRAY_FFTW_PATH}/lib"

if [[ ! -d "$SRC_LIB" ]]; then
    echo "fftw_stage_cray.sh: REFUSED — no lib/ under CRAY_FFTW_PATH." >&2
    echo "  got    ${SRC_LIB}" >&2
    echo "  fix    module load cray-fftw, or set CRAY_FFTW_PATH." >&2
    exit 2
fi

# The one object this stage is about.  Resolve the SONAME spelling to the
# real file so the copy is a file, never a dangling relative symlink.
ENGINE_SONAME="libfftw3.so.mpi31.3"
ENGINE_SRC=""
for cand in "$SRC_LIB/$ENGINE_SONAME" "$SRC_LIB/libfftw3.so.3" \
            "$SRC_LIB/libfftw3.so"; do
    if [[ -e "$cand" ]]; then ENGINE_SRC="$(readlink -f "$cand")"; break; fi
done
if [[ -z "$ENGINE_SRC" ]]; then
    echo "fftw_stage_cray.sh: REFUSED — no double-precision libfftw3 under" >&2
    echo "  ${SRC_LIB}.  Looked for ${ENGINE_SONAME}, libfftw3.so.3," >&2
    echo "  libfftw3.so.  This is not the tree you think it is." >&2
    exit 2
fi

# The SONAME is the name the container's loader will be asked for, so read it
# from the file rather than assuming the filename told the truth.
ENGINE_SONAME="$(readelf -d "$ENGINE_SRC" |
                 sed -n 's/.*SONAME.*\[\(.*\)\].*/\1/p')"
if [[ -z "$ENGINE_SONAME" ]]; then
    echo "fftw_stage_cray.sh: REFUSED — ${ENGINE_SRC} has no SONAME, so the" >&2
    echo "  container loader has no name to resolve it by." >&2
    exit 2
fi

# The three entry points the handler binds.  A tree that does not export all
# three is not an engine, and staging it would move the failure from
# `dlopen` (loud, named) to `dlsym` (loud, but one stage later).
MISSING=""
for sym in fftw_plan_many_dft fftw_execute_dft fftw_destroy_plan; do
    # grep -c, not grep -q -- see the note at the self-check below.
    _n=$(nm -D --defined-only "$ENGINE_SRC" 2>/dev/null |
         grep -cE "[[:space:]]${sym}(@@?[^[:space:]]+)?\$" || true)
    if [[ "${_n:-0}" -eq 0 ]]; then
        MISSING="${MISSING} ${sym}"
    fi
done
if [[ -n "$MISSING" ]]; then
    echo "fftw_stage_cray.sh: REFUSED — ${ENGINE_SRC} does not export the" >&2
    echo "  FFTW3 ADVANCED interface.  Missing:${MISSING}" >&2
    echo "  The handler calls exactly these three; an engine without them" >&2
    echo "  dlopens and then refuses at the first transform." >&2
    exit 2
fi

# --- a stage is ONE engine --------------------------------------------------
#
# Identical rule, and identical reason, to phdf5_stage_cray.sh: `cp` onto a
# populated tree OVERLAYS rather than replaces.  Two FFTW3 builds under one
# mount both resolve — that is the problem, not the fix.  Whichever the
# loader reaches first wins, silently, and nothing on disk records which.
# The stage is ~4 MB and rebuilding it from scratch is always right.
if [[ -d "${LORRAX_FFI_FFTW_DIR}/lib" ]] && \
   [[ -n "$(ls -A "${LORRAX_FFI_FFTW_DIR}/lib" 2>/dev/null)" ]]; then
    if [[ "${LORRAX_FFTW_STAGE_CLOBBER:-0}" != "1" ]]; then
        echo "fftw_stage_cray.sh: REFUSED — the stage is already populated." >&2
        echo "  rule   a stage holds ONE FFTW3.  cp overlays rather than" >&2
        echo "         replaces, and two engines under one mount both" >&2
        echo "         resolve, so nothing fails until the numbers do." >&2
        echo "  got    ${LORRAX_FFI_FFTW_DIR}/lib is non-empty:" >&2
        ls -1 "${LORRAX_FFI_FFTW_DIR}/lib" | sed 's/^/           /' >&2
        echo "  wanted an empty or absent stage dir." >&2
        echo "  fix    rm -rf ${LORRAX_FFI_FFTW_DIR}   (then re-run)" >&2
        echo "         or LORRAX_FFTW_STAGE_CLOBBER=1 to overlay anyway," >&2
        echo "         which you almost certainly do not want." >&2
        exit 2
    fi
    echo "[stage] WARNING: overlaying a populated stage on your say-so" >&2
    echo "[stage]          (LORRAX_FFTW_STAGE_CLOBBER=1)." >&2
fi

echo "[stage] src engine: ${ENGINE_SRC}"
echo "[stage] soname:     ${ENGINE_SONAME}"
echo "[stage] dst:        ${LORRAX_FFI_FFTW_DIR}"
mkdir -p "${LORRAX_FFI_FFTW_DIR}/lib"

REAL_NAME="$(basename "$ENGINE_SRC")"
cp -L "$ENGINE_SRC" "${LORRAX_FFI_FFTW_DIR}/lib/${REAL_NAME}"
chmod u+w "${LORRAX_FFI_FFTW_DIR}/lib/${REAL_NAME}"

# The SONAME spelling is what dlopen() is handed; the bare `.so` spelling is
# the ladder's last candidate.  Both point at the ONE real file, so
# `readlink -f` collapses them and GATE 8 counts one object, not three.
( cd "${LORRAX_FFI_FFTW_DIR}/lib"
  [[ "$REAL_NAME" == "$ENGINE_SONAME" ]] || ln -sf "$REAL_NAME" "$ENGINE_SONAME"
  ln -sf "$REAL_NAME" "libfftw3.so" )

# --- what landed, beside the artifact --------------------------------------
{
    echo "stage=${LORRAX_FFI_FFTW_DIR}"
    echo "src_fftw=${CRAY_FFTW_PATH}"
    echo "engine_file=${ENGINE_SRC}"
    echo "engine_soname=${ENGINE_SONAME}"
    echo "engine_sha256=$(sha256sum "$ENGINE_SRC" | cut -d' ' -f1)"
    echo "precision=double, serial (no _omp, no _threads, no _mpi, no float)"
    echo "staged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "staged_on=$(hostname)"
    echo "staged_by=${USER:-unknown}"
} > "${LORRAX_FFI_FFTW_DIR}/STAGE_PROVENANCE"

# --- self-checks: the two properties the container depends on --------------
#
# Checked HERE as well as in GATE 8 because this is where a second engine or
# an incomplete closure gets INTRODUCED, and the sooner it is named the less
# has to be rebuilt.  Same division of labour as phdf5_stage_cray.sh.
# grep -c, NOT grep -q: under `set -o pipefail` a matching `grep -q` closes
# the pipe, nm dies of SIGPIPE, and the pipeline reports failure -- so a real
# engine reads as absent, intermittently, depending on whether nm's output
# fit the pipe buffer first.  Read all of the input.
_objs=$(for f in "${LORRAX_FFI_FFTW_DIR}"/lib/*; do
            [[ -f "$f" && ! -L "$f" ]] || continue
            _h=$(nm -D --defined-only "$f" 2>/dev/null |
                 grep -cE "$SYM_RE" || true)
            [[ "${_h:-0}" -gt 0 ]] && readlink -f "$f"
        done | sort -u)
_n=$(printf '%s' "$_objs" | grep -c . || true)
if [[ "$_n" -ne 1 ]]; then
    echo "[stage] FAILED: ${_n} staged objects define fftw_plan_many_dft:" >&2
    printf '%s\n' "$_objs" | sed 's/^/[stage]   /' >&2
    echo "[stage]   A stage is ONE engine.  rm -rf the tree and re-run." >&2
    exit 1
fi
echo "[stage] one FFTW3 engine staged: $(basename "$_objs")"

# The closure must be complete with NOTHING mounted but this stage — that is
# what makes it loadable inside a container that has no Cray PE at all.
if ldd "${LORRAX_FFI_FFTW_DIR}/lib/${REAL_NAME}" 2>&1 | grep -q "not found"; then
    echo "[stage] FAILED: the staged engine has unresolved dependencies:" >&2
    ldd "${LORRAX_FFI_FFTW_DIR}/lib/${REAL_NAME}" 2>&1 |
        grep "not found" | sed 's/^/[stage]   /' >&2
    echo "[stage]   It will not dlopen in the container.  Stage the serial" >&2
    echo "[stage]   double-precision engine only (libm + libc closure)." >&2
    exit 1
fi
echo "[stage] closure (bare host): complete"
echo "[stage] NEEDED:"
readelf -d "${LORRAX_FFI_FFTW_DIR}/lib/${REAL_NAME}" |
    grep -E "SONAME|NEEDED" | sed 's/^/[stage]   /'
echo "[stage] spellings:"
ls -la "${LORRAX_FFI_FFTW_DIR}/lib" | sed 's/^/[stage]   /'
echo
echo "[stage] done.  The modulefile mounts this at /lorrax_fftw when"
echo "[stage] LORRAX_FFI_FFTW_DIR=${LORRAX_FFI_FFTW_DIR} (its default)."
echo "[stage] Staging is not the claim — run GATE 8 in-container:"
echo "[stage]   src/ffi/cpp/gate_one_fftw.sh <host.so> [<device.so>]"
