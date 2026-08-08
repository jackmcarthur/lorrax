#!/usr/bin/env bash
# ============================================================================
# verify_ffi_build.sh — THE acceptance contract for a LORRAX FFI library.
#
#   scripts/verify_ffi_build.sh [--leg host|cuda] <so-file>
#
# Every build path in this repository ends here.  config/perlmutter/
# build_ffi_host.sh delegates to it, src/ffi/cpp/build_host.sh and
# src/ffi/cpp/build.sh call it unconditionally, and
# services/distrib_la/tests/test_so_acceptance.py runs the same gates as
# pytest cells.  There is no supported way to produce a LORRAX .so that has
# not been through this file.
#
# ----------------------------------------------------------------------------
# WHY THIS IS A SCRIPT AND NOT A CMake POST_BUILD RULE
# ----------------------------------------------------------------------------
# A post-build rule can only ever ask its question at the moment of the link,
# in the tree that did the linking.  Three of the gates below are not that kind
# of question.
#
#   1. THE ARTIFACT OUTLIVES ITS BUILD TREE.  The defective library this file
#      exists because of — the Aug-7 deployed host lib, `scalapack=0` and two
#      LibSci flavours — has no build directory anywhere on the machine any
#      more.  It is a file in ~/software that nine worktrees pin.  A gate that
#      cannot be pointed at an existing file cannot judge the artifacts that
#      are actually in service, which are the ones that matter.
#
#   2. GATE 7 COMPARES AGAINST THE OTHER POPULATION.  The host leg is built
#      bare-metal against a module; the device leg is built in-container
#      against a bind-mounted stage.  Whether a host .so is correct depends on
#      which HDF5 the CONTAINER will mount, and that can change after the build
#      with no rebuild — a stage gets re-populated, a modulefile's --volume=
#      moves.  The same bytes are correct on Monday and unloadable on Tuesday.
#      Only a re-runnable checker can re-answer that.
#
#   3. THE THREE EXISTING GATE SCRIPTS ARE ALREADY SHELL, and already the
#      single source for the one-MPI, one-HDF5 and engine-identity invariants,
#      each with a long argued comment about why its measurement is the right
#      one.  Reimplementing them in CMake would make two sources for three
#      invariants, which is the failure class this whole branch is about.
#
# A shell verifier can be called from CMake, from a build script, from pytest,
# and by a person holding a .so they found in a scratch directory.  None of
# the other shapes has all four.
#
# ----------------------------------------------------------------------------
# EXPECTATIONS — the parameters, every one of them a documented lever
# ----------------------------------------------------------------------------
# The gates are machine-agnostic; the EXPECTATIONS are the site's.  A gate that
# infers its own expectation from the artifact cannot fail, which is how the
# Aug-7 build passed everything anybody ran on it.
#
#   LORRAX_FFI_EXPECT_LEG          host | cuda.  Inferred from the file name
#                                  when unset, and the inference is announced.
#   LORRAX_FFI_EXPECT_BACKENDS     comma list of what this build was SUPPOSED
#                                  to contain.  Default is the FULL set for the
#                                  leg, so a build that quietly lost one FAILS
#                                  by default and a site that wants fewer says
#                                  so.  That default is the entire lesson of
#                                  the Aug-7 library.
#                                    host: scalapack,gemm,slate,phdf5,fft
#                                    cuda: cusolvermp,cublasmp,cufft,slate,phdf5
#   LORRAX_FFI_EXPECT_HDF5_SOVERSION
#                                  the HDF5 SOVERSION THE RUNTIME WILL MOUNT
#                                  (e.g. 200).  Stated, not inferred: on
#                                  Perlmutter the host build's HDF5 is chosen
#                                  by the Cray compiler wrappers via
#                                  LORRAX_PM_HDF5, while LORRAX_FFI_PHDF5_DIR
#                                  only names the tree to compare against.  A
#                                  build with just the latter set links .so.310
#                                  and is structurally unloadable beside a
#                                  .so.200 device library — that cost a whole
#                                  build on the kchunk branch.
#   LORRAX_FFI_EXPECT_PEER_SO      the OTHER leg's library.  When named, GATE 7
#                                  runs across the pair, which is where a
#                                  SOVERSION split between the legs shows up.
#   LORRAX_FFI_EXPECT_MPI          expected libmpi SONAME fragment (GATE 1).
#   LORRAX_FFI_EXPECT_ABI          expected handler-signature ABI.  Defaults to
#                                  this source tree's
#                                  src/ffi/cpp/common/lorrax_ffi_abi.h.
#   LORRAX_PHDF5_STAGE             the runtime phdf5 stage (GATE 7's have-side).
#   LORRAX_GATE_FFTW_PY            a python that can import jax, which is what
#                                  makes GATE 8 runnable.
#   LORRAX_FFI_VERIFY_STRICT=1     a gate that COULD NOT RUN becomes a failure.
#                                  Off by default because GATE 8 provably
#                                  cannot run on a login node; on for a
#                                  certification run inside an allocation.
#   LORRAX_FFI_VERIFY=off          disable the whole verifier.  Announced on
#                                  every invocation, never silent — the same
#                                  contract src/ffi/gate.py sets for every
#                                  opt-out in this codebase.  An unverified .so
#                                  must not be certified.
#
# ----------------------------------------------------------------------------
# THE DIVISION OF LABOUR WITH THE PYTEST TIER
# ----------------------------------------------------------------------------
# This file checks PATTERNS: at least one exported handler per declared
# backend, and the build stamp agreeing.  It does not hardcode symbol NAMES or
# counts, because a name list here would be a second copy of the loaders'
# target tables and would drift from them.
# services/distrib_la/tests/test_so_acceptance.py checks the exact names, read
# out of the loader's own table, so the two tiers cannot disagree about what
# the library is supposed to export.  Run both.
# ============================================================================
set -uo pipefail

TAG="${GATE_TAG:-verify_ffi_build}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LORRAX_ROOT="${LORRAX_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CPP_DIR="$LORRAX_ROOT/src/ffi/cpp"

say()  { echo "[$TAG] $*"; }
warn() { echo "[$TAG] $*" >&2; }

# --- argument handling ------------------------------------------------------
LEG="${LORRAX_FFI_EXPECT_LEG:-}"
SO=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --leg) LEG="${2:?--leg needs host|cuda}"; shift 2 ;;
        --leg=*) LEG="${1#--leg=}"; shift ;;
        -h|--help)
            sed -n '2,110p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*) warn "unknown option: $1"; exit 2 ;;
        *)  if [[ -n "$SO" ]]; then
                warn "verify one artifact per invocation; got '$SO' and '$1'."
                warn "  The PEER leg is named with LORRAX_FFI_EXPECT_PEER_SO,"
                warn "  because the two legs have different expectations and a"
                warn "  shared argument list would have to guess which is which."
                exit 2
            fi
            SO="$1"; shift ;;
    esac
done

if [[ -z "$SO" ]]; then
    warn "usage: verify_ffi_build.sh [--leg host|cuda] <so-file>"
    exit 2
fi

# A DISABLED VERIFIER SAYS SO, LOUDLY, EVERY TIME.
if [[ "${LORRAX_FFI_VERIFY:-on}" == "off" ]]; then
    warn "VERIFIER DISABLED by LORRAX_FFI_VERIFY=off.  $SO has NOT been"
    warn "checked against the build contract: not its declared backends, not"
    warn "its BLAS flavour count, not its HDF5 pairing, not its ABI.  An"
    warn "unverified .so must not be certified or deployed."
    exit 0
fi

if [[ ! -f "$SO" ]]; then
    warn "REFUSED: no such artifact: $SO"
    exit 1
fi
SO="$(readlink -f "$SO")"

# --- leg inference ----------------------------------------------------------
if [[ -z "$LEG" ]]; then
    case "$(basename "$SO")" in
        *_host.so|*_host.so.*) LEG=host ;;
        liblorrax_ffi.so|liblorrax_ffi.so.*) LEG=cuda ;;
        *)  warn "REFUSED: cannot tell which leg $SO is from its name, and the"
            warn "  leg decides which backends and which symbol names are"
            warn "  expected.  Say so: --leg host|cuda."
            exit 2 ;;
    esac
    say "leg inferred from the file name: $LEG"
fi
case "$LEG" in
    host|cuda) ;;
    *) warn "REFUSED: --leg must be 'host' or 'cuda', got '$LEG'."; exit 2 ;;
esac

# --- expectations -----------------------------------------------------------
# THE DEFAULT IS THE FULL SET.  See the header: a build that silently lost a
# backend must fail, and a site that deliberately builds fewer must say so.
if [[ "$LEG" == host ]]; then
    DEFAULT_BACKENDS="scalapack,gemm,slate,phdf5,fft"
else
    DEFAULT_BACKENDS="cusolvermp,cublasmp,cufft,slate,phdf5"
fi
BACKENDS="${LORRAX_FFI_EXPECT_BACKENDS:-$DEFAULT_BACKENDS}"

# The ABI this SOURCE TREE speaks, from the one header that defines it.
TREE_ABI=""
if [[ -r "$CPP_DIR/common/lorrax_ffi_abi.h" ]]; then
    TREE_ABI="$(sed -n 's/^#define[[:space:]]\+LORRAX_FFI_ABI_VERSION[[:space:]]\+\([0-9]\+\).*/\1/p' \
                "$CPP_DIR/common/lorrax_ffi_abi.h" | head -1)"
fi
EXPECT_ABI="${LORRAX_FFI_EXPECT_ABI:-$TREE_ABI}"

FAILED=0
NOTRUN=0
fail()  { warn "GATE FAILED ($1): ${*:2}"; FAILED=$((FAILED + 1)); }
notrun() { warn "GATE COULD NOT RUN ($1): ${*:2}"; NOTRUN=$((NOTRUN + 1)); }

# ---------------------------------------------------------------------------
# IS THIS A PLACE WHERE THIS LEG'S CLOSURE IS SUPPOSED TO RESOLVE?
#
# Three gates (1, 4, and GATE 7's mapped-objects half) resolve the dependency
# closure with ldd, and their answer is a property of the ENVIRONMENT as much
# as of the artifact.  For the host leg that is fine: it is built bare-metal
# and runs bare-metal, so the build node is a runtime environment for it and an
# unresolved dependency there is a real defect.
#
# For the device leg it is not.  That library is built and run INSIDE the
# Shifter container, against libraries bind-mounted there; on a login node
# `ldd` reports libnccl.so.2, libnvshmem_host.so.3 and libucc.so.1 as `not
# found` for a PERFECTLY CORRECT device library.  Reporting that as a failure
# would teach everyone that the verifier's red means "you ran it in the wrong
# place", which is precisely how the fftw gate got ignored for a day.
#
# So: out of environment, those three gates report COULD NOT RUN and say why.
# src/ffi/cpp/build.sh calls this verifier from INSIDE the container, where
# they are live; LORRAX_FFI_VERIFY_ENV=runtime forces them on anywhere.
# ---------------------------------------------------------------------------
RUNTIME_ENV=1
if [[ "$LEG" == cuda ]]; then
    RUNTIME_ENV=0
    [[ -d /lorrax_phdf5 || -d /lorrax_nvhpc ]] && RUNTIME_ENV=1
fi
[[ "${LORRAX_FFI_VERIFY_ENV:-}" == runtime ]] && RUNTIME_ENV=1
[[ "${LORRAX_FFI_VERIFY_ENV:-}" == build   ]] && RUNTIME_ENV=0

say "======================================================================"
say "artifact  $SO"
say "leg       $LEG"
say "expect    backends=$BACKENDS"
say "          abi=${EXPECT_ABI:-<unknown: no lorrax_ffi_abi.h in this tree>}"
say "          hdf5_soversion=${LORRAX_FFI_EXPECT_HDF5_SOVERSION:-<unstated>}"
say "======================================================================"

# ---------------------------------------------------------------------------
# Readers.  Dump each table ONCE.
#
# Never pipe `nm` into `grep -q` in a loop: under `set -o pipefail`, grep -q
# exits on first match and closes the pipe, nm dies of SIGPIPE, and the
# pipeline reports failure — which reads as "symbol missing" for every symbol
# that IS present.  That cost the Perlmutter script a false gate trip once.
# ---------------------------------------------------------------------------
if ! command -v readelf >/dev/null 2>&1 || ! command -v nm >/dev/null 2>&1; then
    warn "REFUSED: readelf and nm are required (binutils).  Every gate below"
    warn "  reads the ELF; there is no degraded mode that would still mean"
    warn "  anything."
    exit 1
fi

NEEDED="$(readelf -d "$SO" 2>/dev/null | grep NEEDED \
          | sed -n 's/.*Shared library: \[\(.*\)\].*/\1/p')"
DEFINED="$(nm -D --defined-only "$SO" 2>/dev/null | awk '{print $NF}')"
UNDEF="$(nm -D --undefined-only "$SO" 2>/dev/null | awk '{print $NF}')"

say "--- DT_NEEDED ---"
printf '%s\n' "$NEEDED" | sed "s/^/[$TAG]   /"

# ===========================================================================
# GATE 0 — THE BACKENDS THIS BUILD WAS ASKED FOR ARE THE ONES IT CONTAINS.
#
# THE GATE THE AUG-7 LIBRARY NEEDED AND DID NOT GET.  That build ran the
# generic src/ffi/cpp/build_host.sh, which configures no ScaLAPACK; CMake
# printed "ScaLAPACK/BLACS not found" as a WARNING, the link succeeded, and the
# library shipped with `scalapack=0` and zero Scalapack* handlers.  Nothing
# failed.  The defect surfaced later as nineteen contract cells going red in a
# way that needed a person to diagnose.
#
# Two independent readings, because neither implies the other:
#   (a) the build STAMP — what CMake resolved at configure time;
#   (b) the exported SYMBOLS — what actually came out of the link.
# A stamp saying 1 with no symbols is a link that dropped the objects; symbols
# with a stamp saying 0 is a stale build directory.  Both are real.
# ===========================================================================
say "--- GATE 0 (declared backends are present) ---"
STAMP="$(strings -a "$SO" 2>/dev/null | grep -m1 -E '(^|[[:space:]])leg=(host|cuda)[[:space:]]+abi=[0-9]+' || true)"
if [[ -z "$STAMP" ]]; then
    # Fall back to the pre-2026-08-08 host stamp, which had no leg= or abi=.
    STAMP="$(strings -a "$SO" 2>/dev/null | grep -m1 -E '^linked_fftw3=[01] scalapack=[01]' || true)"
fi
if [[ -n "$STAMP" ]]; then
    say "build stamp: $STAMP"
else
    fail 0a "$SO carries NO build stamp, so what it was configured to
      contain cannot be read from the artifact at all.  It predates
      common/build_config.cc for its leg (the device leg had no stamp of any
      kind before 2026-08-08).  This is not a pass and not a warning: a gate
      that scanned nothing must not print PASS.  Rebuild from a tree that
      stamps, or state the risk with LORRAX_FFI_VERIFY=off and do not certify
      the result."
fi

# backend -> (stamp key | -) and the exported-handler pattern.
backend_stamp_key() {
    case "$LEG:$1" in
        host:scalapack)  echo scalapack ;;
        host:gemm)       echo gemm ;;
        host:slate)      echo slate ;;
        host:phdf5)      echo phdf5 ;;
        # The FFT handlers are compiled unconditionally and bind their engine by
        # runtime dlsym, so there is no configure-time key for them.
        # `linked_fftw3` is a DIFFERENT fact (was a dlopen hint recorded) and is
        # legitimately 0 on an MKL site: reported below, never asserted.
        host:fft)        echo - ;;
        cuda:cusolvermp) echo cusolvermp ;;
        cuda:cublasmp)   echo cublasmp ;;
        cuda:cufft)      echo cufft ;;
        cuda:slate)      echo slate ;;
        cuda:phdf5)      echo phdf5 ;;
        *) echo "?" ;;
    esac
}
backend_symbol_re() {
    case "$LEG:$1" in
        host:scalapack)  echo '^Scalapack[A-Za-z0-9]*HostFfi$' ;;
        host:gemm)       echo '^MklBlas[A-Za-z0-9]*HostFfi$' ;;
        host:slate)      echo '^Slate[A-Za-z0-9]*HostFfi$' ;;
        host:phdf5)      echo '^Phdf[A-Za-z0-9]*HostFfi$' ;;
        host:fft)        echo '^MklFft[A-Za-z0-9]*HostFfi$' ;;
        cuda:cusolvermp) echo '^(EighMpFfi|CusolverMp[A-Za-z0-9]*Ffi)$' ;;
        cuda:cublasmp)   echo '^CublasMp[A-Za-z0-9]*Ffi$' ;;
        cuda:cufft)      echo '^Cufft[A-Za-z0-9]*CudaFfi$' ;;
        cuda:slate)      echo '^Slate[A-Za-z0-9]*Ffi$' ;;
        cuda:phdf5)      echo '^Phdf[A-Za-z0-9]*Ffi$' ;;
        *) echo "" ;;
    esac
}

IFS=',' read -r -a _want <<<"$BACKENDS"
for b in "${_want[@]}"; do
    b="$(echo "$b" | tr -d '[:space:]')"
    [[ -n "$b" ]] || continue
    re="$(backend_symbol_re "$b")"
    if [[ -z "$re" ]]; then
        fail 0b "'$b' is not a backend of the $LEG leg.  Known:
      host = scalapack gemm slate phdf5 fft
      cuda = cusolvermp cublasmp cufft slate phdf5"
        continue
    fi
    # On the device leg the host handlers are absent, but keep the two name
    # spaces apart explicitly rather than by luck.
    if [[ "$LEG" == cuda ]]; then
        n="$(printf '%s\n' "$DEFINED" | grep -v 'Host' | grep -cE "$re")"
    else
        n="$(printf '%s\n' "$DEFINED" | grep -cE "$re")"
    fi
    key="$(backend_stamp_key "$b")"
    sval=""
    if [[ "$key" != "-" && -n "$STAMP" ]]; then
        sval="$(printf '%s\n' "$STAMP" | grep -oE "(^|[[:space:]])${key}=[01]" \
                | tr -d '[:space:]' | cut -d= -f2 | head -1)"
    fi
    if [[ "$n" -eq 0 ]]; then
        fail 0c "backend '$b' was declared but this library exports NO
      handler matching $re.
      stamp says ${key}=${sval:-<absent>}.
      This is the Aug-7 defect exactly: a vendor library the configure step
      could not find becomes a CMake WARNING, the link succeeds, and the
      handler group is simply gone.  Either the build is broken, or this site
      genuinely does not build '$b' and must say so:
        LORRAX_FFI_EXPECT_BACKENDS=<the list without '$b'>"
    elif [[ -n "$sval" && "$sval" != "1" ]]; then
        fail 0d "backend '$b' exports $n handler(s) but the build stamp says
      ${key}=${sval}.  Stamp and link disagree — usually a stale build
      directory whose CMakeCache predates the current configure.  Rebuild
      with --fresh."
    else
        say "GATE 0 backend '$b': $n handler(s), stamp ${key}=${sval:-n/a}  OK"
    fi
done
if [[ "$LEG" == host && -n "$STAMP" ]]; then
    say "GATE 0 note: $(printf '%s\n' "$STAMP" | grep -oE 'linked_fftw3=[01]' || echo 'linked_fftw3=?') \
— REPORTED, never gated: which FFT engine binds is a runtime dlsym fact, and 0 is correct on an MKL site."
fi

# ===========================================================================
# GATE 1 — exactly one MPI runtime in the closure.  Delegated: gate_one_mpi.sh
# owns the argument for why it resolves with ldd and dedupes by realpath.
# ===========================================================================
say "--- GATE 1 (one MPI runtime) ---"
if [[ ! -x "$CPP_DIR/gate_one_mpi.sh" ]]; then
    notrun 1 "$CPP_DIR/gate_one_mpi.sh not found (LORRAX_ROOT=$LORRAX_ROOT)."
elif [[ "$RUNTIME_ENV" -eq 0 ]]; then
    notrun 1 "this is not a runtime environment for the '$LEG' leg, so the
      dependency closure cannot be resolved here and the one-MPI invariant
      cannot be checked.  The device library binds its MPI, NCCL and HDF5
      inside the Shifter container; on a login node ldd reports several of
      them 'not found' for a perfectly correct artifact.
      Run it where the library runs — src/ffi/cpp/build.sh does, because it
      builds in-container — or force it with LORRAX_FFI_VERIFY_ENV=runtime."
else
    GATE_TAG="$TAG" "$CPP_DIR/gate_one_mpi.sh" "$SO" "${LORRAX_FFI_EXPECT_MPI:-}" \
        || fail 1 "see gate_one_mpi.sh output above."
fi

# ===========================================================================
# GATE 2 — ONE BLAS, ONE THREADING FLAVOUR.
#
# Generalised from the Perlmutter script's LibSci-only form so a port inherits
# it.  Two threading flavours of one vendor's BLAS in one process — LibSci's
# libsci_gnu.so beside libsci_gnu_mp.so, MKL's sequential beside gnu_thread —
# link cleanly and leave ELF order deciding which one runs.  The Cray runtime
# at least prints [CRAYBLAS_WARNING] on the first BLAS call; MKL prints
# nothing.  MEASURED on the Aug-7 deployed host library: libsci_gnu.so.6 AND
# libsci_gnu_mpi_mp.so.6 AND libsci_gnu_mp.so.6, all three, because the
# cray-libsci module was still loaded at configure time and the CC wrapper
# auto-injected the sequential one.
#
# Two vendors' BLAS at once is the same hazard one level up and is also caught.
# ===========================================================================
say "--- GATE 2 (one BLAS, one threading flavour) ---"
_libsci_seq=$(printf '%s\n' "$NEEDED" | grep -cE '^libsci_gnu(_mpi)?\.so')
_libsci_mp=$(printf '%s\n'  "$NEEDED" | grep -cE '^libsci_gnu(_mpi)?_mp\.so')
_mkl_seq=$(printf '%s\n'    "$NEEDED" | grep -cE '^libmkl_sequential\.so')
_mkl_thr=$(printf '%s\n'    "$NEEDED" | grep -cE '^libmkl_(gnu|intel)_thread\.so')
_vendors=""
[[ $((_libsci_seq + _libsci_mp)) -gt 0 ]] && _vendors="$_vendors libsci"
[[ $(printf '%s\n' "$NEEDED" | grep -cE '^libmkl_') -gt 0 ]] && _vendors="$_vendors mkl"
[[ $(printf '%s\n' "$NEEDED" | grep -cE '^libopenblas') -gt 0 ]] && _vendors="$_vendors openblas"
[[ $(printf '%s\n' "$NEEDED" | grep -cE '^libblis') -gt 0 ]] && _vendors="$_vendors blis"
say "BLAS vendors in DT_NEEDED:${_vendors:- (none directly linked)}"

if [[ "$_libsci_seq" -gt 0 && "$_libsci_mp" -gt 0 ]]; then
    fail 2a "BOTH LibSci threading flavours are linked (sequential=$_libsci_seq
      threaded=$_libsci_mp):
$(printf '%s\n' "$NEEDED" | grep -E '^libsci' | sed 's/^/        /')
      ELF load order then decides which BLAS and which ScaLAPACK runs, and
      nobody re-derives that order after a link-line edit.  The Cray runtime
      reports it as
        [CRAYBLAS_WARNING] Application linked against multiple cray-libsci libraries
      Cause, every time it has happened here: the cray-libsci module was still
      loaded when cmake ran, so the CC wrapper auto-injected -lsci_gnu on top
      of the explicit threaded pair.  Capture the prefix, THEN unload the
      module, THEN configure."
fi
if [[ "$_mkl_seq" -gt 0 && "$_mkl_thr" -gt 0 ]]; then
    fail 2b "BOTH MKL threading layers are linked (sequential=$_mkl_seq
      threaded=$_mkl_thr).  Same hazard as 2a, and MKL prints no warning at
      all.  Pick one threading layer and match SLATE's."
fi
_nvendor=$(printf '%s\n' $_vendors | grep -c . || true)
_gate2_bad=0
if [[ "$_nvendor" -gt 1 ]]; then
    fail 2c "$_nvendor BLAS vendors linked:$_vendors.  Two definitions of
      cblas_dgemm in one process, resolved by load order.  This is why the
      Perlmutter recipe refuses to link MKL for its FFT beside LibSci for its
      ScaLAPACK."
    _gate2_bad=1
fi
[[ "$_libsci_seq" -gt 0 && "$_libsci_mp" -gt 0 ]] && _gate2_bad=1
[[ "$_mkl_seq"    -gt 0 && "$_mkl_thr"   -gt 0 ]] && _gate2_bad=1
[[ "$_gate2_bad" -eq 0 ]] && say "GATE 2 PASSED: one BLAS, one threading flavour"

# ===========================================================================
# GATE 3 — PLATFORM PURITY.  The host leg is CUDA-free BY CONSTRUCTION; that
# is what makes it loadable on a CPU-only node and inside a container with no
# driver.  With craype-accel-nvidia80 or cudatoolkit loaded the CC wrapper
# links libmpi_gtl_cuda, whose libcuda.so.1 dependency defeats the point.
# ===========================================================================
say "--- GATE 3 (platform purity) ---"
if [[ "$LEG" == host ]]; then
    _cuda_needed="$(printf '%s\n' "$NEEDED" | grep -iE 'cuda|nccl|nvshmem|^libcal\.so' || true)"
    if [[ -n "$_cuda_needed" ]]; then
        fail 3 "the host library links CUDA-stack libraries:
$(printf '%s\n' "$_cuda_needed" | sed 's/^/        /')
      Check that craype-accel-nvidia80 and cudatoolkit are unloaded and that
      the SLATE install really is the gpu_backend=none variant."
    else
        say "GATE 3 PASSED: no CUDA-stack library in DT_NEEDED"
    fi
else
    say "GATE 3 N/A on the cuda leg (a CUDA library links CUDA)"
fi

# ===========================================================================
# GATE 5 — THE RUN-TIME-RESOLVED FFT ENGINE IS NOT A LOAD-TIME DEPENDENCY.
#
# Two halves, because neither implies the other, and the history is the reason.
# The structural proof once recorded for this was `nm -D --undefined-only | grep
# -c fftw_` -> 0, which was true while the library was completely unloadable:
# dlsym'ing every entry point drives that count to zero BY CONSTRUCTION.
# Meanwhile the build was passing -lfftw3 plus two more fftw libraries adopted
# by a FindOpenMP misdetection, so the .so carried three fftw DT_NEEDED
# entries.  DT_NEEDED is resolved before any of this library's code runs, and
# the Shifter container does not bind-mount /opt/cray/pe/fftw at all.  Measured
# 2026-08-06: dlopen failed outright and NINETEEN ScaLAPACK/SLATE/GEMM contract
# tests that never perform an FFT reported as SKIPPED, with the suite green at
# 0 failures.  A lost FFT optimisation silently became a lost linear-algebra
# test suite.
# ===========================================================================
say "--- GATE 5 (FFT engine is not a load-time dependency) ---"
_fftw_undef=$(printf '%s\n' "$UNDEF" | grep -c 'fftw_')
_fftw_needed="$(printf '%s\n' "$NEEDED" | grep -i 'fftw' || true)"
if [[ "$_fftw_undef" -ne 0 ]]; then
    fail 5a "$_fftw_undef undefined fftw_ symbols.  The flat-k TU must resolve
      every FFTW3 entry point through mklpin::resolve_sym, never by direct
      call."
fi
if [[ -n "$_fftw_needed" ]]; then
    fail 5b "fftw entries in DT_NEEDED:
$(printf '%s\n' "$_fftw_needed" | sed 's/^/        /')
      An engine resolved at RUN time must not be a LOAD-time dependency.  Any
      fftw here makes the WHOLE library — ScaLAPACK, SLATE, GEMM and phdf5
      handlers included — unloadable wherever that exact SONAME is absent.
      Two known causes: an -lfftw3 on the link line (the CMakeLists records
      the path as a dlopen hint instead), or FindOpenMP adopting cray-fftw's
      libfftw3*_omp as the OpenMP runtime — check that the build unloaded
      cray-fftw before invoking cmake, and check OpenMP_CXX_LIB_NAMES in
      CMakeCache.txt."
fi
[[ "$_fftw_undef" -eq 0 && -z "$_fftw_needed" ]] && \
    say "GATE 5 PASSED: undefined fftw_ symbols=0, fftw in DT_NEEDED=0"

# ===========================================================================
# GATE 6 — THE OpenMP RUNTIME IS AN OpenMP RUNTIME.  gomp (GNU) or iomp5/omp
# (LLVM/Intel) are the legitimate answers; anything else means FindOpenMP
# matched on the substring "omp" in a library that is not one.  That
# misdetection is what put two of the three fftw entries into GATE 5's list,
# and nothing in the build reported it.
# ===========================================================================
say "--- GATE 6 (OpenMP runtime is really OpenMP) ---"
_omp="$(printf '%s\n' "$NEEDED" | grep -E 'omp' || true)"
_bad_omp="$(printf '%s\n' "$_omp" | grep -vE '^lib(gomp|iomp5|omp)\.so' || true)"
if [[ -n "${_bad_omp//[[:space:]]/}" ]]; then
    fail 6 "a non-OpenMP library was adopted as the OpenMP runtime:
$(printf '%s\n' "$_bad_omp" | sed 's/^/        /')
      Check OpenMP_CXX_LIB_NAMES in the build directory's CMakeCache.txt."
else
    say "GATE 6 PASSED: OpenMP entries =${_omp:- (none)}"
fi

# ===========================================================================
# GATE 7 — ONE HDF5, AND THE RUNTIME PROVIDES IT.  Delegated to
# gate_one_hdf5.sh, which owns the argument for why "the SONAME resolves" is
# the wrong invariant (aliasing two HDF5 majors onto one name resolves, and is
# worse than the failure it fixes).
#
# THE PART THIS FILE ADDS: an EXPECTED SOVERSION, stated by the caller.  The
# stage comparison alone cannot catch the trap that cost the kchunk branch a
# whole build, because the host leg's HDF5 is chosen by the Cray compiler
# wrappers (LORRAX_PM_HDF5) while LORRAX_FFI_PHDF5_DIR only names the tree to
# compare against.  Setting only the latter produces a .so.310 library that
# passes a comparison against a .so.310 stage and is unloadable beside the
# .so.200 device library it has to run with.  Stating the number the RUNTIME
# will mount is the check that survives that.
# ===========================================================================
say "--- GATE 7 (one HDF5, and the runtime provides it) ---"
_h5_req="$(printf '%s\n' "$NEEDED" | grep -oE 'libhdf5[A-Za-z0-9_]*\.so\.[0-9]+' | sort -u)"
if [[ -z "$_h5_req" ]]; then
    say "GATE 7 N/A: this artifact links no HDF5 (LORRAX_FFI_HAVE_PHDF5=OFF)."
else
    say "HDF5 requested: $(printf '%s ' $_h5_req)"
    _h5_sov="$(printf '%s\n' "$_h5_req" | grep -oE '[0-9]+$' | sort -u)"
    if [[ -n "${LORRAX_FFI_EXPECT_HDF5_SOVERSION:-}" ]]; then
        if [[ "$_h5_sov" != "$LORRAX_FFI_EXPECT_HDF5_SOVERSION" ]]; then
            fail 7-expect "this library links HDF5 SOVERSION $(printf '%s ' $_h5_sov)
      but the runtime will mount ${LORRAX_FFI_EXPECT_HDF5_SOVERSION}.
      In the container that request is a plain 'not found' and the ENTIRE
      library fails to dlopen — every handler in it, including the ones that
      never touch HDF5.
      On Perlmutter the lever that actually decides this is the MODULE:
        LORRAX_PM_HDF5=cray-hdf5-parallel/<version>
      LORRAX_FFI_PHDF5_DIR does NOT choose it — the Cray compiler wrappers do
      — it only names the tree GATE 7c compares against.  Set BOTH."
        else
            say "GATE 7 SOVERSION $_h5_sov matches the stated runtime expectation"
        fi
    else
        say "GATE 7 note: no LORRAX_FFI_EXPECT_HDF5_SOVERSION stated; the"
        say "  cross-population check below is the only SOVERSION evidence."
    fi
    if [[ -x "$CPP_DIR/gate_one_hdf5.sh" ]]; then
        _peer=()
        if [[ -n "${LORRAX_FFI_EXPECT_PEER_SO:-}" ]]; then
            if [[ -f "${LORRAX_FFI_EXPECT_PEER_SO}" ]]; then
                _peer=("${LORRAX_FFI_EXPECT_PEER_SO}")
                say "peer leg: ${LORRAX_FFI_EXPECT_PEER_SO}"
            else
                fail 7-peer "LORRAX_FFI_EXPECT_PEER_SO=${LORRAX_FFI_EXPECT_PEER_SO}
      does not exist.  An explicit pin that cannot be honoured is a refusal,
      never a fall-through."
            fi
        fi
        # The request/stage halves of GATE 7 are environment-independent — they
        # read SONAMEs out of ELF headers and out of the stage tree — and they
        # are the halves that carry the cross-leg invariant.  The mapped-objects
        # half needs a resolvable closure, which a peer from the OTHER leg does
        # not have here.  Run what can be run and say what was not.
        _closure=on
        if [[ "$RUNTIME_ENV" -eq 0 ]]; then
            _closure=off
        elif [[ ${#_peer[@]} -gt 0 ]]; then
            # The peer is the other leg by construction, so its closure belongs
            # to the other environment.
            _closure=off
            say "GATE 7: closure half OFF for this invocation — the peer leg's"
            say "  dependencies resolve in the other environment.  The"
            say "  SOVERSION comparison across the pair, which is the point of"
            say "  naming a peer, is unaffected."
        fi
        GATE_TAG="$TAG" LORRAX_GATE_ONE_HDF5_CLOSURE="$_closure" \
            "$CPP_DIR/gate_one_hdf5.sh" "$SO" "${_peer[@]}" \
            || fail 7 "see gate_one_hdf5.sh output above."
        [[ "$_closure" == off ]] && notrun 7e "the mapped-objects half of GATE 7
      (are two distinct HDF5 objects reachable at once) was not checked in
      this environment.  Run the verifier where the library runs, or
      LORRAX_FFI_VERIFY_ENV=runtime."
    else
        notrun 7 "$CPP_DIR/gate_one_hdf5.sh not found."
    fi
fi

# ===========================================================================
# GATE 4 — NOTHING LEFT UNRESOLVED AT LOAD TIME.  -Wl,--no-undefined already
# fails the LINK on a missing link-time symbol; this is the other half — a
# NEEDED library that cannot itself be found at run time, which is the failure
# ffi_loader reports as "exists but could not be loaded".
#
# Run LAST of the static gates on purpose: it is the one whose answer depends
# on the environment this script happens to run in, so a person reading the
# output has already seen every environment-independent fact by the time it
# speaks.
# ===========================================================================
say "--- GATE 4 (load-time resolution) ---"
if [[ "$RUNTIME_ENV" -eq 0 ]]; then
    notrun 4 "this is not a runtime environment for the '$LEG' leg.  ldd here
      reports the container-mounted libraries (NCCL, NVSHMEM, UCC, the staged
      HDF5) as 'not found' for a correct artifact, so a failure would mean
      'wrong place', not 'wrong library'.  src/ffi/cpp/build.sh runs this
      verifier in-container, where the gate is live; force it anywhere with
      LORRAX_FFI_VERIFY_ENV=runtime."
elif command -v ldd >/dev/null 2>&1; then
    _ldd="$(ldd -r "$SO" 2>&1)"
    _bad="$(printf '%s\n' "$_ldd" | grep -iE 'not found|undefined symbol' || true)"
    if [[ -n "$_bad" ]]; then
        fail 4 "unresolved at load time in THIS environment:
$(printf '%s\n' "$_bad" | head -20 | sed 's/^/        /')
      If this is the build node and the library is meant to run elsewhere,
      that is not automatically a defect — but it IS the thing to check
      before deploying, and LD_LIBRARY_PATH here must cover every directory
      the run will have (in a container, every RPATH directory must actually
      be bind-mounted)."
    else
        say "GATE 4 PASSED: closure resolves here"
    fi
else
    notrun 4 "ldd is not available, so the closure cannot be resolved.
      readelf -d is not a substitute: it sees direct DT_NEEDED only."
fi

# ===========================================================================
# GATE 8 — ENGINE IDENTITY, WHERE IT CAN RUN.
#
# GATE 5 is half the property and this file cannot fake the other half.  Zero
# fftw in DT_NEEDED means nothing binds at LOAD time; it says nothing about
# which engine — or how many — the process ends up with, because the engine
# arrives by dlopen at first use and no static tool can see it.  Measured
# 2026-08-06 on one .so: the bare host maps cray-fftw and the container maps
# NOTHING, and both pass GATE 5.
#
# gate_one_fftw.sh closes it by driving one real flat-k FFT and reading
# /proc/self/maps in that process.  It needs a python that can import jax, and
# it CANNOT run on a login node (shifter cannot bind-mount $HOME there).  So a
# NOT RUN here is honest and expected; it is counted separately and never
# reported as a pass.
# ===========================================================================
say "--- GATE 8 (FFT engine identity) ---"
if [[ "$LEG" != host ]]; then
    say "GATE 8 N/A: the device leg's FFT is cuFFT, linked, not dlopened."
elif [[ ! -x "$CPP_DIR/gate_one_fftw.sh" ]]; then
    notrun 8 "$CPP_DIR/gate_one_fftw.sh not found."
elif [[ -z "${LORRAX_GATE_FFTW_PY:-}${LORRAX_FFTW3_STAGE:-}" ]]; then
    notrun 8 "no runnable python named.  This gate drives a real FFT and reads
      /proc/self/maps, so it needs a process, not an ELF:
        bare host:    LORRAX_GATE_FFTW_PY=<venv>/bin/python \\
                        src/ffi/cpp/gate_one_fftw.sh $SO
        in-container: LORRAX_FFTW3_STAGE=/lorrax_fftw \\
                        src/ffi/cpp/gate_one_fftw.sh $SO [<device.so>]
      It cannot run on a login node (shifter cannot bind-mount \$HOME there);
      run it inside an allocation."
else
    GATE_TAG="$TAG" "$CPP_DIR/gate_one_fftw.sh" "$SO" \
        || fail 8 "see gate_one_fftw.sh output above."
fi

# ===========================================================================
# GATE 9 — THE HANDLER-SIGNATURE ABI.
#
# The artifact-level half of the check the Python loaders make at dlopen.  Its
# value here is that it fires at BUILD time, in the build's own output, rather
# than at the first read_slabs of a run that has already queued, allocated and
# started.  See src/ffi/cpp/common/lorrax_ffi_abi.h for the rule and for the
# two bumps in two days that motivated it.
# ===========================================================================
say "--- GATE 9 (handler-signature ABI) ---"
_abi_sym="lorrax_ffi_${LEG}_abi_version"
_stamp_abi="$(printf '%s\n' "$STAMP" | grep -oE 'abi=[0-9]+' | cut -d= -f2 | head -1)"
if ! printf '%s\n' "$DEFINED" | grep -qx "$_abi_sym"; then
    # COULD NOT RUN, not FAILED, and the distinction is the same one the
    # Python loaders make.  An UNSTAMPED library was built before 2026-08-08;
    # that is not evidence it is wrong, and the deployed pair plus every
    # library the nine pinning worktrees use is unstamped today.  Calling
    # those FAILED would make the verifier's own red mean "old" far more often
    # than "broken", which is how a gate gets disarmed.
    #
    # It is equally not a PASS: the gate scanned and found nothing to compare.
    # Any library built from this tree stamps, so LORRAX_FFI_VERIFY_STRICT=1
    # — which every certification run should set — still refuses an unstamped
    # artifact, and a fresh build still passes under it.
    notrun 9 "$SO does not export $_abi_sym, so its handler-signature ABI
      cannot be read: built before the stamp existed (2026-08-08).  Not a
      pass and not a defect.  The loaders announce the same fact at dlopen;
      LORRAX_FFI_ABI_STRICT=1 there and LORRAX_FFI_VERIFY_STRICT=1 here both
      turn it into a refusal.  Rebuild from this tree to make it checkable."
elif [[ -z "$EXPECT_ABI" ]]; then
    notrun 9 "the artifact exports $_abi_sym (stamp says abi=${_stamp_abi:-?})
      but no expected version is available: this tree has no
      src/ffi/cpp/common/lorrax_ffi_abi.h and LORRAX_FFI_EXPECT_ABI is unset."
elif [[ "$_stamp_abi" != "$EXPECT_ABI" ]]; then
    fail 9b "ABI MISMATCH: this library speaks abi=${_stamp_abi:-<unreadable>},
      the source tree speaks abi=${EXPECT_ABI}.
      These two cannot be paired.  Mixing them is not a degraded run: it is an
      FFI arity mismatch that surfaces as
        INVALID_ARGUMENT: Wrong number of arguments: expected N but got M
      at the first call that crosses the changed signature — and everything
      off that path is green until then.
      fix: rebuild this leg from this tree.
        host:  bash config/perlmutter/build_ffi_host.sh --fresh
        cuda:  src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh --fresh"
else
    say "GATE 9 PASSED: abi=$_stamp_abi matches this source tree"
fi

# ---------------------------------------------------------------------------
say "======================================================================"
if [[ "$NOTRUN" -gt 0 && "${LORRAX_FFI_VERIFY_STRICT:-0}" == "1" ]]; then
    warn "STRICT: $NOTRUN gate(s) could not run.  A gate that cannot run is"
    warn "  not a gate that passed."
    FAILED=$((FAILED + NOTRUN))
fi
if [[ "$FAILED" -gt 0 ]]; then
    warn "VERIFY FAILED: $FAILED gate(s) failed on $SO"
    [[ "$NOTRUN" -gt 0 ]] && warn "  ($NOTRUN further gate(s) could not run)"
    exit 1
fi
if [[ "$NOTRUN" -gt 0 ]]; then
    say "VERIFY PASSED with $NOTRUN gate(s) NOT RUN — read the reasons above."
    say "  Re-run where they can run, or LORRAX_FFI_VERIFY_STRICT=1 to refuse."
else
    say "VERIFY PASSED: every gate ran and passed on $SO"
fi
exit 0
