#!/usr/bin/env bash
# ============================================================================
# build_mpiwrapper.sh — build the LORRAX MPIwrapper for
# JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi on Frontera.
#
#   config/frontera/build_mpiwrapper.sh [--fresh]
#
# Output: $LORRAX_MPIWRAPPER_PREFIX/lib64/libmpiwrapper.so
#   (default $WORK/lorrax_mpiwrapper/install).  Point MPITRAMPOLINE_LIB at it.
#
# RUN ON A LOGIN NODE, NOT IN THE py312 CONTAINER.  MPIwrapper's CMakeLists
# declares `LANGUAGES C CXX Fortran` and really does compile the Fortran
# bindings; the container has no gfortran.  The host toolchain (gcc 8.3 /
# gfortran, cmake >= 3.20, Intel MPI 2020.4) has everything.
#
# ----------------------------------------------------------------------------
# STATUS: INTERIM STOPGAP.  READ THIS BEFORE ADOPTING IT.
# ----------------------------------------------------------------------------
# A locally-patched MPI shim is a dependency this project should not want to
# ship: every user on every machine would have to build, version and
# understand it.  This script exists because the workaround is needed TODAY and
# an untracked binary in someone's scratch directory is worse than a pinned,
# verified, in-tree recipe -- not because a patched MPI ABI shim is the
# intended architecture.
#
# UPDATE: override (2) is now SUPERSEDED and should not be enabled.
# common.collectives.warm_mesh_cliques() creates each mesh-axis MPI
# communicator (plus the world one) from the Python MAIN THREAD at
# mesh-construction time -- three 8-byte psums, ~150 ms once per process.
# XLA caches communicators per participating-device set and only calls
# CreateCommunicator on a MISS, so every later collective, including the ones
# a pool worker issues inside the BSE Lanczos jit, is a cache hit and never
# reaches the MPI_Is_thread_main guard.  Verified on the real BSE deck with
# this gate UNSET.  It is also SAFER: the override lets XLA call
# MPI_Comm_split (collective over MPI_COMM_WORLD) from arbitrary pool workers,
# and nothing serialises that ACROSS ranks -- a latent deadlock the warm-up
# does not have.
#
# So override (2) stays in the patch only as a fallback / positive control.
# Override (1) -- the MPI_THREAD_MULTIPLE upgrade -- IS STILL REQUIRED, for an
# unrelated reason: XLA's collective OPERATIONS still run on pool threads
# concurrently with h5py/mpi4py collective MPI-IO on the main thread, which is
# the AS.4b ~29% class.  It retires only if jaxlib stops requesting
# MPI_THREAD_FUNNELED -- an upstream change.  That is the remaining exit, and
# it is the only reason this script still needs to exist.
# See docs/dev/mpi_collectives.md and wk_REL/jax_threadmain_alternatives.md.
#
# ----------------------------------------------------------------------------
# WHAT THIS IS AND WHY IT EXISTS
# ----------------------------------------------------------------------------
# jaxlib ships an MPItrampoline-linked XLA:CPU collectives backend.  Under
# `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` every jax CPU collective enters
# MPItrampoline, which dlopen()s the MPIwrapper named by MPITRAMPOLINE_LIB and
# resolves the real MPI through it.  MPIwrapper is therefore the ONE place
# where we can intervene between jaxlib and Intel MPI without patching jaxlib.
#
# We build upstream MPIwrapper v2.11.1 (eschnett/MPIwrapper) at the pinned
# commit below plus exactly ONE patch file, `mpiwrapper/lorrax_thread.patch`,
# which adds two overrides.  Nothing else differs from upstream.
#
#   (1) ALWAYS ON — every MPI_Init / MPI_Init_thread is upgraded to
#       MPI_THREAD_MULTIPLE.  XLA's MpiCollectives::Init() calls
#       MPI_Init_thread(..., MPI_THREAD_FUNNELED, &provided) and never reads
#       `provided`; with mpi4py/h5py collective MPI-IO on the Python main
#       thread and XLA's collectives on an executor thread, a FUNNELED grant
#       is undefined behaviour.  Measured on Frontera as a ~29% segfault/hang
#       rate at P=16 x 8 nodes, provider-independent, always at the zeta-write
#       / V_q boundary where the two threads overlap; two threads of one rank
#       simultaneously inside MPID_Progress_wait.  Upgrading the GRANT to
#       MULTIPLE makes Intel MPI's global lock serialize them.  Requests are
#       upgraded, never downgraded, and the init ORDER is unchanged.
#       (SPEEDUP_SCORECARD.md AS.4b / AS.4c.  Certified 5/5 at P=16 x 8 nodes.)
#
#   (2) OPT-IN, gated on LORRAX_MPI_FORCE_THREAD_MAIN — MPI_Is_thread_main
#       reports true to every caller.  jaxlib 0.9.1
#       xla::cpu::MpiCollectives::CreateCommunicators() opens with
#
#           int flag; MPI_Is_thread_main(&flag);
#           if (!flag) return absl::UnknownError(
#               "MPI: Communicator requested from a thread that is not the "
#               "one MPI was initialized from. ...");
#
#       and only then MPI_Comm_split()s the new clique.  XLA:CPU dispatches
#       collective thunks to intra-op pool workers whenever the ThunkExecutor
#       takes its PARALLEL path, so any clique whose FIRST use is inside a
#       real (non-trivial) jitted program is refused.  This is NOT a thread
#       LEVEL test — MPI_Is_thread_main is false on a non-initialising thread
#       even under MPI_THREAD_MULTIPLE, so no MPI_THREAD_* setting and no
#       collective warm-up ordering can satisfy it.  Confirmed by disassembly
#       of CreateCommunicators in jaxlib/libjax_common.so and by an env-gated
#       negative control (gate OFF: 4 refusals and death; gate ON: clean).
#
#       Legality: MPI_Comm_split from a pool worker is legal MPI only because
#       override (1) has already made the grant MPI_THREAD_MULTIPLE, and
#       because XLA creates cliques in a deterministic order that is identical
#       on every rank.  Blast radius is exactly XLA's CPU collectives —
#       mpi4py, h5py and the FFI host .so all link Intel libmpi.so.12 directly
#       and never route through MPItrampoline, so they never see either
#       override.
#
#       Default OFF: with the variable unset the wrapper's behaviour is
#       byte-for-byte the certified (1)-only wrapper.
#
# See docs/dev/mpi_collectives.md for the full rationale, the evidence, and
# the launch recipe.  Env-var rows: docs/dev/env_vars.md
# (LORRAX_MPI_FORCE_THREAD_MAIN, MPITRAMPOLINE_LIB, LORRAX_MPI_FINALIZE_FIX).
#
# ----------------------------------------------------------------------------
# UPSTREAM SOURCE
# ----------------------------------------------------------------------------
# MPIwrapper is external source under its own licence (see LICENSE.md in the
# checkout), so it is NOT vendored into this repo — only the patch is.  The
# script fetches the pinned commit if LORRAX_MPIWRAPPER_SRC is not supplied.
# Offline builds: hand it an existing checkout at the pinned commit.
# ============================================================================
set -euo pipefail

# --- upstream pin -----------------------------------------------------------
MPIW_REPO="${LORRAX_MPIWRAPPER_REPO:-https://github.com/eschnett/MPIwrapper.git}"
# v2.11.1 + CI update.  Pinned by SHA, not by tag: a tag can be moved.
MPIW_COMMIT="${LORRAX_MPIWRAPPER_COMMIT:-966f4231c96153a08295fc7d0bcbd65e916a73fd}"

# --- paths ------------------------------------------------------------------
: "${LORRAX_ROOT:?set LORRAX_ROOT to the worktree/repo root (contains config/frontera)}"
: "${LORRAX_MPIWRAPPER_STAGE:=${WORK:-$HOME}/lorrax_mpiwrapper}"
: "${LORRAX_MPIWRAPPER_SRC:=$LORRAX_MPIWRAPPER_STAGE/src}"
: "${LORRAX_MPIWRAPPER_BUILD:=$LORRAX_MPIWRAPPER_STAGE/build}"
: "${LORRAX_MPIWRAPPER_PREFIX:=$LORRAX_MPIWRAPPER_STAGE/install}"
: "${LORRAX_IMPI_ROOT:=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64}"

PATCH="$LORRAX_ROOT/config/frontera/mpiwrapper/lorrax_thread.patch"
CMAKE="${LORRAX_CMAKE:-$(command -v cmake || true)}"

[ -f "$PATCH" ] || { echo "[build_mpiw] missing patch: $PATCH" >&2; exit 2; }
[ -n "$CMAKE" ] || { echo "[build_mpiw] no cmake on PATH (module load cmake)" >&2; exit 2; }
[ -x "$LORRAX_IMPI_ROOT/bin/mpicc" ] || {
    echo "[build_mpiw] no Intel MPI at $LORRAX_IMPI_ROOT (set LORRAX_IMPI_ROOT)" >&2
    exit 2; }
command -v gfortran >/dev/null || {
    echo "[build_mpiw] no gfortran on PATH.  MPIwrapper compiles Fortran" >&2
    echo "             bindings; run this on a LOGIN NODE, not inside the" >&2
    echo "             py312 apptainer container." >&2
    exit 2; }

if [ "${1:-}" == "--fresh" ]; then
    rm -rf "$LORRAX_MPIWRAPPER_SRC" "$LORRAX_MPIWRAPPER_BUILD" \
           "$LORRAX_MPIWRAPPER_PREFIX"
fi
mkdir -p "$LORRAX_MPIWRAPPER_STAGE"

# --- 1. upstream checkout at the pinned commit ------------------------------
if [ ! -d "$LORRAX_MPIWRAPPER_SRC/.git" ] && [ ! -f "$LORRAX_MPIWRAPPER_SRC/src/mpiwrapper.cxx" ]; then
    echo "[build_mpiw] cloning $MPIW_REPO -> $LORRAX_MPIWRAPPER_SRC"
    git clone "$MPIW_REPO" "$LORRAX_MPIWRAPPER_SRC"
fi
if [ -d "$LORRAX_MPIWRAPPER_SRC/.git" ]; then
    git -C "$LORRAX_MPIWRAPPER_SRC" fetch --all --tags --quiet || true
    git -C "$LORRAX_MPIWRAPPER_SRC" checkout --quiet "$MPIW_COMMIT"
    HAVE=$(git -C "$LORRAX_MPIWRAPPER_SRC" rev-parse HEAD)
    [ "$HAVE" = "$MPIW_COMMIT" ] || {
        echo "[build_mpiw] FAILED: checkout is $HAVE, want $MPIW_COMMIT" >&2
        exit 1; }
    echo "[build_mpiw] upstream pinned at $MPIW_COMMIT"
else
    echo "[build_mpiw] WARNING: $LORRAX_MPIWRAPPER_SRC is not a git checkout;"
    echo "[build_mpiw]          the upstream pin ($MPIW_COMMIT) is UNVERIFIED."
fi

# --- 2. apply the LORRAX patch (idempotent) ---------------------------------
cd "$LORRAX_MPIWRAPPER_SRC"
if grep -q lorrax_is_thread_main src/mpiwrapper.cxx; then
    echo "[build_mpiw] patch already applied."
else
    echo "[build_mpiw] applying $PATCH"
    patch -p1 --forward < "$PATCH"
fi
# Both overrides must be present.  A patch that applied with fuzz into the
# wrong scope, or a stale checkout, is a silent wrong-binary hazard: the
# wrapper would build, load, and quietly grant FUNNELED.
for sym in lorrax_upgraded_init_thread lorrax_is_thread_main; do
    grep -q "$sym" src/mpiwrapper.cxx || {
        echo "[build_mpiw] FAILED: $sym absent after patching." >&2; exit 1; }
done

# --- 3. configure + build ---------------------------------------------------
# Upstream's own recipe: the MPI it wraps is chosen by MPI_HOME/CMake FindMPI.
# --as-needed keeps libgfortran out of the DT_NEEDED list (the certified
# wk_AS build was linked this way; the .so is dlopen'd by MPItrampoline inside
# the py312 container, which has no libgfortran).
export MPI_HOME="$LORRAX_IMPI_ROOT"
export PATH="$LORRAX_IMPI_ROOT/bin:$PATH"

echo "[build_mpiw] configuring -> $LORRAX_MPIWRAPPER_BUILD"
"$CMAKE" -S "$LORRAX_MPIWRAPPER_SRC" -B "$LORRAX_MPIWRAPPER_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$LORRAX_MPIWRAPPER_PREFIX" \
    -DCMAKE_INSTALL_LIBDIR=lib64 \
    -DCMAKE_MODULE_LINKER_FLAGS="-Wl,--as-needed" \
    -DMPI_HOME="$LORRAX_IMPI_ROOT"

echo "[build_mpiw] compiling..."
"$CMAKE" --build "$LORRAX_MPIWRAPPER_BUILD" --parallel
"$CMAKE" --install "$LORRAX_MPIWRAPPER_BUILD"

# --- 4. verify the artifact -------------------------------------------------
SO="$LORRAX_MPIWRAPPER_PREFIX/lib64/libmpiwrapper.so"
[ -f "$SO" ] || { echo "[build_mpiw] FAILED: no $SO" >&2; exit 1; }
echo "[build_mpiw] --- artifact ---"; ls -lh "$SO"; sha256sum "$SO"

# Verify the OVERRIDES ARE IN THE MACHINE CODE, not merely in the source.
#
# The two overrides are file-static functions that gcc inlines into the
# generated MPIABI entry points, so they leave no callable symbol of their
# own — checking `nm` for them reports MISSING on a perfectly good build.
# Disassembling the two entry points the trampoline actually binds is both
# the strongest and the most stable check.
#
# NOTE on the pipeline style below: every `<tool> | grep -q` under
# `set -o pipefail` is a TRAP — grep exits at the first match, the producer
# dies of SIGPIPE, and pipefail turns 141 into a script failure.  This script
# was written with that bug and it reported a good binary as broken.  Capture
# each tool's output ONCE into a variable and match against that.
# (Same lesson recorded in build_ffi_host.sh's symbol-check block.)
echo "[build_mpiw] --- MPIABI entry points (dynamic) ---"
DYNSYMS="$(nm -D "$SO")"
for s in MPIABI_Init_thread MPIABI_Init MPIABI_Is_thread_main MPIABI_Comm_split; do
    if grep -qE " (T|B|D) $s\$" <<< "$DYNSYMS"; then echo "  OK      $s"
    else echo "  MISSING $s"; exit 1; fi
done

echo "[build_mpiw] --- override (1): THREAD_MULTIPLE upgrade (always on) ---"
# The `required` argument of PMPI_Init_thread is the 3rd (SysV AMD64: %edx).
# The patched entry point hard-sets it to 3 == MPI_THREAD_MULTIPLE and tail-
# jumps.  An UNPATCHED wrapper forwards the caller's value instead, and XLA
# asks for MPI_THREAD_FUNNELED (1) — the ~29% AS.4b crash regime.
DIS_INIT="$(objdump -d --disassemble=MPIABI_Init_thread "$SO" 2>/dev/null || true)"
if grep -qE 'mov +\$0x3,%edx' <<< "$DIS_INIT"; then
    echo "  OK      MPIABI_Init_thread forces required=3 (MPI_THREAD_MULTIPLE)"
else
    echo "[build_mpiw] FAILED: MPIABI_Init_thread does NOT force" >&2
    echo "             MPI_THREAD_MULTIPLE.  This wrapper grants whatever XLA" >&2
    echo "             asks for (FUNNELED) and MUST NOT be used multi-node." >&2
    echo "$DIS_INIT" >&2
    exit 1
fi

echo "[build_mpiw] --- override (2): LORRAX_MPI_FORCE_THREAD_MAIN gate ---"
DIS_ITM="$(objdump -d --disassemble=MPIABI_Is_thread_main "$SO" 2>/dev/null || true)"
STRS="$(strings "$SO")"
OK2=1
grep -qE 'callq? +[0-9a-f]+ <getenv' <<< "$DIS_ITM" || OK2=0
grep -qE 'callq?|jmpq? +[0-9a-f]+ <PMPI_Is_thread_main' <<< "$DIS_ITM" || OK2=0
grep -qx 'LORRAX_MPI_FORCE_THREAD_MAIN' <<< "$STRS" || OK2=0
if [ "$OK2" -eq 1 ]; then
    echo "  OK      MPIABI_Is_thread_main reads LORRAX_MPI_FORCE_THREAD_MAIN"
    echo "  OK      ... and still falls through to PMPI_Is_thread_main when unset"
else
    echo "[build_mpiw] FAILED: the MPI_Is_thread_main override is not in the" >&2
    echo "             machine code (expected a getenv call, the gate name in" >&2
    echo "             .rodata, and a PMPI_Is_thread_main fallthrough)." >&2
    echo "$DIS_ITM" >&2
    exit 1
fi

# Optional reproduction check against a known-good binary.  The build CWD is
# embedded in .note.gnu.build-id and (on TACC) .note.xalt.info, so whole-file
# equality is not achievable; .text equality is, and is what matters.
if [ -n "${LORRAX_MPIWRAPPER_REFERENCE_SO:-}" ] && \
   [ -f "$LORRAX_MPIWRAPPER_REFERENCE_SO" ]; then
    TA=$(mktemp); TB=$(mktemp)
    objcopy -O binary --only-section=.text "$SO" "$TA"
    objcopy -O binary --only-section=.text "$LORRAX_MPIWRAPPER_REFERENCE_SO" "$TB"
    if cmp -s "$TA" "$TB"; then
        echo "[build_mpiw] .text is BIT-IDENTICAL to the reference build."
    else
        echo "[build_mpiw] WARNING: .text differs from" >&2
        echo "             $LORRAX_MPIWRAPPER_REFERENCE_SO" >&2
    fi
    rm -f "$TA" "$TB"
fi

echo "[build_mpiw] --- libraries this .so will load at run time ---"
readelf -d "$SO" | grep NEEDED || true
if readelf -d "$SO" | grep NEEDED | grep -q libgfortran; then
    echo "[build_mpiw] WARNING: this .so needs libgfortran at run time.  The" >&2
    echo "             py312 container has none, so loading it will fail" >&2
    echo "             at runtime.  Rebuild with --as-needed." >&2
fi

cat <<EOF
[build_mpiw] done.

  MPITRAMPOLINE_LIB=$SO

Always-on:  MPI_Init/MPI_Init_thread grant MPI_THREAD_MULTIPLE.
Opt-in:     LORRAX_MPI_FORCE_THREAD_MAIN=1  -> MPI_Is_thread_main reports true,
            which is what lets XLA:CPU create grouped MPI communicators from
            its intra-op pool workers.  Required for BSE and for any program
            whose grouped collectives are first issued inside a jitted region.
            Unset  -> byte-for-byte the certified always-on-only behaviour.

Launch recipe: docs/dev/mpi_collectives.md
EOF
