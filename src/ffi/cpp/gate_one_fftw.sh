#!/usr/bin/env bash
# ===========================================================================
# Post-stage gate: EXACTLY ONE FFTW3 engine may be MAPPED, and it must be the
# one the container stages.
#
#   usage: gate_one_fftw.sh <so-file> [<so-file> ...]
#   env:   LORRAX_FFTW3_STAGE      the bind-mount stage tree (dir with lib/).
#                                  When set, the stage must provide exactly
#                                  one engine and it must be the one that
#                                  ends up bound.
#          LORRAX_GATE_FFTW_PY     python that can `import jax` and the lorrax
#                                  tree (default: python3).  The dynamic leg
#                                  needs both.
#          LORRAX_GATE_ONE_FFTW=off   announced opt-out (never silent).
#
# WHY THIS GATE EXISTS, AND WHY IT IS NOT GATE 5
# ----------------------------------------------
# GATE 5 (config/perlmutter/build_ffi_host.sh) enforces that the FFT engine
# is NOT a load-time dependency: zero `fftw` in DT_NEEDED.  That is a
# property of the ARTIFACT and it is checked at build time.  It says nothing
# about which engine — or how many — a running process ends up with, because
# after GATE 5 the engine arrives by `dlopen` at first use
# (`mklfft/fft_flat_k_ffi.cc`, the stage-1/stage-2/stage-3 ladder).
#
# So the two gates cover the two halves of run-time resolution:
#
#   GATE 5   nothing is bound at LOAD time      (readelf, build host)
#   GATE 8   exactly one thing is bound at RUN  (/proc/self/maps, container)
#
# WHAT MAKES THIS DIFFERENT FROM `ldd`
# ------------------------------------
# gate_one_hdf5.sh can use `ldd`, because HDF5 IS in DT_NEEDED and therefore
# in the closure.  FFTW3 deliberately is not, so `ldd` on these artifacts
# reports NOTHING about the engine — it is invisible to every static tool.
# The engine only becomes observable once a process has actually planned a
# transform.  Hence the dynamic leg drives one real flat-k FFT through the
# normal FFI path and then reads /proc/self/maps in that same process.
#
# WHY "IT RESOLVES" IS NOT THE PROPERTY (the lesson from GATE 7, CLAIMS 110)
# -------------------------------------------------------------------------
# A dynamic linker finds *a* definition, not the right one.  Two concrete
# ways this container can hand you a wrong-but-resolving engine:
#
#   (1) TWO FFTW3 BUILDS VISIBLE.  Stage one engine while another is reachable
#       (a second stage, a leftover LD_LIBRARY_PATH entry, an FFTW that came
#       in transitively) and both map into one process.  Each keeps its own
#       planner state and wisdom; the ladder's `break` on first success hides
#       which one won.  It resolves, so nothing complains.
#
#   (2) libcufftw.  MEASURED 2026-08-06 in `nvcr.io/nvidia/jax:25.04-py3`:
#       the image ships NO FFTW3, but it DOES ship
#       /usr/local/cuda/targets/x86_64-linux/lib/libcufftw.so.11 in the
#       ldconfig cache, and it exports fftw_plan_many_dft, fftw_execute_dft
#       AND fftw_destroy_plan — all three names the ladder binds.  It is a
#       GPU engine wearing the FFTW3 ABI.  Add it to the ladder, or set
#       LORRAX_FFTW3_SO to it, and the HOST handler's refusal turns into 33
#       green cells whose transforms ran somewhere nobody chose.  That is the
#       tempting repair here, exactly as "stage 1.14 beside 1.12" was there.
#
# So the invariant is not "the ladder found something".  It is ONE engine,
# it is MAPPED, and it is the STAGED one.
#
# WHAT THIS GATE RETURNS WHEN THE PROPERTY IS FALSE — stated so no caller has
# to guess, and so this is not another `nm -D | grep -c fftw_` (CLAIMS 88:
# that read 0 while the library was completely unloadable):
#   - any fftw in DT_NEEDED                    -> FAIL 8a, the lines printed
#   - stage dir named but absent/empty         -> CANNOT RUN 8b (not a pass)
#   - stage carrying two engines               -> FAIL 8b, both printed
#   - the dynamic leg cannot run at all        -> CANNOT RUN 8c (not a pass)
#   - zero mapped engines after a real FFT     -> FAIL 8c, ladder trace printed
#   - two or more mapped engines               -> FAIL 8d, every realpath
#   - the bound engine is not the staged one   -> FAIL 8e, got/want printed
# ===========================================================================
set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: gate_one_fftw.sh <so-file> [<so-file> ...]" >&2
    exit 2
fi

TAG="${GATE_TAG:-gate_one_fftw}"
STAGE="${LORRAX_FFTW3_STAGE:-}"
GPY="${LORRAX_GATE_FFTW_PY:-python3}"

# The entry point, as `nm -D` may spell it.  THE VERSION SUFFIX IS NOT
# OPTIONAL DECORATION.  Measured 2026-08-06: this gate's first falsification
# run reported "0 mapped objects define fftw_plan_many_dft" in a process
# whose FFT had just RUN on libcufftw, because CUDA's FFTW3 shim exports
# `fftw_plan_many_dft@@libcufftw.so.11` and a `$`-anchored match skips it.
# A gate that cannot see the one impostor it was written for is worse than
# no gate: it prints PASS.
SYM_RE='[[:space:]]fftw_plan_many_dft(@@?[^[:space:]]+)?$'

if [[ "${LORRAX_GATE_ONE_FFTW:-on}" == "off" ]]; then
    echo "[$TAG] GATE DISABLED by LORRAX_GATE_ONE_FFTW=off — the one-FFTW3" >&2
    echo "[$TAG] invariant is NOT checked for this artifact.  An unchecked" >&2
    echo "[$TAG] .so must not be certified." >&2
    exit 0
fi

for so in "$@"; do
    if [[ ! -f "$so" ]]; then
        echo "[$TAG] GATE FAILED: no such artifact: $so" >&2
        exit 1
    fi
done

# --- 8a: nothing is bound at LOAD time -------------------------------------
# Restated here rather than deferred to GATE 5, because GATE 5 runs on the
# host build only and this gate is handed BOTH legs.  A device .so that
# link-binds an FFTW3 would be just as unloadable in the container.
n_needed=0
for so in "$@"; do
    hits="$(readelf -d "$so" 2>/dev/null | grep NEEDED | grep 'fftw' || true)"
    if [[ -n "$hits" ]]; then
        echo "[$TAG] GATE FAILED (8a): $so has fftw in DT_NEEDED:" >&2
        printf '%s\n' "$hits" | sed "s/^/[$TAG]   /" >&2
        n_needed=$((n_needed + 1))
    fi
done
if [[ "$n_needed" -ne 0 ]]; then
    echo "[$TAG]   A LOAD-time dependency on a version- and MPI-flavour-" >&2
    echo "[$TAG]   stamped SONAME makes the WHOLE library fail to dlopen" >&2
    echo "[$TAG]   wherever that exact string is absent — which is every" >&2
    echo "[$TAG]   container, since the image has no /opt/cray/pe.  Resolve" >&2
    echo "[$TAG]   the engine at RUN time (see GATE 5 in" >&2
    echo "[$TAG]   config/perlmutter/build_ffi_host.sh)." >&2
    exit 1
fi
echo "[$TAG] 8a: fftw entries in DT_NEEDED across $# artifact(s): 0"

# --- 8b: what the STAGE provides -------------------------------------------
STAGE_REAL=""
if [[ -n "$STAGE" ]]; then
    if [[ ! -d "$STAGE/lib" ]]; then
        echo "[$TAG] GATE CANNOT RUN (8b): LORRAX_FFTW3_STAGE=$STAGE has no" >&2
        echo "[$TAG]   lib/ directory, so what the container will provide is" >&2
        echo "[$TAG]   unknown.  This is NOT a pass." >&2
        echo "[$TAG]   fix: run src/ffi/cpp/stage/fftw_stage_cray.sh, or name" >&2
        echo "[$TAG]        the tree with LORRAX_FFI_FFTW_DIR." >&2
        exit 1
    fi
    # NOTE grep -c, NOT grep -q.  Under `set -o pipefail` a `grep -q` that
    # matches early closes the pipe, `nm` dies of SIGPIPE (141), and the
    # PIPELINE reports failure even though the symbol was found -- so the
    # object silently does not get counted.  Measured 2026-08-06: this gate's
    # first run reported "the stage provides 0 engines" against a stage that
    # provides exactly one.  Read all of the input.
    mapfile -t PROV < <(for f in "$STAGE"/lib/*; do
        [[ -f "$f" && ! -L "$f" ]] || continue
        hits=$(nm -D --defined-only "$f" 2>/dev/null |
               grep -cE "$SYM_RE" || true)
        [[ "${hits:-0}" -gt 0 ]] && readlink -f "$f"
    done | sort -u)
    echo "[$TAG] 8b: objects in $STAGE defining fftw_plan_many_dft: ${#PROV[@]}"
    for p in "${PROV[@]}"; do echo "[$TAG]   $p"; done
    if [[ ${#PROV[@]} -ne 1 ]]; then
        echo "[$TAG] GATE FAILED (8b): the stage provides ${#PROV[@]} engines." >&2
        echo "[$TAG]   A stage that carries two lets the ladder bind either" >&2
        echo "[$TAG]   one, and both can end up mapped.  It resolves, so" >&2
        echo "[$TAG]   nothing complains.  Re-stage from ONE module:" >&2
        echo "[$TAG]     rm -rf $STAGE" >&2
        echo "[$TAG]     module load PrgEnv-gnu cray-fftw" >&2
        echo "[$TAG]     LORRAX_FFI_FFTW_DIR=$STAGE \\" >&2
        echo "[$TAG]       src/ffi/cpp/stage/fftw_stage_cray.sh" >&2
        exit 1
    fi
    STAGE_REAL="${PROV[0]}"
fi

# --- 8c/8d/8e: what actually gets MAPPED, in one process, after a real FFT --
#
# The engine is bound by a function-local static inside the handler, so it
# does not exist until a transform is planned.  Nothing short of running one
# observes the real state; a script that re-implements the ladder observes
# its own behaviour instead.  So: drive make_flat_k_fft_ffi exactly as
# tests/test_fft_flat_k_numerics.py does, then read /proc/self/maps.
GATE_STAGE_REAL="$STAGE_REAL" "$GPY" - "$@" <<'PY'
import ctypes, os, subprocess, sys

TAG = os.environ.get("GATE_TAG", "gate_one_fftw")
STAGE_REAL = os.environ.get("GATE_STAGE_REAL", "")


def say(msg):
    print("[%s] %s" % (TAG, msg))


def die(msg_lines, rc=1):
    for m in msg_lines:
        sys.stderr.write("[%s] %s\n" % (TAG, m))
    sys.exit(rc)


# Load the artifacts into THIS process, globally, exactly as the runtime
# does.  RTLD_GLOBAL matters: it is what lets the handler's stage-1 resolver
# see anything the ladder brings in.
for p in sys.argv[1:]:
    try:
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        say("8c: dlopen OK   %s" % p)
    except OSError as e:
        die(["GATE FAILED (8c): %s will not dlopen here:" % p,
             "  %s" % e,
             "  The one-engine invariant cannot be checked against a",
             "  library that does not load.  Fix the closure first",
             "  (gate_one_hdf5.sh / gate_one_mpi.sh)."])

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P
    from ffi.fft import make_flat_k_fft_ffi
except Exception as e:                                    # noqa: BLE001
    die(["GATE CANNOT RUN (8c): the dynamic leg needs jax and the lorrax",
         "  tree importable, and they are not:",
         "  %s: %s" % (type(e).__name__, e),
         "  This is NOT a pass — the engine is invisible to every static",
         "  tool, so a run that could not plan a transform has measured",
         "  nothing.  Run this gate in-container with PYTHONPATH set, or",
         "  state the risk with LORRAX_GATE_ONE_FFTW=off."])

# One real transform through the normal path.  Tiny (4x4x4, trail 1): this
# gate is about WHICH engine, never about the numbers — those are
# tests/test_fft_flat_k_numerics.py's job.
try:
    mesh = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    fn = make_flat_k_fft_ffi(mesh, (4, 4, 4), P(None, None, None, None),
                             kind="fftn", norm=None, out_spec=None)
    x = np.zeros((64, 1), dtype=np.complex128)
    np.asarray(jax.jit(fn)(jnp.asarray(x)))
    say("8c: one flat-k FFT planned and executed — the engine is now bound")
except Exception as e:                                    # noqa: BLE001
    die(["GATE FAILED (8c): the flat-k FFT handler did not run:",
         "  %s: %s" % (type(e).__name__, str(e).replace("\n", "\n[%s]   " % TAG)),
         "  If this says 'no FFTW3 engine in this process', the ladder",
         "  found nothing: no engine is staged where the container can",
         "  reach it.  Zero mapped engines is a FAIL, not a skip."])

# Everything is loaded.  Now ask the kernel what is mapped, and ask each
# mapped file whether it DEFINES the entry point — a property, not a name
# pattern, so libcufftw and libmkl_rt are caught without being listed.
paths = set()
with open("/proc/self/maps") as fh:
    for line in fh:
        parts = line.rstrip("\n").split(None, 5)
        if len(parts) == 6 and parts[5].startswith("/"):
            paths.add(parts[5])

providers = set()
for p in sorted(paths):
    try:
        out = subprocess.run(["nm", "-D", "--defined-only", p],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        continue
    for ln in out.stdout.splitlines():
        # Versioned spellings count: libcufftw exports
        # `fftw_plan_many_dft@@libcufftw.so.11`, and it is precisely the
        # object this gate must not miss.
        sym = ln.rsplit(None, 1)[-1] if ln.split() else ""
        if sym == "fftw_plan_many_dft" or sym.startswith("fftw_plan_many_dft@"):
            providers.add(os.path.realpath(p))
            break

say("8d: distinct MAPPED objects defining fftw_plan_many_dft: %d"
    % len(providers))
for p in sorted(providers):
    say("      %s" % p)


class _DlInfo(ctypes.Structure):
    _fields_ = [("dli_fname", ctypes.c_char_p),
                ("dli_fbase", ctypes.c_void_p),
                ("dli_sname", ctypes.c_char_p),
                ("dli_saddr", ctypes.c_void_p)]


def _who_answers():
    """The object the dynamic linker ACTUALLY hands the handler.

    The /proc/self/maps scan above counts candidates; this names the winner,
    and it does so through the same resolver the handler uses.  Two
    independent observations of the same fact -- if they disagree, neither
    is trustworthy and the gate says so rather than picking one.
    """
    libdl = ctypes.CDLL("libdl.so.2")
    libdl.dlsym.restype = ctypes.c_void_p
    libdl.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    addr = libdl.dlsym(ctypes.c_void_p(0), b"fftw_plan_many_dft")
    if not addr:
        return None
    info = _DlInfo()
    if not libdl.dladdr(ctypes.c_void_p(addr), ctypes.byref(info)):
        return None
    return os.path.realpath(info.dli_fname.decode()) if info.dli_fname else None


answered = _who_answers()
say("8d: the object that ANSWERED fftw_plan_many_dft (dladdr): %s"
    % (answered or "<unresolvable>"))
if answered and answered not in providers:
    providers.add(answered)
    say("8d: ...which the /proc/self/maps scan had NOT counted; adding it.")

if len(providers) == 0:
    die(["GATE FAILED (8d): the FFT ran but nothing mapped defines",
         "  fftw_plan_many_dft.  Either `nm` is unavailable here (then this",
         "  gate cannot run and must not pass) or the process is in a state",
         "  nobody predicted.  Do not certify."])
if len(providers) != 1:
    die(["GATE FAILED (8d): %d distinct FFTW3 engines are mapped into one"
         % len(providers),
         "  process.  The ladder binds whichever it reached first and the",
         "  other sits there with its own planner state.  It RESOLVES,",
         "  which is why nothing complained.  Remove the second engine",
         "  from the container's reach — do not pick a winner with",
         "  LORRAX_FFTW3_SO."] + ["  %s" % p for p in sorted(providers)])

bound = answered or sorted(providers)[0]

# 8e: the mapped engine must be the STAGED engine.  This is the leg that
# catches libcufftw: exactly one object would be mapped, the count would
# pass, and the host handler would be running on the GPU.
if STAGE_REAL:
    if os.path.realpath(bound) != os.path.realpath(STAGE_REAL):
        die(["GATE FAILED (8e): the mapped engine is not the staged one.",
             "  got    %s" % bound,
             "  wanted %s" % STAGE_REAL,
             "  A different object answering to the FFTW3 ABI is not the",
             "  same engine.  libcufftw.so.11 ships in this image and",
             "  exports all three entry points; binding it would run the",
             "  HOST handler's transforms on the GPU with nobody having",
             "  chosen that."])
    say("8e: the mapped engine IS the staged one")
else:
    say("8e: no LORRAX_FFTW3_STAGE named — engine identity NOT checked")
    say("      (count is 1, provenance is unverified; name the stage to")
    say("       make this leg meaningful)")

say("GATE 8 (one FFTW3 engine, mapped, and it is the staged one) PASSED")
say("  engine: %s" % bound)
PY
rc=$?
[[ $rc -ne 0 ]] && exit $rc
exit 0
