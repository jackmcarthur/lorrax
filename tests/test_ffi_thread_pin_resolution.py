"""FFI thread-pin symbol resolution: what the RTLD_DEFAULT/RTLD_NEXT drift costs.

Gate for the 2026-07-30 FFI C++ divergence audit.  Three families copied the
MKL thread-pin block and the copies drifted on the resolver line:

    scalapack/cpp/blacs_grid.h      bare dlsym(RTLD_DEFAULT, ...)
    mklfft/cpp/fft_flat_k_ffi.cc    bare dlsym(RTLD_DEFAULT, ...)
    mklblas/cpp/gemm_batch_ffi.cc   dlsym(RTLD_DEFAULT) then dlsym(RTLD_NEXT)

They really were three independent resolutions, not one shared inline — each
is a function-local static, invisible to `strings` and visible to `nm -C`:

    $ nm -C lorrax_ffi_unified/build_host_C64/liblorrax_ffi_host.so \\
        | grep 'mkl_set_num_threads_local_ptr()::fn'
    b lorrax_ffi::mklfft::mkl_set_num_threads_local_ptr()::fn
    b lorrax_ffi::mklblas::mkl_set_num_threads_local_ptr()::fn
    u lorrax_ffi::scalapack::mkl_set_num_threads_local_ptr()::fn

WHAT THIS TEST EXISTS TO PIN DOWN.  The audit brief predicted a behavioural
consequence: "under a local-scope dlopen, GEMM would pin MKL threads and FFT
would silently not".  The first version of this test asserted exactly that
and FAILED — the bare form resolved the symbol under RTLD_LOCAL too.  The
prediction rests on the man page's summary of RTLD_DEFAULT ("the process's
global symbol scope"), which describes a caller in the MAIN EXECUTABLE.
glibc's `_dl_sym` resolves RTLD_DEFAULT against the CALLER's link-map scope,
and for a dlopen'd object that scope includes its own DT_NEEDED closure —
where libmkl_intel_lp64.so actually lives.  The two resolvers are therefore
EQUIVALENT for this symbol in every configuration reachable here, and the
extraction into common/cpp/mkl_thread_pin.h is a maintenance fix, not a
correctness fix.

So the tests below assert equivalence, across the full six-way matrix, and
the file's job is to make it loud if that ever stops being true (a different
libc, a musl container, a dlmopen namespace).  That is a real question with
a real answer, unlike the fiction the first draft encoded.

NOT-VOID CHECKS.  Two assertions here are built to fail rather than to pass:
`test_absent_symbol_resolves_to_null` pins the miss path (if it went green
by resolving something, every "found it" result would be meaningless), and
`test_dt_needed_carries_the_vendor_lib` proves the probe .so reproduces the
real library's shape instead of testing a symbol that is simply absent.

Everything is built with plain g++ against the real repo header — no MKL, no
XLA, no MPI, no container — so this runs on a login node.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HEADER = REPO / "src" / "ffi" / "common" / "cpp" / "mkl_thread_pin.h"

# A stand-in for libmkl_intel_lp64.so.  Only the symbol NAME matters to
# dlsym; the body records that it was actually called, so the pin RAII can be
# checked end to end rather than just "the pointer was non-null".
FAKE_MKL_SRC = r"""
extern "C" {
static int g_local = 0;
int MKL_Set_Num_Threads_Local(int n) { int prev = g_local; g_local = n; return prev; }
int MKL_Get_Max_Threads(void) { return 28; }
int fake_mkl_current(void) { return g_local; }
}
"""

# probe_dep: stands in for liblorrax_ffi_host.so -- libfakemkl is in its
# DT_NEEDED, exactly as libmkl_intel_lp64 is in the real library's.
# probe_nodep: same code, NO dependency, so the vendor lib has to be found
# through whatever scope the driver put it in.
# probe_bare()   is the pre-audit mklfft / blacs_grid form.
# probe_shared() calls the resolver in the real repo header under test.
PROBE_SRC = r"""
#include <dlfcn.h>
#include "mkl_thread_pin.h"

extern "C" {

int probe_bare(void) {
    return dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads_Local") != nullptr;
}

int probe_shared(void) {
    return lorrax_ffi::mklpin::resolve_sym("MKL_Set_Num_Threads_Local") != nullptr;
}

// Miss path.  If this ever answers 1 the resolver is finding something it
// should not and every "found it" result in this file is worthless.
int probe_absent(void) {
    return lorrax_ffi::mklpin::resolve_sym(
        "LORRAX_definitely_not_a_real_symbol_9f3a") != nullptr;
}

int probe_log_here(const char* env) {
    return lorrax_ffi::mklpin::log_here(env) ? 1 : 0;
}
}
"""

# Only linked into probe_dep, which has the vendor lib in DT_NEEDED.
PIN_SRC = r"""
#include "mkl_thread_pin.h"
extern "C" int MKL_Set_Num_Threads_Local(int);
extern "C" int fake_mkl_current(void);
// Returns inside*100 + after, so one int carries both observations.
extern "C" int probe_pin_roundtrip(void) {
    MKL_Set_Num_Threads_Local(28);          // ambient "global" setting
    int inside = -1;
    {
        lorrax_ffi::mklpin::MklThreadScope s(4);
        inside = fake_mkl_current();
    }
    return inside * 100 + fake_mkl_current();
}
"""

# Driver: dlopen the vendor lib (optionally) and then the probe, each with an
# explicit scope, and call one entry point.  Run as a SUBPROCESS per matrix
# cell -- an RTLD_GLOBAL dlopen is irreversible within a process, so doing
# this in-process would let earlier cells contaminate later ones.
DRIVER_SRC = r"""
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
// argv: <vendor.so|-> <G|L> <probe.so> <G|L> <fn> [str-arg]
int main(int argc, char** argv) {
    if (strcmp(argv[1], "-") != 0) {
        int vf = RTLD_NOW | (strcmp(argv[2], "G") == 0 ? RTLD_GLOBAL : RTLD_LOCAL);
        if (!dlopen(argv[1], vf)) { fprintf(stderr, "vendor: %s\n", dlerror()); return 2; }
    }
    int pf = RTLD_NOW | (strcmp(argv[4], "G") == 0 ? RTLD_GLOBAL : RTLD_LOCAL);
    void* h = dlopen(argv[3], pf);
    if (!h) { fprintf(stderr, "probe: %s\n", dlerror()); return 2; }
    void* f = dlsym(h, argv[5]);
    if (!f) { fprintf(stderr, "sym %s: %s\n", argv[5], dlerror()); return 3; }
    int rc = (argc > 6) ? ((int(*)(const char*))f)(argv[6]) : ((int(*)(void))f)();
    printf("%d\n", rc);
    return 0;
}
"""


def _cxx() -> str:
    return os.environ.get("CXX", "g++")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build the fake vendor lib, both probe .so variants and the driver."""
    if not HEADER.is_file():
        pytest.fail(f"header under test is missing: {HEADER}")
    try:
        subprocess.run([_cxx(), "--version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        pytest.skip(f"no C++ compiler ({_cxx()}) available")

    b = tmp_path_factory.mktemp("thread_pin_probe")
    (b / "fakemkl.cc").write_text(FAKE_MKL_SRC)
    (b / "probe.cc").write_text(PROBE_SRC)
    (b / "pin.cc").write_text(PIN_SRC)
    (b / "driver.c").write_text(DRIVER_SRC)

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:  # pragma: no cover
            pytest.fail(f"build failed: {' '.join(map(str, cmd))}\n{r.stderr}")

    inc = f"-I{HEADER.parent}"
    run([_cxx(), "-std=c++17", "-fPIC", "-shared", "-O0",
         "-o", b / "libfakemkl.so", b / "fakemkl.cc"])
    # WITHOUT the vendor lib in DT_NEEDED.
    run([_cxx(), "-std=c++17", "-fPIC", "-shared", "-O0", inc,
         "-o", b / "libprobe_nodep.so", b / "probe.cc", "-ldl"])
    # WITH it -- the real library's shape.  --no-as-needed keeps the
    # DT_NEEDED entry even though probe.cc alone would not require it.
    run([_cxx(), "-std=c++17", "-fPIC", "-shared", "-O0", inc,
         "-o", b / "libprobe_dep.so", b / "probe.cc", b / "pin.cc",
         "-L", str(b), "-Wl,--no-as-needed", "-lfakemkl",
         f"-Wl,-rpath,{b}", "-ldl"])
    run(["gcc", "-o", b / "driver", b / "driver.c", "-ldl"])
    return b


def call(built, *, probe: str, probe_scope: str, fn: str,
         vendor_scope: str | None = None, arg: str | None = None,
         env: dict | None = None) -> int:
    vendor = str(built / "libfakemkl.so") if vendor_scope else "-"
    cmd = [str(built / "driver"), vendor, vendor_scope or "L",
           str(built / f"libprobe_{probe}.so"), probe_scope, fn]
    if arg is not None:
        cmd.append(arg)
    e = dict(os.environ)
    e.pop("LORRAX_TEST_LOG_GATE", None)
    if env:
        e.update(env)
    r = subprocess.run(cmd, capture_output=True, text=True, env=e)
    assert r.returncode == 0, f"driver failed rc={r.returncode}: {r.stderr}"
    return int(r.stdout.strip())


# --------------------------------------------------------------------------
#  Shape checks -- these make the matrix below mean something.
# --------------------------------------------------------------------------
def test_dt_needed_carries_the_vendor_lib(built):
    """probe_dep must reproduce the real .so's shape: vendor lib in DT_NEEDED.

    Without this, the "found it" cells would be testing a symbol that is
    reachable for some other reason, and the six-way matrix would be void.
    """
    out = subprocess.run(["readelf", "-d", str(built / "libprobe_dep.so")],
                         capture_output=True, text=True).stdout
    assert "libfakemkl.so" in out, f"probe_dep lacks the DT_NEEDED entry:\n{out}"


def test_nodep_probe_really_has_no_dependency(built):
    out = subprocess.run(["readelf", "-d", str(built / "libprobe_nodep.so")],
                         capture_output=True, text=True).stdout
    assert "libfakemkl.so" not in out, f"probe_nodep should have no dep:\n{out}"


def test_absent_symbol_resolves_to_null(built):
    """The miss path must miss.  A resolver that always answers "found" would
    make every other assertion in this file pass for the wrong reason."""
    assert call(built, probe="dep", probe_scope="G", fn="probe_absent") == 0


# --------------------------------------------------------------------------
#  THE MATRIX.  Both resolvers, six reachable configurations, same answer.
#  If a future libc/loader breaks the equivalence, these fail and the
#  mkl_thread_pin.h header note needs re-measuring -- that is the point.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "probe,vendor_scope,probe_scope,expect",
    [
        # Vendor in the probe's own DT_NEEDED -- the real library's shape.
        # RTLD_DEFAULT searches the CALLER's scope, which includes DT_NEEDED,
        # so the load mode of the probe itself is irrelevant.
        ("dep",   None, "L", 1),
        ("dep",   None, "G", 1),
        # Vendor dlopen'd separately into a LOCAL scope: unreachable to both.
        ("nodep", "L",  "L", 0),
        ("nodep", "L",  "G", 0),
        # Vendor dlopen'd separately into the GLOBAL scope: reachable to both.
        ("nodep", "G",  "L", 1),
        ("nodep", "G",  "G", 1),
    ],
)
def test_bare_and_shared_resolvers_agree(built, probe, vendor_scope,
                                         probe_scope, expect):
    bare = call(built, probe=probe, probe_scope=probe_scope,
                vendor_scope=vendor_scope, fn="probe_bare")
    shared = call(built, probe=probe, probe_scope=probe_scope,
                  vendor_scope=vendor_scope, fn="probe_shared")
    assert bare == expect, (
        f"bare dlsym(RTLD_DEFAULT) gave {bare}, expected {expect} "
        f"(probe={probe} vendor_scope={vendor_scope} probe_scope={probe_scope})"
    )
    assert shared == bare, (
        "The two resolvers DISAGREE on this platform: bare="
        f"{bare} shared={shared} (probe={probe} vendor_scope={vendor_scope} "
        f"probe_scope={probe_scope}).  The equivalence table in "
        "src/ffi/common/cpp/mkl_thread_pin.h was measured on gcc 8.3.0/glibc "
        "and is now wrong for this toolchain -- re-measure it before "
        "trusting either form."
    )


def test_pin_scope_pins_and_restores(built):
    """MklThreadScope must set the thread-local team and put the old value back."""
    got = call(built, probe="dep", probe_scope="G", fn="probe_pin_roundtrip")
    inside, after = divmod(got, 100)
    assert inside == 4, f"scope did not pin (inside={inside})"
    assert after == 28, f"scope did not restore (after={after})"


# --------------------------------------------------------------------------
#  Rank scoping of the opt-in C++ debug-log knobs.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,rank,expect",
    [
        (None, "0", 0),      # unset -> off everywhere
        (None, "7", 0),
        ("1", "0", 1),       # set -> on at rank 0
        ("1", "7", 0),       # set -> OFF elsewhere: this is the new rank guard
        ("all", "7", 1),     # explicit opt-out of the rank guard
        ("ALL", "7", 1),     # case-insensitive
        ("*", "7", 1),
        ("yes", "7", 0),     # any other value keeps the rank-0 default
    ],
)
def test_log_gate_rank_scoping(built, value, rank, expect):
    """LORRAX_{MKLFFT,CUFFT}_LOG: presence-tested, rank-0 by default, 'all' opts out.

    Before the audit these two families had NO rank guard, so an opt-in debug
    log was multiplied by the process count -- and at the mklfft
    descriptor-commit site by the OpenMP team size on top of that.
    """
    name = "LORRAX_TEST_LOG_GATE"
    env = {"SLURM_PROCID": rank}
    for stale in ("PMI_RANK", "OMPI_COMM_WORLD_RANK"):
        env[stale] = ""
    if value is not None:
        env[name] = value
    got = call(built, probe="dep", probe_scope="G", fn="probe_log_here",
               arg=name, env=env)
    assert got == expect
