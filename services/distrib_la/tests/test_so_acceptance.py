"""The C++ tier: the build acceptance checks, as pytest cells.

Checks 1-4 are BUILD_NOTES.md's four post-build commands.  Check 5 ratchets
the ``libslate``/``libblaspp`` SONAME collision the dlopen-order rule was
written for.  Check 6 and its two companions (2026-08-08) are the
artifact-level ratchet on KNOWN_FAILURES **L1**: the two platform libraries
are dlopened RTLD_GLOBAL into one process, so no name LORRAX'S OWN SOURCE
defines may be defined by both.  Twenty-five were, before the fix.


There are NO C++-level tests anywhere in LORRAX (survey 1, S8: ``find
src/ffi/cpp -iname '*test*'`` is empty).  The nearest thing is a block of
shell in ``config/perlmutter/build_ffi_host.sh``'s seven gates plus the
four post-build commands ``BUILD_NOTES.md`` writes out, which are run by
hand after a rebuild and by nobody afterwards.  This file makes them a
suite: the same assertions, against the .so a run is actually PINNED to,
so a defective library is caught by the test suite rather than by
nineteen contract cells failing in a way that needs a person to diagnose.

THE DEFECT THIS EXISTS FOR IS REAL AND RECENT.  The deployed Aug-7 host
library was built with the generic ``src/ffi/cpp/build_host.sh`` instead
of the Perlmutter script, so it carries ``scalapack=0`` and exports ZERO
``Scalapack*HostFfi`` symbols (survey 2's headline).  CMake printed
"ScaLAPACK/BLACS not found" as a WARNING and the build succeeded.  Every
check below would have caught it at the artifact level in milliseconds.

ARTIFACT-LEVEL, and that is the point.  These cells never dlopen anything:
they read the ELF.  A load-time probe answers a different and
environment-dependent question (survey 2: ``$LORRAX_SHIFTER`` over ssh does
not apply ``--volume``/``--env``, which voided every in-container load
observation that survey made and left only nm/readelf/strings standing).
The contract suite covers the load question, from inside a launch where
the answer means something.

ENV-PARAMETERIZED: ``LORRAX_FFI_HOST_SO`` / ``LORRAX_FFI_SO``, the SAME
pins ``distrib_la.loader`` reads, so this tier checks the library the run
would use rather than one it went looking for.  With no pin the cells skip
naming the variable — and that skip is on the machine profile's allowlist
OFF Perlmutter and REMOVED from it ON Perlmutter (conftest ``_allowed_for``),
because BUILD_NOTES.md pins the host lib there and a skip would mean the
tier evaporated on the only machine that has a library to check.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

from distrib_la.loader import _HOST_TARGET_SYMBOLS, _PLATFORMS

# The seven cells below are ENVIRONMENT-GATED, not expected-to-fail logic:
# they pass against the fresh matmul FFI builds the GEMM lane produced and
# fail against every older pin, including the campaign's.  Named skips, not
# xfail — see claim 209 for the measured before/after sets.
_NEEDS_GEMM_LANE_SO = (
    "requires a rebuilt host .so containing the ScaLAPACK/PBLAS GEMM "
    "handler — unskip when that artifact is pinned")


#: The ScaLAPACK host handlers.  BUILD_NOTES.md's check 2 counts
#: them; naming them means a build that exported three and a typo passes
#: the count and fails here.
_SCALAPACK_SYMBOLS = tuple(sorted(
    sym for sym in _HOST_TARGET_SYMBOLS.values() if "Scalapack" in sym))

_SCALAPACK_GEMM_TARGET = "lorrax_scalapack_batched_gemm"
_SCALAPACK_GEMM_EXPORT = "ScalapackBatchedGemmHostFfi"
_SCALAPACK_API_SYMBOLS = (
    "pzheevd_", "pdsyevd_", "pzgetrf_", "pdgetrf_", "pzgetrs_",
    "pdgetrs_", "pzgemm_", "pdgemm_", "numroc_", "Cblacs_gridinfo",
    "descinit_", "Csys2blacs_handle", "Cblacs_gridinit",
)

# ---------------------------------------------------------------------------
# WHAT LORRAX ITSELF DEFINES — the set that must not be shared.
# ---------------------------------------------------------------------------
# ``src/ffi/cpp/exports_{cuda,host}.map`` localise everything LORRAX itself
# compiles, and ``cpp/common/c_abi.h`` gives the host leg's ctypes entry
# points a ``_host`` suffix, so the set below is what the two libraries are
# allowed to still put in ``.dynsym`` — and by construction the two legs'
# versions of it are DISJOINT.
#
# The third-party names both files also define (``slate::``, ``blas::``,
# ``xla::ffi::``, libstdc++ internals — ~234 of them) are deliberately NOT
# localised.  They are ODR-CORRECT duplicates: same headers, same source,
# weak COMDAT copies the C++ ABI intends the dynamic linker to merge.
# Localising them was measured, on 2026-08-08, to segfault five SLATE host
# cells (exports_cuda.map carries the five-arm table).  Their sharing is the
# SAME defect as ``libslate.so.2`` / ``libblaspp.so.2`` resolving out of two
# different builds under one SONAME — check 5's territory, and what
# ``_open_cuda_before_host`` exists for.  Not L1, and not claimed here.


def _is_lorrax_owned(sym: str) -> bool:
    """Did LORRAX's own source define this name?

    Three families, and they are all of them:
      * anything mentioning ``lorrax`` — the mangled ``namespace lorrax_ffi``
        internals, ``std::`` templates instantiated on our types, our
        function-local statics and their guard variables, the extern "C"
        CUDA kernel launchers, and the host build-config stamp;
      * the ``lrx_*`` ctypes lifecycle ABI;
      * the ``*Ffi`` XLA handler registration symbols.
    Everything else in either table came out of a third-party header.
    """
    return ("lorrax" in sym) or sym.startswith("lrx_") or sym.endswith("Ffi")


# The build-config stamp and the handler-signature ABI, per leg.  Both are
# `lorrax_ffi`-named, so both would be caught by the version scripts' denylist
# and by ``_is_lorrax_owned`` above; both are deliberately exported, and each
# leg's pair carries that leg's infix, so the two sanctioned surfaces stay
# disjoint and check 6's intersection stays empty.  See
# src/ffi/cpp/exports_{host,cuda}.map and cpp/common/lorrax_ffi_abi.h.
_STAMP_EXPORTS = {
    "CUDA": frozenset(
        {"lorrax_ffi_cuda_build_config", "lorrax_ffi_cuda_abi_version"}),
    "cpu": frozenset(
        {"lorrax_ffi_host_build_config", "lorrax_ffi_host_abi_version"}),
}


def _sanctioned_cuda(sym: str) -> bool:
    """liblorrax_ffi.so: bare ``*Ffi`` handlers, unsuffixed ``lrx_*``, the
    two ``_cuda`` stamp entry points."""
    if sym.endswith("Ffi"):
        return not sym.endswith("HostFfi")
    if sym.startswith("lrx_"):
        return not sym.endswith("_host")
    return sym in _STAMP_EXPORTS["CUDA"]


def _sanctioned_host(sym: str) -> bool:
    """liblorrax_ffi_host.so: ``*HostFfi``, ``lrx_*_host``, the two ``_host``
    stamp entry points."""
    if sym.endswith("Ffi"):
        return sym.endswith("HostFfi")
    if sym.startswith("lrx_"):
        return sym.endswith("_host")
    return sym in _STAMP_EXPORTS["cpu"]


_SANCTIONED_EXPORTS = {
    "CUDA": (
        _sanctioned_cuda,
        "the *Ffi XLA handler registration symbols, the lrx_* ctypes ABI, "
        "and the two lorrax_ffi_cuda_* stamp entry points",
    ),
    "cpu": (
        _sanctioned_host,
        "the *HostFfi XLA handler registration symbols, the lrx_*_host "
        "ctypes ABI, and the two lorrax_ffi_host_* stamp entry points",
    ),
}


def _pinned(platform: str) -> str:
    """The .so this run is pinned to, or SKIP naming the variable."""
    env = _PLATFORMS[platform]["env"]
    path = os.environ.get(env)
    if not path:
        pytest.skip(
            f"{env} is not set, so there is no pinned "
            f"{_PLATFORMS[platform]['so_name']} to accept.  This tier is "
            f"env-parameterized on purpose: it checks the library a RUN "
            f"would load, not one it went looking for.  BUILD_NOTES.md "
            f"pins it on Perlmutter; covered by the Perlmutter leg.")
    if not os.path.exists(path):
        pytest.fail(
            f"{env}={path!r} does not exist.  An explicit pin that cannot "
            f"be honored is a REFUSAL, never a fall-through — the same "
            f"rule distrib_la.loader._locate_so applies, and for the same "
            f"reason: a stale pin used to leave no trace while the run "
            f"silently used some other build.", pytrace=False)
    return path


def _tool(name: str) -> str:
    if shutil.which(name) is None:
        pytest.skip(f"{name} is not on PATH; the ELF acceptance checks are "
                    f"binutils-driven (that is what BUILD_NOTES.md's four "
                    f"commands are).  Covered by any leg with binutils.")
    return name


def _readelf_needed(so: str) -> list[str]:
    out = subprocess.run([_tool("readelf"), "-d", so], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, timeout=120)
    assert out.returncode == 0, f"readelf -d failed:\n{out.stdout[-2000:]}"
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out.stdout)


def _defined_symbols(so: str) -> set[str]:
    out = subprocess.run([_tool("nm"), "-D", "--defined-only", so],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, timeout=180)
    assert out.returncode == 0, f"nm -D failed:\n{out.stdout[-2000:]}"
    return {ln.split()[-1] for ln in out.stdout.splitlines() if ln.split()}


# ---------------------------------------------------------------------------
# BUILD_NOTES.md's four checks
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason=_NEEDS_GEMM_LANE_SO)
def test_check_1_the_host_library_was_built_with_scalapack():
    """``strings <so> | grep 'scalapack='`` must say ``scalapack=1``.

    Read out of the file's bytes rather than through ``strings``: the
    build stamps a literal ``scalapack=<0|1>`` into its build_config
    string, and searching for it exactly removes both a tool dependency
    and the chance of matching some other occurrence.
    """
    so = _pinned("cpu")
    with open(so, "rb") as fh:
        blob = fh.read()
    found = sorted({m.decode() for m in
                    re.findall(rb"scalapack=[01]", blob)})
    assert found, (
        f"{so} carries no scalapack= stamp at all, so its build config "
        f"cannot be read.  Built by a script that does not stamp one?")
    assert found == ["scalapack=1"], (
        f"{so} reports {found}.  This is the defect survey 2 found in the "
        f"deployed Aug-7 library: a build that used the generic "
        f"src/ffi/cpp/build_host.sh instead of "
        f"config/perlmutter/build_ffi_host.sh gets 'ScaLAPACK/BLACS not "
        f"found' as a CMake WARNING and succeeds with no ScaLAPACK at all.")


def test_check_2_all_scalapack_handlers_are_exported():
    """Every ScaLAPACK handler declared by the loader must be exported.

    The count AND the names.  A count alone passes a build that exported
    four symbols one of which was misspelled, and the names come from
    ``distrib_la.loader``'s own table — so this cell also pins that the
    table and the library have not drifted apart, which is the failure
    ``probe_target`` reports as "loaded but does not export".
    """
    so = _pinned("cpu")
    syms = _defined_symbols(so)
    missing = [s for s in _SCALAPACK_SYMBOLS if s not in syms]
    assert not missing, f"{so} does not export {missing}"
    grepped = sum(1 for s in syms if "Scalapack" in s)
    assert grepped == len(_SCALAPACK_SYMBOLS), (
        f"{so} exports {grepped} Scalapack* symbols, want exactly "
        f"{len(_SCALAPACK_SYMBOLS)} "
        f"({sorted(s for s in syms if 'Scalapack' in s)})")


def test_exact_scalapack_gemm_loader_target():
    """Pin the public target spelling to the exact C++ export."""
    assert _HOST_TARGET_SYMBOLS.get(
        _SCALAPACK_GEMM_TARGET) == _SCALAPACK_GEMM_EXPORT
    source_path = os.path.join(
        _repo_root(), "src", "ffi", "cpp", "scalapack", "gemm_ffi.cc")
    with open(source_path) as fh:
        source = fh.read()
    assert re.search(
        rf"XLA_FFI_DEFINE_HANDLER_SYMBOL\(\s*"
        rf"{re.escape(_SCALAPACK_GEMM_EXPORT)}\s*,", source)


def test_exact_scalapack_gemm_export_is_present():
    """Make every pinned host artifact carry that exact export."""
    so = _pinned("cpu")
    assert _SCALAPACK_GEMM_EXPORT in _defined_symbols(so), (
        f"{so} does not export {_SCALAPACK_GEMM_EXPORT} for "
        f"{_SCALAPACK_GEMM_TARGET}")


def test_check_3_no_fftw_in_the_dynamic_dependencies():
    """``readelf -d <so> | grep NEEDED | grep -c fftw`` must be 0 — GATE 5b.

    THE 2026-08-06 LOSS, at the artifact level.  The Aug-5 host library
    carried a DT_NEEDED on cray-fftw's ``libfftw3.so.mpi31.3``, which the
    Shifter container does not mount.  dlopen failed, and because both
    probes turned every negative answer into a skip, NINETEEN
    ScaLAPACK/SLATE/GEMM contract cells — none of which perform an FFT —
    were reported as "19 skipped" beside "0 failed".  The suite was green.
    """
    so = _pinned("cpu")
    needed = _readelf_needed(so)
    fftw = [n for n in needed if "fftw" in n.lower()]
    assert not fftw, (
        f"{so} has a DT_NEEDED on {fftw}.  GATE 5b: the host library must "
        f"not hard-link an FFTW the container does not mount — that is "
        f"the dependency whose dlopen failure was reported as 19 skips on "
        f"2026-08-06.  Full NEEDED list: {needed}")


@pytest.mark.skip(reason=_NEEDS_GEMM_LANE_SO)
def test_check_4_exactly_one_libsci_flavour():
    """``readelf -d <so> | grep NEEDED | grep -cE 'libsci_gnu(_mpi)?\\.so'``
    must be 0 — GATE 2.

    The deployed Aug-7 library links BOTH LibSci flavours (sequential and
    threaded).  The loader warns (``CRAYBLAS_WARNING``) and then ELF order
    decides which BLAS actually runs, which is a performance and
    reproducibility fork nobody chose.  GATE 2 catches it and never ran on
    that build.
    """
    so = _pinned("cpu")
    needed = _readelf_needed(so)
    bad = [n for n in needed if re.match(r"libsci_gnu(_mpi)?\.so", n)]
    assert not bad, (
        f"{so} has DT_NEEDED on {bad}.  GATE 2: linking more than one "
        f"LibSci flavour leaves ELF order deciding which BLAS runs.  Full "
        f"NEEDED list: {needed}")


# ---------------------------------------------------------------------------
# Two more the loader's own table earns for free
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason=_NEEDS_GEMM_LANE_SO)
def test_every_host_handler_this_package_claims_is_actually_exported():
    """The service's target table vs the library, in full.

    ``probe_target``'s third reason — "loaded but does not export" — is
    the genuine partial build, and this is the cell that would name which
    handler went missing before a single test tried to call it.
    """
    so = _pinned("cpu")
    syms = _defined_symbols(so)
    missing = sorted(f"{t} -> {s}" for t, s in _HOST_TARGET_SYMBOLS.items()
                     if s not in syms)
    assert not missing, (
        f"{so} is missing handlers distrib_la's host table claims: "
        f"{missing}")


@pytest.mark.skip(reason=_NEEDS_GEMM_LANE_SO)
def test_the_device_library_exports_what_the_cuda_table_claims():
    """Same check on the CUDA side, when a device .so is pinned.

    Survey 2 verified the deployed device library is CORRECT — every
    cuSOLVERMp and SLATE handler present — so this cell is expected green
    and exists so that stops being something a person has to remember.
    """
    so = _pinned("CUDA")
    syms = _defined_symbols(so)
    table = _PLATFORMS["CUDA"]["targets"]
    missing = sorted(f"{t} -> {s}" for t, s in table.items() if s not in syms)
    assert not missing, (
        f"{so} is missing handlers distrib_la's CUDA table claims: "
        f"{missing}")


def test_the_acceptance_checks_can_fail(tmp_path):
    """RED TWIN for the whole tier.

    Every cell above reads a real ELF, so on a machine with a GOOD library
    they all pass and none of them demonstrates it could do otherwise.
    Point the same readers at a file that is NOT that library and each
    must refuse.  Without this, a broken ``_readelf_needed`` that returned
    ``[]`` for everything would make checks 3 and 4 pass unconditionally.

    A scratch file rather than this module, which was the first draft and
    FAILED: this file quotes ``scalapack=1`` in its own prose, so check
    1's byte search found it and the twin asserted the opposite of what it
    meant.  A red twin that is wrong about its own control is worse than
    none.
    """
    not_a_lib = str(tmp_path / "not_a_library.bin")
    with open(not_a_lib, "wb") as fh:
        fh.write(b"this is not an ELF file\n" * 64)
    with open(not_a_lib, "rb") as fh:
        assert b"scalapack=1" not in fh.read()      # check 1's finding
    if shutil.which("readelf") and shutil.which("nm"):
        out = subprocess.run(["readelf", "-d", not_a_lib],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=60)
        needed = re.findall(r"Shared library: \[([^\]]+)\]", out.stdout)
        assert not needed, "a .py file reported DT_NEEDED entries"
        # ...and the symbol reader finds nothing to claim, so check 2's
        # assertion would fire rather than pass vacuously.
        out = subprocess.run(["nm", "-D", "--defined-only", not_a_lib],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=60)
        assert not {ln.split()[-1] for ln in out.stdout.splitlines()
                    if ln.split() and not ln.startswith("nm:")} & set(
                        _SCALAPACK_SYMBOLS)


def test_a_pin_that_does_not_exist_is_a_failure_not_a_skip(monkeypatch):
    """An explicit pin that cannot be honored REFUSES.

    The same rule ``distrib_la.loader._locate_so`` applies, and the reason
    is the one recorded there (2026-08-06): a stale or mistyped pin used
    to be appended to the candidate list, so the run silently used the
    in-tree build and the pin left no trace anywhere.
    """
    monkeypatch.setenv("LORRAX_FFI_HOST_SO", "/nonexistent/liblorrax.so")
    with pytest.raises(pytest.fail.Exception, match="does not exist"):
        _pinned("cpu")


# ---------------------------------------------------------------------------
# The SONAME collision, at the artifact level
# ---------------------------------------------------------------------------

def _shared_defined(so_a: str, so_b: str) -> list[str]:
    """Defined symbol names exported by BOTH ELF files, sorted.

    One function so check 6 and its red twin cannot use different readers —
    the whole failure mode being guarded against here is a check that
    reports "nothing shared" because the reader returned nothing.
    """
    return sorted(_defined_symbols(so_a) & _defined_symbols(so_b))


@pytest.mark.skip(reason=_NEEDS_GEMM_LANE_SO)
def test_check_6_the_two_libraries_share_no_defined_symbol_of_ours():
    """THE L1 RATCHET: nothing LORRAX defines is defined by both .so files.

    ``ffi_loader`` and ``distrib_la.loader`` both dlopen their .so
    ``RTLD_GLOBAL``, and a CUDA-capable process legitimately has BOTH open —
    CUDA handlers on the device, host phdf5 reads on the CPU backend.  ld.so
    answers a name from the FIRST object that defined it, for the whole
    process, INCLUDING for the second library's own internal calls.  So any
    name both files define is a name one library can be made to execute the
    other's code for.

    MEASURED on the pre-fix pair (deployed device .so + ``build_host_h200``,
    both at ``96a6399``): 259 shared defined names, **25 of them LORRAX's
    own** — the nine ``lrx_*`` entry points and sixteen mangled
    ``lorrax_ffi::phdf5::*``, among them ``open_ctx``, ``close_ctx``,
    ``ensure_dataset``, ``ensure_read_buf``, ``ensure_pinned`` and
    ``~PhdfCtx``.  And they were not the same functions: ``PhdfCtx`` carries
    the CUDA stream/event/pinned members under ``#ifndef
    LORRAX_FFI_NO_CUDA``, so ONE type name had TWO layouts and both
    libraries exported one ``open_ctx(...) -> PhdfCtx*``.  A handler handed
    the other build's ctx read its fields at the wrong offsets, which is
    what KNOWN_FAILURES B1's ``offset_base=[0,0,0,4596944070643295330]``
    (a float64 decoded as an int64) and the xdist arm's ``phdf5 read:
    ctx_handle is null`` both are.

    ZERO, not "few".  The sanctioned surfaces are disjoint BY CONSTRUCTION
    (``*Ffi`` vs ``*HostFfi``, ``lrx_*`` vs ``lrx_*_host``) and everything
    else of ours is localised by ``exports_{cuda,host}.map``.  One shared
    name means one of those two mechanisms stopped working.

    SECOND ASSERTION, and it is the sharper one: no shared name may have C
    LINKAGE.  A mangled name that both files define is a weak COMDAT copy of
    somebody's inline or template — the C++ ABI merges those on purpose, and
    that is what the ~234 remaining shared names are.  An UNMANGLED shared
    name is one a human deliberately exported, and there is no legitimate
    reason for two libraries in one process to both export one.

    THIS CELL SUPERSEDES check 5 AS THE STATEMENT OF THE PROBLEM.  Check 5
    ratchets the ``libslate``/``libblaspp`` SONAME collision, which is a
    DIFFERENT and still-live defect about which BUILD of a third-party
    library answers; this one is about which of OUR OWN definitions answers.
    Fixing the second did not fix the first, and the load-order rule still
    exists for the first.
    """
    shared = _shared_defined(_pinned("cpu"), _pinned("CUDA"))

    ours = [s for s in shared if _is_lorrax_owned(s)]
    assert not ours, (
        f"the two platform libraries both define {len(ours)} symbol(s) that "
        f"LORRAX's own source produced: {ours[:40]}"
        f"{' ...' if len(ours) > 40 else ''}.\n"
        f"Both are dlopened RTLD_GLOBAL, so the first one loaded answers "
        f"every one of these for BOTH — including for the other library's "
        f"internal calls.  That is KNOWN_FAILURES L1.  Usual causes, in "
        f"order of likelihood:\n"
        f"  1. one of the two .so files predates 2026-08-08 (a mixed pin: "
        f"an old host lib still exports the UNSUFFIXED lrx_* names, which "
        f"is exactly a 9-symbol intersection).  Rebuild BOTH.\n"
        f"  2. -Wl,--version-script=src/ffi/cpp/exports_*.map fell off a "
        f"link line — check CMakeCache.txt.\n"
        f"  3. a new exported name was added to one leg and, by the same "
        f"spelling, to the other.")

    c_linkage = [s for s in shared if not s.startswith("_Z")]
    assert not c_linkage, (
        f"the two platform libraries both define {len(c_linkage)} "
        f"C-LINKAGE symbol(s): {c_linkage[:40]}.  Mangled shared names are "
        f"weak COMDAT copies the C++ ABI merges on purpose; an unmangled "
        f"one was deliberately exported by somebody, and two libraries in "
        f"one RTLD_GLOBAL process must not both export the same one.")


@pytest.mark.skip(reason=_NEEDS_GEMM_LANE_SO)
def test_each_library_exports_only_its_sanctioned_surface():
    """No LORRAX internal leaks out of either .so — the other half of L1.

    Check 6 measures the INTERSECTION, which two libraries can keep empty
    while both still export their entire innards (they would just have to
    disagree about every name).  This cell measures what each one exports on
    its own.  Scope is LORRAX's own symbols, for the reason given at
    ``_is_lorrax_owned``: the third-party names are ODR-correct duplicates
    that the C++ ABI wants merged, and localising them was measured to
    segfault SLATE.

    The strongest single line of it is that NO MANGLED name of ours
    survives — a ``_ZN10lorrax_ffi...`` in either table means a
    ``namespace lorrax_ffi`` internal is back on the dynamic table, which is
    the exact shape of L1.
    """
    for platform in ("cpu", "CUDA"):
        so = _pinned(platform)
        sanctioned, described = _SANCTIONED_EXPORTS[platform]
        ours = {s for s in _defined_symbols(so) if _is_lorrax_owned(s)}
        stray = sorted(s for s in ours if not sanctioned(s))
        assert not stray, (
            f"{so} exports {len(stray)} LORRAX-owned symbol(s) outside its "
            f"sanctioned surface ({described}): {stray[:40]}"
            f"{' ...' if len(stray) > 40 else ''}.\n"
            f"src/ffi/cpp/exports_"
            f"{'host' if platform == 'cpu' else 'cuda'}.map localises "
            f"everything LORRAX defines; anything of ours still on the "
            f"dynamic table is a name the OTHER platform library can answer "
            f"for under RTLD_GLOBAL.")
        assert not [s for s in ours if s.startswith("_Z")], (
            f"{so} still exports mangled LORRAX C++ symbols — the "
            f"`namespace lorrax_ffi` internals are on the dynamic table "
            f"again, which is the shape of KNOWN_FAILURES L1.")


@pytest.mark.skip(reason=_NEEDS_GEMM_LANE_SO)
def test_the_shared_symbol_check_can_fail():
    """RED TWIN for check 6 and the surface check.

    Check 6 asserts an EMPTY intersection, which is what a broken
    ``_defined_symbols`` returning ``set()`` would also produce — the
    tautology this project post-mortems.  Three constructions, all using the
    real readers on real ELF files, no compiler and no synthetic data:

    1. INTERSECT THE HOST LIBRARY WITH ITSELF.  Every symbol it has must
       come back shared.  If ``nm -D --defined-only`` ever stopped returning
       names, this is the cell that says so.
    2. THE THIRD-PARTY OVERLAP IS STILL THERE.  Check 6 is scoped to
       LORRAX's own symbols, and that scope is only honest if the reader can
       see the names outside it — the two libraries DO still share ~234
       mangled third-party names, and a check-6 that passed because the
       intersection was empty for the wrong reason would show up here.
    3. THE TWO SANCTIONED SURFACES DO NOT OVERLAP, in both directions.  That
       disjointness (``lrx_*`` vs ``lrx_*_host``, ``*Ffi`` vs ``*HostFfi``)
       is what makes an empty intersection possible at all, so a predicate
       loose enough to accept both legs would silently void check 6.
    """
    host = _pinned("cpu")
    self_shared = _shared_defined(host, host)
    assert self_shared, (
        "the host library intersected with ITSELF came back empty — "
        "_defined_symbols is not reading the symbol table, so check 6's "
        "empty intersection means nothing.")
    assert len(self_shared) == len(_defined_symbols(host))

    shared = _shared_defined(host, _pinned("CUDA"))
    assert shared, (
        "the two libraries share NO defined symbol at all, not even a "
        "third-party template instantiation.  Either one of them was not "
        "read, or somebody localised the weak COMDAT copies the C++ ABI "
        "merges on purpose — which is measured (2026-08-08) to segfault "
        "the SLATE host handlers.  See src/ffi/cpp/exports_cuda.map.")
    assert all(s.startswith("_Z") for s in shared)

    host_syms = _defined_symbols(host)
    cuda_syms = _defined_symbols(_pinned("CUDA"))
    sanctioned_cuda, _ = _SANCTIONED_EXPORTS["CUDA"]
    accepted = sorted(s for s in host_syms if sanctioned_cuda(s))
    assert not accepted, (
        f"the CUDA sanctioned-surface predicate accepts host symbols "
        f"{accepted} — the two surfaces overlap, so 'each library exports "
        f"only its own surface' no longer implies they cannot collide.")
    sanctioned_host, _ = _SANCTIONED_EXPORTS["cpu"]
    accepted = sorted(s for s in cuda_syms if sanctioned_host(s))
    assert not accepted, (
        f"the host sanctioned-surface predicate accepts CUDA symbols "
        f"{accepted} — same overlap, other direction.")


def test_check_5_the_two_libraries_still_share_their_slate_soname():
    """A RATCHET on the reason ``_open_cuda_before_host`` exists.

    SCOPE, since 2026-08-08: this cell is about the SONAME of a THIRD-PARTY
    library, and nothing else.  The separate question — whether the two
    files share any of their OWN defined symbols — was a real defect
    (KNOWN_FAILURES L1), it is fixed, and check 6 above is its ratchet.
    Do not read this cell as covering it.

    Both platform libraries carry ``NEEDED libslate.so.2`` and ``NEEDED
    libblaspp.so.2``, resolved out of different builds (the host lib's
    DT_RPATH names a ``gpu_backend=none`` blaspp whose
    ``blas::get_device_count()`` is a compiled-in 0; the CUDA lib's
    DT_RUNPATH names the CUDA one).  ld.so keys a loaded object by SONAME,
    so the FIRST of the two dlopened decides for both, and only one order
    survives -- which is why both loaders open CUDA first.

    This cell fails the day somebody gives the two stacks distinct
    sonames, or statically links one of them, or drops SLATE from the host
    build.  That is the GOOD failure: it is the signal to delete
    ``_open_cuda_before_host`` from both loaders and the cells under 'THE
    SONAME RACE' in test_distrib_la_contract.py that guard it, rather than
    leave a workaround nobody can date.  Read the failure message before
    deleting anything.
    """
    shared = sorted(set(_readelf_needed(_pinned("cpu")))
                    & set(_readelf_needed(_pinned("CUDA"))))
    collide = [n for n in shared if "slate" in n or "blaspp" in n]
    assert collide, (
        "the two platform libraries no longer share a SLATE/blaspp SONAME "
        f"(shared DT_NEEDED: {shared}).  The dlopen-order rule "
        "distrib_la.loader._open_cuda_before_host enforces was written for "
        "exactly that collision and is now dead weight -- delete it from "
        "BOTH loaders (services/distrib_la/src/distrib_la/loader.py and "
        "src/ffi/common/ffi_loader.py) together with the cells under "
        "'THE SONAME RACE' in test_distrib_la_contract.py and the "
        "loader cells in tests/test_gpu_pinning.py.")


# ---------------------------------------------------------------------------
# THE FULL GATE SET, as one cell — scripts/verify_ffi_build.sh
# ---------------------------------------------------------------------------
# The four checks above are four of ten.  The other six (one MPI runtime,
# one BLAS threading flavour, OpenMP-is-OpenMP, HDF5 pairing, handler ABI)
# lived only in config/perlmutter/build_ffi_host.sh, which means they ran
# exactly once, at build time, on one machine, and never again — not on the
# artifact somebody pinned three days later, and not at all on the device leg
# or on Frontera.  scripts/verify_ffi_build.sh is the machine-agnostic form of
# all ten, and this cell is how a user VALIDATES A BUILD BY RUNNING PYTEST.
#
# It is the same file the build scripts call, so there is no possibility of the
# suite certifying something the build would have rejected, or the reverse.
# ---------------------------------------------------------------------------

def _repo_root():
    """The monorepo root, or SKIP.

    This package is independently installable; a standalone install has no
    ``scripts/`` and no ``src/ffi/cpp``, and the verifier reads both.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    if not os.path.exists(os.path.join(root, "scripts", "verify_ffi_build.sh")):
        pytest.skip(
            "monorepo wiring: scripts/verify_ffi_build.sh is not reachable "
            f"from {here} (standalone install of this service).  The verifier "
            "is the LORRAX repo's file; covered by any monorepo run.")
    return root


def test_scalapack_symbol_checker_owns_the_exact_13_symbol_surface():
    """Keep the executable provider gate aligned with C++ declarations."""
    root = _repo_root()
    checker = os.path.join(
        root, "src", "ffi", "cpp", "scalapack", "check_symbol_contract.sh")
    out = subprocess.run(
        ["bash", checker, "--print-all"], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=60)
    assert out.returncode == 0, out.stdout
    assert tuple(out.stdout.splitlines()) == _SCALAPACK_API_SYMBOLS

    header = os.path.join(
        root, "src", "ffi", "cpp", "scalapack", "blacs_grid.h")
    with open(header) as fh:
        source = fh.read()
    missing = [
        symbol for symbol in _SCALAPACK_API_SYMBOLS
        if re.search(rf"\b{re.escape(symbol)}\s*\(", source) is None
    ]
    assert not missing, f"{header} lacks declarations for {missing}"


def test_machine_scripts_do_not_gate_scalapack_on_slate():
    """ScaLAPACK/PBLAS is a provider contract, not a SLATE side effect."""
    root = _repo_root()
    frontera_path = os.path.join(
        root, "config", "frontera", "build_ffi_host.sh")
    perlmutter_path = os.path.join(
        root, "config", "perlmutter", "build_ffi_host.sh")
    with open(frontera_path) as fh:
        frontera = fh.read()
    with open(perlmutter_path) as fh:
        perlmutter = fh.read()

    provider = frontera.index("# ScaLAPACK provider.")
    slate = frontera.index("# Optional SLATE provider")
    assert provider < slate
    assert "LORRAX_SLATE_HOST_INSTALL_DIR" not in frontera[provider:slate]
    assert "-DLORRAX_HOST_HAVE_SCALAPACK=ON" in frontera
    assert '_expect_backends="scalapack,gemm,phdf5,fft"' in frontera

    assert "-DLORRAX_HOST_HAVE_SCALAPACK=ON" in perlmutter
    assert "-DLORRAX_HOST_HAVE_SLATE=OFF" in perlmutter
    assert "LORRAX_SLATE_HOST_INSTALL_DIR" not in perlmutter


def _run_verifier(so: str, leg: str, extra_env: dict | None = None):
    root = _repo_root()
    env = dict(os.environ)
    env.setdefault("LORRAX_ROOT", root)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", os.path.join(root, "scripts", "verify_ffi_build.sh"),
         "--leg", leg, so],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=600, env=env)


def test_the_host_library_passes_every_gate():
    """The whole contract, on the .so this run is pinned to.

    Expectations are stated, not inferred — that is the entire lesson of the
    Aug-7 library, which passed everything anybody ran on it because nothing
    anywhere recorded what it was supposed to contain.  The host leg of a
    LORRAX build has five backends; a site that builds fewer says so through
    ``LORRAX_FFI_EXPECT_BACKENDS``, and this cell honours that lever rather
    than hardcoding the full set, so it is correct on a phdf5-only Frontera
    build too.
    """
    so = _pinned("cpu")
    out = _run_verifier(so, "host")
    assert out.returncode == 0, (
        f"scripts/verify_ffi_build.sh REFUSED the pinned host library.\n"
        f"  LORRAX_FFI_HOST_SO={so}\n"
        f"Every gate names its own fix; the output follows.\n\n{out.stdout}")


def test_the_device_library_passes_every_gate():
    """Same contract, device leg.

    This leg had NO artifact-level acceptance of any kind before 2026-08-08 —
    not even a build stamp, so `strings liblorrax_ffi.so | grep abi=` on the
    deployed library returns nothing at all.
    """
    so = _pinned("CUDA")
    out = _run_verifier(so, "cuda")
    assert out.returncode == 0, (
        f"scripts/verify_ffi_build.sh REFUSED the pinned device library.\n"
        f"  LORRAX_FFI_SO={so}\n\n{out.stdout}")


def test_the_two_pinned_libraries_agree_about_hdf5():
    """GATE 7 ACROSS THE PAIR — the check no single-artifact gate can make.

    THE FAILURE THIS IS FOR IS THE DEPLOYED PAIR ITSELF.  Measured 2026-08-08:
    ``~/software/lorrax_ffi_2026-08-07/liblorrax_ffi_host.so`` links
    ``libhdf5_parallel_gnu.so.310`` and its own deployed partner
    ``liblorrax_ffi.so`` links ``libhdf5_parallel_gnu_123.so.200``.  Each is
    self-consistent; together they are two HDF5 majors, and whichever the
    container does not mount takes its WHOLE library down at dlopen —
    ScaLAPACK, SLATE, GEMM and phdf5 handlers alike, none of which touch HDF5.
    That is the 19-fails-with-'cannot open ...so.310' arm in BUILD_NOTES.md.

    The gate is on the PAIR because neither artifact alone is wrong.
    """
    host = _pinned("cpu")
    cuda = _pinned("CUDA")
    out = _run_verifier(host, "host",
                        {"LORRAX_FFI_EXPECT_PEER_SO": cuda})
    assert out.returncode == 0, (
        f"the two pinned libraries do not form a loadable pair.\n"
        f"  LORRAX_FFI_HOST_SO={host}\n"
        f"  LORRAX_FFI_SO     ={cuda}\n\n{out.stdout}")


def test_the_verifier_refuses_something(tmp_path):
    """RED TWIN for the tier above.

    Three green cells against three good libraries demonstrate nothing about
    whether the verifier can say no.  Point it at a file that is not a LORRAX
    library and it must refuse — and specifically it must refuse rather than
    crash, because a non-zero exit from a traceback would make every cell
    above pass for the wrong reason.
    """
    _repo_root()
    not_a_lib = str(tmp_path / "not_a_library.so")
    with open(not_a_lib, "wb") as fh:
        fh.write(b"\x7fELF" + b"not really\n" * 64)
    out = _run_verifier(not_a_lib, "host")
    assert out.returncode == 1, (
        f"the verifier did not refuse a non-library (rc={out.returncode}):\n"
        f"{out.stdout}")
    assert "GATE FAILED" in out.stdout, (
        f"the verifier refused without naming a gate:\n{out.stdout}")


# ---------------------------------------------------------------------------
# THE HANDLER-SIGNATURE ABI
# ---------------------------------------------------------------------------

def _header_abi_version():
    """The number in the C++ header, or SKIP."""
    root = _repo_root()
    hdr = os.path.join(root, "src", "ffi", "cpp", "common", "lorrax_ffi_abi.h")
    if not os.path.exists(hdr):
        pytest.skip(f"monorepo wiring: {hdr} not reachable.")
    with open(hdr) as fh:
        m = re.search(r"^#define\s+LORRAX_FFI_ABI_VERSION\s+(\d+)",
                      fh.read(), re.M)
    assert m, f"{hdr} does not define LORRAX_FFI_ABI_VERSION"
    return int(m.group(1))


def test_the_python_mirror_matches_the_cpp_definition():
    """THE DRIFT DETECTOR for a number that has to exist twice.

    A C++ macro cannot be imported, so the version is defined in
    ``src/ffi/cpp/common/lorrax_ffi_abi.h`` and mirrored in
    ``distrib_la.loader``.  A copy nothing compares is exactly how the ten
    eigh-dispatch ladders drifted; this cell is the comparison.
    """
    from distrib_la.loader import LORRAX_FFI_ABI_VERSION
    assert LORRAX_FFI_ABI_VERSION == _header_abi_version(), (
        f"distrib_la.loader.LORRAX_FFI_ABI_VERSION="
        f"{LORRAX_FFI_ABI_VERSION} but "
        f"src/ffi/cpp/common/lorrax_ffi_abi.h says {_header_abi_version()}.  "
        f"The header is the definition; move the mirror, in the same commit "
        f"as the signature change that caused the bump.")


def test_the_pinned_host_library_speaks_this_packages_abi():
    """The stamp on the artifact, against the mirror.

    Skips honestly rather than failing on a PRE-STAMP library: those exist in
    numbers on Perlmutter today (the whole deployed pair is unstamped) and
    "built before the mechanism" is not "built wrong".  What it refuses is a
    STAMPED library carrying a different number.
    """
    so = _pinned("cpu")
    from distrib_la.loader import LORRAX_FFI_ABI_VERSION
    syms = _defined_symbols(so)
    if "lorrax_ffi_host_abi_version" not in syms:
        pytest.skip(
            f"{so} carries no handler-ABI stamp (pre-2026-08-08 build), so "
            f"there is nothing to compare.  Covered by any leg pinned to a "
            f"library built from this tree; the loader announces the same "
            f"fact at dlopen and LORRAX_FFI_ABI_STRICT=1 turns it into a "
            f"refusal.")
    with open(so, "rb") as fh:
        blob = fh.read()
    found = sorted({m.decode() for m in re.findall(rb"abi=[0-9]+", blob)})
    assert found == [f"abi={LORRAX_FFI_ABI_VERSION}"], (
        f"{so} stamps {found}; this package speaks "
        f"abi={LORRAX_FFI_ABI_VERSION}.  These cannot be paired — the "
        f"mismatch is invisible until the first call crossing a changed "
        f"signature, which is why the number exists.")


def test_a_stamped_old_library_produces_the_named_refusal(tmp_path):
    """RED TWIN for the ABI mechanism, end to end through the loader.

    THE POINT IS THE SUBSTITUTION.  Every other cell here reads an ELF; this
    one builds a real shared object that exports a real
    ``lorrax_ffi_host_abi_version`` returning a STALE number, hands it to the
    loader's own dlopen path, and requires the refusal to name both versions
    and the rebuild command — not the ``INVALID_ARGUMENT: Wrong number of
    arguments`` that a mispaired library produced before this branch, and not
    a generic "could not load".

    A twenty-line .so rather than monkeypatching the check, because the thing
    under test is whether the check RUNS on a library the loader actually
    opened, in the right ORDER: before argtypes are declared and before any
    target is registered.  A monkeypatched twin would pass with the call site
    in the wrong place.
    """
    import ctypes
    from distrib_la.loader import (AbiMismatch, LORRAX_FFI_ABI_VERSION,
                                   _check_abi)
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler on PATH to build the stale-ABI twin; "
                    "covered by any leg with a toolchain (every build node).")
    stale = LORRAX_FFI_ABI_VERSION - 1
    src = tmp_path / "stale_abi.c"
    src.write_text(
        "int lorrax_ffi_host_abi_version(void) { return %d; }\n" % stale)
    lib_path = tmp_path / "libstale_abi.so"
    build = subprocess.run([cc, "-shared", "-fPIC", "-o", str(lib_path),
                            str(src)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=180)
    assert build.returncode == 0, f"could not build the twin:\n{build.stdout}"

    lib = ctypes.CDLL(str(lib_path))
    with pytest.raises(AbiMismatch) as excinfo:
        _check_abi(lib, "cpu", str(lib_path))
    msg = str(excinfo.value)
    assert f"speaks abi={stale}" in msg, msg
    assert f"this tree speaks abi={LORRAX_FFI_ABI_VERSION}" in msg, msg
    assert "build_ffi_host.sh" in msg, (
        "the refusal must name the rebuild command, or it is the same "
        f"dead end as the runtime error it replaces:\n{msg}")
    # ...and the OTHER direction: a library stamping the current version is
    # accepted, so this cell is not passing because _check_abi refuses
    # everything.
    ok_src = tmp_path / "ok_abi.c"
    ok_src.write_text("int lorrax_ffi_host_abi_version(void) { return %d; }\n"
                      % LORRAX_FFI_ABI_VERSION)
    ok_path = tmp_path / "libok_abi.so"
    assert subprocess.run([cc, "-shared", "-fPIC", "-o", str(ok_path),
                           str(ok_src)], timeout=180).returncode == 0
    _check_abi(ctypes.CDLL(str(ok_path)), "cpu", str(ok_path))
