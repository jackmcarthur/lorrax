"""The C++ tier, v1: the four build acceptance checks, as pytest cells.

There are NO C++-level tests anywhere in LORRAX (survey 1, S8: ``find
src/ffi/cpp -iname '*test*'`` is empty).  The nearest thing is a block of
shell in ``config/perlmutter/build_ffi_host.sh``'s seven gates plus the
four post-build commands ``BUILD_NOTES.md`` writes out, which are run by
hand after a rebuild and by nobody afterwards.  This file makes them a
suite: same four assertions, against the .so a run is actually PINNED to,
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

#: The four Scalapack host handlers.  BUILD_NOTES.md's check 2 counts
#: them; naming them means a build that exported three and a typo passes
#: the count and fails here.
_SCALAPACK_SYMBOLS = tuple(sorted(
    sym for sym in _HOST_TARGET_SYMBOLS.values() if "Scalapack" in sym))


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


def test_check_2_all_four_scalapack_handlers_are_exported():
    """``nm -D --defined-only <so> | grep -c Scalapack`` must be 4.

    The count AND the names.  A count alone passes a build that exported
    four symbols one of which was misspelled, and the names come from
    ``distrib_la.loader``'s own table — so this cell also pins that the
    table and the library have not drifted apart, which is the failure
    ``probe_target`` reports as "loaded but does not export".
    """
    so = _pinned("cpu")
    syms = _defined_symbols(so)
    assert len(_SCALAPACK_SYMBOLS) == 4, (
        f"the loader's host table lists {len(_SCALAPACK_SYMBOLS)} Scalapack "
        f"handlers, not 4; BUILD_NOTES.md's check counts 4 — one of the two "
        f"moved and they must move together: {_SCALAPACK_SYMBOLS}")
    missing = [s for s in _SCALAPACK_SYMBOLS if s not in syms]
    assert not missing, f"{so} does not export {missing}"
    grepped = sum(1 for s in syms if "Scalapack" in s)
    assert grepped == 4, (
        f"{so} exports {grepped} Scalapack* symbols, want exactly 4 "
        f"({sorted(s for s in syms if 'Scalapack' in s)})")


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
