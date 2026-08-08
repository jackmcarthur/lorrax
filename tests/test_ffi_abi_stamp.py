"""The handler-signature ABI: one definition, two mirrors, no drift.

``LORRAX_FFI_ABI_VERSION`` is defined once, in
``src/ffi/cpp/common/lorrax_ffi_abi.h``, and has to exist as a Python constant
in two places that cannot import a C++ macro: lorrax's own
``src/ffi/common/ffi_loader.py`` and the independently-installable
``services/distrib_la``.  Three copies of one number is the shape of every
defect the services phase was extracted to prevent — so the copies are
compared, here and in ``services/distrib_la/tests/test_so_acceptance.py``.

These cells are pure AST/text and run in milliseconds on any machine.  They
never dlopen anything: whether a LIBRARY's stamp matches is the acceptance
tier's question, asked of the .so a run is actually pinned to.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_HEADER = _REPO / "src" / "ffi" / "cpp" / "common" / "lorrax_ffi_abi.h"


def _header_version() -> int:
    text = _HEADER.read_text()
    m = re.search(r"^#define\s+LORRAX_FFI_ABI_VERSION\s+(\d+)", text, re.M)
    assert m, f"{_HEADER} does not define LORRAX_FFI_ABI_VERSION"
    return int(m.group(1))


def _mirror_version(path: Path) -> int:
    """Read the constant TEXTUALLY, not by importing.

    Importing ``distrib_la.loader`` needs jax and lxkit; importing
    ``ffi.common.ffi_loader`` needs jax too.  This cell's question is about
    three characters in three files and should not be able to fail because a
    backend is missing.
    """
    m = re.search(r"^LORRAX_FFI_ABI_VERSION\s*=\s*(\d+)", path.read_text(), re.M)
    assert m, f"{path} does not define LORRAX_FFI_ABI_VERSION"
    return int(m.group(1))


def test_the_header_defines_a_version():
    assert _header_version() >= 1


@pytest.mark.parametrize("mirror", [
    "src/ffi/common/ffi_loader.py",
    "services/distrib_la/src/distrib_la/loader.py",
])
def test_every_python_mirror_matches_the_header(mirror):
    """THE DRIFT DETECTOR.

    Fails the day somebody bumps the C++ header for a signature change and
    forgets a loader — which would leave that loader silently ACCEPTING the
    library it should refuse, i.e. exactly the pre-2026-08-08 behaviour, with
    a mechanism in place that looks like it is working.
    """
    path = _REPO / mirror
    assert _mirror_version(path) == _header_version(), (
        f"{mirror} says abi={_mirror_version(path)}, "
        f"{_HEADER.relative_to(_REPO)} says abi={_header_version()}.  The "
        f"header is the definition.  Move the mirror in the same commit as "
        f"the signature change that caused the bump.")


def test_the_build_stamp_carries_the_version():
    """The C++ side actually bakes it in.

    ``common/build_config.cc`` is what puts ``abi=<N>`` into the .so's
    .rodata and exports ``lorrax_ffi_{host,cuda}_abi_version``.  If it stopped
    including the header, every artifact-level ABI gate would go quietly
    vacuous: GATE 11 would report "no stamp" for every fresh build and the
    loaders would take the announce-and-proceed branch forever.
    """
    cc = (_REPO / "src" / "ffi" / "cpp" / "common" / "build_config.cc").read_text()
    assert '#include "lorrax_ffi_abi.h"' in cc
    assert "LORRAX_FFI_ABI_VERSION" in cc
    for sym in ("lorrax_ffi_host_abi_version", "lorrax_ffi_cuda_abi_version"):
        assert sym in cc, f"build_config.cc no longer exports {sym}"


@pytest.mark.parametrize("mapfile,leg", [
    ("src/ffi/cpp/exports_host.map", "host"),
    ("src/ffi/cpp/exports_cuda.map", "cuda"),
])
def test_the_version_script_still_exports_the_stamp(mapfile, leg):
    """THE OTHER WAY THIS MECHANISM GOES VACUOUS, and it did.

    The ODR fix links a version script into each leg whose ``local:`` clause
    is the denylist ``*lorrax_ffi*`` — which matches
    ``lorrax_ffi_<leg>_abi_version`` and ``lorrax_ffi_<leg>_build_config``
    exactly as it matches a mangled internal.  Localising them does not fail
    the link, does not fail any build gate, and does not fail a run: the
    library stops exporting the symbol, the loaders take the "built before the
    stamp existed" branch, and GATE 11 reports COULD NOT RUN.  Every artifact
    then looks like a legitimate pre-2026-08-08 library and every pairing is
    accepted.

    That is precisely how this mechanism disarms without anybody noticing, so
    the exception is asserted here rather than left to be re-discovered by the
    first mispaired run.  Text, not ELF, so it runs on any machine and fails
    in the same commit that removes the line.
    """
    text = (_REPO / mapfile).read_text()
    # The `global:` clause is everything between the two clause keywords where
    # they appear as keywords -- alone on their own line.  Both files discuss
    # `local: *` in prose above, so a plain substring split lands in a comment.
    m = re.search(r"^\s*global:\s*$(.*?)^\s*local:\s*$", text, re.M | re.S)
    assert m, f"{mapfile} has no `global:` clause at all"
    head = m.group(1)
    for sym in (f"lorrax_ffi_{leg}_abi_version",
                f"lorrax_ffi_{leg}_build_config"):
        assert re.search(rf"^\s*{sym};", head, re.M), (
            f"{mapfile} no longer names {sym} in its `global:` clause, so the "
            f"`local: *lorrax_ffi*` denylist below will hide it.  The symbol "
            f"is an API: both loaders dlsym the ABI accessor at dlopen and "
            f"the acceptance tier reads the stamp.  Hiding it disarms the "
            f"pairing refusal silently.")


def test_the_two_legs_do_not_share_a_stamp_symbol():
    """Per-leg names, and the reason is not style.

    Both libraries are dlopened RTLD_GLOBAL in a GPU process and already share
    sixteen symbol names (KNOWN_FAILURES L1, the cross-.so ODR hazard).  A
    shared spelling for the ABI accessor would let whichever library loaded
    first answer for both — so a mispaired process would read the ABI of the
    library that is not the one being checked, and certify itself.
    """
    cc = (_REPO / "src" / "ffi" / "cpp" / "common" / "build_config.cc").read_text()
    assert "LORRAX_FFI_NO_CUDA" in cc, (
        "build_config.cc no longer selects its symbol names per leg")
