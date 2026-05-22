"""Build smoke test: check that liblorrax_ffi.so has no missing ELF dependencies.

Runs ``ldd build/liblorrax_ffi.so | grep 'not found'`` and asserts the output
is empty.  Catches the SONAME / RPATH regression class documented in
agent_1.md items #13, #14, #15.

Skip policy: if the .so does not exist at any of the candidate paths, the test
is skipped cleanly.  CI runs without building the FFI; the test is only
meaningful when the library has been compiled (Perlmutter or a local CUDA
build).

The .so search order mirrors ``ffi.common.ffi_loader._candidate_paths()``:
  1. ``LORRAX_FFI_SO`` env var (absolute path).
  2. ``src/ffi/common/cpp/build/liblorrax_ffi*.so`` relative to repo root.
  3. Any ``liblorrax_ffi*.so`` on ``sys.path``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _find_so() -> Path | None:
    """Return the first existing liblorrax_ffi.so candidate, or None."""
    env_override = os.environ.get("LORRAX_FFI_SO")
    if env_override:
        p = Path(env_override)
        if p.is_file():
            return p

    # Locate repo root: walk up from this file until we find src/ffi.
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "src" / "ffi" / "common" / "cpp" / "build"
        if candidate.is_dir():
            matches = sorted(candidate.glob("liblorrax_ffi*.so"))
            if matches:
                return matches[0]

    # Check sys.path entries.
    for entry in sys.path:
        p = Path(entry)
        if p.is_dir():
            matches = sorted(p.glob("liblorrax_ffi*.so"))
            if matches:
                return matches[0]

    return None


def test_ldd_no_missing_symbols() -> None:
    """ldd liblorrax_ffi.so must report no 'not found' libraries."""
    so_path = _find_so()
    if so_path is None:
        pytest.skip(
            "liblorrax_ffi.so not found — build with "
            "'bash src/ffi/common/cpp/build.sh' or set LORRAX_FFI_SO."
        )

    result = subprocess.run(
        ["ldd", str(so_path)],
        capture_output=True,
        text=True,
    )
    # ldd exits non-zero on some platforms even when the output is useful;
    # focus on the content rather than the return code.
    not_found_lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if "not found" in line
    ]
    assert not_found_lines == [], (
        f"Missing ELF dependencies in {so_path}:\n"
        + "\n".join(f"  {ln}" for ln in not_found_lines)
        + "\n\nFull ldd output:\n"
        + result.stdout
    )
