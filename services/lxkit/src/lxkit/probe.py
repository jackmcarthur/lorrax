"""Probe vocabulary — ABSENT vs BROKEN, and the three-way unusable reason.

POLICY ONLY.  The symbol tables, library search paths and build hints that
answer "is this handler here?" belong to the service that owns the ``.so``;
what lives here is the shape every service's answer must take, so that a
harness can act on it without knowing whose library it was.

The two exception types split the one question every skip decision turns on::

    ABSENT   the library was never built here.  Nothing is wrong; these
             tests do not apply on this machine.  A skip is honest.
    BROKEN   the library WAS built, is sitting right there, and will not
             dlopen.  Something IS wrong, and it is exactly the kind of
             wrong a test suite exists to catch.

:func:`lxkit.testing.absent_or_broken` is the harness arm that consumes the
split.  Collapsing the two is how the LORRAX host leg was lost on
2026-08-06 — 19 ScaLAPACK/SLATE/GEMM contract cells reported "19 skipped"
alongside "0 failed" and the suite looked green.

The three reason constructors are the *unusable* taxonomy, ported from
``ffi_loader.probe_target`` (96a6399, src/ffi/common/ffi_loader.py:438-497).
They exist because the three cases have three different fixes and a bare
bool tells somebody to rebuild a library that is fine.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

__all__ = [
    "ProbeResult", "AVAILABLE", "LibraryNotBuilt", "LibraryUnusable",
    "unknown_target", "not_loadable", "missing_symbol",
]


class ProbeResult(NamedTuple):
    """``(ok, reason)`` — what a probe answers, and why.

    A ``NamedTuple`` on purpose: :attr:`lxkit.gate.Gate.probe` is declared
    as ``(target, platform) -> tuple[bool, str]``, so a probe may return
    this or a plain pair and both unpack identically.  ``reason`` is never
    empty — a refusal without a reason is the bare bool this module exists
    to replace.
    """

    ok: bool
    reason: str


#: The positive answer.  Its wording is part of the contract: harnesses
#: report the reason string for both outcomes.
AVAILABLE = ProbeResult(True, "available")


class LibraryNotBuilt(FileNotFoundError):
    """No library exists for this platform anywhere on the search path.

    This is the LEGITIMATE platform gate: a CPU-only laptop, or a checkout
    that was never built.  A caller may reasonably skip on it.
    """


class LibraryUnusable(OSError):
    """A library WAS located but cannot be used.

    Distinct from :class:`LibraryNotBuilt` on purpose.  "The library is
    absent" and "the library is present and broken" want opposite
    responses -- the first is an environment fact, the second is a DEFECT,
    and collapsing them into one exception is how a broken dependency
    closure gets reported as a platform skip.  Live example, measured
    2026-08-06 inside the Shifter container on nid001644: the host .so at
    src/ffi/cpp/build_host/liblorrax_ffi_host.so exists, but /opt/cray/pe
    is not bind-mounted, so all three libfftw3*.so.mpi31.3 RPATH deps are
    "not found" and 19 Tier-1 cells reported themselves as skipped.
    """


def unknown_target(target: str, platform: str,
                   known: Iterable[str] = ()) -> ProbeResult:
    """Not a target of this platform's library at all — a typo or a
    wrong-platform request.  Nothing to build, nothing to fix in the .so."""
    names = ", ".join(sorted(known))
    return ProbeResult(False, (
        f"unknown target: {target!r} is not a target of the {platform} "
        f"library" + (f" (known: {names})" if names else "")))


def not_loadable(platform: str, exc: BaseException, *,
                 target: str = "", hint: str = "") -> ProbeResult:
    """The library is missing, or loading it FAILED.

    One of the libraries it in turn needs could not be found, a
    glibc/GLIBCXX mismatch, a wrong ``LD_LIBRARY_PATH``.  **The handler may
    well be compiled; nothing about the build is wrong** — and saying
    otherwise is the defect this taxonomy removes.  Conflating this case
    with :func:`missing_symbol` produces an error that tells you to rebuild
    a library that is perfectly fine while the actual cause (one missing
    entry in ``LD_LIBRARY_PATH``) goes unmentioned; on Frontera that
    silently downgraded an Nx1 mesh to ``native`` with a "not compiled"
    diagnosis (found by wk_P G4, 2026-07-25).

    ``hint`` is the service's own path/dependency advice — which env var
    pins the ``.so``, which libraries ``LD_LIBRARY_PATH`` must cover.
    """
    note = (f"  NOTE: this says nothing about whether {target} is compiled — "
            f"fix the library path/dependencies first." if target else "")
    return ProbeResult(False, (
        f"the {platform} library could not be loaded: "
        f"{type(exc).__name__}: {exc}{note}"
        + (f"  {hint}" if hint else "")))


def missing_symbol(target: str, symbol: str, path: str, *,
                   build_hint: str = "") -> ProbeResult:
    """Loaded, but the handler symbol is absent.

    The genuine partial-build case — this, and ONLY this, is a "rebuild the
    library" problem, so it is the only reason that carries a build hint.
    """
    return ProbeResult(False, (
        f"loaded {path} but it does not export {symbol} — this library was "
        f"built WITHOUT the {target} handler."
        + (f"  Rebuild with {build_hint}." if build_hint else "")))
