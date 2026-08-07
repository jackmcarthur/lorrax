"""The shared pytest harness for LORRAX services (``pytest11`` plugin).

Installed as a plugin entry point, so a service suite gets the autouse
:func:`gate_state` fixture by installing lxkit and nothing else; the helpers
are plain functions, imported where they are used.

This module is the ONE place in lxkit that imports pytest.  It is NOT
imported by ``lxkit/__init__.py`` — the package's stdlib-only property (see
the package docstring) covers the runtime modules, and a test harness that
dragged pytest into every production import would break it.  jax stays lazy
here too: :func:`require_devices` imports it inside the function body, so
this plugin loads in a jax-free interpreter.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from typing import Iterable, NamedTuple, Sequence

import pytest

from lxkit.gate import reset_gate_state

__all__ = [
    "OK", "ABSENT", "BROKEN", "absent_or_broken",
    "gate_state", "require_devices",
    "HostileGeometry", "hostile_extents",
    "IsolationRun", "import_isolation",
    "machine_profile", "assert_skips_match_profile",
]


# ---------------------------------------------------------------------------
# ABSENT IS A SKIP.  BUILT-AND-BROKEN IS A FAILURE.
# ---------------------------------------------------------------------------
# Ported from tests/test_ffi_linalg_contract.py:106-257 (96a6399).  Both
# probes there used to answer one question -- "does the .so load?" -- and turn
# any negative answer into a skip.  That collapses two situations that mean
# opposite things:
#
#   ABSENT   the library was never built here.  Nothing is wrong; these tests
#            do not apply on this machine.  A skip is the honest report.
#   BROKEN   the library WAS built, is sitting right there, and will not
#            dlopen.  Something IS wrong, and it is exactly the kind of wrong
#            that a test suite exists to catch.
#
# Reporting BROKEN as a skip is how the host leg was lost on 2026-08-06: the
# .so carried a DT_NEEDED on cray-fftw's `libfftw3.so.mpi31.3`, which the
# Shifter container does not mount, so dlopen failed and NINETEEN
# ScaLAPACK/SLATE/GEMM contract tests -- none of which perform an FFT -- were
# reported as "19 skipped" alongside "0 failed".  The suite looked green.  A
# skip reads as "not applicable on this platform"; what it meant was "did not
# run".  Nothing in the output distinguished them.
#
# This is the lxkit.gate rule applied to test collection: an explicit request
# that cannot be honored REFUSES rather than silently downgrading.  A built
# .so is an explicit request.
#
# WHY THE FAILURE IS PER-TEST AND NOT A COLLECTION ERROR.  Raising at import
# would be louder still, but it would take the whole module down -- including
# the cells on the other platform, which on 2026-08-06 were genuinely passing
# 33/33.  Losing real signal to report a different defect is the same mistake
# in the other direction.  Each affected cell fails, naming the library and
# the loader error; everything unaffected still reports.
#
# WHO DECIDES ABSENT-VS-BROKEN.  The loader does, when it can:
# lxkit.probe.LibraryNotBuilt / LibraryUnusable are the split, because the
# loader knows which candidate path it tried and why it gave up, which a
# stat() from out here can only guess at.

OK = "ok"
ABSENT = "absent"       # not built here — skip is honest
BROKEN = "broken"       # built and will not load — this is a defect


def absent_or_broken(state: str, reason: str):
    """Decorator: run / skip / refuse, per the rule above.

    ``state`` is one of :data:`OK` / :data:`ABSENT` / :data:`BROKEN`;
    ``reason`` is the probe's own text (:class:`lxkit.probe.ProbeResult`),
    quoted verbatim into the skip or the failure.  An unrecognized state
    raises at decoration time rather than silently running the cell — a
    typo'd state that quietly meant OK is the same hole in a different wall.
    """
    if state not in (OK, ABSENT, BROKEN):
        raise ValueError(
            f"absent_or_broken: state={state!r} is not one of "
            f"{(OK, ABSENT, BROKEN)}")

    def decorate(fn):
        if state == OK:
            return fn
        if state == ABSENT:
            return pytest.mark.skip(reason=reason or "")(fn)

        @functools.wraps(fn)
        def _refuse(*args, **kwargs):
            pytest.fail(reason, pytrace=False)
        return _refuse
    return decorate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def gate_state():
    """Clear the process-global announcement memo around every test.

    ``lxkit.gate._ANNOUNCED`` is per-PROCESS, so without this any test that
    asserts on an announcement is order-dependent: whichever cell reaches
    the gate first burns the key and every later cell observes silence and
    passes for the wrong reason.  In lorrax at 96a6399 ``reset_gate_state()``
    had ZERO callers, so that hazard was live tree-wide.

    Autouse and BOTH SIDES: a test that deliberately burns a key must not
    leak it forward either.
    """
    reset_gate_state()
    yield
    reset_gate_state()


def require_devices(n: int) -> int:
    """SKIP unless this process sees at least ``n`` jax devices; returns the
    count.

    **Skips, never asserts.**  Suite-wide convention, and a recorded defect:
    ``tests/KNOWN_FAILURES.md`` lists 11 cells (``test_contract_bands`` (9) +
    ``test_projection_lgemm`` (2)) that FAILED on the bare 1-device leg
    purely because they wrote ``assert n_dev >= 4``.  Device count is a
    property of how the leg was launched (``XLA_FLAGS
    --xla_force_host_platform_device_count``, which must be set before the
    first jax import, or a real ``srun -n 4``), not a property of the code
    under test.  The covering leg is named in the skip reason so the skip
    is answerable rather than merely quiet.
    """
    try:
        import jax
    except Exception as exc:                              # noqa: BLE001
        pytest.skip(f"jax is not importable here ({exc}); this cell needs "
                    f">= {n} devices — covered by the emulated 4-device leg")
    have = jax.device_count()
    if have < n:
        pytest.skip(
            f"needs >= {n} devices, have {have}.  Set XLA_FLAGS="
            f"--xla_force_host_platform_device_count={n} BEFORE the first jax "
            f"import, or run the real multi-process leg")
    return have


# ---------------------------------------------------------------------------
# Hostile geometry (charter: mandatory for every mesh-touching service)
# ---------------------------------------------------------------------------

class HostileGeometry(NamedTuple):
    """One non-divisible logical extent and the padded shape it lives in."""

    name: str
    logical: tuple[int, ...]
    padded: tuple[int, ...]


def _next_prime(n: int) -> int:
    """Smallest prime strictly greater than ``n``."""
    c = max(2, n + 1)
    while True:
        if c > 2 and c % 2 == 0:
            c += 1
            continue
        d, ok = 3, c != 4
        while d * d <= c and ok:
            ok = c % d != 0
            d += 2
        if ok:
            return c
        c += 1


def _pad_up(logical: Sequence[int], mesh_shape: Sequence[int]) -> tuple:
    """Round each extent up to the next multiple of its mesh axis."""
    return tuple(-(-n // m) * m for n, m in zip(logical, mesh_shape))


def hostile_extents(mesh_shape: Sequence[int]) -> tuple[HostileGeometry, ...]:
    """Parametrization data: extents that do NOT divide ``mesh_shape``.

    The five families are the ones measured end-to-end on Perlmutter job
    56389339 (4 nodes / 16 ranks, 4x4 mesh, each case written from a padded
    sharded operand and compared against a single-rank reference plus an
    explicit zero-check of the pad region), generalized off the mesh shape:
    prime extents on both axes, a tighter prime pair, non-divisible on one
    axis only, fewer slices than ranks, and empty tiles on one axis.  At
    ``mesh_shape=(4, 4)`` it reproduces that job's table exactly --
    (17,23) (13,17) (17,16) (1,1) (2,16) -- which is what makes the
    generalization checkable rather than decorative.

    Two axes only: this is the ('x','y') device mesh every LORRAX service
    shards over.
    """
    if len(mesh_shape) != 2 or any(m < 1 for m in mesh_shape):
        raise ValueError(
            f"hostile_extents expects a 2-axis mesh of positive extents, "
            f"got {tuple(mesh_shape)!r}")
    mx, my = mesh_shape
    p = _next_prime
    rows = (
        ("prime-both-axes",         (p(4 * mx), p(5 * my))),
        ("prime-both-axes-tighter", (p(3 * mx), p(4 * my))),
        ("nondivisible-axis0-only", (p(4 * mx), 4 * my)),
        ("fewer-slices-than-ranks", (1, 1)),
        ("empty-tiles-on-axis0",    (max(1, mx // 2), 4 * my)),
    )
    return tuple(HostileGeometry(name, logical, _pad_up(logical, mesh_shape))
                 for name, logical in rows)


# ---------------------------------------------------------------------------
# Import isolation — what makes "standalone" falsifiable
# ---------------------------------------------------------------------------

_SENTINEL = "LXKIT_ISOLATION "

_PROBE = r'''
import json, os, sys
{preamble}
import {pkg} as _probed
_forbidden = {forbidden!r}
_loaded = sorted({{m.split(".")[0] for m in sys.modules}} & set(_forbidden))
_reachable = []
for _entry in sys.path:
    for _root in _forbidden:
        if (os.path.isfile(os.path.join(_entry, _root, "__init__.py"))
                or os.path.isfile(os.path.join(_entry, _root + ".py"))):
            _reachable.append([_root, _entry])
print({sentinel!r} + json.dumps({{
    "file": os.path.realpath(getattr(_probed, "__file__", "") or ""),
    "loaded": _loaded,
    "reachable": _reachable,
    "cwd": os.getcwd(),
}}))
'''


class IsolationRun(NamedTuple):
    """What the isolated subprocess reported."""

    stdout: str
    stderr: str
    file: str                     #: realpath of the imported package
    loaded: tuple[str, ...]       #: forbidden roots found in sys.modules
    reachable: tuple[tuple[str, str], ...]   #: (root, sys.path entry) pairs


def import_isolation(pkg: str, forbidden_roots: Iterable[str], *,
                     src_dir: str | None = None,
                     extra_path: Sequence[str] = (),
                     preamble: str = "",
                     check_path: bool = True) -> IsolationRun:
    """Import ``pkg`` in a subprocess with nothing but ``src_dir`` on the
    path, and ASSERT it dragged in none of ``forbidden_roots``.

    This is the charter's falsifiability mechanism: "standalone" is a claim
    about what happens with the rest of the monorepo absent, and the only
    way to observe that is a process where it IS absent.

    Why each piece:

    ``python -S``
        MEASURED, not defensive.  The parent is typically a venv with the
        monorepo EDITABLE-installed -- ``.venv/.../site-packages/
        __editable__.lorrax-0.1.0.pth`` puts ``<tree>/src`` on the path of
        every subprocess of that interpreter, whatever ``PYTHONPATH`` says.
        ``-S`` skips ``site`` and therefore every ``.pth``, so the child's
        path is exactly the stdlib plus what this function put there.  A
        leg that needs pytest or jax back asks for them by name via
        ``extra_path`` (the site-packages DIRECTORY carries no ``.pth``
        processing when it arrives through ``PYTHONPATH``).
    ``PYTHONPATH = src_dir`` (+ ``extra_path``)
        Set, not popped.  The parent's ``PYTHONPATH`` is discarded outright.
    ``cwd`` outside the repo
        A temp dir, so ``sys.path[0]`` cannot smuggle sibling packages in.
    assert on ``sys.modules`` **and** on ``sys.path``
        ``sys.modules`` proves the package did not IMPORT the forbidden
        roots; ``sys.path`` proves it COULD not have -- the second is what
        makes the first evidence about the package rather than about the
        machine, and it is the assertion that catches an ``lx``-baked
        ``PYTHONPATH`` or a ``.pth`` that slipped past ``-S``.
        ``check_path=False`` is for the deliberate
        forbidden-roots-are-present positive runs (jax installed; the
        monorepo put back on the path on purpose), where the path half is
        inapplicable by construction.
    assert on stdout CONTENT, not the return code
        A probe that dies before printing, or an interpreter that prints a
        warning and exits 0, both return 0-ish signals.  The payload line
        is the evidence.

    ``preamble`` is injected before the import -- the red twin's handle: a
    deliberate ``import <forbidden>`` there MUST make this function raise,
    or the check is a tautology.
    """
    forbidden = tuple(forbidden_roots)
    if not forbidden:
        raise ValueError("import_isolation: forbidden_roots must be non-empty "
                         "(a check with nothing to find cannot fail)")
    if src_dir is None:
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.origin:
            raise ValueError(f"import_isolation: cannot locate {pkg!r} to "
                             f"derive src_dir; pass it explicitly")
        pkg_dir = os.path.dirname(os.path.realpath(spec.origin))
        src_dir = os.path.dirname(pkg_dir)
    src_dir = os.path.realpath(src_dir)
    path = [src_dir, *(os.path.realpath(p) for p in extra_path)]

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    script = _PROBE.format(pkg=pkg, forbidden=forbidden, preamble=preamble,
                           sentinel=_SENTINEL)
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([sys.executable, "-S", "-c", script], cwd=tmp,
                              env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        out = proc.stdout.decode()
        err = proc.stderr.decode()

    payload = None
    for line in out.splitlines():
        if line.startswith(_SENTINEL):
            payload = json.loads(line[len(_SENTINEL):])
    assert payload is not None, (
        f"isolated import of {pkg!r} produced no {_SENTINEL.strip()} line "
        f"(rc={proc.returncode})\nPYTHONPATH={env['PYTHONPATH']}\n"
        f"--- stdout ---\n{out}\n--- stderr ---\n{err}")
    assert proc.returncode == 0, (
        f"isolated import of {pkg!r} exited {proc.returncode}\n"
        f"--- stdout ---\n{out}\n--- stderr ---\n{err}")

    run = IsolationRun(out, err, payload["file"],
                       tuple(payload["loaded"]),
                       tuple(tuple(r) for r in payload["reachable"]))
    assert run.file.startswith(src_dir + os.sep), (
        f"isolated import resolved {pkg!r} to {run.file!r}, which is not "
        f"under the src dir under test ({src_dir!r}) — the run measured "
        f"some other copy")
    assert not run.loaded, (
        f"importing {pkg!r} pulled {list(run.loaded)} into sys.modules; "
        f"{pkg!r} is not standalone with respect to {list(forbidden)}\n"
        f"--- stdout ---\n{out}")
    if check_path:
        assert not run.reachable, (
            f"{list(forbidden)} were importable from sys.path in the "
            f"'isolated' run ({list(run.reachable)}), so a clean "
            f"sys.modules proves nothing about {pkg!r}.  PYTHONPATH="
            f"{env['PYTHONPATH']}")
    return run


# ---------------------------------------------------------------------------
# Skip-honesty (charter: an unexpected skip is a FAILURE)
# ---------------------------------------------------------------------------
# DELIBERATELY UNIMPLEMENTED.  The charter requires one gate per service
# asserting that observed skips match the machine's declared expected-backend
# profile (Perlmutter: scalapack, slate, cusolvermp, phdf5 MUST probe
# available).  No such mechanism exists anywhere in the tree at 96a6399 --
# tests/profiles/ holds xprof traces, not machine profiles -- so there is
# nothing here to port and nothing measured to encode.  The design is
# settled (DESIGN_distrib_la.md, "Test architecture": PROFILES keyed on
# NERSC_HOST/LX_MACHINE with an explicit `unknown` row that asserts nothing;
# positive half fails with the three-way probe reason; negative half diffs
# observed skips against an allowed-skip predicate plus a
# minimum-collected floor; red twins for both) but the profile ROWS are a
# measurement that only the real machine can supply.
#
# **distrib_la step 2 (service test suite) fills these in**, with the
# Perlmutter probe results as its evidence, and lifts them here once the
# shape has survived one real service.  A plausible stub returning an empty
# profile would be worse than none: every skip would match it and the gate
# would report green while asserting nothing -- the exact failure mode
# (2026-08-06, 19 cells) this whole section exists to prevent.

_NOT_YET = (
    "lxkit.testing.{name} is not implemented yet.  Skip-honesty profiles are "
    "filled in by distrib_la step 2 (service test suite) from real Perlmutter "
    "probe results -- see DESIGN_distrib_la.md 'Test architecture'.  lxkit "
    "ships the refusal rather than a permissive stub: a profile that asserts "
    "nothing turns the skip-honesty gate green while it measures nothing.")


def machine_profile(machine: str | None = None):
    """The declared expected-backend profile for this machine.  **Stub.**"""
    raise NotImplementedError(_NOT_YET.format(name="machine_profile"))


def assert_skips_match_profile(*args, **kwargs):
    """Fail when observed skips diverge from the profile.  **Stub.**"""
    raise NotImplementedError(
        _NOT_YET.format(name="assert_skips_match_profile"))
