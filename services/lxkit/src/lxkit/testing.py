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
from typing import Callable, Iterable, Mapping, NamedTuple, Sequence

import pytest

from lxkit.gate import reset_gate_state

__all__ = [
    "OK", "ABSENT", "BROKEN", "absent_or_broken",
    "gate_state", "require_devices",
    "HostileGeometry", "hostile_extents",
    "IsolationRun", "import_isolation", "dep_dirs",
    "MustRow", "AllowedSkip", "MachineProfile", "PROFILES",
    "detect_machine", "machine_profile", "must_rows_for",
    "assert_must_probe", "assert_skips_match_profile",
    "arm_skip_honesty", "observed_skips",
    "ArmedScope", "armed_scopes", "check_armed_scope",
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


def require_devices(n: int, platform: str | None = None) -> int:
    """SKIP unless this process sees at least ``n`` jax devices; returns the
    count.

    ``platform`` asks about ONE backend rather than the default one, and
    the emulated tier has to: ``--xla_force_host_platform_device_count``
    creates HOST devices, so on any box with a GPU the default backend is
    ``cuda``, ``jax.device_count()`` answers 1, and an emulated-4-device
    cell skips on the machine where the flag worked perfectly.  Measured
    on the WSL dev box (one CudaDevice) while writing distrib_la's L-b
    tier.  ``platform="cpu"`` asks the question the flag actually
    answers.

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
    try:
        have = (jax.device_count() if platform is None
                else len(jax.devices(platform)))
    except Exception as exc:                              # noqa: BLE001
        pytest.skip(f"needs >= {n} devices on platform {platform!r}, which "
                    f"this process has no backend for ({exc})")
    if have < n:
        where = "" if platform is None else f" on platform {platform!r}"
        pytest.skip(
            f"needs >= {n} devices{where}, have {have}.  Set XLA_FLAGS="
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


def _declared_requirements(name: str) -> tuple[str, ...]:
    """What ``name``'s installed distribution says it needs.

    From the METADATA the installer wrote, via stdlib
    ``importlib.metadata`` — no third-party resolver, and no hand-kept
    list.  Environment markers are DROPPED rather than evaluated: a
    requirement that does not apply here simply will not resolve, and
    :func:`dep_dirs` already drops what it cannot find, so evaluating
    markers would add a parser to save nothing.
    """
    try:
        from importlib.metadata import requires as _requires
        reqs = _requires(name) or []
    except Exception:                                          # noqa: BLE001
        return ()
    out = []
    for req in reqs:
        head = req.split(";")[0].strip()
        for sep in ("<", ">", "=", "!", "~", "[", "(", " "):
            head = head.split(sep)[0]
        head = head.strip().replace("-", "_")
        if head:
            out.append(head)
    return tuple(out)


def dep_dirs(names: Iterable[str], *, closure: bool = True) -> list[str]:
    """The directories THIS interpreter resolved ``names`` from.

    NOT ``sysconfig.get_paths()['purelib']``, which is the obvious answer
    for "give the child its dependencies back" and is WRONG on the machine
    that matters.  MEASURED inside the Perlmutter Shifter image
    (``lorrax_J070``, 2026-08-07, via ``lx run python3``)::

        sys.executable  /usr/bin/python3
        purelib         /usr/local/lib/python3.12/dist-packages
        jax.__file__    /opt/jax/jax/__init__.py     <- NOT under purelib
        pytest.__file__ ~/.local/perlmutter/.../site-packages/pytest/...

    The image's jax is a source checkout reached through an editable
    finder hook installed by ``site``, and :func:`import_isolation` runs
    the child under ``python -S``, which skips ``site`` and therefore
    every ``.pth`` and every such hook.  A child handed ``purelib`` had no
    jax at all, the probe died before printing its payload line, and cells
    with nothing to do with isolation failed:

        lx test services/lxkit/tests services/distrib_la/tests
          -> 8 failed, 107 passed, 1 skipped     (2026-08-07, before)

    Asking each dependency where it ACTUALLY lives is right in both places
    — a venv answers ``purelib`` for all of them, the container answers
    ``/opt/jax`` for jax and ``dist-packages`` for the rest — and it keeps
    the child's path a LIST OF NAMES.  That is the load-bearing part:
    naming a dependency is the claim, and anything unnamed must stay
    unreachable, which is what the isolation check measures.

    ``closure=True`` follows each name's DECLARED requirements
    (:func:`_declared_requirements`) transitively.  Also measured, also on
    Perlmutter: naming ``pytest`` is not enough to import pytest.  The
    child got the directory ``_pytest`` lives in and died anyway ---

        File ".../_pytest/_io/terminalwriter.py", line 13, in <module>
          import pygments
        ModuleNotFoundError: No module named 'pygments'

    --- because ``pygments`` is installed somewhere else on that machine.
    Hand-maintaining a package's transitive closure in a test file is a
    losing game with a quiet failure mode, and the installer already
    wrote the answer down.  Set ``closure=False`` to hand over exactly the
    names given, which is what a cell asserting something about a
    SPECIFIC dependency wants.

    Naming a dependency stays the claim either way: the closure of a
    declared dependency is a declared dependency, and the forbidden roots
    are asserted unreachable regardless of how many legitimate library
    directories arrive.

    A name this interpreter cannot resolve is silently dropped: the child
    would not have been able to import it either, and a hard error here
    would turn "scipy is not installed" into a test failure about
    isolation.  Namespace packages are dropped too — they have no single
    directory to hand over.
    """
    out: list[str] = []
    seen: set[str] = set()
    queue = list(names)
    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        if closure:
            queue.extend(r for r in _declared_requirements(name)
                         if r not in seen)
        try:
            spec = importlib.util.find_spec(name)
        except Exception:                                      # noqa: BLE001
            continue
        if spec is None or not spec.origin or spec.origin == "namespace":
            continue
        pkg = os.path.dirname(os.path.realpath(spec.origin))
        root = os.path.dirname(pkg) if os.path.basename(pkg) == name else pkg
        if root not in out:
            out.append(root)
    return out


def import_isolation(pkg: str, forbidden_roots: Iterable[str], *,
                     src_dir: str | None = None,
                     extra_path: Sequence[str] = (),
                     deps: Iterable[str] = (),
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
        leg that needs pytest or jax back asks for them BY NAME via
        ``deps`` — :func:`dep_dirs` resolves each one to the directory it
        actually lives in, which is not ``purelib`` on the machine that
        matters (see :func:`dep_dirs`).  A directory carries no ``.pth``
        processing when it arrives through ``PYTHONPATH``.
    ``PYTHONPATH = src_dir`` (+ ``deps`` + ``extra_path``)
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
    path = list(dict.fromkeys([src_dir, *dep_dirs(deps),
                               *(os.path.realpath(p) for p in extra_path)]))

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
# SKIP-HONESTY.  An unexpected skip is a FAILURE.
# ---------------------------------------------------------------------------
# A skip reads as "not applicable on this machine".  When that is false it
# is the quietest way a suite can lose coverage, and this tree has the
# receipt: on 2026-08-06 nineteen ScaLAPACK/SLATE contract cells reported
# "19 skipped" beside "0 failed" because a built-and-unloadable .so was
# probed as absent.  Nothing in the output distinguished "does not apply"
# from "did not run".
#
# :func:`absent_or_broken` fixes the per-library half of that.  This
# section fixes the per-MACHINE half, in the two directions a skip can lie:
#
#   POSITIVE  the machine DECLARES which backends must be there.  On
#             Perlmutter scalapack, slate, cusolvermp and phdf5 must all
#             probe available; if one does not, that is a defect in the
#             build or the environment and the gate FAILS with the probe's
#             own three-way reason (unknown target / library would not
#             load / loaded but does not export), because those three have
#             three different fixes.
#
#   NEGATIVE  every skip the session actually emitted is diffed against an
#             ALLOWLIST, and each allowed row names the leg that DOES run
#             the cell (the tests/KNOWN_FAILURES.md discipline: a skip
#             with no covering leg is evaporated coverage wearing a green
#             dot).  A skip that matches nothing fails the session.
#
# ...plus a MINIMUM-COLLECTED FLOOR, because both halves above are vacuous
# in a session that collected nothing.  A 38-byte junitxml parses as zero
# tests and zero failures; `lx` reports pytest rc=5 ("no tests collected")
# as its own kind of not-a-pass for the same reason.  Judge by artifacts.
#
# THE `unknown` PROFILE ASSERTS NOTHING, EXPLICITLY.  A laptop has no
# ScaLAPACK and never will, and a profile that demanded one would train
# everybody to ignore the gate.  It is a named row rather than a
# fall-through so that "this machine makes no promises" is a decision in
# the table instead of an accident of lookup.


class MustRow(NamedTuple):
    """One backend a machine PROMISES is available.

    ``target``/``platform`` are the FFI vocabulary of ``service``'s own
    loader, not lxkit's: lxkit owns the policy, each service owns its
    tables.  The gate is handed a probe per service (see
    :func:`assert_must_probe`) and a row whose service has no probe is
    reported as UNASSERTED rather than passing quietly.
    """

    name: str
    service: str
    target: str
    platform: str


class AllowedSkip(NamedTuple):
    """One skip this machine is allowed to emit, and what covers it.

    ``where`` matches a substring of the test's nodeid, ``why`` a
    substring of the skip reason; an empty string matches anything, and
    BOTH must match for the row to apply.  ``covered_by`` names the leg
    that actually runs the cell — required, not decorative: a skip nobody
    can point at a covering run for is lost coverage, and writing the leg
    down is what makes that checkable at review time.
    """

    where: str
    why: str
    covered_by: str


class MachineProfile(NamedTuple):
    """What a machine promises, and what it is allowed to skip."""

    machine: str
    must: tuple[MustRow, ...]
    allowed_skips: tuple[AllowedSkip, ...]
    min_collected: int
    #: True for the ``unknown`` row: both halves become no-ops, loudly.
    asserts_nothing: bool = False

    def must_for(self, service: str) -> tuple[MustRow, ...]:
        return tuple(r for r in self.must if r.service == service)


#: Skips EVERY machine may emit, whatever else it promises.  These are
#: properties of the invocation rather than of the machine: a suite run
#: outside the monorepo cannot exercise the with-lorrax legs, and a suite
#: run in one process cannot exercise a four-process mesh.  Each still
#: names its covering leg.
_UNIVERSAL_SKIPS = (
    AllowedSkip("", "needs >= 4 devices",
                "the service-only leg (`pytest services/<name>/tests`), "
                "whose conftest sets XLA_FLAGS before the first jax import"),
    AllowedSkip("", "needs >= 2 devices", "the same service-only leg"),
    AllowedSkip("", "real multi-process",
                "leg L-c: `lx run -N 1 -G 4 -n 4 python3 -m "
                "<module> --mesh 2x2`"),
    AllowedSkip("", "single-process", "leg L-c (the CLI mode of the "
                "same file, under srun -n 4)"),
    AllowedSkip("", "no lorrax src/", "the monorepo run; a standalone "
                "install has no lorrax to be isolated from"),
    AllowedSkip("", "no monorepo", "the monorepo run"),
    AllowedSkip("", "jax is not installed",
                "any leg with jax; the jax-free legs assert the property "
                "that does not need it"),
)

#: Perlmutter's declared expected-backend profile.  The four MUST rows are
#: the charter's, and they are measurable today: survey 2 verified all
#: eleven ScaLAPACK symbols in LibSci 25.09.0, the SLATE host and device
#: builds, cuSOLVERMp 0.7.2 and the phdf5 handlers at the ARTIFACT level
#: (nm/readelf), and BUILD_NOTES.md pins the host .so those symbols live
#: in.  ``cusolvermp`` is CUDA-only by construction, so its row is
#: asserted on a GPU leg; ``scalapack`` is host-only for the same reason.
_PERLMUTTER = MachineProfile(
    machine="perlmutter",
    must=(
        MustRow("scalapack", "distrib_la", "lorrax_scalapack_eigh", "cpu"),
        MustRow("slate", "distrib_la", "lorrax_slate_potrf", "cpu"),
        MustRow("cusolvermp", "distrib_la", "lorrax_cusolvermp_eigh", "CUDA"),
        # Not distrib_la's library to probe: phdf5 belongs to slab_io,
        # which is not a service yet.  The row is declared HERE because
        # the promise is the MACHINE's, and a gate that only listed rows
        # it could already check would silently shrink to whatever was
        # convenient.  distrib_la's gate reports it UNASSERTED; the
        # slab_io retrofit (charter wave 1b) supplies the probe.
        MustRow("phdf5", "slab_io", "lorrax_phdf5_read", "cpu"),
    ),
    allowed_skips=_UNIVERSAL_SKIPS + (
        AllowedSkip("slate_eigh_true_eigenvectors_cpu", "SIGSEGV",
                    "nothing — bug L-2 is CARRIED, not covered: SLATE's "
                    "host heev faults deterministically and a SIGSEGV "
                    "takes the whole pytest process down.  See "
                    "docs/dev/linalg_ffi.md; the CPU eigh contract is "
                    "covered by the ScaLAPACK cells instead"),
    ),
    min_collected=1,
    asserts_nothing=False,
)

#: The explicit nothing-promised row.  NOT a fall-through.
_UNKNOWN = MachineProfile(
    machine="unknown",
    must=(),
    allowed_skips=(AllowedSkip("", "", "this machine declares no profile"),),
    min_collected=1,
    asserts_nothing=True,
)

PROFILES: Mapping[str, MachineProfile] = {
    "perlmutter": _PERLMUTTER,
    "unknown": _UNKNOWN,
}


def detect_machine() -> str:
    """Which profile this process is running under.

    ``LX_MACHINE`` wins so a leg can say what it is (and so the red twins
    can name a machine that does not exist); ``NERSC_HOST`` is the site's
    own variable and is what makes a Perlmutter run self-identifying with
    no configuration.  Anything unrecognized is ``unknown``, which asserts
    nothing — the honest answer for a machine nobody has measured.
    """
    for var in ("LX_MACHINE", "NERSC_HOST"):
        val = (os.environ.get(var) or "").strip().lower()
        if val:
            return val
    return "unknown"


def machine_profile(machine: str | None = None) -> MachineProfile:
    """The declared profile for ``machine`` (default: this machine).

    An unrecognized name resolves to :data:`PROFILES` ``['unknown']`` — a
    row that asserts nothing and says so — rather than raising, because a
    new NERSC host appearing must not turn every service suite red.
    """
    name = (machine or detect_machine()).strip().lower()
    return PROFILES.get(name, _UNKNOWN._replace(machine=name))


def must_rows_for(service: str, machine: str | None = None):
    return machine_profile(machine).must_for(service)


# --- positive half ---------------------------------------------------------

def assert_must_probe(profile: MachineProfile,
                      probes: Mapping[str, Callable[[str, str], object]],
                      *, service: str | None = None,
                      fail: Callable[..., None] | None = None) -> list:
    """FAIL when a MUST row of ``profile`` does not probe available.

    ``probes`` maps a service name to its ``probe_target(target,
    platform) -> (ok, reason)``.  The reason is quoted VERBATIM into the
    failure: it is the three-way taxonomy (unknown target / library would
    not load / loaded but does not export the symbol), and those three
    have three different fixes, so collapsing them into "unavailable"
    throws away the only part of the message anybody can act on.

    ``pytrace=False``: the traceback would be this function, which is
    never the interesting frame.  The interesting frame is a build script.

    Returns the rows it could not assert (no probe for that service), so
    the caller can report the GAP rather than let it pass as a success.
    """
    fail = fail or (lambda msg, **kw: pytest.fail(msg, pytrace=False))
    if profile.asserts_nothing:
        return []
    rows = profile.must if service is None else profile.must_for(service)
    unasserted, broken = [], []
    for row in rows:
        probe = probes.get(row.service)
        if probe is None:
            unasserted.append(row)
            continue
        ok, reason = probe(row.target, row.platform)
        if not ok:
            broken.append((row, reason))
    if broken:
        lines = [f"machine profile {profile.machine!r} declares "
                 f"{len(rows)} backend(s) that MUST be available here; "
                 f"{len(broken)} is/are not.  An unavailable MUST row is a "
                 f"DEFECT in this machine's build or environment, not a "
                 f"platform this suite does not apply to."]
        for row, reason in broken:
            lines.append(f"\n  {row.name}  ({row.service}: target "
                         f"{row.target!r} on platform {row.platform!r})\n"
                         f"    {reason}")
        fail("\n".join(lines))
    return unasserted


# --- negative half ---------------------------------------------------------

class ObservedSkip(NamedTuple):
    nodeid: str
    reason: str
    path: str = ""          #: absolute file the cell came from, when known


def _skip_is_allowed(skip: ObservedSkip, allowed) -> AllowedSkip | None:
    for row in allowed:
        if row.where in skip.nodeid and row.why in skip.reason:
            return row
    return None


def assert_skips_match_profile(skips: Sequence[ObservedSkip],
                               profile: MachineProfile, *,
                               collected: int,
                               extra_allowed: Sequence[AllowedSkip] = (),
                               min_collected: int | None = None) -> None:
    """Raise ``AssertionError`` when the session's skips break the profile.

    PURE on purpose: it takes the observed skips and returns/raises, so it
    is testable without a pytest session — which is what lets its red twin
    be a direct call rather than an elaborate fixture.  The pytest wiring
    (:func:`arm_skip_honesty`) is thirty lines on top of this.

    The floor is checked FIRST.  A session that collected nothing has no
    skips to diff, so every other assertion here would pass vacuously —
    that is the 38-byte-junitxml failure mode, and checking it last would
    reproduce it.
    """
    floor = profile.min_collected if min_collected is None else min_collected
    if collected < floor:
        raise AssertionError(
            f"skip-honesty: the session collected {collected} test(s), "
            f"below the floor of {floor}.  A session that collected "
            f"nothing skips nothing and fails nothing, so every other "
            f"check here would pass while measuring zero coverage.  "
            f"(A 38-byte junitxml parses as 0 tests and 0 failures; "
            f"pytest rc=5 is not a pass.)")
    if profile.asserts_nothing:
        return
    allowed = tuple(profile.allowed_skips) + tuple(extra_allowed)
    bad = [s for s in skips if _skip_is_allowed(s, allowed) is None]
    if bad:
        lines = [f"skip-honesty: machine profile {profile.machine!r} does "
                 f"not allow {len(bad)} of the {len(skips)} skip(s) this "
                 f"session emitted.  On this machine an unexpected skip is "
                 f"a FAILURE: it is how coverage evaporates without "
                 f"anything turning red."]
        for s in bad:
            lines.append(f"\n  {s.nodeid}\n    reason: {s.reason.strip()}")
        lines.append(
            f"\nIf the skip is legitimate, add an lxkit.testing.AllowedSkip "
            f"row NAMING THE LEG that runs the cell instead — a skip with "
            f"no covering leg is lost coverage, and the allowlist is where "
            f"that claim is written down.  Allowed here: "
            f"{[(r.where or '*', r.why or '*') for r in allowed]}")
        raise AssertionError("\n".join(lines))


# --- the pytest wiring -----------------------------------------------------
# The hooks are ALWAYS collecting and NEVER asserting until a suite arms
# them.  Recording a skip costs a list append; asserting on a session the
# author never declared a profile for would make lxkit's plugin fail other
# people's suites, which is how a good gate gets deleted.

_OBSERVED: list = []
_ROOTDIR: dict = {}


class ArmedScope(NamedTuple):
    """One service's armed gate.  See :data:`_ARMED`."""

    scope: str                      #: realpath of the directory, "" = unscoped
    profile: MachineProfile
    extra_allowed: tuple
    min_collected: int | None


# ---------------------------------------------------------------------------
# ONE ROW PER SCOPE, NOT ONE SLOT.  This is the cross-service process-state
# discipline (POST_WAVE_CLEANUP item 2).
# ---------------------------------------------------------------------------
# ``_ARMED`` used to be a single dict that ``arm_skip_honesty`` overwrote, so
# a second calling service did not add a second gate — it REPLACED the first
# one's scope and allowlist.  MEASURED at the merged head, full ``services/``
# run on WSL: distrib_la, symmetry_maps and vcoul all arm unconditionally and
# conftests load in directory order, so the surviving scope was VCOUL's and
# BOTH of the other two gates were silently inert.  wfn_loader's
# ``test_this_gate_did_not_disarm_distrib_las`` is the cell that says so out
# loud, and it was RED.
#
# Two services then paid for the single slot in their own source: zeta_loader
# arms only when the slot is free and otherwise re-implements lxkit's
# ``pytest_sessionfinish`` locally, and wfn_loader gave up on lxkit's gate
# entirely and hand-rolled a service-local collector.  Both wrote down that
# the real fix is a LIST of scopes and that lxkit was frozen that wave.  This
# is that fix: services register under their OWN directory and can no more
# see each other's rows than they can see each other's fixtures.
#
# Keyed by realpath, so re-arming the SAME scope (symmetry_maps does this from
# ``pytest_collection_modifyitems`` once the invocation is known) updates one
# row instead of appending a stale duplicate.
_ARMED: dict = {}

#: Per scope: nodeids seen in ANY phase whose file is under it.  Needed
#: because the pytest-xdist CONTROLLER has an empty ``session.items`` (the
#: workers did the collecting) while every report still passes through its
#: ``pytest_runtest_logreport`` — see :func:`pytest_sessionfinish`.
_SEEN_IN_SCOPE: dict = {}

#: ``(nodeid, when)`` pairs already recorded.  THE HOOKS ARE REGISTERED ONCE
#: PER CONFTEST THAT IMPORTS THEM, and several do; pytest registers hook
#: implementations per plugin MODULE, so the same function object arriving
#: through three conftests is called three times for every report.  That used
#: to double- and triple-count every skip, which is why the old docstrings
#: told services NOT to import the hooks — a rule that cannot be enforced and
#: was the reason two suites forked the sessionfinish body.  Deduping here
#: makes N registrations exactly equivalent to one, so importing the hooks is
#: safe and the rule can go.
_RECORDED: set = set()

#: Sessions whose sessionfinish has already run, same reason.
_FINISHED: set = set()


def observed_skips() -> tuple:
    """Every skip this pytest session has emitted so far."""
    return tuple(_OBSERVED)


def armed_scopes() -> tuple:
    """Every :class:`ArmedScope` armed in this session, in arming order."""
    return tuple(_ARMED.values())


def arm_skip_honesty(profile: MachineProfile | None = None, *,
                     scope: str = "",
                     extra_allowed: Sequence[AllowedSkip] = (),
                     min_collected: int | None = None) -> MachineProfile:
    """Turn the negative half on for THIS session.  Call from a conftest.

    One call per service suite — the charter's "one gate per service" — and
    services do NOT disarm each other: each registers a row under its own
    ``scope`` (see :data:`_ARMED`), and :func:`pytest_sessionfinish` runs
    every row.  Calling it twice for the same scope re-arms that row only.

    Returns the profile in force so the caller can report it.

    ``scope`` is a DIRECTORY, and passing it is close to mandatory in a
    monorepo.  A service's allowlist describes the SERVICE's skips; without
    a scope the gate would also judge every skip the surrounding suite
    emitted — on Perlmutter that is ~45 device-count skips in lorrax's own
    tests, none of which this service has any business ruling on, and the
    gate would fail for reasons that are not about it.  A gate that fires
    on other people's business is a gate that gets disarmed.

    A DIRECTORY and not a nodeid prefix, because nodeids are relative to
    whatever pytest picked as rootdir and that changes with the
    invocation: ``pytest services/distrib_la/tests`` from the monorepo
    root resolves rootdir to the SERVICE (its pyproject.toml is the
    nearest one above the argument), so the same cell is
    ``services/distrib_la/tests/test_x.py::t`` in one run and
    ``tests/test_x.py::t`` in the next.  A prefix match would have
    silently matched nothing on the standalone invocation — the gate would
    have reported "not selected, inert" and asserted zero, which is the
    quiet-failure shape this whole section exists to remove.  The
    directory is resolved against the session's collected items, whose
    paths are absolute.
    """
    prof = profile or machine_profile()
    key = os.path.realpath(scope) if scope else ""
    _ARMED[key] = ArmedScope(key, prof, tuple(extra_allowed), min_collected)
    return prof


def pytest_configure(config):
    """Remember the rootdir, so a report's path can be made absolute.

    ``report.fspath`` is ROOTDIR-RELATIVE and a report is all the
    pytest-xdist controller ever sees, so without this the controller
    cannot tell whether a skip came from the armed scope.
    """
    _ROOTDIR["path"] = str(getattr(config, "rootpath", "")
                           or getattr(config, "rootdir", "") or "")


def _report_path(report) -> str:
    loc = getattr(report, "fspath", "") or ""
    if not loc:
        return ""
    root = _ROOTDIR.get("path", "")
    return os.path.realpath(os.path.join(root, str(loc)) if root
                            else str(loc))


def pytest_runtest_logreport(report):
    """Record every skip, wherever it came from.

    ``report.longrepr`` for a skip is ``(file, lineno, "Skipped: <reason>")``
    — the reason as the REPORTER saw it, which is what a human reads and
    therefore what the allowlist must match against.  Reaching into
    ``item.own_markers`` instead would miss every in-body ``pytest.skip``,
    and in-body is where this tree's device-count and library skips live.
    """
    # IDEMPOTENT across registrations — see :data:`_RECORDED`.
    stamp = (report.nodeid, getattr(report, "when", ""))
    if stamp in _RECORDED:
        return
    _RECORDED.add(stamp)
    if report.skipped:
        longrepr = getattr(report, "longrepr", None)
        reason = ""
        if isinstance(longrepr, tuple) and len(longrepr) == 3:
            reason = str(longrepr[2])
        elif longrepr is not None:
            reason = str(longrepr)
        if reason.startswith("Skipped: "):
            reason = reason[len("Skipped: "):]
        _OBSERVED.append(ObservedSkip(report.nodeid, reason,
                                      _report_path(report)))
    path = _report_path(report)
    if not path:
        return
    for key in _ARMED:
        if key and path.startswith(key + os.sep):
            _SEEN_IN_SCOPE.setdefault(key, set()).add(report.nodeid)


def check_armed_scope(row: "ArmedScope", session) -> str:
    """Run ONE armed row against this session.  ``""`` when it is satisfied.

    Split out of :func:`pytest_sessionfinish` so the per-row decision is a
    plain function: it is what makes "several services armed at once" a loop
    instead of a second policy, and it is directly callable from a test.
    """
    total = int(getattr(session, "testscollected", 0) or 0)
    if row.scope:
        root = row.scope + os.sep
        items = getattr(session, "items", None) or []
        if items:
            mine = {str(getattr(it, "nodeid", "")) for it in items
                    if os.path.realpath(str(getattr(it, "path", None)
                                            or getattr(it, "fspath", ""))
                                        ).startswith(root)}
            skips = [s for s in observed_skips() if s.nodeid in mine]
            in_scope = len(mine)
        else:
            # THE pytest-xdist CONTROLLER.  Its session.items is EMPTY --
            # the workers did the collecting -- so an items-only scope
            # made this gate silently inert under `-n 4`, which is how
            # `lx test` runs every suite on Perlmutter.  A skip-honesty
            # gate that evaporates on the machine it was written for is
            # the joke telling itself; MEASURED 2026-08-07, where a
            # cusolvermp-needs-a-true-2D-mesh skip sailed through an
            # allowlist that did not contain it.  Every report still
            # reaches this process, so scope by the report's own path.
            skips = [s for s in observed_skips() if s.path.startswith(root)]
            in_scope = len(_SEEN_IN_SCOPE.get(row.scope, ()))
    else:
        skips, in_scope = list(observed_skips()), total

    # DELIBERATELY DESELECTED is not the same as EMPTY.  A run that said
    # --no-services, -m something-else or -k narrowed this suite away
    # collected other things; failing its floor would turn a deliberate
    # narrowing into a red, and this tree has exactly the deselection
    # switches that make that a daily occurrence.  A session that
    # collected NOTHING AT ALL still trips the floor, which is the case
    # the floor is actually about.
    if in_scope == 0 and total > 0:
        return ""
    try:
        assert_skips_match_profile(
            skips, row.profile, collected=in_scope,
            extra_allowed=row.extra_allowed,
            min_collected=row.min_collected)
    except AssertionError as exc:                              # noqa: BLE001
        return str(exc)
    return ""


def pytest_sessionfinish(session, exitstatus):
    """Run the negative half for EVERY armed scope.

    A violation is reported to the terminal AND turns the session red.
    Both: the exit status is what CI reads, the text is what a human acts
    on, and this tree's standing rule is to judge by the artifact.

    Every row runs even when an earlier one failed: the rows belong to
    different services, and reporting only the first would make the second
    service's defect invisible until the first was fixed.

    Idempotent across registrations (:data:`_RECORDED`): several conftests
    import this function, pytest registers each conftest as its own plugin,
    and without the guard the gate would run once per importer and print
    every violation that many times.
    """
    stamp = id(session)
    if stamp in _FINISHED:
        return
    _FINISHED.add(stamp)
    failures = [(row, msg) for row in _ARMED.values()
                if (msg := check_armed_scope(row, session))]
    if not failures:
        return
    for row, msg in failures:
        title = f"SKIP-HONESTY GATE FAILED ({row.scope or 'unscoped'})"
        try:
            tw = session.config.get_terminal_writer()
            tw.line("")
            tw.line(title, red=True, bold=True)
            for line in msg.splitlines():
                tw.line(line, red=True)
        except Exception:                                      # noqa: BLE001
            print(f"{title}\n{msg}")
    session.exitstatus = 1
