"""``--help`` and a bad flag must not start a distributed runtime.

Every driver runs ``initialize_communicator_stack`` at MODULE scope, so
before the seam that :mod:`runtime.cli_seam` introduces, asking a driver
to describe itself paid the whole bring-up -- and on a box whose FFI
library is not built it printed no usage at all, exiting 1 at
``_enforce_required_ffi``.

Two independent halves, because either alone is defeatable:

  * STATIC.  The seam call exists at module scope ABOVE the startup call.
    AST, not grep: a comment saying "we answer --help early" is not the
    call being made.
  * BEHAVIOURAL.  The driver is actually launched.  The FFI library is
    pointed at a path that does not exist, so ANY process that reaches
    the runtime cannot exit 0 -- which is what makes "exit 0 with usage"
    proof that the runtime did not run, rather than a hope.

Both halves carry their falsifying twin: a bad flag must exit 2 (not
sail past into a run), a GOOD argv must still start the runtime (the
seam must not swallow launches), and a driver on the debt list below
must FAIL the behavioural assertion -- which is what shows the
assertion can fail at all.

THE SEAM'S THREE PRECONDITIONS ARE GATED TOO.  Each is one edit away
from being lost, and none of them announces itself by failing a driver:

  * the seam call sits under ``if __name__ == "__main__":``.  A driver
    imported as a LIBRARY must never consult argv -- ``bse.exciton_bands``
    imports ``bandstructure.htransform`` and ``gw.sigma_dispatch`` imports
    ``gw.kin_ion_io``, both with argv belonging to a different program.
  * nothing above the startup call imports jax, directly or through a
    module that does.  jax latches x64 and the platform at ITS import, so
    an import that beats ``runtime.set_default_env`` to it silently costs
    the run its 64-bit values (measured 2026-08-27; see
    ``runtime.own_x64_on_a_live_jax``).
  * the SEAMED/debt census is DERIVED from ``src/`` rather than listed, so
    a driver that lands unseamed turns this file red instead of simply
    being absent from a table nobody re-derives.
"""
from __future__ import annotations

import ast
import functools
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")

#: Drivers whose parser needs nothing but ``argparse``, so it can be
#: defined above the startup call.  These carry the seam.
SEAMED = {
    "gw.gw_jax": "gw/gw_jax.py",
    "gw.kin_ion_io": "gw/kin_ion_io.py",
    "gw.downfold_cli": "gw/downfold_cli.py",
    "centroid.kmeans_cli": "centroid/kmeans_cli.py",
}

#: DEBT, and the reason each one is debt rather than an omission: the
#: parser's ``choices=``/defaults come from a module that imports jax, so
#: hoisting the parser above the startup call would drag jax above the
#: env defaults -- the exact ordering the startup call exists to protect.
#: Measured 2026-08-27 by importing the named module in a fresh
#: interpreter and checking ``'jax' in sys.modules``.
#:
#: THE TARGET IS ZERO AND THIS COUNT MUST ONLY GO DOWN.  Closing one of
#: these means splitting the jax-free constants out of the named module,
#: which is a decision about that module, not about the driver.
NO_SEAM_DEBT = {
    "bse.bse_jax": ("bse/bse_jax.py", "common.band_degeneracy"),
    "bse.exciton_bands": ("bse/exciton_bands.py", "gw.gw_config"),
    "bandstructure.htransform": ("bandstructure/htransform.py", "gw.gw_config"),
    "psp.get_dipole_mtxels": ("psp/get_dipole_mtxels.py", "common.mtxel_sweep"),
}

#: A RATCHET, NOT A CEILING, and the difference is the whole point: with
#: ``<=`` here, seaming a driver and forgetting to lower this number left the
#: recorded target above the truth, and the next unseamed driver could land
#: back into the gap without a red cell.  ``==`` means closing debt is not
#: finished until this constant is edited, which is the act that records it.
DEBT_TARGET = 4


def _source(rel: str) -> str:
    with open(os.path.join(_SRC, rel), encoding="utf-8") as fh:
        return fh.read()


def _dotted(path: str) -> str:
    """``<src>/gw/gw_jax.py`` -> ``gw.gw_jax``; packages -> their package."""
    rel = os.path.relpath(path, _SRC)[:-3].replace(os.sep, ".")
    return rel[: -len(".__init__")] if rel.endswith(".__init__") else rel


def _startup_census() -> dict:
    """``{dotted: [lineno]}`` for EVERY module under ``src/`` that calls
    ``initialize_communicator_stack`` at module scope.

    DERIVED, never listed.  A hand-kept roster of drivers can only fail when
    a developer edits it, which is the one moment it is least likely to be
    wrong; this walks the tree instead, so a driver that lands next week is
    in the census the day it lands.
    """
    out = {}
    for dirpath, dirnames, filenames in os.walk(_SRC):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            try:
                hits = _module_scope_call_lines(
                    text, "initialize_communicator_stack")
            except SyntaxError:
                continue
            if hits:
                out[_dotted(path)] = hits
    return out


def _module_scope_call_lines(source: str, name: str) -> list:
    """Line numbers of module-scope calls to ``name``. Recursion stops at defs.

    ``if __name__ == "__main__":`` at module scope IS module scope, which
    is why this walks ``If``/``Try``/``With`` but not ``FunctionDef``.
    """
    out = []

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            return

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

        def visit_Call(self, node):
            fn = node.func
            spelled = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if spelled == name:
                out.append(node.lineno)
            self.generic_visit(node)

    V().visit(ast.parse(source))
    return sorted(out)


# ---------------------------------------------------------------------------
# 1.  STATIC — the seam is above the startup call
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", sorted(SEAMED))
def test_the_seam_runs_before_the_startup_call(mod):
    src = _source(SEAMED[mod])
    seam = _module_scope_call_lines(src, "refuse_bad_argv_before_startup")
    start = _module_scope_call_lines(src, "initialize_communicator_stack")
    assert seam, (
        f"{mod} has no module-scope refuse_bad_argv_before_startup() call. "
        f"A comment about answering --help early is not the call.")
    assert start, f"{mod} no longer initializes the runtime at module scope"
    assert max(seam) < min(start), (
        f"{mod}: seam at {seam}, startup at {start}.  The seam is BELOW the "
        f"startup call, so --help still pays the bring-up.")


@pytest.mark.parametrize("mod", sorted(SEAMED))
def test_the_seam_and_main_share_one_parser_factory(mod):
    """No second source of truth: both call the same factory by name."""
    tree = ast.parse(_source(SEAMED[mod]))
    factories = {n.name for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)
                 and n.name in ("build_parser", "build_argparser")}
    assert len(factories) == 1, (
        f"{mod} declares {sorted(factories)} parser factories; the seam and "
        f"main() must share exactly one")
    factory = factories.pop()
    called = _module_scope_call_lines(_source(SEAMED[mod]), factory)
    assert called, f"{mod}'s seam does not call {factory}()"
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    in_main = [n for n in ast.walk(main) if isinstance(n, ast.Call)
               and (getattr(n.func, "id", None) == factory
                    or getattr(n.func, "attr", None) == factory)]
    assert in_main, (
        f"{mod}.main() builds its parser some other way; the seam would then "
        f"be validating argv against a DIFFERENT parser than the run uses")


def _parent_map(tree):
    return {child: parent for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)}


def _is_main_guard(test) -> bool:
    """``__name__ == "__main__"``, written either way round."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)):
        return False
    sides = [test.left, test.comparators[0]]
    names = {n.id for n in sides if isinstance(n, ast.Name)}
    consts = {n.value for n in sides if isinstance(n, ast.Constant)}
    return "__name__" in names and "__main__" in consts


_BLOCK_NODES = (ast.If, ast.Try, ast.With, ast.For, ast.While, ast.Module,
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@pytest.mark.parametrize("mod", sorted(SEAMED))
def test_the_seam_call_is_under_a_main_guard(mod):
    """THE GUARD IS LOAD-BEARING, NOT DECORATION.

    Dedent the seam out of its guard and every cell in this file still
    passes (measured), while ``import gw.kin_ion_io`` from any program with
    its own argv dies at ``argparse`` -- ``gw.sigma_dispatch`` imports that
    module, ``bse.exciton_bands`` imports ``bandstructure.htransform``, and
    neither owns the argv the seam would parse.  The rule the docstring at
    ``runtime/cli_seam.py:56-60`` states, as a check.
    """
    tree = ast.parse(_source(SEAMED[mod]))
    parents = _parent_map(tree)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and (getattr(n.func, "id", None)
                  or getattr(n.func, "attr", None))
             == "refuse_bad_argv_before_startup"]
    assert calls, f"{mod} has no refuse_bad_argv_before_startup() call at all"
    for call in calls:
        node = parents.get(call)
        while node is not None and not isinstance(node, _BLOCK_NODES):
            node = parents.get(node)
        assert isinstance(node, ast.If) and _is_main_guard(node.test), (
            f"{mod}: the seam call at line {call.lineno} is not directly "
            f"under `if __name__ == \"__main__\":` (nearest enclosing block is "
            f"{type(node).__name__}).  Importing this driver as a LIBRARY "
            f"would then parse somebody else's argv and SystemExit inside "
            f"their import.")


def _module_scope_imports(source: str) -> list:
    """``[(lineno, absolute module name)]`` for module-scope imports.

    Same scope rule as :func:`_module_scope_call_lines`: ``if``/``try``/
    ``with`` bodies count (they run on import), ``def``/``class`` bodies do
    not.  Relative imports are skipped -- they cannot name jax.
    """
    out = []

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            return

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

        def visit_Import(self, node):
            out.extend((node.lineno, a.name) for a in node.names)

        def visit_ImportFrom(self, node):
            if not node.level and node.module:
                out.append((node.lineno, node.module))

    V().visit(ast.parse(source))
    return out


@pytest.mark.parametrize("mod", sorted(SEAMED))
def test_no_module_scope_import_above_the_startup_call_names_jax(mod):
    """THE CONSTRAINT THE WHOLE SEAM RESTS ON (``runtime/cli_seam.py:29-34``).

    jax latches x64 and the platform when IT is imported.  One ``import
    jax`` above the startup call therefore costs the run its 64-bit values
    with nothing on screen -- and every one of the four debt entries below
    exists because its parser needs a jax-importing module, so whoever
    closes that debt is one line away from doing exactly this.

    STATIC because it must fire on the spelling, in milliseconds, on a box
    with no runtime; the transitive half (a module above the seam that
    pulls jax indirectly) is measured by
    :func:`test_nothing_above_the_startup_call_pulls_jax`.
    """
    src = _source(SEAMED[mod])
    start = min(_module_scope_call_lines(src, "initialize_communicator_stack"))
    guilty = [(ln, name) for ln, name in _module_scope_imports(src)
              if ln < start and name.split(".")[0] in ("jax", "jaxlib")]
    assert not guilty, (
        f"{mod} imports jax at module scope ABOVE the startup call at line "
        f"{start}: {guilty}.  jax reads JAX_ENABLE_X64/JAX_PLATFORMS at its "
        f"own import, so this silently drops the run to float32.")


def test_the_census_of_drivers_is_derived_from_the_tree():
    """A DRIVER THAT LANDS UNSEAMED MUST TURN THIS FILE RED.

    Both tables below are hand-written, so on their own they can only fail
    when someone edits them.  This re-derives the population — every module
    under ``src/`` that calls ``initialize_communicator_stack`` at module
    scope — and requires the two tables to account for exactly it.  Same
    shape as ``tests/test_service_path_bootstrap.py``'s
    ``_ALL_MODULE_SCOPE_DOOR_CONSUMERS`` cell, for the same reason.
    """
    census = set(_startup_census())
    listed = set(SEAMED) | set(NO_SEAM_DEBT)
    assert census == listed, (
        f"drivers in the tree but in neither table: {sorted(census - listed)}\n"
        f"tabled but no longer a module-scope driver: {sorted(listed - census)}\n"
        f"A new driver belongs in SEAMED (with the seam) or in NO_SEAM_DEBT "
        f"(with the jax-importing module that blocks it, and DEBT_TARGET "
        f"raised only by the owner).")


def test_the_debt_list_is_exactly_the_unseamed_drivers():
    """Two-sided.  A new unseamed driver fails; so does closing one silently."""
    seamed_now, unseamed_now = set(), set()
    for mod, rel in list(SEAMED.items()) + [
            (m, v[0]) for m, v in NO_SEAM_DEBT.items()]:
        src = _source(rel)
        if _module_scope_call_lines(src, "refuse_bad_argv_before_startup"):
            seamed_now.add(mod)
        else:
            unseamed_now.add(mod)
    assert seamed_now == set(SEAMED), (
        f"seam present on {sorted(seamed_now)} but the table says "
        f"{sorted(SEAMED)}")
    assert unseamed_now == set(NO_SEAM_DEBT), (
        f"unseamed drivers are {sorted(unseamed_now)} but the debt table says "
        f"{sorted(NO_SEAM_DEBT)}")
    assert len(NO_SEAM_DEBT) == DEBT_TARGET, (
        f"{len(NO_SEAM_DEBT)} drivers cannot answer --help without a runtime "
        f"but the recorded target is {DEBT_TARGET}.  Closing one means LOWERING "
        f"this constant in the same edit; raising it is an owner decision.")


@functools.lru_cache(maxsize=None)
def _import_pulls_jax(dotted: str) -> bool:
    """Does a bare ``import <dotted>`` leave jax in ``sys.modules``?

    MEASURED, in a fresh interpreter, because that is the claim: this test
    process has imported jax long before it reads this file, so an
    in-process check would answer True for every module in the tree.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p])
    p = subprocess.run(
        [sys.executable, "-c",
         f"import sys, {dotted}; print('JAX', 'jax' in sys.modules)"],
        capture_output=True, text=True, timeout=300, env=env)
    assert p.returncode == 0, (
        f"import {dotted} failed, so the debt reason cannot be measured:\n"
        f"{p.stderr[-2000:]}")
    return "JAX True" in p.stdout


@pytest.mark.parametrize("mod", sorted(NO_SEAM_DEBT))
def test_each_debt_entry_names_a_module_that_really_imports_jax(mod):
    """The debt's REASON has to be true, or the list is an excuse list.

    Two halves, and the name of this test is the DYNAMIC one: the named
    module is imported in a fresh interpreter and must actually pull jax.
    The static half (the driver still imports it) is what makes the entry
    about THIS driver rather than about a module it stopped using.
    """
    rel, blocker = NO_SEAM_DEBT[mod]
    assert _import_pulls_jax(blocker), (
        f"{mod} is on the debt list because importing {blocker} pulls jax "
        f"above the startup call -- but a fresh interpreter importing "
        f"{blocker} does NOT load jax any more.  The seam is now possible: "
        f"move {mod} to SEAMED and lower DEBT_TARGET.")
    tree = ast.parse(_source(rel))
    targets = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            targets.add(n.module)
        elif isinstance(n, ast.Import):
            targets.update(a.name for a in n.names)
    assert blocker in targets, (
        f"{mod} is on the debt list because of {blocker}, but {blocker} is "
        f"not imported there any more -- re-check whether the seam is now "
        f"possible instead of leaving the entry")


def test_the_jax_import_measurement_can_answer_no():
    """THE CONTROL for the cell above.  ``argparse`` is what every seamed
    parser needs and nothing more; a probe that answered "pulls jax" for it
    would make all four debt reasons above vacuous."""
    assert not _import_pulls_jax("argparse")


# ---------------------------------------------------------------------------
# 2.  BEHAVIOURAL — launch the driver with the FFI deliberately missing
# ---------------------------------------------------------------------------

def _launch(mod, *args):
    """Run ``python -m mod args`` with an FFI path that does not exist.

    That is the load-bearing part: `_enforce_required_ffi` (step 6b of the
    startup call) refuses on an unloadable library, so a process that
    reaches the runtime CANNOT exit 0.  ``returncode == 0`` is therefore
    the whole proof here, and it is a sharp one.

    THERE IS DELIBERATELY NO "the banner did not print" ASSERTION.  It
    would be a tautology inside this harness: the FFI refusal is step 6b
    and the startup report is step 8, so under a broken ``.so`` no process
    can print the banner however far it got -- measured 2026-08-27, two
    drivers that DO reach the runtime (``psp.get_dipole_mtxels``,
    ``bse.bse_jax``) exit 1 with zero occurrences of "LORRAX runtime".
    A check that cannot fail is not evidence (TASTE), and it would also
    couple this file to the report's prose.
    """
    env = dict(os.environ)
    env["LORRAX_FFI_SO"] = "/nonexistent/liblorrax_ffi.so"
    env["LORRAX_FFI_HOST_SO"] = "/nonexistent/liblorrax_ffi_host.so"
    env["LORRAX_DEBUG_PRINT"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [_SRC, os.path.join(_REPO, "services", "distrib_la", "src")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return subprocess.run([sys.executable, "-m", mod, *args],
                          capture_output=True, text=True, timeout=300, env=env)


_LIBRARY_IMPORT_PROBE = """
import sys, importlib
import runtime


class Reached(Exception):
    pass


def _stub(*a, **k):
    raise Reached('jax' in sys.modules)


runtime.initialize_communicator_stack = _stub
sys.argv = [{mod!r}, '--definitely-not-a-flag', '--help']
try:
    importlib.import_module({mod!r})
except Reached as exc:
    print('REACHED_STARTUP jax_loaded=%s' % exc.args[0])
except SystemExit as exc:
    print('SYSTEMEXIT code=%r' % (exc.code,))
else:
    print('NO_STARTUP_CALL')
"""


@functools.lru_cache(maxsize=None)
def _library_import(mod: str) -> str:
    """Import ``mod`` AS A LIBRARY, under an argv that belongs to nobody.

    The startup call is replaced by a sentinel, so this needs no FFI and no
    devices: it measures exactly the module body ABOVE the startup call.
    Reports whether that body consulted argv (``SYSTEMEXIT``) and whether
    it had already pulled jax when the startup call was reached.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p])
    p = subprocess.run(
        [sys.executable, "-c", _LIBRARY_IMPORT_PROBE.format(mod=mod)],
        capture_output=True, text=True, timeout=300, env=env)
    assert p.returncode == 0, (
        f"the library-import probe for {mod} did not finish:\n"
        f"{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p.stdout.strip()


@pytest.mark.parametrize("mod", sorted(SEAMED))
def test_a_library_import_never_consults_argv(mod):
    """THE BEHAVIOURAL TWIN of the ``__main__``-guard cell above."""
    out = _library_import(mod)
    assert out.startswith("REACHED_STARTUP"), (
        f"importing {mod} as a library under a hostile argv answered {out!r}. "
        f"`SYSTEMEXIT` means the seam ran outside its `__main__` guard and "
        f"argparse ended somebody else's program; `NO_STARTUP_CALL` means "
        f"this probe no longer reaches the thing it claims to measure.")


@pytest.mark.parametrize("mod", sorted(SEAMED))
def test_nothing_above_the_startup_call_pulls_jax(mod):
    """The TRANSITIVE half of the static jax cell: measured, not parsed.

    A module above the startup call that imports jax itself is invisible to
    the AST scan and just as fatal -- that is precisely why the four debt
    entries are debt.
    """
    out = _library_import(mod)
    assert out == "REACHED_STARTUP jax_loaded=False", (
        f"{mod}: {out}.  Something imported above the startup call has "
        f"already pulled jax, so jax read JAX_ENABLE_X64/JAX_PLATFORMS "
        f"before runtime.set_default_env could set them.")


@pytest.mark.parametrize("mod", sorted(SEAMED))
def test_help_exits_zero_without_a_runtime(mod):
    p = _launch(mod, "--help")
    assert p.returncode == 0, (
        f"{mod} --help exited {p.returncode}.  With no loadable FFI that "
        f"means it reached the runtime.\n{p.stderr[-2000:]}")
    assert p.stdout.startswith("usage:"), (
        f"{mod} --help printed no usage first:\n{p.stdout[:500]}")


@pytest.mark.parametrize("mod", sorted(SEAMED))
def test_a_bad_flag_is_refused_without_a_runtime(mod):
    """RED TWIN.  A rejected argv must exit 2 here, not sail into a run."""
    p = _launch(mod, "--definitely-not-a-flag")
    assert p.returncode == 2, (
        f"{mod} --definitely-not-a-flag exited {p.returncode}, want argparse's "
        f"2\n{p.stderr[-2000:]}")


def test_the_seam_does_not_swallow_a_good_argv():
    """RED TWIN.  Acceptable argv must fall through into the real startup.

    Uses the FFI-less launcher on purpose: the run is EXPECTED to fail at
    the FFI gate, and that failure is the evidence the seam let it past.
    """
    p = _launch("gw.downfold_cli", "--print-schema")
    assert p.returncode != 0, (
        "downfold_cli --print-schema exited 0 with no loadable FFI; the seam "
        "answered a launch it should have passed through")
    assert not p.stdout.startswith("usage:")


@pytest.mark.parametrize("mod", sorted(NO_SEAM_DEBT))
def test_the_behavioural_assertion_can_fail(mod):
    """THE CONTROL.  An unseamed driver must NOT pass the check above.

    Without this, `test_help_exits_zero_without_a_runtime` could be
    asserting something every driver in the tree already satisfies.
    """
    p = _launch(mod, "--help")
    assert not (p.returncode == 0 and p.stdout.startswith("usage:")), (
        f"{mod} is on the debt list but answers --help without a runtime; "
        f"move it to SEAMED and lower DEBT_TARGET")
