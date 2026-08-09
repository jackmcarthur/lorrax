"""No LORRAX module may inject an out-of-repo absolute path into sys.path.

RED TWIN for completeness-audit P21.

``common.jax_profile.profile_section`` inserted
``/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling`` at **sys.path[0]**
of every LORRAX process that ran a GN-PPM Sigma, to import a ``pf`` helper
that has not existed there for months.  Nothing crashed and nothing warned:
the CM printed a plausible ``[pf] sigma_ppm 1.484s`` line and produced no
trace at all, because ``pf.region()`` is only a ``TraceAnnotation`` and the
session starter was never called.

The danger is not the dead helper, it is the SHADOWING.  ``sys.path[0]`` wins
over the repo, so any module in that directory whose name collides with one
of ours -- ``pf``, ``analyze_trace``, ``analyze_hlo_dump`` -- would have been
imported in preference to the repo copy, out of a tree under nobody version
control.

The load-bearing cell here is the SOURCE SCAN, not the import check: the
injection fired inside ``__enter__``, at Sigma-execution time, so merely
importing the pipeline never triggered it and no import-time assertion could
have caught it.  Catching this class needs the source, and the source scan is
what fails on the pre-P21 tree.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"

# Names that exist BOTH in the repo / profiling tooling and in the sandbox
# tree the deleted shim pointed at.  A sandbox copy of any of these would
# silently win over ours.
_COLLIDING = ("pf", "analyze_trace", "analyze_hlo_dump", "analyze_compile_log",
              "run_profiled")


def _sys_path_injections(path: pathlib.Path):
    """Every ``sys.path.insert/append`` in ``path`` whose argument is a string
    LITERAL.  Repo-relative constructions (``Path(__file__)...``) are not
    literals and are deliberately not flagged -- they cannot escape the
    checkout, which is the property under test.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:                                    # pragma: no cover
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute) or f.attr not in ("insert", "append"):
            continue
        owner = f.value                       # sys.path.X / _sys.path.X
        if not (isinstance(owner, ast.Attribute) and owner.attr == "path"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("/"):
                    out.append((node.lineno, arg.value))
    return out


def test_no_absolute_out_of_repo_sys_path_injection_in_src():
    """THE RED: this cell fails on the pre-P21 tree, at jax_profile.py."""
    offenders = []
    for py in sorted(_SRC.rglob("*.py")):
        for lineno, literal in _sys_path_injections(py):
            offenders.append(f"{py.relative_to(_REPO)}:{lineno} -> {literal!r}")
    assert not offenders, (
        "A LORRAX module inserts an absolute, out-of-repo path into "
        "sys.path:\n  " + "\n  ".join(offenders) +
        "\nsys.path[0] wins over the repo, so a colliding module name in "
        "that tree is imported in preference to ours.  Build paths from "
        "__file__ instead.")


def test_no_sandbox_path_precedes_the_repo_on_sys_path():
    """The shadowing property, which is the one the repo actually owns.

    ``profile_section`` inserted its sandbox directory at ``sys.path[0]`` --
    ahead of everything, including our own ``src`` -- so a colliding module
    name there won.  That is what this cell forbids.

    It deliberately does NOT demand that no sandbox path exist at all,
    because one does and it is not the repo doing it: the ``lx`` harness
    puts ``/pscratch/sd/j/jackm/lorrax_sandbox/sources`` on ``PYTHONPATH``,
    LAST, after the checkout src.  Trailing, it shadows nothing; leading, it
    would shadow everything.  Asserting absence would make this cell fail for
    a reason no commit in this repo can fix, which is how a gate becomes
    noise.  Position is the property; position is what is asserted.
    """
    import gw.ppm_pipeline                                    # noqa: F401
    src = str(_SRC.resolve())
    resolved = [str(pathlib.Path(p).resolve()) if p else p for p in sys.path]
    if src not in resolved:                                   # pragma: no cover
        pytest.skip(f"repo src not on sys.path ({src}); nothing to order against")
    src_at = resolved.index(src)
    ahead = [p for p in resolved[:src_at] if "lorrax_sandbox" in p]
    assert not ahead, (
        f"sandbox path(s) ahead of the repo src on sys.path: {ahead}\n"
        f"(repo src is at index {src_at}).  Anything there shadows our own "
        f"modules by name.")


@pytest.mark.parametrize("name", _COLLIDING)
def test_colliding_tool_names_never_resolve_to_a_sandbox_copy(name):
    """Absent, or ours -- never a copy out of the sandbox tree."""
    import gw.ppm_pipeline                                    # noqa: F401
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):                          # pragma: no cover
        return
    if spec is None or not spec.origin:
        return
    origin = pathlib.Path(spec.origin).resolve()
    assert "lorrax_sandbox" not in str(origin), (
        f"{name!r} resolves to a sandbox copy at {origin}")


def test_profile_section_stays_deleted():
    """The shim itself does not come back."""
    import common.jax_profile as jp
    assert not hasattr(jp, "profile_section"), (
        "common.jax_profile.profile_section is back.  It produced no trace, "
        "printed a plausible timing line that was really the job of "
        "timing.section, and injected a sandbox path at sys.path[0].  The "
        "working entry points are trace_section / annotation / "
        "step_annotation.")
