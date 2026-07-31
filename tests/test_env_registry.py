"""Registry-enforcement gate: every LORRAX_* env read has an env_vars.md row.

RUNNABLE ON THE LOGIN NODE with plain ``python3`` — no jax, no h5py::

    cd /work2/08271/jackmc/frontera/lorrax
    python3 tests/test_env_registry.py

(also collects under pytest.)

Why this file exists
--------------------
``docs/dev/env_vars.md`` decayed twice: knobs were added, the page was
not, and the drift went unnoticed because the only check was
``tools/env_audit.py`` run BY HAND — which was additionally a silent
no-op on this login node's python 3.7 (``ast.Str`` vs ``ast.Constant``:
it walked the whole tree, matched nothing, printed nothing, exited 0 —
a FALSE-CLEAN sweep), and blind to helper-mediated reads
(``env_bool(...)`` / ``Gate(env=...)``), which are the majority since
the grammar unification.

This gate closes the loop mechanically:

* the PYTHON side reuses ``tools.env_audit``'s visitor (raw reads AND
  helper-mediated reads AND ``Gate(env=...)``), so the tool and the gate
  cannot disagree about what a "read" is — and the tool's own
  ``selftest()`` runs here, so an interpreter whose AST the visitor
  cannot see fails the SUITE instead of producing an empty report;
* the C++ side is a regex scan for ``getenv("...")`` and the two
  read-wrappers this tree uses (``log_here`` / ``env_flag``), which is
  what the AST walk cannot see;
* every ``LORRAX_*`` name found on either side must be covered by a row
  token in ``docs/dev/env_vars.md`` (exact, ``PREFIX*`` glob, or
  ``PREFIX{A,B}`` brace form).

Every auditor has a NEGATIVE CONTROL beside it, because an auditor that
has never been shown failing is not evidence.

Scope note: enforcement is LORRAX_* only.  External names (JAX_*, XLA_*,
SLURM_*, MKL_*...) are registered too, but their spelling authority is
not this repo, so a missing external row is a docs task, not a gate.
"""
import ast
import importlib.util
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
_REGISTRY = os.path.join(_REPO, "docs", "dev", "env_vars.md")

# ---------------------------------------------------------------------------
# Load tools/env_audit.py without a package (tools/ has no __init__)
# ---------------------------------------------------------------------------


def _load_env_audit():
    name = "lorrax_tools_env_audit"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, "tools", "env_audit.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


env_audit = _load_env_audit()


# ---------------------------------------------------------------------------
# Registry vocabulary
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{2,}\*?")
_BRACE_RE = re.compile(r"([A-Z][A-Z0-9_]*_)\{([A-Z0-9_,]+)\}(\*?)")


def registry_vocabulary(text):
    """``(exact_names, glob_prefixes)`` from the env_vars.md text.

    Three spellings count as a row token: ``LORRAX_FOO`` (exact),
    ``LORRAX_FOO*`` (prefix glob — the section-4 build-var idiom), and
    ``LORRAX_FOO_{A,B}[*]`` (brace alternation, expanded).
    """
    exact, globs = set(), set()
    for prefix, alts, star in _BRACE_RE.findall(text):
        for alt in alts.split(","):
            alt = alt.strip()
            if not alt:
                continue
            if star:
                globs.add(prefix + alt)
            else:
                exact.add(prefix + alt)
    # Strip brace groups so their fragments don't register as exact names.
    stripped = _BRACE_RE.sub(" ", text)
    for tok in _NAME_RE.findall(stripped):
        if tok.endswith("*"):
            globs.add(tok[:-1])
        else:
            exact.add(tok)
    return exact, globs


def covered(name, exact, globs):
    return name in exact or any(name.startswith(g) for g in globs)


def _registry_text():
    with open(_REGISTRY, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# The two scans
# ---------------------------------------------------------------------------

def python_read_sites():
    """``{name: [(file, line, default)]}`` for every env read under src/.

    Delegates to ``tools.env_audit.collect`` — ONE visitor for the tool
    and the gate.
    """
    return env_audit.collect(_SRC)


_CPP_EXTS = (".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp")
#: getenv("X") plus the two read-wrappers the C++ tree funnels through
#: (mkl_thread_pin.h log_here; phdf5 context.cc env_flag).  Whitespace
#: tolerated around the argument.
_CPP_READ_RE = re.compile(
    r"\b(?:getenv|log_here|env_flag)\s*\(\s*\"([A-Z][A-Z0-9_]*)\"")


def cpp_read_sites(root=None):
    """``{name: [(file, line)]}`` from a regex scan of the C++ sources."""
    root = root or _SRC
    found = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(_CPP_EXTS):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            with open(path, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    for name in _CPP_READ_RE.findall(line):
                        found.setdefault(name, []).append((rel, i))
    return found


# ---------------------------------------------------------------------------
# 1. The walker can see, ON THIS INTERPRETER
# ---------------------------------------------------------------------------

def test_env_audit_selftest_passes_here():
    """The false-clean guard: py3.7's ast.Str must not blind the visitor.

    If this fails, EVERY green result below is meaningless — which is why
    it is a test and not a comment.
    """
    assert env_audit.selftest() == 0, (
        "tools/env_audit.py cannot see its own fixture on python %s — the "
        "registry gate below would pass vacuously" % sys.version.split()[0])


def test_python_scan_sees_the_tree_at_all():
    """A LORRAX tree with zero env reads does not exist; empty = broken."""
    hits = python_read_sites()
    lorrax = [n for n in hits if n.startswith("LORRAX_")]
    assert len(lorrax) >= 30, (
        "the python scan found only %d LORRAX_* reads under src/ — the "
        "2026-07-31 baseline is ~50; the scanner has gone blind, not the "
        "tree clean" % len(lorrax))


def test_cpp_scan_sees_the_tree_at_all():
    hits = cpp_read_sites()
    lorrax = [n for n in hits if n.startswith("LORRAX_")]
    assert len(lorrax) >= 15, (
        "the C++ regex scan found only %d LORRAX_* reads — baseline is "
        "~28; suspect the regex or the extension list" % len(lorrax))


# ---------------------------------------------------------------------------
# 2. THE GATE — every LORRAX_* read site has a registry row
# ---------------------------------------------------------------------------

def test_every_python_lorrax_read_has_a_registry_row():
    exact, globs = registry_vocabulary(_registry_text())
    hits = python_read_sites()
    missing = []
    for name in sorted(hits):
        if not name.startswith("LORRAX_"):
            continue
        if not covered(name, exact, globs):
            sites = ", ".join("%s:%d" % (f, l) for f, l, _ in hits[name][:3])
            missing.append("%s  (read at %s)" % (name, sites))
    assert not missing, (
        "LORRAX_* env vars read under src/ with NO docs/dev/env_vars.md "
        "row — every new env var needs a registry row (project doctrine):"
        "\n  " + "\n  ".join(missing))


def test_every_cpp_lorrax_read_has_a_registry_row():
    exact, globs = registry_vocabulary(_registry_text())
    hits = cpp_read_sites()
    missing = []
    for name in sorted(hits):
        if not name.startswith("LORRAX_"):
            continue
        if not covered(name, exact, globs):
            sites = ", ".join("%s:%d" % (f, l) for f, l in hits[name][:3])
            missing.append("%s  (read at %s)" % (name, sites))
    assert not missing, (
        "LORRAX_* vars read by the C++ FFI side with no env_vars.md row "
        "(the AST tool cannot see these, which is how they went "
        "unregistered before):\n  " + "\n  ".join(missing))


# ---------------------------------------------------------------------------
# 3. Negative controls
# ---------------------------------------------------------------------------

def test_registry_coverage_can_fail():
    exact, globs = registry_vocabulary(
        "`LORRAX_REAL_KNOB` and `LORRAX_FAMILY_*` and "
        "`LORRAX_PAIR_{ONE,TWO}` are rows")
    assert covered("LORRAX_REAL_KNOB", exact, globs)
    assert covered("LORRAX_FAMILY_ANYTHING", exact, globs)
    assert covered("LORRAX_PAIR_ONE", exact, globs)
    assert covered("LORRAX_PAIR_TWO", exact, globs)
    assert not covered("LORRAX_NEVER_WRITTEN_DOWN", exact, globs), (
        "the coverage check passes a name the registry does not carry — "
        "the gate is decorative")
    assert not covered("LORRAX_PAIR_THREE", exact, globs), (
        "brace expansion invented an alternative that is not in the text")


def test_cpp_regex_can_fail_and_can_match():
    fixture = (
        '  const char* a = std::getenv("LORRAX_FAKE_CPP_KNOB");\n'
        '  static const bool on = mklpin::log_here("LORRAX_FAKE_LOG");\n'
        '  bool w = env_flag("LORRAX_FAKE_FLAG", true);\n'
        '  int unrelated = foo("LORRAX_NOT_A_READ");\n')
    names = set(_CPP_READ_RE.findall(fixture))
    assert names == {"LORRAX_FAKE_CPP_KNOB", "LORRAX_FAKE_LOG",
                     "LORRAX_FAKE_FLAG"}, names


def test_python_visitor_sees_helper_reads():
    """The exact blindness that produced the false-clean sweep."""
    import collections
    hits = collections.defaultdict(list)
    src = (
        'import os\n'
        'a = env_bool("LORRAX_T_BOOL", True)\n'
        'b = env_float("LORRAX_T_FLOAT", 1e-6, refuse=True)\n'
        'c = _env_falsy("LORRAX_T_FALSY")\n'
        'd = Gate(env="LORRAX_T_GATE", target="t", platforms=(),\n'
        '         modes=(), default="off", off_label="x")\n'
        'e = os.environ.get("LORRAX_T_RAW", "")\n')
    env_audit.EnvReadVisitor("<t>", hits).visit(ast.parse(src))
    assert set(hits) == {"LORRAX_T_BOOL", "LORRAX_T_FLOAT",
                         "LORRAX_T_FALSY", "LORRAX_T_GATE",
                         "LORRAX_T_RAW"}, sorted(hits)


def test_registry_file_exists_and_is_a_registry():
    text = _registry_text()
    assert "LORRAX environment variables" in text.splitlines()[0], (
        "docs/dev/env_vars.md no longer starts as the registry; the "
        "vocabulary scrape below is scraping something else")
    exact, globs = registry_vocabulary(text)
    assert len(exact) > 100, (
        "only %d exact tokens scraped from env_vars.md — the scrape or "
        "the page broke" % len(exact))


# ===========================================================================
# Standalone runner — same pattern as tests/test_layering.py
# ===========================================================================

def _main():
    n_run, failures = 0, []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        n_run += 1
        try:
            fn()
        except Exception as exc:            # noqa: BLE001 — this IS the tally
            msg = (str(exc).splitlines() or [""])[0]
            failures.append("FAIL %s — %s: %s"
                            % (name, type(exc).__name__, msg))
            print(failures[-1])
        else:
            print("ok   %s" % name)
    print("\n%d/%d passed, %d failed"
          % (n_run - len(failures), n_run, len(failures)))
    for line in failures:
        print(line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
