"""Env-var grammar + process-global-state gates for the GW init/config layer.

RUNNABLE ON THE LOGIN NODE with plain ``python3`` — no jax, no h5py::

    cd /work2/08271/jackmc/frontera/lorrax
    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 tests/test_env_grammar.py

(also collects under pytest).

Why this file exists
--------------------
Three defects, all of the same shape — a boolean env knob parsed by hand:

1. ``LORRAX_EXIT_AFTER_ZETA=0`` used a bare presence test
   (``gw_init.py:1007``) and therefore ended a production run with
   ``SystemExit(0)``.  Setting a debug knob to "off" terminated the job
   *successfully*; nothing downstream could tell that from completion.
2. ``LORRAX_MALLOC_TRIM=OFF`` used a case-SENSITIVE
   ``not in ("0", "off", "false")`` (``isdf_fitting.py:907``) and left the
   trim hook ON.  Its documented sibling ``LORRAX_MALLOC_TUNE=OFF`` works,
   so the two disagree.
3. Six different boolean grammars were live in the tree at once (see
   ``test_vocabulary_has_not_drifted`` for the four named ones).

The tests below are in three layers, because a behavioural test of a
helper cannot prove the *call sites* were converted:

* **behaviour** — what :func:`gw.gw_config.env_bool` returns per token;
* **call-site audit** — an ``ast`` scan of the four owned source files,
  which reads the shipped code without importing it (``gw_init`` and
  ``isdf_fitting`` both ``import jax`` at module scope, so importing them
  needs a GPU node; parsing them does not);
* **drift** — the recognised token set is compared against every other
  live copy of it in the tree, so a future edit to any one of them fails
  here instead of splitting the grammar again.

Every auditor in this file has a NEGATIVE CONTROL beside it
(``test_auditor_*_can_fail``) that feeds it a synthetic source string
containing the defect and asserts the auditor reports it.  An auditor
that has never been shown failing is not evidence.
"""
import ast
import importlib.util
import os
import sys
import types

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# Isolated loaders — import ONE source file without executing its package
# ``__init__`` (``src/common/__init__.py`` and ``src/gw/__init__.py`` both
# pull in jax).  Same pattern as tests/test_sanity_gates.py.
# ---------------------------------------------------------------------------

def _load_isolated(mod_name, relpath):
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(_SRC, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pkg(name, relpath):
    p = sys.modules.setdefault(name, types.ModuleType(name))
    p.__path__ = [os.path.join(_SRC, relpath)]
    return p


_common = _pkg("common", "common")
_common.units = _load_isolated("common.units", "common/units.py")
_pkg("gw", "gw")
_pkg("ffi", "ffi")
_pkg("ffi.common", "ffi/common")

gw_config = _load_isolated("gw.gw_config", "gw/gw_config.py")
gw_output = _load_isolated("gw.gw_output", "gw/gw_output.py")
gate = _load_isolated("ffi.gate", "ffi/gate.py")
import runtime as _runtime            # noqa: E402  (os + subprocess only)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _Log:
    """Collecting print_fn so a test can assert on what the operator sees."""

    def __init__(self):
        self.lines = []

    def __call__(self, *a, **k):
        self.lines.append(" ".join(str(x) for x in a))

    @property
    def text(self):
        return "\n".join(self.lines)


class _Env:
    """Context manager that sets/clears env vars and always restores them."""

    def __init__(self, **kw):
        self._want = kw
        self._old = {}

    def __enter__(self):
        for k, v in self._want.items():
            self._old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def _reset_announce():
    """Drop env_bool's once-per-(name,value) announcement memo."""
    fn = getattr(gw_config, "reset_env_announce_state", None)
    if fn is not None:
        fn()


# ---------------------------------------------------------------------------
# AST utilities (py3.7-compatible: the parser emits ast.Str there, not
# ast.Constant)
# ---------------------------------------------------------------------------

def _const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if node.__class__.__name__ == "Str":            # py<3.8
        return node.s
    return None


def _func_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _parent_map(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _env_read_name(node):
    """Env-var name if ``node`` reads ``os.environ``/``os.getenv``, else None.

    Returns the literal name, or ``"<dynamic>"`` when the key is not a
    string literal (a dynamic key is itself worth flagging).
    """
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "get":
            base = fn.value
            if isinstance(base, ast.Attribute) and base.attr == "environ":
                if node.args:
                    return _const_str(node.args[0]) or "<dynamic>"
                return "<dynamic>"
        if _func_name(fn) == "getenv":
            if node.args:
                return _const_str(node.args[0]) or "<dynamic>"
            return "<dynamic>"
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Attribute) and base.attr == "environ":
            sl = node.slice
            # py<3.9 wraps the key in ast.Index
            sl = getattr(sl, "value", sl)
            return _const_str(sl) or "<dynamic>"
    return None


_TRANSPARENT_CALLS = {"int", "float", "str", "len"}
_TRANSPARENT_METHODS = {"strip", "lstrip", "rstrip", "lower", "upper", "casefold"}


def _bool_consumed(node, parents):
    """True when ``node``'s value is used to make a boolean decision."""
    cur = node
    while True:
        par = parents.get(cur)
        if par is None:
            return False
        if isinstance(par, (ast.If, ast.While, ast.IfExp)) and par.test is cur:
            return True
        if isinstance(par, ast.Assert) and par.test is cur:
            return True
        if isinstance(par, ast.comprehension) and cur in par.ifs:
            return True
        if isinstance(par, ast.UnaryOp) and isinstance(par.op, ast.Not):
            return True
        if isinstance(par, ast.BoolOp):
            return True
        if isinstance(par, ast.Compare) and any(
                isinstance(o, (ast.In, ast.NotIn)) for o in par.ops):
            return True
        if isinstance(par, ast.Call) and _func_name(par.func) == "bool":
            return True
        if isinstance(par, ast.Call) and (
                _func_name(par.func) in _TRANSPARENT_CALLS
                or (isinstance(par.func, ast.Attribute)
                    and par.func.attr in _TRANSPARENT_METHODS)):
            cur = par
            continue
        if isinstance(par, ast.Attribute):
            cur = par
            continue
        return False


def audit_source(src, filename="<src>"):
    """Return (raw_reads, bool_reads) for one python source string.

    ``raw_reads``  : [(name, lineno)] — every ``os.environ``/``os.getenv``.
    ``bool_reads`` : [(name, lineno)] — the subset used as a boolean.
    """
    tree = ast.parse(src, filename)
    parents = _parent_map(tree)
    raw, boolish = [], []
    for node in ast.walk(tree):
        name = _env_read_name(node)
        if name is None:
            continue
        raw.append((name, getattr(node, "lineno", -1)))
        if _bool_consumed(node, parents):
            boolish.append((name, getattr(node, "lineno", -1)))
    return raw, boolish


def audit_env_bool_calls(src, filename="<src>"):
    """Knob names passed as the first literal arg of ``env_bool(...)``."""
    tree = ast.parse(src, filename)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _func_name(node.func) == "env_bool":
            if node.args:
                nm = _const_str(node.args[0])
                if nm:
                    found.append((nm, node.lineno))
    return found


def _literal_tuple_from_source(relpath, varname):
    """Read a module-level ``varname = (...)`` string tuple WITHOUT importing.

    Used for drift checks against modules that import jax.
    """
    with open(os.path.join(_SRC, relpath)) as fh:
        tree = ast.parse(fh.read(), relpath)
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name) and t.id == varname:
                return tuple(ast.literal_eval(node.value))
    raise AssertionError(
        "%s does not define a module-level %s" % (relpath, varname))


# ---------------------------------------------------------------------------
# The four files this workstream owns, and the knobs whose grammar it owns.
# ---------------------------------------------------------------------------

OWNED_FILES = (
    "gw/gw_init.py",
    "gw/gw_config.py",
    "gw/gw_output.py",
    "gw/isdf_fitting.py",
    # P1.3 (2026-07-31): isdf/core.py's local ``_env_bool`` was retired —
    # the module imports ``gw_config.env_bool`` now — which moved its four
    # boolean knobs under this scan.  Scanning the file is what stops the
    # divergent-parser class from recurring silently: a fifth grammar
    # would have to appear IN one of these files to escape notice, and the
    # scan below flags any raw boolean env read in them.
    "isdf/core.py",
)

#: Boolean knobs whose ONLY readers live in the owned files, so their
#: grammar can be fixed here without splitting the knob across packages.
OWNED_BOOL_KNOBS = (
    "LORRAX_EXIT_AFTER_ZETA",
    "LORRAX_MALLOC_TRIM",
    "LORRAX_MEM_DEBUG",
    "LORRAX_RCHUNK_DEBUG",
    # Converted 2026-07-30 across ALL FIVE readers at once (gw_init.py x3,
    # gw/v_q_g_flat.py, gw/screening.py), which is what let it move from
    # CROSS_FILE_BOOL_KNOBS to here.  The whole-tree gate lives in
    # tests/test_crossfile_requests.py::
    # test_no_module_anywhere_reads_force_full_bz_by_hand.
    "LORRAX_FORCE_FULL_BZ",
    # P1.3: the isdf/core.py knobs, owned since that file joined
    # OWNED_FILES (its readers all route through gw_config.env_bool now).
    "LORRAX_ZETA_RANK_LOG",
    "LORRAX_COLLECTIVE_CHUNK_LOG",
    # Read once, in gw_init's ζ-reuse gate — previously through a lazy
    # ``from isdf.core import _env_bool`` (the last consumer of that
    # helper); collapsed onto the module-scope env_bool import.
    "LORRAX_FORCE_REFIT",
)

#: Boolean knobs that are ALSO read outside the owned files.  Converting
#: one here alone would split the knob's grammar between packages (the same
#: value meaning different things in two modules), which is worse than a
#: single wrong grammar.  Filed as cross-file requests; exempted here so
#: this gate stays green on a correct tree.
CROSS_FILE_BOOL_KNOBS = {
    # EMPTY since P1.3: LORRAX_RCHUNK_DEBUG's second reader (isdf/core.py,
    # historically a bare presence test under which ``=0`` meant ON) now
    # reads through gw_config.env_bool like gw/isdf_fitting.py, and
    # isdf/core.py itself is in OWNED_FILES.  The dict stays so the next
    # genuinely cross-file knob has somewhere to be filed WITH its request
    # number instead of silently exempted.
}


def _read(relpath):
    with open(os.path.join(_SRC, relpath)) as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# 1. Behaviour of the canonical helper
# ---------------------------------------------------------------------------

FALSE_SPELLINGS = ("0", "off", "false", "no",
                   "OFF", "False", "NO", "Off", "  0  ", " off ")
TRUE_SPELLINGS = ("1", "on", "true", "yes",
                  "ON", "True", "YES", "On", "  1  ", " yes ")


def test_env_bool_exists():
    assert hasattr(gw_config, "env_bool"), (
        "gw.gw_config.env_bool is missing — the four owned files have no "
        "canonical boolean env grammar to route through.")


def test_env_bool_unset_and_empty_take_the_default():
    for default in (True, False):
        with _Env(LORRAX_TEST_KNOB=None):
            assert gw_config.env_bool("LORRAX_TEST_KNOB", default) is default
        for blank in ("", " ", "\t"):
            with _Env(LORRAX_TEST_KNOB=blank):
                assert gw_config.env_bool(
                    "LORRAX_TEST_KNOB", default) is default, blank


def test_env_bool_false_spellings():
    for default in (True, False):
        for tok in FALSE_SPELLINGS:
            with _Env(LORRAX_TEST_KNOB=tok):
                assert gw_config.env_bool("LORRAX_TEST_KNOB", default) is False, (
                    "%r must parse FALSE (default=%s)" % (tok, default))


def test_env_bool_true_spellings():
    for default in (True, False):
        for tok in TRUE_SPELLINGS:
            with _Env(LORRAX_TEST_KNOB=tok):
                assert gw_config.env_bool("LORRAX_TEST_KNOB", default) is True, (
                    "%r must parse TRUE (default=%s)" % (tok, default))


def test_env_bool_unrecognised_token_is_announced_not_silent():
    """A typo must not resolve quietly in either direction."""
    log = _Log()
    _reset_announce()
    with _Env(LORRAX_TEST_KNOB="tru"):
        got = gw_config.env_bool("LORRAX_TEST_KNOB", True, print_fn=log)
    assert got is False, (
        "unrecognised tokens follow isdf.core._env_bool (-> False); "
        "diverging would split the grammar for knobs both modules read")
    assert "LORRAX_TEST_KNOB" in log.text and "tru" in log.text, log.text
    assert "LORRAX SANITY" in log.text, (
        "the announcement must carry the project's grep-able SANITY marker; "
        "got: %r" % log.text)


def test_env_bool_announces_once_per_value():
    """Knobs read once per r-chunk must not spam the log."""
    log = _Log()
    _reset_announce()
    with _Env(LORRAX_TEST_KNOB="wat"):
        for _ in range(5):
            gw_config.env_bool("LORRAX_TEST_KNOB", False, print_fn=log)
    assert len(log.lines) == 1, log.lines


# ---------------------------------------------------------------------------
# 2. The three named defects
# ---------------------------------------------------------------------------

def test_defect1_exit_after_zeta_off_does_not_exit():
    """DEFECT 1 — ``LORRAX_EXIT_AFTER_ZETA=0`` must NOT end the run.

    The old ``if os.environ.get("LORRAX_EXIT_AFTER_ZETA"):`` at
    gw_init.py:1007 fell through to ``raise SystemExit(0)`` at :1011 for
    every non-empty value, so "off" ended a production run with rc=0.
    """
    for tok in FALSE_SPELLINGS:
        with _Env(LORRAX_EXIT_AFTER_ZETA=tok):
            assert gw_config.env_bool("LORRAX_EXIT_AFTER_ZETA", False) is False, (
                "LORRAX_EXIT_AFTER_ZETA=%r would exit the job" % tok)
    with _Env(LORRAX_EXIT_AFTER_ZETA="1"):
        assert gw_config.env_bool("LORRAX_EXIT_AFTER_ZETA", False) is True
    with _Env(LORRAX_EXIT_AFTER_ZETA=None):
        assert gw_config.env_bool("LORRAX_EXIT_AFTER_ZETA", False) is False


def test_defect2_malloc_trim_off_disables_trim():
    """DEFECT 2 — ``LORRAX_MALLOC_TRIM=OFF`` must disable the trim hook.

    The old case-SENSITIVE ``not in ("0", "off", "false")`` at
    isdf_fitting.py:907 missed ``""``, ``"no"`` and every uppercase
    spelling, while its documented sibling LORRAX_MALLOC_TUNE=OFF works
    (runtime._env_falsy).
    """
    for tok in FALSE_SPELLINGS:
        with _Env(LORRAX_MALLOC_TRIM=tok):
            assert gw_config.env_bool("LORRAX_MALLOC_TRIM", True) is False, (
                "LORRAX_MALLOC_TRIM=%r left the trim hook ON" % tok)
    with _Env(LORRAX_MALLOC_TRIM=None):
        assert gw_config.env_bool("LORRAX_MALLOC_TRIM", True) is True


def test_defect2_agrees_with_its_documented_sibling():
    """MALLOC_TRIM and MALLOC_TUNE must answer the same for every token.

    They are advertised as a pair in ``docs/dev/env_vars.md:116-117``; the
    defect was that only one of them honoured ``OFF``.
    """
    for tok in FALSE_SPELLINGS + TRUE_SPELLINGS:
        with _Env(LORRAX_MALLOC_TRIM=tok, LORRAX_MALLOC_TUNE=tok):
            trim = gw_config.env_bool("LORRAX_MALLOC_TRIM", True)
            tune = not _runtime._env_falsy("LORRAX_MALLOC_TUNE")
            assert trim is tune, (
                "%r: MALLOC_TRIM=%s but MALLOC_TUNE=%s" % (tok, trim, tune))
    with _Env(LORRAX_MALLOC_TRIM=None, LORRAX_MALLOC_TUNE=None):
        assert gw_config.env_bool("LORRAX_MALLOC_TRIM", True) is True
        assert _runtime._env_falsy("LORRAX_MALLOC_TUNE") is False


def test_blank_is_unset_in_every_vocabulary_including_runtime():
    """Blank means UNSET everywhere, ``runtime._env_falsy`` included.

    Replaces the self-retiring gate that PINNED the divergence: while
    ``""`` was in ``runtime._FALSY_TOKENS``, an explicit
    ``export LORRAX_MALLOC_TUNE=`` DISABLED a default-on knob, whereas
    ``ffi/gate.py`` ("unset or whitespace always maps to the gate's
    declared default"), ``isdf.core._env_bool``,
    ``file_io._slab_io_mpi_host._env_flag`` and this module's ``env_bool``
    all read blank as unset.

    A blank export is what a shell leaves behind for ``export X=$UNDEFINED``,
    so the old rule turned a typo in a variable NAME into a silent knob flip.
    Both spellings of blank are checked — ``""`` and whitespace — because the
    parser strips before comparing and only the stripped form was ever the
    documented case.
    """
    for blank in ("", " ", "\t"):
        with _Env(LORRAX_MALLOC_TUNE=blank):
            assert _runtime._env_falsy("LORRAX_MALLOC_TUNE") is False, (
                "runtime._env_falsy treats %r as FALSE; every other LORRAX "
                "vocabulary treats blank as unset (= the knob's default)"
                % blank)
        # ... and the default is honoured, not hard-coded to "on".
        with _Env(LORRAX_MALLOC_TUNE=blank):
            assert _runtime._env_falsy("LORRAX_MALLOC_TUNE", "0") is True
        with _Env(LORRAX_TEST_KNOB=blank):
            assert gw_config.env_bool("LORRAX_TEST_KNOB", True) is True
            assert gw_config.env_bool("LORRAX_TEST_KNOB", False) is False
    # A real falsy token must still be honoured — this is the half of the
    # parser the fix must NOT have broken.
    for tok in ("0", "off", "OFF", " no ", "False"):
        with _Env(LORRAX_MALLOC_TUNE=tok):
            assert _runtime._env_falsy("LORRAX_MALLOC_TUNE") is True, tok


def test_defect3_vocabulary_has_not_drifted():
    """DEFECT 3 — one recognised token set, checked against every live copy.

    Three named vocabularies remain in the tree and must stay set-equal:
      * ``ffi/gate.py::MODE_SPELLINGS``  (two-valued since 2026-08-06 —
        the ``auto`` token was deleted with the auto tier it named; the
        resolvers stay separate, the tokens agree);
      * ``runtime.__init__._FALSY_TOKENS``     (the falsy set exactly — the
        ``""`` it used to carry was the blank-token divergence, now fixed);
      * ``file_io/_slab_io_mpi_host.py::_TRUE`` (read from source: imports
        jax at package scope).

    ``isdf/core.py::_ENV_TRUE`` — the fourth copy this test used to pin —
    was RETIRED by P1.3: the module imports ``gw_config.env_bool``
    instead.  The companion check below asserts the copy stays dead.
    """
    assert set(gw_config._ENV_TRUE) == set(gate.MODE_SPELLINGS["on"])
    assert set(gw_config._ENV_FALSE) == set(gate.MODE_SPELLINGS["off"])
    assert set(_runtime._FALSY_TOKENS) == set(gw_config._ENV_FALSE)
    assert set(gw_config._ENV_TRUE) == set(
        _literal_tuple_from_source("file_io/_slab_io_mpi_host.py", "_TRUE"))
    # ``auto`` must stay out of the two-valued sets.
    assert "auto" not in set(gw_config._ENV_TRUE) | set(gw_config._ENV_FALSE)
    # ...and out of the gate vocabulary: a token with no resolver branch in
    # enabled()/resolve()/enforce() would accept ``=auto`` and run as ``on``
    # in silence (deleted 2026-08-06; decisions.md 2026-08-01 killed the tier).
    assert "auto" not in gate.MODE_SPELLINGS and "auto" not in gate.MODE_HELP


def test_isdf_core_grammar_copy_stays_dead():
    """P1.3 ratchet: isdf/core.py must IMPORT the grammar, not re-grow it.

    The failure this pins: someone re-adds a local ``_env_bool``/
    ``_ENV_TRUE`` to isdf/core.py "to avoid the import", and the tree is
    back to two parsers whose drift is invisible until a knob splits.
    Parsed, not imported (isdf.core imports jax at module scope).
    """
    src = _read("isdf/core.py")
    tree = ast.parse(src, "isdf/core.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert node.name != "_env_bool", (
                "isdf/core.py re-defines _env_bool at line %d; import "
                "gw_config.env_bool instead (P1.3)" % node.lineno)
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            assert not (isinstance(t, ast.Name) and t.id == "_ENV_TRUE"), (
                "isdf/core.py re-defines _ENV_TRUE; the vocabulary lives "
                "in gw_config (P1.3)")
    # and the import is really there, spelled from gw.gw_config
    has_import = any(
        isinstance(n, ast.ImportFrom) and n.module == "gw.gw_config"
        and any(a.name == "env_bool" for a in n.names)
        for n in ast.walk(tree))
    assert has_import, (
        "isdf/core.py no longer imports env_bool from gw.gw_config — its "
        "boolean knobs have no grammar")


# ---------------------------------------------------------------------------
# 3. Call-site audit — the shipped files, parsed not imported
# ---------------------------------------------------------------------------

def test_owned_knobs_are_read_only_through_env_bool():
    """No raw ``os.environ`` read of an owned boolean knob may survive."""
    offenders = []
    for rel in OWNED_FILES:
        raw, _ = audit_source(_read(rel), rel)
        for name, line in raw:
            if name in OWNED_BOOL_KNOBS:
                offenders.append("%s:%d %s" % (rel, line, name))
    assert not offenders, (
        "these boolean knobs are still parsed by hand instead of "
        "gw_config.env_bool:\n  " + "\n  ".join(offenders))


def test_owned_knobs_each_have_an_env_bool_call_site():
    """The positive half: the knob must still be reachable.

    Without this, deleting a knob's only read would pass the negative
    check above — a green gate over a knob that no longer works.
    """
    seen = set()
    for rel in OWNED_FILES:
        for name, _ in audit_env_bool_calls(_read(rel), rel):
            seen.add(name)
    missing = [k for k in OWNED_BOOL_KNOBS if k not in seen]
    assert not missing, (
        "no env_bool() call site for: %s" % ", ".join(missing))


#: Functions allowed to read DYNAMIC env keys in boolean context: launcher
#: PMI/PMIx presence probes iterate over environment names, which is not a
#: boolean LORRAX knob and has no literal key to route through env_bool.
#: Each entry is asserted to still exist, so the exemption cannot outlive
#: the function it excuses.
_DYNAMIC_PROBE_FUNCS = (
    ("gw/gw_config.py", "_mpi_launcher_env"),
)


def _func_line_span(src: str, funcname: str):
    # max-lineno walk instead of end_lineno: the suites run on py3.7,
    # where ast nodes have no end_lineno.
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == funcname:
            last = max((n.lineno for n in ast.walk(node)
                        if hasattr(n, "lineno")), default=node.lineno)
            return node.lineno, last
    return None


def test_no_hand_rolled_boolean_env_parse_in_owned_files():
    """Broader net: any LORRAX_* env read used as a boolean is a defect."""
    offenders = []
    for rel in OWNED_FILES:
        src = _read(rel)
        spans = []
        for f, fn in _DYNAMIC_PROBE_FUNCS:
            if f == rel:
                span = _func_line_span(src, fn)
                assert span is not None, (
                    "%s: exempted dynamic-probe function %s no longer "
                    "exists — remove it from _DYNAMIC_PROBE_FUNCS" % (rel, fn))
                spans.append(span)
        _, boolish = audit_source(src, rel)
        for name, line in boolish:
            if name in CROSS_FILE_BOOL_KNOBS:
                continue
            if name == "<dynamic>" and any(a <= line <= b for a, b in spans):
                continue
            if name.startswith("LORRAX_") or name == "<dynamic>":
                offenders.append("%s:%d %s" % (rel, line, name))
    assert not offenders, (
        "hand-rolled boolean env parses (route through gw_config.env_bool, "
        "or add to CROSS_FILE_BOOL_KNOBS with a request number):\n  "
        + "\n  ".join(offenders))


# --- negative controls for the auditors -------------------------------------

_BAD_PRESENCE = 'import os\nif os.environ.get("LORRAX_EXIT_AFTER_ZETA"):\n    raise SystemExit(0)\n'
_BAD_CASE = 'import os\nif os.environ.get("LORRAX_MALLOC_TRIM", "1") not in ("0", "off"):\n    x = 1\n'
_BAD_BOOLCALL = 'import os\nd = bool(os.environ.get("LORRAX_RCHUNK_DEBUG"))\n'
_BAD_NOT = 'import os\nif not os.environ.get("LORRAX_MEM_DEBUG"):\n    pass\n'
_BAD_INT = 'import os\nif bool(int(os.environ.get("LORRAX_FORCE_FULL_BZ", "0"))):\n    pass\n'
_BAD_GETENV = 'import os\nif os.getenv("LORRAX_MEM_DEBUG"):\n    pass\n'
_BAD_SUBSCRIPT = 'import os\nif os.environ["LORRAX_MEM_DEBUG"]:\n    pass\n'
_GOOD = ('from .gw_config import env_bool\n'
         'if env_bool("LORRAX_EXIT_AFTER_ZETA", False):\n    pass\n')


def test_auditor_can_fail_on_every_defect_shape():
    """NEGATIVE CONTROL — the scanner must report each known bad shape."""
    for label, src in (("presence", _BAD_PRESENCE), ("case", _BAD_CASE),
                       ("bool()", _BAD_BOOLCALL), ("not", _BAD_NOT),
                       ("bool(int())", _BAD_INT), ("getenv", _BAD_GETENV),
                       ("subscript", _BAD_SUBSCRIPT)):
        raw, boolish = audit_source(src, label)
        assert raw, "%s: scanner saw no env read at all" % label
        assert boolish, "%s: scanner did not flag the boolean use" % label


def test_auditor_is_quiet_on_the_fixed_shape():
    """NEGATIVE CONTROL — and must NOT fire on the canonical call."""
    raw, boolish = audit_source(_GOOD, "good")
    assert raw == [] and boolish == []
    assert audit_env_bool_calls(_GOOD, "good") == [("LORRAX_EXIT_AFTER_ZETA", 2)]


def test_auditor_can_fail_on_a_missing_env_bool_call():
    """NEGATIVE CONTROL for the positive half."""
    assert audit_env_bool_calls('x = 1\n', "empty") == []


# ---------------------------------------------------------------------------
# 4. XLA GPU-memory env readers — honesty of the reported peak
# ---------------------------------------------------------------------------
#
# jaxlib 0.9.1 ``generate_pjrt_gpu_plugin_options``
# (site-packages/jaxlib/xla_client.py:181-222) is the authority:
#     allocator = os.getenv('XLA_PYTHON_CLIENT_ALLOCATOR', 'default').lower()
#     ... must be one of default | platform | bfc | cuda_async
#     preallocate = os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE', '')
#     if preallocate: options['preallocate'] = preallocate not in ('false','False','0')
#     memory_fraction = os.getenv('XLA_CLIENT_MEM_FRACTION', '')
#     deprecated      = os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION', '')
# Everything here mirrors that parse exactly; the tests below are what
# stops the mirror drifting from the original.

def test_xla_mem_env_helper_exists():
    assert hasattr(gw_config, "resolve_xla_gpu_memory_env")


def test_allocator_parse_is_case_insensitive_like_jax():
    """jax lowercases the value; a case-sensitive == "platform" does not."""
    for tok in ("platform", "PLATFORM", " Platform "):
        with _Env(XLA_PYTHON_CLIENT_ALLOCATOR=tok):
            r = gw_config.resolve_xla_gpu_memory_env()
            assert r.allocator == "platform", tok
            assert r.peak_is_faithful is False, tok


def test_platform_allocator_is_BLIND_not_merely_under_reporting():
    """Measured (allocator workstream, job 7882447): under ``platform``
    ``memory_stats()`` returns ``bytes_limit=0`` AND ``peak_bytes_in_use=0``.

    The old comment at gw_init.py:491-494 called this "under-reports",
    which is a different and much weaker claim: a low number can still be
    read as a lower bound, a ZERO cannot.  It also means any figure the
    ζ-fit does print under ``platform`` came from the nvidia-smi FALLBACK
    in ``fit_zeta_to_h5._track_peak``, which samples the whole GPU —
    including other processes — not this run's arena.
    """
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR="platform"):
        r = gw_config.resolve_xla_gpu_memory_env()
        assert r.allocator == "platform"
        assert r.peak_accounting == "none"
        assert r.peak_is_faithful is False
        assert "0" in r.caveat() or "no arena" in r.caveat().lower(), r.caveat()
        assert "under-report" not in r.caveat().lower(), (
            "the measurement contradicts 'under-reports': %r" % r.caveat())


def test_cuda_async_is_not_caveated_as_under_reporting():
    """Measured: ``peak_bytes_in_use`` is IDENTICAL under BFC and cuda_async
    (1.000 and 6.500 GB, job 7882447).  The premise of the original branch —
    "cuda_async returns freed transients so the reading under-reports" —
    was not reproduced.

    It is what ``config/frontera/ffi_env.sh:24`` deploys, so getting this
    wrong caveats (or fails to caveat) the project's own FFI runs.
    Transient-heavy kernels were NOT tested, so the note must say that
    rather than claim either way.
    """
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR="cuda_async"):
        r = gw_config.resolve_xla_gpu_memory_env()
        assert r.allocator == "cuda_async"
        assert r.peak_accounting == "arena"
        assert r.peak_is_faithful is True
        assert r.caveat() == "", (
            "an unqualified caveat here contradicts the measurement: %r"
            % r.caveat())
        assert "transient" in r.peak_note.lower(), (
            "the untested case must be stated, not silently claimed: %r"
            % r.peak_note)


def test_bfc_allocators_are_faithful():
    for tok, name in ((None, "default"), ("default", "default"), ("bfc", "bfc")):
        with _Env(XLA_PYTHON_CLIENT_ALLOCATOR=tok):
            r = gw_config.resolve_xla_gpu_memory_env()
            assert r.allocator == name, tok
            assert r.peak_accounting == "arena", tok
            assert r.peak_is_faithful is True, tok
            assert r.caveat() == "", tok


def test_tf_gpu_allocator_is_inert_and_changes_no_verdict():
    """``TF_GPU_ALLOCATOR`` is a TensorFlow variable, inert for JAX
    (measured; ``src/runtime/__init__.py:231``).  The old branch OR'd it
    into the caveat test, so a stale export — and
    ``config/modulefiles/lorrax/0.1.0.lua:131`` still sets it — would have
    caveated a perfectly faithful BFC peak.
    """
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR=None,
              TF_GPU_ALLOCATOR="cuda_malloc_async"):
        r = gw_config.resolve_xla_gpu_memory_env()
        assert r.allocator == "default"
        assert r.peak_is_faithful is True
        assert r.caveat() == "", r.caveat()
        assert r.tf_gpu_allocator_is_inert is True


# --- the client, not the environment ----------------------------------------
#
# Job 7882443 (allocator workstream) found kin_ion_io pre- and post-refactor
# ending with IDENTICAL os.environ but DIFFERENT clients (bytes_limit 11.805
# vs 0.000), because ``bootstrap()`` -> ``fallback_to_cpu_if_no_gpu_backend()``
# calls ``jax.devices()``, which IS backend init — so a ``setdefault`` after
# that point sets a string and changes nothing.  os.environ is therefore a
# FALSE WITNESS for allocator state, and anything that reports memory has to
# corroborate it against the live client.

def test_pool_classifier_exists():
    assert hasattr(gw_config, "classify_xla_pool")


def test_client_agreeing_with_env_is_quiet():
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR=None):
        v = gw_config.classify_xla_pool(
            {"bytes_limit": 11805000000, "bytes_in_use": 1000,
             "peak_bytes_in_use": 6500000000}, backend="gpu")
        assert v.accounting_present is True
        assert v.env_agrees is True
        assert v.disagreement == ""


def test_client_disagreeing_with_env_is_loud():
    """env says BFC (accounting) but the client has none — the exact
    signature of an allocator decided before the variable was set."""
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR=None):
        v = gw_config.classify_xla_pool(
            {"bytes_limit": 0, "peak_bytes_in_use": 0}, backend="gpu")
        assert v.accounting_present is False
        assert v.env_agrees is False
        assert "backend init" in v.disagreement, v.disagreement
    # ...and the other direction: env says platform, client has an arena.
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR="platform"):
        v = gw_config.classify_xla_pool(
            {"bytes_limit": 11805000000, "peak_bytes_in_use": 1}, backend="gpu")
        assert v.env_agrees is False
        assert v.disagreement != ""


def test_missing_stats_on_a_cpu_backend_is_not_a_disagreement():
    """CPU has no arena accounting by design — crying wolf there would
    train operators to ignore the warning that matters on GPU."""
    for stats in (None, {}):
        with _Env(XLA_PYTHON_CLIENT_ALLOCATOR=None):
            v = gw_config.classify_xla_pool(stats, backend="cpu")
            assert v.accounting_present is False
            assert v.env_agrees is True
            assert v.disagreement == ""


def test_pool_classifier_names_the_peak_source():
    """gw_init prints a number whose SOURCE differs by allocator; the
    caption has to say which one produced it."""
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR=None):
        assert gw_config.classify_xla_pool(
            {"bytes_limit": 1, "peak_bytes_in_use": 2},
            backend="gpu").peak_source == "arena"
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR="platform"):
        assert gw_config.classify_xla_pool(
            {"bytes_limit": 0, "peak_bytes_in_use": 0},
            backend="gpu").peak_source == "nvidia-smi"


#: memory_stats() signatures MEASURED on Quadro RTX 5000 / jax 0.9.1,
#: job 7882478 (source-pinned, manifest verified at start and end).
#: 1.07 GB allocated then freed in every cell.
MEASURED_STATS = {
    # allocator unset -> BFC.  Everything populated; the peak survives free.
    "default": {"num_allocs": 243, "bytes_in_use": 7680,
                "peak_bytes_in_use": 2147491584, "bytes_limit": 12675219456},
    # cuda_async.  peak_bytes_in_use IDENTICAL to BFC — but bytes_limit is 0,
    # so a banner that computes ``avail = limit - in_use`` prints a NEGATIVE
    # number here.
    "cuda_async": {"num_allocs": 243, "bytes_in_use": 2624,
                   "peak_bytes_in_use": 2147486280, "bytes_limit": 0},
    # platform.  memory_stats() is the EMPTY DICT — not zeros, nothing.
    "platform": {},
}


def test_classifier_matches_the_measured_signatures():
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR=None):
        v = gw_config.classify_xla_pool(MEASURED_STATS["default"],
                                        backend="gpu")
        assert v.accounting_present is True and v.env_agrees is True
        assert v.peak_source == "arena"
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR="cuda_async"):
        v = gw_config.classify_xla_pool(MEASURED_STATS["cuda_async"],
                                        backend="gpu")
        assert v.accounting_present is True, (
            "cuda_async reports bytes_limit=0 with a VALID peak; a "
            "limit-only test would call this 'no accounting'")
        assert v.env_agrees is True and v.peak_source == "arena"
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR="platform"):
        v = gw_config.classify_xla_pool(MEASURED_STATS["platform"],
                                        backend="gpu")
        assert v.accounting_present is False and v.env_agrees is True
        assert v.peak_source == "nvidia-smi"


def _banner_with_stats(stats, alloc, backend="gpu"):
    log = _Log()

    class _Dev:
        def memory_stats(self):
            return stats

    fake_jax = types.ModuleType("jax")
    fake_jax.local_devices = lambda: [_Dev()]
    old = sys.modules.get("jax")
    sys.modules["jax"] = fake_jax
    try:
        with _Env(XLA_PYTHON_CLIENT_ALLOCATOR=alloc):
            gw_output.print_banner(
                backend=backend, n_devices=1, grid_x=1, grid_y=1, n_procs=1,
                device_kind="Quadro RTX 5000", print_fn=log)
    finally:
        if old is None:
            sys.modules.pop("jax", None)
        else:
            sys.modules["jax"] = old
    return log


def test_banner_never_prints_a_negative_available_pool():
    """cuda_async reports ``bytes_limit=0`` with real in_use/peak values.

    ``avail = limit - in_use`` is then negative — a number that cannot
    exist.  Measured signature, job 7882478.
    """
    log = _banner_with_stats(MEASURED_STATS["cuda_async"], "cuda_async")
    assert "-" not in log.text.split("XLA pool:")[-1].split("\n")[0], log.text
    assert "peak" in log.text.lower() or "in_use" in log.text, log.text


def test_banner_reports_the_peak_when_the_limit_is_unknown():
    log = _banner_with_stats(MEASURED_STATS["cuda_async"], "cuda_async")
    assert "2.15" in log.text, (
        "the measured 2.147 GB peak must still reach the operator:\n%s"
        % log.text)


def test_banner_announces_an_env_client_disagreement():
    log = _Log()

    class _NoArena:
        device_kind = "rtx"

        def memory_stats(self):
            return {"bytes_limit": 0, "bytes_in_use": 0,
                    "peak_bytes_in_use": 0}

    fake_jax = types.ModuleType("jax")
    fake_jax.local_devices = lambda: [_NoArena()]
    old = sys.modules.get("jax")
    sys.modules["jax"] = fake_jax
    try:
        with _Env(XLA_PYTHON_CLIENT_ALLOCATOR=None):
            gw_output.print_banner(
                backend="gpu", n_devices=1, grid_x=1, grid_y=1, n_procs=1,
                device_kind="rtx", print_fn=log)
    finally:
        if old is None:
            sys.modules.pop("jax", None)
        else:
            sys.modules["jax"] = old
    assert "LORRAX SANITY" in log.text, log.text
    assert "backend init" in log.text, log.text


def test_gw_init_corroborates_the_env_against_the_client():
    """Source-level: the ζ-fit peak caption must read the live client."""
    src = _read("gw/gw_init.py")
    assert "classify_xla_pool" in src, (
        "gw_init captions the ζ-fit peak from os.environ alone; os.environ "
        "is a false witness for allocator state (job 7882443)")
    tree = ast.parse(src, "gw_init.py")
    calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
             and _func_name(n.func) in ("memory_stats", "classify_xla_pool")]
    assert len(calls) >= 2, calls


def test_allocator_outside_jax_vocabulary_is_flagged():
    """jax RAISES on an unknown value; we must not report it as faithful."""
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR="platfrom"):
        r = gw_config.resolve_xla_gpu_memory_env()
        assert r.allocator_is_valid is False
        assert r.peak_is_faithful is False


def test_preallocate_reports_the_effective_value_not_the_raw_string():
    """jax's own test is ``not in ('false','False','0')`` — case-SENSITIVE.

    ``XLA_PYTHON_CLIENT_PREALLOCATE=FALSE`` therefore leaves preallocation
    ON inside jax while reading as "off" to a human.  Unset means ON too
    (the option is simply not passed, and XLA's default preallocates).
    """
    cases = {None: True, "": True, "false": False, "False": False, "0": False,
             "FALSE": True, "no": True, "off": True, "true": True, "1": True}
    for tok, want in cases.items():
        with _Env(XLA_PYTHON_CLIENT_PREALLOCATE=tok):
            r = gw_config.resolve_xla_gpu_memory_env()
            assert r.preallocate is want, (
                "PREALLOCATE=%r: jax resolves %s, helper said %s"
                % (tok, want, r.preallocate))


def test_preallocate_flags_the_case_trap():
    with _Env(XLA_PYTHON_CLIENT_PREALLOCATE="FALSE"):
        r = gw_config.resolve_xla_gpu_memory_env()
        assert r.preallocate_looks_like_a_typo is True
    with _Env(XLA_PYTHON_CLIENT_PREALLOCATE="false"):
        assert gw_config.resolve_xla_gpu_memory_env(
            ).preallocate_looks_like_a_typo is False


def test_mem_fraction_reads_the_current_var_not_only_the_deprecated_one():
    """jaxlib 0.9.1 renamed the knob to ``XLA_CLIENT_MEM_FRACTION``.

    gw_output.py:138 reported only ``XLA_PYTHON_CLIENT_MEM_FRACTION``, so a
    run using the current name showed "mem_fraction: unset" while a
    fraction was in force.
    """
    with _Env(XLA_CLIENT_MEM_FRACTION="0.7",
              XLA_PYTHON_CLIENT_MEM_FRACTION=None):
        r = gw_config.resolve_xla_gpu_memory_env()
        assert r.mem_fraction == "0.7"
        assert r.mem_fraction_var == "XLA_CLIENT_MEM_FRACTION"
    with _Env(XLA_CLIENT_MEM_FRACTION=None,
              XLA_PYTHON_CLIENT_MEM_FRACTION="0.85"):
        r = gw_config.resolve_xla_gpu_memory_env()
        assert r.mem_fraction == "0.85"
        assert r.mem_fraction_deprecated is True
    # Setting BOTH makes jax raise at backend init; say so before it does.
    with _Env(XLA_CLIENT_MEM_FRACTION="0.7",
              XLA_PYTHON_CLIENT_MEM_FRACTION="0.85"):
        r = gw_config.resolve_xla_gpu_memory_env()
        assert r.mem_fraction_conflict is True


def test_banner_labels_gpu_only_knobs_on_a_cpu_backend():
    """The XLA pool knobs are read ONLY by the CUDA plugin's option builder.

    Printing "XLA preallocate: false" on a CPU run states a GPU fact about
    a run that has no GPU.
    """
    log = _Log()
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR="platform",
              XLA_PYTHON_CLIENT_PREALLOCATE="false"):
        gw_output.print_banner(
            backend="cpu", n_devices=2, grid_x=1, grid_y=2, n_procs=2,
            device_kind="cpu", print_fn=log)
    assert "preallocate" not in log.text.lower() or "cpu" in log.text.lower(), (
        "CPU banner printed GPU allocator state unqualified:\n%s" % log.text)
    assert "not applicable" in log.text.lower() or "ignored" in log.text.lower(), (
        "CPU banner must say the GPU pool knobs do not apply:\n%s" % log.text)


def test_banner_reports_the_allocator_on_gpu():
    log = _Log()
    with _Env(XLA_PYTHON_CLIENT_ALLOCATOR="cuda_async",
              XLA_PYTHON_CLIENT_PREALLOCATE="false"):
        gw_output.print_banner(
            backend="gpu", n_devices=1, grid_x=1, grid_y=1, n_procs=1,
            device_kind="rtx", print_fn=log)
    assert "cuda_async" in log.text, log.text


def test_banner_memory_stats_failure_is_reported_not_swallowed():
    """``except Exception: pass`` around the pool read hid every failure.

    An operator then sees a banner with no pool line and no reason.
    """
    log = _Log()
    marker = object()

    class _BoomDevice:
        def memory_stats(self):
            raise RuntimeError("memory_stats unavailable on this backend")

    fake_jax = types.ModuleType("jax")
    fake_jax.local_devices = lambda: [_BoomDevice()]
    old = sys.modules.get("jax")
    sys.modules["jax"] = fake_jax
    try:
        gw_output.print_banner(
            backend="gpu", n_devices=1, grid_x=1, grid_y=1, n_procs=1,
            device_kind="rtx", print_fn=log)
    finally:
        if old is None:
            sys.modules.pop("jax", None)
        else:
            sys.modules["jax"] = old
    assert "memory_stats unavailable" in log.text or "pool" in log.text.lower(), (
        "the pool read failed silently; banner was:\n%s" % log.text)
    assert marker is marker


# ---------------------------------------------------------------------------
# 4b. Numeric knobs — a swallowed parse is the same defect class
# ---------------------------------------------------------------------------

def test_env_float_announces_a_bad_value():
    """``ISDF_CHUNK_TARGET_UTILIZATION`` used a bare
    ``except Exception: chunk_utilization = 0.0``.

    Its two siblings twenty lines below (``ISDF_ZCT_STAGE_CAP_GB`` /
    ``_FRAC``) already announce, with the comment "Swallowing this left the
    user believing a cap was in force when it was not -- an OOM later, with
    no clue."  Exactly the same is true of the utilization.
    """
    assert hasattr(gw_config, "env_float")
    log = _Log()
    _reset_announce()
    with _Env(ISDF_CHUNK_TARGET_UTILIZATION="0.97"):
        assert gw_config.env_float(
            "ISDF_CHUNK_TARGET_UTILIZATION", 0.0, print_fn=log) == 0.97
    assert log.lines == []
    with _Env(ISDF_CHUNK_TARGET_UTILIZATION=None):
        assert gw_config.env_float(
            "ISDF_CHUNK_TARGET_UTILIZATION", 0.0, print_fn=log) == 0.0
    assert log.lines == []
    with _Env(ISDF_CHUNK_TARGET_UTILIZATION="high"):
        assert gw_config.env_float(
            "ISDF_CHUNK_TARGET_UTILIZATION", 0.0, print_fn=log) == 0.0
    assert "LORRAX SANITY" in log.text, log.text
    assert "high" in log.text and "0.0" in log.text, log.text


def test_zct_cap_resolver_states_every_reason_it_is_not_set():
    assert hasattr(gw_config, "resolve_zct_stage_cap")
    r = gw_config.resolve_zct_stage_cap
    log = _Log()
    # explicit GB wins, clamped to the budget
    assert r("4", None, per_device_gb=10.0, total_gb=16.0, print_fn=log) == 4.0
    assert r("40", None, per_device_gb=10.0, total_gb=16.0, print_fn=log) == 10.0
    assert log.lines == []
    # fraction of the physical card
    assert r(None, "0.5", per_device_gb=10.0, total_gb=16.0,
             print_fn=log) == 8.0
    # a bad GB value announces
    log = _Log()
    assert r("big", None, per_device_gb=10.0, total_gb=16.0, print_fn=log) is None
    assert "LORRAX SANITY" in log.text
    # a bad fraction announces
    log = _Log()
    assert r(None, "half", per_device_gb=10.0, total_gb=16.0,
             print_fn=log) is None
    assert "LORRAX SANITY" in log.text
    # ...and the CPU case: total_gb=0 means there is no card to take a
    # fraction OF.  Previously the whole branch was skipped by an
    # ``and jax.default_backend() in ("gpu","cuda")`` guard, so a user who
    # set ISDF_ZCT_STAGE_CAP_FRAC on a CPU run got no cap and no message.
    log = _Log()
    assert r(None, "0.5", per_device_gb=10.0, total_gb=0.0,
             print_fn=log) is None
    assert "LORRAX SANITY" in log.text, log.text
    assert "ISDF_ZCT_STAGE_CAP_FRAC" in log.text
    # unset stays quiet
    log = _Log()
    assert r(None, None, per_device_gb=10.0, total_gb=0.0, print_fn=log) is None
    assert log.lines == []


def test_config_uses_the_announcing_numeric_helpers():
    """Source-level: the swallowing parses must be gone."""
    src = _read("gw/gw_config.py")
    tree = ast.parse(src, "gw_config.py")
    # No ``except Exception``/bare except may wrap an ISDF_* float parse.
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body = ast.dump(ast.Module(body=node.body, type_ignores=[])
                        if hasattr(ast, "type_ignores") else
                        ast.Module(body=node.body))
        if "ISDF_CHUNK_TARGET_UTILIZATION" not in body:
            continue
        for h in node.handlers:
            if not any(isinstance(n, ast.Call) for n in ast.walk(h)):
                bad.append(node.lineno)
    assert not bad, (
        "gw_config.py:%s still swallows a bad ISDF_CHUNK_TARGET_UTILIZATION "
        "without telling anybody" % bad)


# ---------------------------------------------------------------------------
# 5. Truncating-knob provenance guard
# ---------------------------------------------------------------------------

def test_truncation_knobs_are_named_in_gw_init():
    """``LORRAX_MAX_RCHUNKS=N`` breaks the r-chunk loop early, and the
    writer still calls ``mark_zeta_done`` (isdf_fitting.py:1098).  If
    ``gw_init`` then stamps ``fit_provenance`` (gw_init.py:478-486), the
    TRUNCATED ζ becomes reusable by a later production run in the same
    directory — silently wrong physics with no warning at all.

    gw_init must therefore know which knobs truncate a fit.
    """
    assert hasattr(gw_config, "ZETA_TRUNCATING_ENV_KNOBS")
    assert "LORRAX_MAX_RCHUNKS" in gw_config.ZETA_TRUNCATING_ENV_KNOBS
    assert hasattr(gw_config, "active_zeta_truncating_knobs")
    with _Env(LORRAX_MAX_RCHUNKS=None):
        assert gw_config.active_zeta_truncating_knobs() == []
    with _Env(LORRAX_MAX_RCHUNKS="2"):
        assert gw_config.active_zeta_truncating_knobs() == [
            ("LORRAX_MAX_RCHUNKS", "2")]
    # An empty value is not a truncation request.
    with _Env(LORRAX_MAX_RCHUNKS=""):
        assert gw_config.active_zeta_truncating_knobs() == []


def test_gw_init_consults_the_truncation_guard_before_stamping():
    """Source-level: the provenance stamp must be guarded."""
    src = _read("gw/gw_init.py")
    assert "active_zeta_truncating_knobs" in src, (
        "gw_init stamps fit_provenance unconditionally; a LORRAX_MAX_RCHUNKS "
        "run therefore leaves a truncated ζ that _zeta_reuse_ok will reuse")
    tree = ast.parse(src, "gw_init.py")
    stamp_lines = [n.lineno for n in ast.walk(tree)
                   if isinstance(n, ast.Call)
                   and _func_name(n.func) == "stamp_fit_provenance"]
    guard_lines = [n.lineno for n in ast.walk(tree)
                   if isinstance(n, ast.Call)
                   and _func_name(n.func) == "active_zeta_truncating_knobs"]
    assert stamp_lines and guard_lines, (stamp_lines, guard_lines)
    assert min(guard_lines) < min(stamp_lines), (
        "the truncation guard must be evaluated before the stamp")


# ---------------------------------------------------------------------------

def _main():
    fns = [(n, o) for n, o in sorted(globals().items())
           if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in fns:
        try:
            fn()
        except Exception as exc:            # noqa: BLE001 — this IS the tally
            failed.append((name, exc))
            print("FAIL %s\n     %s: %s" % (name, type(exc).__name__, exc))
        else:
            print("ok   %s" % name)
    print("\n%d/%d passed, %d failed" % (len(fns) - len(failed), len(fns),
                                         len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
