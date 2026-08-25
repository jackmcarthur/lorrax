"""Layer L-a: source/AST guards over the PRODUCERS this door reads after.

Two guards, and neither is about ``zeta_loader``'s own code.  That is the
point of the service owning the FORMAT CONTRACT and not just the reader
(survey §6.2): the writer is split across three modules — ``isdf_fitting``
for payload and create-order and ``isdf_header`` for the metadata groups,
while the reader is one class, and
"a service that owns the read contract but not the write contract cannot
enforce it" is exactly how the striping defect and the ``zeta_is_done``-
never-read defect both survived.

Pure ``ast``.  No jax, no h5, no file writes, milliseconds — EXCEPT the two
``inspect``-based cells at the bottom, which import ``isdf.core`` and
therefore jax; they are kept in their own section so the AST half runs in a
jax-free interpreter.

EVERY CHECKER HERE IS A FUNCTION OF SOURCE TEXT, not of a path.  That is what
lets each one be run over a synthetic BAD source in its red twin: a guard
whose only input is the real file can be shown passing and can never be shown
failing, which is the shape of every guard in this tree that turned out to be
measuring nothing.
"""

from __future__ import annotations

import ast
import os
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))
_REPO = os.path.dirname(_SERVICES)
_SRC = os.path.join(_REPO, "src")

ISDF_FITTING = os.path.join(_SRC, "gw", "isdf_fitting.py")
ISDF_CORE = os.path.join(_SRC, "isdf", "core.py")


def _read(path: str) -> str:
    if not os.path.isfile(path):
        pytest.skip(f"no lorrax host tree at {path!r} (standalone install); "
                    f"these guards read the monorepo's own producers")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    raise AssertionError(
        f"no function named {name!r} in this source.  A guard that cannot "
        f"find its target must FAIL, not pass vacuously — a rename would "
        f"otherwise silently retire the check.")


def _call_name(node: ast.AST) -> str:
    """``f(...)`` -> ``'f'``; ``a.b.f(...)`` -> ``'f'``; else ``''``."""
    if not isinstance(node, ast.Call):
        return ""
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""


# ===========================================================================
# GUARD 1 — the zeta_q.h5 striping create-order  (design D7, survey R2)
# ===========================================================================

def check_striping_create_order(source: str,
                                func_name: str = "fit_zeta_to_h5") -> list:
    """Violations of the ζ inode create-order.  ``[]`` means the order holds.

    Four claims about ``func_name``'s body:

    1. a ``with SlabIO(..., mode='w', ...)`` context exists — the inode is
       created COLLECTIVELY, which is the only create that carries the
       striping hints in its ``MPI_Info``;
    2. it PRECEDES every ``copy_mf_header`` call in source order;
    3. no ``copy_mf_header`` call passes ``dst_mode='w'`` — the header
       append must keep the inode it finds, not make a new one;
    4. no ``_replace_inode_for_write`` call remains — ``SlabIO(mode='w')``
       runs that same rank-0-unlink+barrier helper internally, so an
       explicit call would be doing the job twice.

    Returns strings rather than raising so the caller can assert on the SET
    of violations; a checker that raised on the first would make a red twin
    prove only that something is wrong, not that the right things are.
    """
    tree = ast.parse(source)
    fn = _find_function(tree, func_name)
    bad: list[str] = []

    slabio_w_lines, copy_lines = [], []
    for node in ast.walk(fn):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                call = item.context_expr
                if _call_name(call) != "SlabIO":
                    continue
                for kw in call.keywords:
                    if (kw.arg == "mode" and isinstance(kw.value, ast.Constant)
                            and kw.value.value == "w"):
                        slabio_w_lines.append(node.lineno)
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "copy_mf_header":
                copy_lines.append(node.lineno)
                for kw in node.keywords:
                    if kw.arg == "dst_mode":
                        val = getattr(kw.value, "value", "<non-literal>")
                        if val != "a":
                            bad.append(
                                f"copy_mf_header at line {node.lineno} passes "
                                f"dst_mode={val!r}; only 'a' keeps the inode "
                                f"SlabIO created (dst_mode='w' is the defect "
                                f"that cost zeta_q.h5 its stripe count)")
            elif name == "_replace_inode_for_write":
                bad.append(
                    f"_replace_inode_for_write at line {node.lineno}: "
                    f"SlabIO(mode='w') runs that helper internally, so an "
                    f"explicit call unlinks the inode a second time")

    if not slabio_w_lines:
        bad.append(
            f"{func_name} has no `with SlabIO(..., mode='w', ...)` context: "
            f"nothing creates the inode COLLECTIVELY, so the file takes the "
            f"directory default stripe layout and ROMIO pins collective "
            f"buffering to one aggregator (cb_nodes = min(stripe_count, "
            f"nranks)) at any rank count")
    elif copy_lines and min(copy_lines) < min(slabio_w_lines):
        bad.append(
            f"copy_mf_header at line {min(copy_lines)} runs BEFORE the "
            f"SlabIO(mode='w') create at line {min(slabio_w_lines)}: rank-0 "
            f"serial h5py creates the inode, MPI-IO never sees it, and the "
            f"striping hints are silently discarded")
    return bad


def test_the_production_zeta_writer_creates_the_inode_collectively():
    """THE STRIPING REGRESSION GUARD — the assertion that actually pins it.

    The fix landed at ``96a6399`` and is measured end to end (before:
    ``stripe_count 1`` at c960 and c1440 against a directory default of 1
    and a sibling ``isdf_tensors_*.h5`` of 4; after: 4; cost of the extra
    collective create+close on an empty file: 4 ms at 1 rank).  IT IS NOT
    RE-OPENED HERE.  What is re-opened is the claim that it is guarded.

    ``tests/test_file_io.py:1118``'s docstring says of itself:

        "This is the only assertion that catches a regression to
        ``copy_mf_header(..., dst_mode='w')``"

    **That is FALSE as written** (survey §3).  Both striping cells build
    their file with ``_build_zeta_like_production`` (:1040), which
    RE-IMPLEMENTS the create order by hand — ``SlabIO(mode='w')`` →
    ``copy_mf_header(dst_mode='a')`` → payload.  NEITHER calls
    ``fit_zeta_to_h5``.  So a revert of ``isdf_fitting.py``'s ordering
    tomorrow leaves the hand-rolled builder doing the right thing and both
    cells stay green: the PROPERTY is tested, the PRODUCTION WRITER'S USE of
    the property is not.  On top of that the policy half
    (``test_zeta_q_inode_gets_striping_policy``) skips wherever ``lfs`` is
    absent, which is always in the production container and always here.

    This cell reads the writer's own source.  It runs on WSL, in
    milliseconds, needs no Lustre and no MPI, and it is the check that turns
    red when someone reverts the order.
    """
    bad = check_striping_create_order(_read(ISDF_FITTING))
    assert bad == [], (
        "gw.isdf_fitting.fit_zeta_to_h5's ζ inode create-order regressed:\n  "
        + "\n  ".join(bad))


def test_the_striping_guard_is_not_vacuous():
    """The checker FOUND the things it is checking, on the real file.

    ``[]`` violations is the same answer for "the order is right" and for
    "the walk matched nothing" — a guard whose target moved would report
    green forever.  So the positive facts are asserted separately: the
    function exists, it contains a ``SlabIO(mode='w')`` create, and it
    contains a ``copy_mf_header`` call for that create to precede.
    """
    tree = ast.parse(_read(ISDF_FITTING))
    fn = _find_function(tree, "fit_zeta_to_h5")
    calls = [_call_name(n) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    assert "copy_mf_header" in calls, (
        "fit_zeta_to_h5 no longer calls copy_mf_header, so the ordering "
        "half of the striping guard is measuring nothing")
    assert "write_isdf_header" in calls
    withs = [n for n in ast.walk(fn) if isinstance(n, (ast.With, ast.AsyncWith))
             for it in n.items if _call_name(it.context_expr) == "SlabIO"]
    assert withs, "fit_zeta_to_h5 no longer opens SlabIO in a with-statement"


#: The order as it stood BEFORE ``96a6399`` — rank-0 serial h5py creates the
#: inode, MPI-IO never sees it, the file takes the directory default.  Kept
#: as source text rather than described in prose: this is the thing the guard
#: has to catch, and a red twin that paraphrases it proves nothing.
_OLD_ORDER = '''
def fit_zeta_to_h5(wfn, sym, meta, mesh_xy, output_file):
    _replace_inode_for_write(output_file)
    if jax.process_index() == 0:
        copy_mf_header(_wfn_src_path, output_file, dst_mode='w')
        write_isdf_header(output_file, _isdf_hdr, mode='a')
    with SlabIO(output_file, mode='a', mesh=mesh_xy) as io:
        io.create_dataset('zeta_q_G', shape=shape, dtype=np.complex128)
'''

#: Right order, wrong mode — the single-token revert.
_RIGHT_ORDER_WRONG_MODE = '''
def fit_zeta_to_h5(wfn, sym, meta, mesh_xy, output_file):
    with SlabIO(output_file, mode='w', mesh=mesh_xy):
        pass
    if jax.process_index() == 0:
        copy_mf_header(_wfn_src_path, output_file, dst_mode='w')
'''

#: The collective create is THERE but on the wrong side of the header
#: append — the ordering defect in isolation, with nothing else wrong.
_ORDER_INVERTED = '''
def fit_zeta_to_h5(wfn, sym, meta, mesh_xy, output_file):
    if jax.process_index() == 0:
        copy_mf_header(_wfn_src_path, output_file, dst_mode='a')
    with SlabIO(output_file, mode='w', mesh=mesh_xy):
        pass
'''

#: No collective create at all.
_NO_COLLECTIVE_CREATE = '''
def fit_zeta_to_h5(wfn, sym, meta, mesh_xy, output_file):
    if jax.process_index() == 0:
        copy_mf_header(_wfn_src_path, output_file, dst_mode='a')
    with SlabIO(output_file, mode='a', mesh=mesh_xy) as io:
        io.write_slab('zeta_q_G', A)
'''

#: The order as it stands today, reduced.  The POSITIVE control: a checker
#: that flagged everything would pass its red twins and fail the real file.
_CURRENT_ORDER = '''
def fit_zeta_to_h5(wfn, sym, meta, mesh_xy, output_file):
    with SlabIO(output_file, mode='w', mesh=mesh_xy):
        pass
    if jax.process_index() == 0:
        copy_mf_header(_wfn_src_path, output_file, dst_mode='a')
        write_isdf_header(output_file, _isdf_hdr, mode='a')
'''


def test_the_striping_guard_can_fail_on_the_old_order():
    """RED TWIN.  The pre-96a6399 source, run through the same checker.

    All three defects at once, each named separately — because the fixes
    differ: the explicit ``_replace_inode_for_write`` is a leftover, the
    ``dst_mode='w'`` is the header call re-creating the file, and the third
    is that the ONLY SlabIO context in that version was ``mode='a'`` for the
    payload, so nothing created the inode collectively at all.  (The
    ordering violation proper needs a ``mode='w'`` create on the wrong side
    of the append, which is :data:`_ORDER_INVERTED` below — the old code
    could not exhibit it, because it had no collective create to misplace.)
    """
    bad = check_striping_create_order(_OLD_ORDER)
    joined = " | ".join(bad)
    assert len(bad) == 3, joined
    assert "dst_mode='w'" in joined
    assert "_replace_inode_for_write" in joined
    assert "no `with SlabIO(..., mode='w', ...)` context" in joined


def test_the_striping_guard_can_fail_on_each_defect_alone():
    """RED TWIN, one defect at a time — the guard is not one big or.

    A checker that only fired when everything was wrong would sail through
    the single-token revert, which is the realistic regression.
    """
    only_mode = check_striping_create_order(_RIGHT_ORDER_WRONG_MODE)
    assert len(only_mode) == 1 and "dst_mode='w'" in only_mode[0]

    no_create = check_striping_create_order(_NO_COLLECTIVE_CREATE)
    assert len(no_create) == 1
    assert "no `with SlabIO(..., mode='w', ...)` context" in no_create[0]
    assert "cb_nodes" in no_create[0]           # the MECHANISM, in the report

    inverted = check_striping_create_order(_ORDER_INVERTED)
    assert len(inverted) == 1
    assert "runs BEFORE the SlabIO(mode='w') create" in inverted[0]
    assert "striping hints are silently discarded" in inverted[0]


def test_the_striping_guard_passes_the_current_order():
    """POSITIVE CONTROL for the twins above: the shape it must accept."""
    assert check_striping_create_order(_CURRENT_ORDER) == []


def test_the_striping_guard_refuses_a_source_without_its_target():
    """A renamed or deleted writer must FAIL, not pass vacuously."""
    with pytest.raises(AssertionError, match="no function named"):
        check_striping_create_order("def something_else():\n    pass\n")


# ===========================================================================
# GUARD 2 — zeta_rcond: five mirrors collapsed to one constant (design D6)
# ===========================================================================
# Survey §4.1: SIX copies of 1e-8, "kept in sync by comment", with
# gw_config.py:776 saying outright "Mirrored by the isdf/core.py +
# gw/isdf_fitting.py signature defaults" and NOTHING in the tree asserting
# that they agree.  Survey §4.2: the value moved three times in ONE DAY
# (1e-10 -> 1e-6 -> 1e-8, 2026-07-21), one of those moves explicitly
# re-freezing a gate.  The VALUE is the owner's (§R19, confirm-not-tune);
# the AGREEMENT is this suite's.

#: The five signatures that used to carry their own literal.
_MIRRORS = (
    ("gw.isdf_fitting", "fit_zeta_to_h5"),
    ("isdf.core", "_factor_c_q_replicated"),
    ("isdf.core", "_factor_c_q_replicated_qparallel"),
    ("isdf.core", "_factor_c_q_distributed_rank_truncate"),
    ("isdf.core", "factor_c_q"),
)


def check_no_literal_zeta_rcond_default(source: str) -> list:
    """Function defs whose ``zeta_rcond`` default is a LITERAL.  THE RATCHET.

    The collapse is worth nothing if the next signature to be written
    re-types ``zeta_rcond: float = 1e-8``: the copies come back one at a
    time, each individually harmless, and the comment that says they are
    mirrored goes on being the only thing holding them together.  This
    refuses a literal default anywhere, so the only spelling that survives
    review is the imported constant.

    EXACT NAME MATCH.  ``transverse_zeta_rcond`` is a DIFFERENT knob
    (``1e-10``, no env twin by policy, scorecard AV) and is deliberately
    NOT ratcheted here — it has one definition, not five, and folding it in
    would be this guard quietly acquiring a second subject.
    """
    tree = ast.parse(source)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        positional = list(args.posonlyargs) + list(args.args)
        pairs = list(zip(positional[len(positional) - len(args.defaults):],
                         args.defaults))
        pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults)
                  if d is not None]
        for arg, default in pairs:
            if arg.arg != "zeta_rcond":
                continue
            if isinstance(default, ast.Constant) and not isinstance(
                    default.value, (str, bytes, type(None), bool)):
                bad.append(
                    f"{node.name} (line {node.lineno}) declares "
                    f"zeta_rcond={default.value!r} as a LITERAL default.  "
                    f"It must be gw.gw_config.ZETA_RCOND_DEFAULT — five "
                    f"mirrors of one number kept in sync by comment is what "
                    f"this collapse removed, and the number moved three "
                    f"times in one day (2026-07-21) while they were mirrors")
    return bad


def check_zeta_rcond_defaults_name_the_constant(source: str) -> list:
    """Every ``zeta_rcond`` default, and whether it is the constant NAME.

    Returns ``(function_name, spelling)`` pairs.  The positive half of the
    ratchet: "no literals" is satisfiable by having no defaults at all, and
    a signature that dropped its default would change the call contract
    silently.
    """
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        positional = list(args.posonlyargs) + list(args.args)
        pairs = list(zip(positional[len(positional) - len(args.defaults):],
                         args.defaults))
        pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults)
                  if d is not None]
        for arg, default in pairs:
            if arg.arg == "zeta_rcond":
                found.append((node.name,
                              default.id if isinstance(default, ast.Name)
                              else ast.dump(default)))
    return found


@pytest.mark.parametrize("path", [ISDF_CORE, ISDF_FITTING],
                         ids=["isdf.core", "gw.isdf_fitting"])
def test_no_function_carries_a_literal_zeta_rcond_default(path):
    """THE ANTI-REINTRODUCTION RATCHET.  Pure AST, no jax."""
    bad = check_no_literal_zeta_rcond_default(_read(path))
    assert bad == [], (f"{os.path.basename(path)} re-grew a literal "
                       f"zeta_rcond default:\n  " + "\n  ".join(bad))


@pytest.mark.parametrize("path,want", [(ISDF_CORE, 4), (ISDF_FITTING, 1)],
                         ids=["isdf.core", "gw.isdf_fitting"])
def test_every_zeta_rcond_default_names_the_constant(path, want):
    """The positive half, and the census: FOUR in isdf.core, ONE in
    isdf_fitting — the five mirrors of survey §4.1, now all one name.

    Counting them is what makes this non-vacuous: "every default is the
    constant" is trivially true of a file with no defaults.
    """
    found = check_zeta_rcond_defaults_name_the_constant(_read(path))
    assert len(found) == want, (
        f"{os.path.basename(path)} has {len(found)} zeta_rcond defaults, "
        f"expected {want} (survey §4.1's mirror census): {found}")
    wrong = [(n, s) for n, s in found if s != "ZETA_RCOND_DEFAULT"]
    assert wrong == [], f"not the constant: {wrong}"


@pytest.mark.parametrize("path", [ISDF_CORE, ISDF_FITTING],
                         ids=["isdf.core", "gw.isdf_fitting"])
def test_both_producers_import_the_constant_from_gw_config(path):
    """The import edge D6 rides on, asserted rather than assumed.

    Direction is safe and FREE because ``isdf/core.py`` already imports
    ``gw.gw_config`` at module scope (documented L1→L1; ``gw_config`` is
    deliberately jax-free), so the collapse added ZERO new import edges.  If
    someone re-routes the constant through a new module, that claim stops
    being true and this cell says so.
    """
    tree = ast.parse(_read(path))
    ok = any(isinstance(n, ast.ImportFrom)
             and (n.module or "").endswith("gw_config")
             and any(a.name == "ZETA_RCOND_DEFAULT" for a in n.names)
             for n in ast.walk(tree))
    assert ok, (f"{os.path.basename(path)} does not import "
                f"ZETA_RCOND_DEFAULT from gw_config")


def test_the_zeta_rcond_ratchet_can_fail():
    """RED TWIN.  A synthetic def with the literal back in the signature.

    Two spellings, because both are what a reintroduction actually looks
    like: an annotated keyword-only default and a plain positional one.
    """
    reintroduced = (
        "def solve_zeta(a, b, *, zeta_rcond: float = 1e-8):\n"
        "    return a\n"
        "\n"
        "def factor_c_q(a, zeta_rcond=1e-10):\n"
        "    return a\n"
    )
    bad = check_no_literal_zeta_rcond_default(reintroduced)
    assert len(bad) == 2, bad
    joined = " | ".join(bad)
    assert "solve_zeta" in joined and "1e-08" in joined
    assert "factor_c_q" in joined and "1e-10" in joined
    assert "ZETA_RCOND_DEFAULT" in joined      # the FIX, in the message

    # POSITIVE CONTROL: the constant spelling passes, and the sibling knob
    # is deliberately untouched by this ratchet.
    good = ("def factor_c_q(a, *, zeta_rcond: float = ZETA_RCOND_DEFAULT,\n"
            "               transverse_zeta_rcond: float = 1e-10):\n"
            "    return a\n")
    assert check_no_literal_zeta_rcond_default(good) == []
    assert check_zeta_rcond_defaults_name_the_constant(good) == \
        [("factor_c_q", "ZETA_RCOND_DEFAULT")]


def test_the_constant_is_still_1e_8():
    """VALUES UNCHANGED is part of the D6 contract.

    ``gw.gw_config`` is deliberately jax-free, so this pin costs no import.
    The VALUE is the owner's question (§R19 records that lowering it on a
    noise-floor argument would have cost a 5000 eV error and that 1e-8 is
    "doing exactly the job it was measured into") — this suite pins, it does
    not tune.  ``1e-10 -> 1e-6 -> 1e-8`` all happened on 2026-07-21; the pin
    is what makes a fourth move deliberate.
    """
    if not os.path.isdir(_SRC):
        pytest.skip("no lorrax host tree (standalone install)")
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from gw.gw_config import ZETA_RCOND_DEFAULT, _DEFAULTS
    assert ZETA_RCOND_DEFAULT == 1e-8
    assert _DEFAULTS["zeta_rcond"] == ZETA_RCOND_DEFAULT
    assert _DEFAULTS["zeta_rcond"] is ZETA_RCOND_DEFAULT, (
        "the deck default is a COPY of the constant rather than the "
        "constant: an equal-but-separate float is exactly the mirror this "
        "collapse removed")


# ===========================================================================
# GUARD 2b — the same agreement through inspect.  IMPORTS jax.
# ===========================================================================
# Split from the AST half deliberately: `isdf.core` imports jax at module
# scope, so these two cells cost a jax import that the whole section above
# does not.  Keeping them apart means the ratchet still runs in a jax-free
# interpreter, which is where a source-level guard belongs.

def _import_mirrors():
    if not os.path.isdir(_SRC):
        pytest.skip("no lorrax host tree (standalone install)")
    pytest.importorskip("jax")
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    import importlib
    out = []
    for mod_name, fn_name in _MIRRORS:
        mod = importlib.import_module(mod_name)
        out.append((f"{mod_name}.{fn_name}", getattr(mod, fn_name)))
    return out


def test_the_five_zeta_rcond_defaults_agree_at_runtime():
    """``inspect.signature`` on all five — the survey's §4.5 recommendation.

    The AST half proves the SOURCE says ``ZETA_RCOND_DEFAULT``; this proves
    the name resolves to the same object at import time in every one of the
    five.  They are different claims: a module that shadowed the name, or
    imported it from a stale copy, would satisfy the first and fail this.
    """
    import inspect
    from gw.gw_config import ZETA_RCOND_DEFAULT

    seen = {}
    for label, fn in _import_mirrors():
        params = inspect.signature(fn).parameters
        assert "zeta_rcond" in params, f"{label} lost its zeta_rcond parameter"
        seen[label] = params["zeta_rcond"].default
    assert len(seen) == 5, seen
    wrong = {k: v for k, v in seen.items() if v != ZETA_RCOND_DEFAULT}
    assert wrong == {}, (
        f"the zeta_rcond mirrors drifted from gw_config's "
        f"ZETA_RCOND_DEFAULT={ZETA_RCOND_DEFAULT!r}: {wrong}")
    assert set(seen.values()) == {1e-8}


def test_the_runtime_agreement_check_can_fail():
    """RED TWIN for the runtime half, with the same comparison.

    A stub whose default drifted must be caught by the identical expression
    the real check uses — asserting through the same ``inspect.signature``
    read is what makes this a twin of the check rather than a restatement
    of ``!=``.
    """
    import inspect
    from gw.gw_config import ZETA_RCOND_DEFAULT

    def drifted(a, *, zeta_rcond: float = 1e-6):
        return a

    def missing(a):
        return a

    params = inspect.signature(drifted).parameters
    assert params["zeta_rcond"].default != ZETA_RCOND_DEFAULT
    assert "zeta_rcond" not in inspect.signature(missing).parameters
