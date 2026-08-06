"""Varying-manual-axes marking: the version table, and a tree-wide AST census.

From jax 0.7.0 ``shard_map`` tracks varying manual axes (VMA) and a
scan / fori_loop / while_loop carry whose init is REPLICATED but whose body
writes a device-varying value is rejected at trace time.  Ten BSE tests failed
that way on the 0.7.0 container, all at one site.

Two things are worth pinning, and neither needs a GPU:

1. **The version decision table.**  The defect that motivated this file was a
   compatibility path nobody could see: ``try: lax.pcast / except
   AttributeError: identity``, commented "jax <= 0.8 has no VMA tracking".
   Tracking actually starts at 0.7.0 and 0.7.x has no ``lax.pcast``, so the
   except-branch installed a NO-OP on exactly the versions that enforce the
   rule.  ``vma.select_mode`` is pure, so every row — including the two the
   running container cannot exhibit — is checkable by construction.

2. **The census.**  The tests only reach the sites they reach; the tree is
   larger than the tests.  ``test_no_unmarked_replicated_carry`` re-derives,
   from the AST, every loop carry that is reachable from a VMA-CHECKED
   ``shard_map`` and initialised by a shape/dtype-only constructor, and
   requires it to be marked.  A new unmarked accumulator fails here on the
   login node instead of on a container nobody runs today.

Facts the census encodes, all MEASURED on the 0.7.0 container (JID 56405696,
``_vma_reports/vma_probe*.py``), because each one changes the scope:

  * ``jnp.zeros_like(x)`` PROPAGATES x's varying axes — those carries need no
    mark.  Only shape/dtype constructors start with an empty set.
  * ``lax.scan`` with ``init=None`` has no carry and cannot mismatch.
  * ``check_rep=False`` (experimental spelling) / ``check_vma=False`` (the
    ``jax.shard_map`` spelling) switch the check OFF, so those bodies are
    exempt.  The two kwargs are NOT interchangeable: passing the wrong one is
    a TypeError on 0.7.0, which is what ``test_check_kwarg_matches_import``
    guards.
  * Over-marking is NOT safe: a carry that is invariant by construction and
    leaves through a replicated ``out_specs`` entry FAILS if marked.  So the
    census reports sites; it does not authorise a blanket edit.
"""
from __future__ import annotations

import ast
import os

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))


# ===========================================================================
# 1.  the version decision table, by construction
# ===========================================================================

def test_select_mode_table():
    """Every row of the measured table, including the two 0.7.0 cannot show.

        jax     VMA tracked?   lax.pvary            lax.pcast
        0.5.3   no             absent               absent
        0.7.0   YES            present              absent
        0.7.2   YES            present              absent
        0.9.0   YES            present (deprecated) present
    """
    from common.vma import select_mode

    # 0.5.3 — no tracking, neither spelling.  Identity is the ONLY behaviour.
    assert select_mode(False, False, (0, 5, 3)) == "identity"
    # 0.7.0 / 0.7.2 / 0.8 — tracking, pvary only.  This is the range the old
    # try/except guard silently no-op'd.
    assert select_mode(False, True, (0, 7, 0)) == "lax.pvary"
    assert select_mode(False, True, (0, 7, 2)) == "lax.pvary"
    assert select_mode(False, True, (0, 8, 1)) == "lax.pvary"
    # 0.9 — both present; the newer spelling wins.
    assert select_mode(True, True, (0, 9, 0)) == "lax.pcast"
    assert select_mode(True, False, (0, 10, 0)) == "lax.pcast"


def test_select_mode_refuses_a_tracking_jax_with_no_spelling():
    """The case the replaced guard got wrong: announce and refuse, never degrade.

    A future jax that renames the primitive again must not fall through to the
    identity.  That is the whole defect: an unmarked carry on a jax that
    enforces marking fails at trace time with a message about varying manual
    axes and no hint that a compatibility path chose to do nothing.
    """
    from common.vma import VmaSupportError, select_mode

    with pytest.raises(VmaSupportError) as exc:
        select_mode(False, False, (0, 9, 0))
    msg = str(exc.value)
    assert "lax.pcast" in msg and "lax.pvary" in msg, \
        "the refusal must name both spellings it looked for"
    assert "src/common/vma.py" in msg, "the refusal must name the fix"


def test_mark_varying_is_not_the_identity_on_a_tracking_jax():
    """The single assertion the replaced guard would have failed.

    On 0.5.3 the identity is correct; on anything >= 0.7.0 an identity here
    means every marked carry in the tree is unmarked and the shard_maps that
    need it fail.
    """
    import jax
    from common import vma

    ver = tuple(jax.version.__version_info__)
    if ver < vma.VMA_TRACKING_SINCE:
        assert vma.vma_mode().startswith("identity"), (
            "jax %s does not track VMA, so the marking must be the identity; "
            "got %s" % (ver, vma.vma_mode()))
    else:
        assert not vma.vma_mode().startswith("identity"), (
            "jax %s TRACKS varying manual axes, so mark_varying must resolve "
            "to a real primitive; it resolved to %s, which is the exact "
            "no-op-on-the-version-that-needs-it defect this module replaced"
            % (ver, vma.vma_mode()))


# ===========================================================================
# 2.  the tree-wide census
# ===========================================================================

LOOPS = ("scan", "while_loop", "fori_loop")
SHARDMAP_NAMES = ("shard_map", "_shard_map_fn", "smap", "_smap")
# Constructors taking SHAPE and DTYPE only: no data flows in, so the result's
# varying-manual-axes set is empty and a body that writes varying data into it
# breaks the carry-type check.
PURE_REPLICATED_CTORS = ("zeros", "ones", "empty", "eye", "arange",
                         "linspace", "full", "tri", "identity")
# *_like INHERITS the operand's axes (measured), so these are already varying.
LIKE_CTORS = ("zeros_like", "ones_like", "empty_like", "full_like")
# an init already routed through a marking primitive
MARK_FNS = ("mark_varying", "pvary", "pcast", "_pcast")


def _name_of(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_shardmap_call(node):
    if not isinstance(node, ast.Call):
        return False
    n = _name_of(node.func)
    if n is None:
        return False
    tail = n.rsplit(".", 1)[-1]
    if tail in SHARDMAP_NAMES:
        return True
    if tail == "partial" and node.args:
        a0 = _name_of(node.args[0])
        if a0 and a0.rsplit(".", 1)[-1] in SHARDMAP_NAMES:
            return True
    return False


def _vma_checked(call_node):
    """False when this shard_map turns the check off."""
    if not isinstance(call_node, ast.Call):
        return True
    for kw in call_node.keywords:
        if kw.arg in ("check_rep", "check_vma"):
            if isinstance(kw.value, ast.Constant) and kw.value.value is False:
                return False
    return True


def _loop_kind(node):
    if not isinstance(node, ast.Call):
        return None
    n = _name_of(node.func)
    if n is None:
        return None
    tail = n.rsplit(".", 1)[-1]
    return tail if tail in LOOPS and ("lax" in n or n == tail) else None


def _carry_init(node, kind):
    for kw in node.keywords:
        if kw.arg in ("init", "init_val"):
            return kw.value
    idx = {"scan": 1, "fori_loop": 3, "while_loop": 2}[kind]
    return node.args[idx] if len(node.args) > idx else None


def _classify(expr, scopes, depth=0):
    """REPLICATED | MARKED | INHERITS | FROM_DATA | CONST for one carry element."""
    kinds = set()

    def walk(e, d):
        if isinstance(e, ast.Call):
            nm = (_name_of(e.func) or "").rsplit(".", 1)[-1]
            if nm in PURE_REPLICATED_CTORS:
                kinds.add("REPLICATED")
                return
            if nm in LIKE_CTORS:
                kinds.add("INHERITS")
                return
            if nm in MARK_FNS:
                kinds.add("MARKED")
                return
        if isinstance(e, ast.Attribute) and e.attr in (
                "shape", "dtype", "size", "ndim", "T"):
            return                       # metadata, not a data flow
        if isinstance(e, ast.Name):
            if d < 4:
                for sc in scopes:
                    if e.id in sc:
                        walk(sc[e.id], d + 1)
                        return
            kinds.add("FROM_DATA")
            return
        if isinstance(e, ast.Constant):
            return
        for ch in ast.iter_child_nodes(e):
            walk(ch, d)

    if expr is None:
        return "CONST"
    walk(expr, depth)
    for k in ("MARKED", "FROM_DATA", "INHERITS", "REPLICATED"):
        if k in kinds:
            return k
    return "CONST"


def _assigns(node):
    out = {}
    body = node.body if isinstance(node.body, list) else [node.body]
    for stmt in body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                    and isinstance(n.targets[0], ast.Name):
                out.setdefault(n.targets[0].id, n.value)
    return out


def _census_module(rel, src):
    tree = ast.parse(src, filename=rel)
    funcs, qual_of = {}, {}

    def index(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + ch.name
            elif isinstance(ch, ast.Lambda):
                q = prefix + "<lambda@%d>" % ch.lineno
            else:
                index(ch, prefix)
                continue
            funcs[q] = ch
            qual_of[id(ch)] = q
            index(ch, q + ".")

    index(tree, "")

    entry = {}

    def add(q, checked):
        entry[q] = entry.get(q, False) or checked

    def by_tail(nm, checked):
        tail = nm.rsplit(".", 1)[-1]
        for q in funcs:
            if q == tail or q.endswith("." + tail):
                add(q, checked)

    for q, node in funcs.items():
        for dec in getattr(node, "decorator_list", []):
            if _is_shardmap_call(dec) or (
                    _name_of(dec) or "").rsplit(".", 1)[-1] in SHARDMAP_NAMES:
                add(q, _vma_checked(dec))
    for node in ast.walk(tree):
        if not _is_shardmap_call(node):
            continue
        args = list(node.args) + [kw.value for kw in node.keywords
                                  if kw.arg in (None, "f", "fun")]
        if not args:
            continue
        a0, chk = args[0], _vma_checked(node)
        if isinstance(a0, ast.Lambda):
            if id(a0) in qual_of:
                add(qual_of[id(a0)], chk)
        elif isinstance(a0, ast.Call):
            inner = _name_of(a0.func)
            if inner and inner.rsplit(".", 1)[-1] == "partial" and a0.args:
                nm = _name_of(a0.args[0])
                if nm:
                    by_tail(nm, chk)
        else:
            nm = _name_of(a0)
            if nm:
                by_tail(nm, chk)

    calls = {}
    for q, node in funcs.items():
        got = set()
        body = node.body if isinstance(node.body, list) else [node.body]
        for stmt in body:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Call):
                    nm = _name_of(n.func)
                    if nm:
                        tail = nm.rsplit(".", 1)[-1]
                        got.update(q2 for q2 in funcs
                                   if q2 == tail or q2.endswith("." + tail))
        got.update(q2 for q2 in funcs if q2.startswith(q + "."))
        calls[q] = got

    seen, stack = set(), [q for q, c in entry.items() if c]
    while stack:
        q = stack.pop()
        if q in seen:
            continue
        seen.add(q)
        stack.extend(calls.get(q, ()))

    bad = []
    for q in sorted(seen):
        node = funcs.get(q)
        if node is None:
            continue
        scopes = []
        parts = q.split(".")
        for i in range(len(parts), 0, -1):
            anc = ".".join(parts[:i])
            if anc in funcs:
                scopes.append(_assigns(funcs[anc]))
        body = node.body if isinstance(node.body, list) else [node.body]
        for stmt in body:
            for n in ast.walk(stmt):
                kind = _loop_kind(n)
                if not kind:
                    continue
                # attribute the site to its innermost reachable function
                inner = [q2 for q2 in seen
                         if q2.startswith(q + ".") and q2 in funcs
                         and any(x is n for x in ast.walk(funcs[q2]))]
                if inner:
                    continue
                init = _carry_init(n, kind)
                elems = (list(init.elts)
                         if isinstance(init, (ast.Tuple, ast.List)) else [init])
                for j, e in enumerate(elems):
                    if _classify(e, scopes) == "REPLICATED":
                        bad.append((rel, n.lineno, kind, q, j))
    return bad


def _iter_src():
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", "build", "build_host")]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                p = os.path.join(dirpath, fn)
                try:
                    yield os.path.relpath(p, SRC), open(p, encoding="utf-8").read()
                except (OSError, UnicodeDecodeError):
                    continue


def test_no_unmarked_replicated_carry_under_a_checked_shard_map():
    """Every shape/dtype-constructed loop carry inside a CHECKED shard_map is marked.

    This is the census, not the ten tests.  The ten failures on 0.7.0 all
    landed on ONE of the eight sites (bse_ring_comm.apply_V_ring's A0); the
    other seven were only reachable once that one was fixed, or by callers the
    suite does not exercise.  A site added later gets caught here.

    If this fails on a site you believe is invariant by construction (its body
    feeds only psum/pmax results), do NOT mark it — marking a carry that
    leaves through a replicated out_specs entry is itself an error on 0.7.0.
    Either leave the shard_map's check off, or restructure.
    """
    bad = []
    for rel, src in _iter_src():
        if "shard_map" not in src:
            continue
        bad.extend(_census_module(rel, src))
    assert not bad, (
        "unmarked replicated loop carry under a VMA-checked shard_map — these "
        "raise 'the varying manual axes do not match' on jax >= 0.7.0:\n"
        + "\n".join("  %s:%d  %s in %s (carry element %d)" % b for b in bad))


def test_check_kwarg_matches_the_shard_map_that_was_imported():
    """``check_rep=`` and ``check_vma=`` belong to DIFFERENT shard_map symbols.

    Measured on jax 0.7.0::

        jax.shard_map                        -> (f, out_specs, axis_names,
                                                 in_specs, mesh, check_vma)
        jax.experimental.shard_map.shard_map -> (f, mesh, in_specs, out_specs,
                                                 check_rep, auto)

    Passing the wrong one raises TypeError; it is not ignored.  Roughly sixty
    shard_maps in this tree rely on ``check_rep=False`` to switch VMA checking
    off, and every one of them must therefore be calling the EXPERIMENTAL
    symbol.  A file that switches its import to ``from jax import shard_map``
    while keeping ``check_rep=`` breaks on every jax >= 0.7 at once.
    """
    offenders = []
    for rel, src in _iter_src():
        if "check_rep" not in src:
            continue
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError:
            continue
        # AST, not grep: this module's own docstring names the kwarg.
        uses = any(isinstance(n, ast.keyword) and n.arg == "check_rep"
                   for n in ast.walk(tree))
        if not uses:
            continue
        if "from jax.experimental.shard_map import shard_map" in src \
                or "from jax.experimental import shard_map" in src:
            continue
        offenders.append(rel)
    assert not offenders, (
        "these files pass check_rep= but do not import shard_map from "
        "jax.experimental.shard_map; on jax >= 0.7.0 jax.shard_map has no "
        "check_rep kwarg and raises TypeError:\n  " + "\n  ".join(offenders))
