"""AST gate: ζ reuse stays live for bispinor runs, and never drops Σ^B.

Source-level, jax-free, runs on the login python — the production check
is a two-leg job (fit, then reuse in the same directory, compared
EXACT-0), which is far too expensive to be a regression test.  What is
cheap to pin is the SHAPE of the code that job certified:

1. ``fit_zeta``'s reuse decision is not gated on ``cfg.bispinor``.
   The line ``_reuse = (not cfg.bispinor) and _zeta_reuse_ok(...)``
   switched the cache off for the entire bispinor run and cost a
   measured 318 s of a 660 s GW wall on every rerun (b600 bispinor,
   job 7885966: charge 181.85 s + transverse 135.91 s).

2. Every ``_zeta_reuse_ok`` call passes ``n_rmu_expected`` — the probe
   that catches a good header sitting over a ζ of the wrong μ extent.

3. The bispinor reuse path returns a REBUILT ``transverse_wfn_data``,
   never ``None``.  ``None`` there flows into the Σ kernels, whose Σ^B
   fold-in is a silent no-op on ``None``: rc=0 with Σ^B dropped.  That
   is the exact failure commit 3d89885 fixed on the restart round-trip
   and it must not come back through the reuse door.

4. ``_transverse_wfn_data`` has exactly two call sites — the fit path
   and the reuse path.  Bit-identity of the two legs rests on both
   sampling ψ through the same code; a third, open-coded sampling site
   is how that guarantee would rot.

5. The transverse ζ files get a provenance stamp.  Without it every
   bispinor rerun refits: rule 4 of ``_zeta_reuse_ok`` is "no
   provenance ⇒ refit", and the μ_L loop used to write none.
"""
from __future__ import annotations

import ast
import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "gw",
                   "gw_init.py")


def _fit_zeta_tree():
    with open(SRC, encoding="utf-8") as fh:
        mod = ast.parse(fh.read(), filename=SRC)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "fit_zeta":
            return mod, node
    raise AssertionError("gw_init.fit_zeta not found")


def _calls(tree, name):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id == name:
                out.append(n)
            elif isinstance(f, ast.Attribute) and f.attr == name:
                out.append(n)
    return out


def test_reuse_decision_is_not_gated_on_bispinor():
    _, fit_zeta = _fit_zeta_tree()
    seen = 0
    for n in ast.walk(fit_zeta):
        if not isinstance(n, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "_reuse"
                   for t in n.targets):
            continue
        seen += 1
        # The first assignment must be the bare reuse check.  A BoolOp
        # here is how the bispinor gate was spelled.
        if seen == 1:
            assert isinstance(n.value, ast.Call), (
                "fit_zeta's `_reuse = ...` is not a bare _zeta_reuse_ok "
                "call — the bispinor gate (or another short-circuit) is "
                "back, which switches the ζ cache off for bispinor runs")
            assert isinstance(n.value.func, ast.Name) and \
                n.value.func.id == "_zeta_reuse_ok"
    assert seen >= 1, "no `_reuse` assignment in fit_zeta"


def test_every_reuse_check_probes_the_dataset_extent():
    _, fit_zeta = _fit_zeta_tree()
    calls = _calls(fit_zeta, "_zeta_reuse_ok")
    assert len(calls) == 2, (
        "expected exactly two _zeta_reuse_ok call sites in fit_zeta "
        "(charge, transverse); found %d" % len(calls))
    for c in calls:
        assert any(kw.arg == "n_rmu_expected" for kw in c.keywords), (
            "a _zeta_reuse_ok call omits n_rmu_expected, so a ζ whose "
            "dataset extent disagrees with its header would be reused")


def test_bispinor_reuse_path_returns_rebuilt_transverse_data():
    _, fit_zeta = _fit_zeta_tree()
    returns = [n for n in ast.walk(fit_zeta) if isinstance(n, ast.Return)]
    tuples = [r for r in returns
              if isinstance(r.value, ast.Tuple) and len(r.value.elts) == 3]
    assert tuples, "fit_zeta returns no (path, mem_est, transverse) tuple"
    rebuilt = [r for r in tuples
               if isinstance(r.value.elts[2], ast.Call)
               and isinstance(r.value.elts[2].func, ast.Name)
               and r.value.elts[2].func.id == "_transverse_wfn_data"]
    assert rebuilt, (
        "no return in fit_zeta re-samples ψ at the transverse centroids. "
        "The bispinor reuse path must rebuild transverse_wfn_data; "
        "returning None there silently drops Σ^B (rc=0, wrong physics).")
    # And the None-returning tuple that remains must be the
    # non-bispinor one — i.e. it is guarded by `not cfg.bispinor`.
    nones = [r for r in tuples
             if isinstance(r.value.elts[2], ast.Constant)
             and r.value.elts[2].value is None]
    for r in nones:
        assert _guarded_by_not_bispinor(fit_zeta, r), (
            "a `return ..., None` in fit_zeta is not guarded by "
            "`not cfg.bispinor`; on a bispinor run that drops Σ^B")


def _guarded_by_not_bispinor(fit_zeta, target):
    """True when ``target`` sits under an ``if not cfg.bispinor:`` body."""
    for n in ast.walk(fit_zeta):
        if not isinstance(n, ast.If):
            continue
        t = n.test
        if not (isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not)):
            continue
        v = t.operand
        if not (isinstance(v, ast.Attribute) and v.attr == "bispinor"):
            continue
        if any(x is target for b in n.body for x in ast.walk(b)):
            return True
    return False


def test_transverse_psi_has_exactly_two_call_sites():
    mod, _ = _fit_zeta_tree()
    calls = _calls(mod, "_transverse_wfn_data")
    assert len(calls) == 2, (
        "expected exactly two _transverse_wfn_data call sites (fit path "
        "and reuse path); found %d.  Both legs must sample ψ through "
        "the same code or their bit-identity is not structural."
        % len(calls))


def test_transverse_zeta_files_are_stamped():
    _, fit_zeta = _fit_zeta_tree()
    stamps = _calls(fit_zeta, "stamp_fit_provenance")
    assert len(stamps) >= 2, (
        "fit_zeta stamps fit_provenance on fewer than two ζ files.  The "
        "three transverse ζ need their own stamps or every bispinor "
        "rerun refits (no provenance ⇒ refit).")
    provs = _calls(fit_zeta, "_provenance_T")
    assert provs, "no per-vertex transverse provenance is built"


if __name__ == "__main__":
    _fails = 0
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS %s" % _name)
            except AssertionError as _e:
                _fails += 1
                print("FAIL %s: %s" % (_name, _e))
    print("%d failure(s)" % _fails)
    sys.exit(1 if _fails else 0)
