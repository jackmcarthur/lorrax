"""AST gate: ζ reuse stays live for bispinor runs, and never drops Σ^B.

Source-level, jax-free, runs on the login python — the production check
is a two-leg job (fit, then reuse in the same directory, compared
EXACT-0), which is far too expensive to be a regression test.  What is
cheap to pin is the SHAPE of the code that job certified:

1. The one pre-fit contract validates charge and every transverse artifact
   independently; a failed later channel cannot invalidate an accepted earlier
   channel.
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

5. Only missing channels reach the incumbent fit/stamp path; accepted files
   are never opened by the writer during resume.

6. The transverse ζ files get a provenance stamp.  Without it every
   bispinor rerun refits: rule 4 of ``_zeta_reuse_ok`` is "no
   provenance ⇒ refit", and the μ_L loop used to write none.
"""
from __future__ import annotations

import ast
import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "gw",
                   "gw_init.py")


def _function_tree(name):
    with open(SRC, encoding="utf-8") as fh:
        mod = ast.parse(fh.read(), filename=SRC)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return mod, node
    raise AssertionError("gw_init.%s not found" % name)


def _fit_zeta_tree():
    return _function_tree("fit_zeta")


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


def test_reuse_decisions_are_independent_and_use_one_validator():
    _, contract = _function_tree("_resolve_zeta_fit_contract")
    direct_assigns = {
        t.id: n for n in contract.body if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)
    }
    charge = direct_assigns.get("reuse_charge")
    transverse = direct_assigns.get("reuse_transverse")
    assert charge is not None and transverse is not None, (
        "the zeta contract must publish direct charge and transverse reuse "
        "verdicts before any fit planning")
    assert len(_calls(charge.value, "_zeta_reuse_ok")) == 1
    assert len(_calls(transverse.value, "_zeta_reuse_ok")) == 1
    assert charge.lineno < transverse.lineno
    # Both assignments are direct function-body statements, rather than the
    # transverse probe living under ``if reuse_charge``.  Thus a bad charge
    # artifact cannot prevent authentication/reuse of completed current files.
    assert charge in contract.body and transverse in contract.body


def test_every_reuse_check_probes_the_dataset_extent():
    _, contract = _function_tree("_resolve_zeta_fit_contract")
    calls = _calls(contract, "_zeta_reuse_ok")
    assert len(calls) == 2, (
        "expected exactly two _zeta_reuse_ok call sites in the contract "
        "(charge, transverse); found %d" % len(calls))
    for c in calls:
        assert any(kw.arg == "n_rmu_expected" for kw in c.keywords), (
            "a _zeta_reuse_ok call omits n_rmu_expected, so a ζ whose "
            "dataset extent disagrees with its header would be reused")


def test_reuse_contract_precedes_and_bypasses_fit_only_planners():
    """A complete cache never enters either charge or transverse fit HWM."""
    _, contract = _function_tree("_resolve_zeta_fit_contract")
    assert not _calls(contract, "_plan_gflat_chunks_for_channel")
    assert not _calls(contract, "load_centroids_band_chunked")

    _, prepare = _function_tree("prepare_isdf_and_wavefunctions")
    resolves = _calls(prepare, "_resolve_zeta_fit_contract")
    plans = _calls(prepare, "_plan_gflat_chunks_for_channel")
    assert len(resolves) == 1 and len(plans) == 1
    assert resolves[0].lineno < plans[0].lineno
    guarded = [n for n in ast.walk(prepare)
               if isinstance(n, ast.If)
               and ast.unparse(n.test) == "not charge_zeta_reused"
               and any(x is plans[0] for x in ast.walk(n))]
    assert len(guarded) == 1
    assert ast.unparse(guarded[0].test) == "not charge_zeta_reused"

    _, fit_zeta = _fit_zeta_tree()
    transverse_plans = _calls(fit_zeta, "_plan_gflat_chunks_for_channel")
    assert len(transverse_plans) == 1
    reuse_if = next(n for n in ast.walk(fit_zeta)
                    if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "zeta_contract.reuse")
    assert reuse_if.lineno < transverse_plans[0].lineno
    assert any(isinstance(n, ast.Return) for n in ast.walk(reuse_if))
    transverse_plan_guard = [n for n in ast.walk(fit_zeta)
                             if isinstance(n, ast.If)
                             and ast.unparse(n.test) == "not all(_reuse_T)"
                             and any(x is transverse_plans[0]
                                     for x in ast.walk(n))]
    assert len(transverse_plan_guard) == 1


def test_only_missing_channels_reach_fit_writers():
    _, fit_zeta = _fit_zeta_tree()
    fits = _calls(fit_zeta, "fit_zeta_to_h5")
    assert len(fits) == 2, (
        "expected the incumbent charge and transverse fit call sites only; "
        "found %d" % len(fits))
    charge_guards = [n for n in ast.walk(fit_zeta)
                     if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "_reuse_charge"
                     and any(x is fits[0] for stmt in n.orelse
                             for x in ast.walk(stmt))]
    assert len(charge_guards) == 1, (
        "the charge writer is not exclusively in the non-reuse branch")

    mu_loops = [n for n in ast.walk(fit_zeta)
                if isinstance(n, ast.For)
                and ast.unparse(n.target) == "mu_L"
                and any(x is fits[1] for x in ast.walk(n))]
    assert len(mu_loops) == 1
    loop = mu_loops[0]
    skip = next((n for n in loop.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "_reuse_T[mu_L - 1]"), None)
    assert skip is not None and any(isinstance(n, ast.Continue)
                                    for n in ast.walk(skip)), (
        "an accepted transverse artifact does not continue past the writer")
    assert skip.lineno < fits[1].lineno


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
