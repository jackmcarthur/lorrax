"""Host-only checked-root cache routing; no JAX/device solve or speed claim."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class RootArray(np.ndarray):
    """Actual small numerical roots with a device-buffer lifetime witness."""
    def is_deleted(self):
        return getattr(self, "deleted", False)


def owner():
    path = Path(__file__).parents[1]/"src/gw/response_bank.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_coulomb_batch")
    calls = []
    def solve(meta, basis, resource, layout, mesh, span, execute):
        calls.append((layout, span, execute))
        return roots(span[1]-span[0])
    scope = dict(_COULOMB_ROOTS={}, dense_layout=lambda meta, cfg, mesh, n: cfg["linalg"],
        _coulomb_roots=solve, _span_rows=lambda mesh, width: lambda value, start: value[start:start+width],
        _span_cursor=lambda mesh, cover, a, b: a)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    return scope, calls


def roots(nq):
    matrices = np.stack([np.array([[3.+q, .1+.2j], [.1-.2j, 2.+q]]) for q in range(nq)])
    values, vectors = np.linalg.eigh(matrices)
    h = ((vectors*np.sqrt(values)[:, None])@vectors.conj().swapaxes(-1, -2)).view(RootArray)
    hi = ((vectors/np.sqrt(values)[:, None])@vectors.conj().swapaxes(-1, -2)).view(RootArray)
    return h, hi, [2]*nq


def fixture():
    scope, calls = owner()
    meta = SimpleNamespace(mu_basis=SimpleNamespace(n_packed=2, n_logical=2))
    bank = dict(coulomb=dict(operator="authenticated-V"))
    mesh = object()
    checked = roots(4)
    key = ("authenticated-V", (0, 4), 2, 2, "local", mesh)
    scope["_COULOMB_ROOTS"][key] = checked
    return scope, calls, meta, bank, mesh, checked, key


def test_face_constructor_borrows_exact_bank_roots_and_supports_without_solve():
    scope, calls, meta, bank, mesh, checked, _ = fixture()
    receipt = {}
    actual = scope["_coulomb_batch"](meta, dict(linalg="distributed"), bank, mesh,
        (0, 4), object(), cache_config=dict(linalg="local"), cache_receipt=receipt)
    assert actual is checked and not calls
    assert actual[0] is checked[0] and actual[1] is checked[1] and actual[2] is checked[2]
    assert receipt["root_cache"] == dict(hit=True, solve_layout="distributed", source_layout="local",
        requested_span=[0, 4], covering_span=[0, 4], fresh_solve=False)


def test_cover_span_preserves_root_bits_support_order_and_existing_backing():
    scope, calls, meta, bank, mesh, checked, _ = fixture()
    checked[2][:] = [1, 2, 0, 2]
    receipt = {}
    actual = scope["_coulomb_batch"](meta, dict(linalg="distributed"), bank, mesh,
        (1, 3), object(), cache_config=dict(linalg="local"), cache_receipt=receipt)
    assert not calls and actual[2] == [2, 0]
    for got, whole in zip(actual[:2], checked[:2]):
        assert np.shares_memory(got, whole)
        np.testing.assert_array_equal(got.view(np.uint64), whole[1:3].view(np.uint64))
    assert receipt["root_cache"]["covering_span"] == [0, 4]
    assert len(scope["_COULOMB_ROOTS"]) == 1  # No second retained full-q resource.


@pytest.mark.parametrize("mismatch", ["absent", "deleted_h", "deleted_hi", "operator", "extent", "mesh"])
def test_missing_or_invalid_bank_cover_keeps_distributed_fresh_admission(mismatch):
    scope, calls, meta, bank, mesh, checked, key = fixture()
    if mismatch == "absent":
        scope["_COULOMB_ROOTS"].clear()
    elif mismatch.startswith("deleted"):
        checked[0 if mismatch == "deleted_h" else 1].deleted = True
    elif mismatch == "operator":
        bank["coulomb"]["operator"] = "different-V"
    elif mismatch == "extent":
        meta.mu_basis.n_logical = 1
    else:
        mesh = object()
    execute, receipt = object(), {}
    result = scope["_coulomb_batch"](meta, dict(linalg="distributed"), bank, mesh,
        (1, 3), execute, cache_config=dict(linalg="local"), cache_receipt=receipt)
    assert calls == [("distributed", (1, 3), execute)]
    assert receipt["root_cache"]["fresh_solve"] is True
    assert result[0] is not checked[0]


def test_unrequested_other_layout_is_not_a_global_cache_identity_relaxation():
    scope, calls, meta, bank, mesh, checked, _ = fixture()
    scope["_coulomb_batch"](meta, dict(linalg="distributed"), bank, mesh, (0, 4), object())
    assert calls[0][0] == "distributed"


def test_constructor_requests_original_cache_dial_but_retains_face_fallback():
    root = Path(__file__).parents[1]/"src/gw"
    tree = ast.parse((root/"shared_pole_constructor.py").read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "response_coulomb_powers")
    assert isinstance(call.args[1], ast.Name) and call.args[1].id == "coulomb_config"
    assert any(k.arg == "cache_config" and isinstance(k.value, ast.Name)
        and k.value.id == "config" for k in call.keywords)
    assert "if execution == 'face' else config" in (root/"shared_pole_constructor.py").read_text()
    wrapper = next(n for n in ast.parse((root/"w_isdf.py").read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name == "response_coulomb_powers")
    native = next(n for n in ast.walk(wrapper) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "_coulomb_batch")
    assert {k.arg for k in native.keywords} >= {"cache_config", "cache_receipt"}
