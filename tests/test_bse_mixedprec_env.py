"""The documented TF32 escape keeps its exact-1 grammar before release."""
import jax
import pytest

from bse.w_ladder_mixedprec import _refuse_unpinned_matmul_precision


@pytest.mark.parametrize("value", [None, "", "0", "1", "true", "on", "yes", " 1 ", "2"])
def test_only_exact_one_bypasses_the_precision_guard(monkeypatch, value):
    name = "LORRAX_MIXEDPREC_ALLOW_TF32"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    with jax.default_matmul_precision("default"):
        if value == "1":
            _refuse_unpinned_matmul_precision()
        else:
            with pytest.raises(RuntimeError, match="complex64 solve requires"):
                _refuse_unpinned_matmul_precision()


def test_pinned_precision_needs_no_escape(monkeypatch):
    monkeypatch.delenv("LORRAX_MIXEDPREC_ALLOW_TF32", raising=False)
    with jax.default_matmul_precision("highest"):
        _refuse_unpinned_matmul_precision()
