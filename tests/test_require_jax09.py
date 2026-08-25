"""The documented all-driver JAX/JAXLIB preflight is tracked and can refuse."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / "tools" / "require_jax09.py"
_SPEC = importlib.util.spec_from_file_location("lorrax_require_jax09", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CHECK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECK)


@pytest.mark.parametrize(
    "versions, override, expected",
    [
        ({"jax": "0.9.1", "jaxlib": "0.9.1"}, "", 0),
        ({"jax": "0.8.2", "jaxlib": "0.8.2"}, "", 86),
        ({"jax": "0.9.1", "jaxlib": "0.9.1"}, "1", 86),
    ],
)
def test_jax09_preflight_has_positive_and_negative_arms(
        monkeypatch, capsys, versions, override, expected):
    monkeypatch.setattr(_CHECK, "_version", versions.__getitem__)
    monkeypatch.setenv("LORRAX_JAX_UNSUPPORTED_OK", override)
    assert _CHECK.main() == expected
    output = capsys.readouterr()
    if expected == 0:
        assert "JAX09_ENV_OK" in output.out
    else:
        assert "JAX09_ENV_REFUSED" in output.err
