"""Tests for the proposed JAX-support startup gate.

These are negative-controlled: each refusal is proved to fire by CONSTRUCTING
the unsupported condition, not by asserting that today's environment happens
to be fine.  A gate that only ever sees a good environment is untested.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/src")

from common import jax_support as js  # noqa: E402


# ---------------------------------------------------------------- version leg
@pytest.mark.parametrize("vi,supported", [
    ((0, 9, 0), True),
    ((0, 9, 1), True),
    ((0, 9, 99), True),
    ((0, 10, 0), False),   # above the window
    ((0, 5, 3), False),    # the Perlmutter container, measured 2026-08-06
    ((0, 8, 9), False),    # below the floor
])
def test_version_window(monkeypatch, vi, supported):
    monkeypatch.setattr(js, "describe", lambda: {
        "version": ".".join(map(str, vi)), "version_info": vi,
        "release_version": None, "is_dev_build": True,
        "file": "/fake/jax/__init__.py", "shard_map_top_level": False,
    })
    problems = js.check_version()
    assert (problems == []) is supported, problems


def test_dev_stamp_does_not_smuggle_a_bad_version_through(monkeypatch):
    """0.5.3.dev20260806 is stamped TODAY; a string compare would be fooled."""
    monkeypatch.setattr(js, "describe", lambda: {
        "version": "0.5.3.dev20260806", "version_info": (0, 5, 3),
        "release_version": None, "is_dev_build": True,
        "file": "/opt/jax/jax/__init__.py", "shard_map_top_level": False,
    })
    problems = js.check_version()
    assert problems, "a 0.5.3 dev build must not satisfy a >=0.9.0 floor"
    assert "0.5.3.dev20260806" in problems[0]


def test_empty_version_info_refuses(monkeypatch):
    monkeypatch.setattr(js, "describe", lambda: {
        "version": "weird", "version_info": (), "release_version": None,
        "is_dev_build": True, "file": "/x", "shard_map_top_level": False,
    })
    assert js.check_version(), "an unidentifiable jax must not pass"


# ------------------------------------------------------------ private arity leg
def _fake_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def test_arity_mismatch_is_detected(monkeypatch):
    """The measured Perlmutter-container shape must be caught.

    jax 0.5.3: _hash_accelerator_config(hash_obj, accelerators, backend) -> 3
    this tree patches it with 2 -> TypeError on the first jit compile.
    """
    def three(hash_obj, accelerators, backend):
        pass

    fake = _fake_module("jax._src.cache_key", _hash_accelerator_config=three)
    real_import = importlib.import_module

    def fake_import(name):
        if name == "jax._src.cache_key":
            return fake
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    problems = js.check_private_arity()
    assert any("_hash_accelerator_config takes 3" in p for p in problems), problems


def test_absent_private_symbol_is_detected(monkeypatch):
    """backend_compile_and_load is ABSENT on 0.5.3 — the SILENT degradation."""
    fake = _fake_module("jax._src.compiler")  # no backend_compile_and_load
    real_import = importlib.import_module

    def fake_import(name):
        if name == "jax._src.compiler":
            return fake
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    problems = js.check_private_symbols()
    assert any("backend_compile_and_load is ABSENT" in p for p in problems), problems


# ------------------------------------------------------------------ enforce leg
def test_enforce_refuses_with_the_standard_shape(monkeypatch):
    monkeypatch.delenv(js.OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(js, "check_version", lambda: ["jax 0.5.3 from /opt/jax"])
    monkeypatch.setattr(js, "check_private_arity", lambda: [])
    monkeypatch.setattr(js, "check_private_symbols", lambda: [])
    with pytest.raises(js.JaxSupportError) as exc:
        js.enforce()
    text = str(exc.value)
    for field in ("REFUSED:", "got", "want", "fix", "doc"):
        assert field in text, f"refusal is missing {field!r}: {text}"
    assert js.RULE_UNSUPPORTED_VERSION in text


def test_override_downgrades_to_one_announced_line(monkeypatch):
    monkeypatch.setenv(js.OVERRIDE_ENV, "1")
    monkeypatch.setattr(js, "check_version", lambda: ["jax 0.5.3"])
    monkeypatch.setattr(js, "check_private_arity", lambda: [])
    monkeypatch.setattr(js, "check_private_symbols", lambda: [])
    said = []
    js.enforce(announce=said.append)  # must NOT raise
    assert len(said) == 1 and js.OVERRIDE_ENV in said[0], said


def test_override_must_be_exactly_1_not_any_truthy_string(monkeypatch):
    """'0', 'false', 'no' must NOT disable the gate."""
    monkeypatch.setattr(js, "check_version", lambda: ["jax 0.5.3"])
    monkeypatch.setattr(js, "check_private_arity", lambda: [])
    monkeypatch.setattr(js, "check_private_symbols", lambda: [])
    for val in ("0", "false", "no", ""):
        monkeypatch.setenv(js.OVERRIDE_ENV, val)
        with pytest.raises(js.JaxSupportError):
            js.enforce()


def test_clean_environment_does_not_refuse(monkeypatch):
    monkeypatch.delenv(js.OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(js, "check_version", lambda: [])
    monkeypatch.setattr(js, "check_private_arity", lambda: [])
    monkeypatch.setattr(js, "check_private_symbols", lambda: [])
    js.enforce()  # must not raise


# --------------------------------------------------------- live-environment leg
def test_describe_reports_the_real_jax():
    """Scope: this asserts the SHAPE of describe(), not that jax is supported."""
    d = js.describe()
    assert set(d) >= {"version", "version_info", "release_version",
                      "is_dev_build", "file", "shard_map_top_level"}
    assert isinstance(d["version"], str) and d["version"]
