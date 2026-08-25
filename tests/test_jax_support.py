"""Tests for the JAX-support startup gate (WIRED 2026-08-06, step 5b).

These are negative-controlled: each refusal is proved to fire by CONSTRUCTING
the unsupported condition, not by asserting that today's environment happens
to be fine.  A gate that only ever sees a good environment is untested.

Since the gate is wired, three further things are pinned here that were not
before, because a wired gate can break a run: that the window is the one
``pyproject.toml`` declares, that the JAX THIS PROCESS IS RUNNING satisfies it
(so a container regression fails a test rather than every job), and that the
startup path really calls it.
"""
from __future__ import annotations

import importlib
import inspect
import sys
import types

import pytest

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/src")

# IMPORT jax HERE, AT COLLECTION, AND ON PURPOSE.
#
# Two tests below stub ``importlib.import_module`` to inject a wrong-arity /
# missing-symbol fake into the private-API probes.  jax's own package init
# ALSO calls ``importlib.import_module`` (plugin discovery, with a second
# positional argument the stub does not take), so if this file's stub is what
# first triggers ``import jax``, jax comes out half-initialised and every
# later test that touches it dies on "partially initialized module 'jax' has
# no attribute 'version'".
#
# Until 2026-08-06 this happened not to bite, for a reason nobody chose: the
# module under test was ``common.jax_support``, and importing it ran
# ``common/__init__.py``, which imports jax.  The gate moved to
# ``runtime.jax_support`` — a package that must stay jax-free at import — so
# the accident is gone.  Stating the dependency is the fix; relying on
# another package's import side effect was never one.
import jax  # noqa: E402,F401
from runtime import jax_support as js  # noqa: E402


# ---------------------------------------------------------------- version leg
@pytest.mark.parametrize("vi,supported", [
    ((0, 7, 0), False),    # retired CUDA-12 launcher
    ((0, 7, 2), False),
    ((0, 9, 1), True),     # current Perlmutter/Frontera generation
    ((0, 9, 99), True),
    ((0, 10, 0), False),   # above the window
    ((0, 6, 9), False),    # below the floor: no VMA tracking, no lax.pvary
    ((0, 5, 3), False),    # the OLD Perlmutter container, measured 2026-08-06
])
def test_version_window(monkeypatch, vi, supported):
    monkeypatch.setattr(js, "describe", lambda: {
        "version": ".".join(map(str, vi)), "version_info": vi,
        "release_version": None, "is_dev_build": True,
        "file": "/fake/jax/__init__.py", "shard_map_top_level": False,
        "jaxlib_version": "0.9.1", "jaxlib_version_info": (0, 9, 1),
        "jaxlib_file": "/fake/jaxlib/__init__.py",
    })
    problems = js.check_version()
    assert (problems == []) is supported, problems


def test_dev_stamp_does_not_smuggle_a_bad_version_through(monkeypatch):
    """0.5.3.dev20260806 is stamped TODAY; a string compare would be fooled.

    Both containers restamp: the 0.5.3 image and the 0.7.0 image BOTH print
    ``.dev20260806``.  Only ``__version_info__`` separates them.
    """
    monkeypatch.setattr(js, "describe", lambda: {
        "version": "0.5.3.dev20260806", "version_info": (0, 5, 3),
        "release_version": None, "is_dev_build": True,
        "file": "/opt/jax/jax/__init__.py", "shard_map_top_level": False,
        "jaxlib_version": "0.9.1", "jaxlib_version_info": (0, 9, 1),
        "jaxlib_file": "/fake/jaxlib/__init__.py",
    })
    problems = js.check_version()
    assert problems, "a 0.5.3 dev build must not satisfy the 0.9-series gate"
    assert "0.5.3.dev20260806" in problems[0]


def test_empty_version_info_refuses(monkeypatch):
    monkeypatch.setattr(js, "describe", lambda: {
        "version": "weird", "version_info": (), "release_version": None,
        "is_dev_build": True, "file": "/x", "shard_map_top_level": False,
        "jaxlib_version": "0.9.1", "jaxlib_version_info": (0, 9, 1),
        "jaxlib_file": "/fake/jaxlib/__init__.py",
    })
    assert js.check_version(), "an unidentifiable jax must not pass"


def test_jaxlib_must_independently_be_in_the_09_series(monkeypatch):
    monkeypatch.setattr(js, "describe", lambda: {
        "version": "0.9.1", "version_info": (0, 9, 1),
        "release_version": "0.9.1", "is_dev_build": False,
        "file": "/fake/jax/__init__.py", "shard_map_top_level": True,
        "jaxlib_version": "0.8.2", "jaxlib_version_info": (0, 8, 2),
        "jaxlib_file": "/wrong/jaxlib/__init__.py",
    })
    problems = js.check_version()
    assert len(problems) == 1
    assert "jaxlib 0.8.2" in problems[0]


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
    monkeypatch.setattr(js, "check_version", lambda: ["jax 0.5.3 from /opt/jax"])
    monkeypatch.setattr(js, "check_private_arity", lambda: [])
    monkeypatch.setattr(js, "check_private_symbols", lambda: [])
    with pytest.raises(js.JaxSupportError) as exc:
        js.enforce()
    text = str(exc.value)
    for field in ("REFUSED:", "got", "want", "fix", "doc"):
        assert field in text, f"refusal is missing {field!r}: {text}"
    assert js.RULE_UNSUPPORTED_VERSION in text


def test_unsupported_version_cannot_be_downgraded(monkeypatch):
    monkeypatch.setattr(js, "check_version", lambda: ["jax 0.5.3"])
    monkeypatch.setattr(js, "check_private_arity", lambda: [])
    monkeypatch.setattr(js, "check_private_symbols", lambda: [])
    said = []
    with pytest.raises(js.JaxSupportError):
        js.enforce(announce=said.append)
    assert said == []


def test_clean_environment_does_not_refuse(monkeypatch):
    monkeypatch.setattr(js, "check_version", lambda: [])
    monkeypatch.setattr(js, "check_private_arity", lambda: [])
    monkeypatch.setattr(js, "check_private_symbols", lambda: [])
    js.enforce()  # must not raise


# --------------------------------------------------------- live-environment leg
def test_describe_reports_the_real_jax():
    """Scope: this asserts the SHAPE of describe(), not that jax is supported."""
    d = js.describe()
    assert set(d) >= {"version", "version_info", "release_version",
                      "is_dev_build", "file", "jaxlib_version",
                      "jaxlib_version_info", "jaxlib_file",
                      "shard_map_top_level"}
    assert isinstance(d["version"], str) and d["version"]


# ------------------------------------------------------- the gate is now WIRED
def test_the_jax_this_process_runs_actually_satisfies_the_wired_gate():
    """The gate is enforced at startup, so an unsupported stack breaks EVERY run.

    Scope: this is about the interpreter running pytest, which on Perlmutter is
    the Shifter image and on Frontera is the venv.  It is here so an image
    regression costs one red test with a named reason instead of every job
    refusing at step 5b with the same message and no owner.
    """
    problems = (js.check_version() + js.check_private_arity()
                + js.check_private_symbols())
    assert problems == [], (
        "the jax in this process does not satisfy the gate that "
        "runtime.initialize_communicator_stack enforces at step 5b: %s"
        % "; ".join(problems))


def test_the_declared_window_matches_pyproject():
    """``SUPPORTED_MIN``/``MAX`` and pyproject.toml must not drift apart.

    The module says it is "the thing pyproject.toml means".  Nothing enforced
    that, which is how the packaging floor and the running stack came to
    differ by four minor versions with no diff to point at.
    """
    import re

    root = __file__.rsplit("/tests/", 1)[0]
    with open(root + "/pyproject.toml", encoding="utf-8") as fh:
        text = fh.read()

    want_lo = ".".join(map(str, js.SUPPORTED_MIN))
    want_hi = ".".join(map(str, js.SUPPORTED_MAX_EXCLUSIVE))
    specs = re.findall(r'"jax(?:lib)?(?:\[[a-z0-9]+\])?([^"]*)"', text)
    assert specs, "no jax specifier found in pyproject.toml"
    for spec in specs:
        assert spec == f">={want_lo},<{want_hi}", (
            "pyproject.toml pins jax %r but runtime/jax_support.py declares "
            "the window [%s, %s)" % (spec, want_lo, want_hi))


def test_the_startup_path_calls_the_gate():
    """Wiring is the whole point; an unwired gate is a comment.

    Read the AST of ``initialize_communicator_stack`` rather than running it
    (running it builds a mesh and needs devices) and rather than searching its
    text: the docstring names ``prepare_mesh`` several hundred characters
    before the code calls it, so a substring compare answers a question about
    prose, not about order of execution.
    """
    import ast
    import runtime

    tree = ast.parse(inspect.getsource(runtime.initialize_communicator_stack))
    at = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            at.setdefault(node.func.id, node.lineno)

    assert "_enforce_supported_jax" in at, (
        "runtime.initialize_communicator_stack no longer CALLS the jax "
        "support gate; it was wired at step 5b on 2026-08-06")
    assert "prepare_mesh" in at, "prepare_mesh is no longer called here"
    # The gate must run BEFORE the mesh, which performs the first jit.
    assert at["_enforce_supported_jax"] < at["prepare_mesh"], (
        "the jax gate must run before prepare_mesh(): the failure it catches "
        "is a TypeError inside the first jit")
    assert "jax_support" in inspect.getsource(runtime._enforce_supported_jax)
