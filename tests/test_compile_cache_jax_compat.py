"""Unit tests for the ``jax._src`` generation shims in ``jax_compile_cache``.

These need no devices and no ``jax.distributed``.  Every case CONSTRUCTS the
condition it is about — a fake ``jax._src`` submodule with the 0.5.3 shape, or
the 0.9 shape, or neither — rather than asserting that today's environment
happens to be fine.  A test that only passes because the machine it ran on was
healthy is the kind of check this whole workstream exists to replace.

The shims are numbered "COMPAT shim N of 5" in the module under test:

  1  cache_key._hash_accelerator_config          3 params on 0.5.3, 2 on 0.9
  2  compilation_cache.get_executable_and_time   3 params on 0.5.3, 4 on 0.9
  3  VerificationCache / compilation_cache_check_contents   absent on 0.5.3
  4  compiler.backend_compile_and_load           absent on 0.5.3
  5  the missing ``executable_devices`` argument — NOT shimmable, degrades
"""
from __future__ import annotations

import inspect
import types

import jax  # noqa: F401  -- populates the jax._src submodule attributes
import jax._src as jax_src
import pytest

from common import jax_compile_cache as jcc


@pytest.fixture(autouse=True)
def _clean_compat_state():
    """Each test starts with no announcement memo and rank 0 speaking."""
    jcc._COMPAT_SAID.clear()
    old_idx, jcc._STATE.proc_idx = jcc._STATE.proc_idx, 0
    old_rebind, jcc._PEER_REBIND = jcc._PEER_REBIND, None
    yield
    jcc._COMPAT_SAID.clear()
    jcc._STATE.proc_idx = old_idx
    jcc._PEER_REBIND = old_rebind


def _fake_cache_key(n_accel_params: int):
    """A ``jax._src.cache_key`` with the accelerator hook at a chosen arity."""
    calls = []

    def _hash_string(hash_obj, s):
        calls.append(s)

    def _opts(hash_obj, options, strip_device_assignment=False):
        calls.append(("opts", strip_device_assignment))

    if n_accel_params == 3:
        def _accel(hash_obj, accelerators, backend):
            raise AssertionError("the original must be replaced, not called")
    else:
        def _accel(hash_obj, accelerators):
            raise AssertionError("the original must be replaced, not called")

    mod = types.SimpleNamespace(
        _hash_string=_hash_string,
        _hash_serialized_compile_options=_opts,
        _hash_accelerator_config=_accel,
    )
    return mod, calls


class _Devices:
    """Minimal stand-in for the ``np.ndarray`` of devices JAX passes in."""

    def __init__(self, devices):
        self.flat = list(devices)


def _device(kind="A100", platform="gpu"):
    return types.SimpleNamespace(platform=platform, device_kind=kind)


# ---------------------------------------------------------------------------
# shim 1 — the accelerator-config hook's arity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_params", [2, 3])
def test_invariant_key_hook_accepts_both_call_shapes(monkeypatch, capsys,
                                                     n_params):
    """JAX 0.5.3 calls the hook with a third ``backend``; 0.9 does not."""
    mod, calls = _fake_cache_key(n_params)
    monkeypatch.setattr(jax_src, "cache_key", mod, raising=False)

    jcc._install_invariant_key_patch()
    devices = _Devices([_device("A100"), _device("A100")])
    if n_params == 3:
        mod._hash_accelerator_config(object(), devices, "backend-object")
    else:
        mod._hash_accelerator_config(object(), devices)

    assert len(calls) == 1
    assert calls[0].startswith("lorrax-canon:gpu:2:A100,A100:")


def test_invariant_key_announces_only_on_the_divergent_arity(monkeypatch,
                                                             capsys):
    mod_09, _ = _fake_cache_key(2)
    monkeypatch.setattr(jax_src, "cache_key", mod_09, raising=False)
    jcc._install_invariant_key_patch()
    assert "_hash_accelerator_config" not in capsys.readouterr().out

    jcc._COMPAT_SAID.clear()
    mod_05, _ = _fake_cache_key(3)
    monkeypatch.setattr(jax_src, "cache_key", mod_05, raising=False)
    jcc._install_invariant_key_patch()
    out = capsys.readouterr().out
    assert "jax-compat" in out
    assert "_hash_accelerator_config takes 3 parameters" in out


# ---------------------------------------------------------------------------
# shim 2 — the lookup hook's arity
# ---------------------------------------------------------------------------
def _fake_compilation_cache(n_get_params: int, *, verification: bool):
    seen = []

    if n_get_params == 3:
        def _get(cache_key, compile_options, backend):
            seen.append((cache_key, compile_options, backend))
            return "executable", 1.0
    else:
        def _get(cache_key, compile_options, backend, executable_devices):
            seen.append((cache_key, compile_options, backend,
                         executable_devices))
            return "executable", 1.0

    def _in_cache(backend, cache_key):
        return True

    mod = types.SimpleNamespace(
        get_executable_and_time=_get, is_executable_in_cache=_in_cache)
    if verification:
        mod.VerificationCache = lambda base: ("verified", base)
    return mod, seen


@pytest.mark.parametrize("n_params", [3, 4])
def test_agreement_hook_forwards_whatever_jax_hands_it(monkeypatch, n_params):
    mod, seen = _fake_compilation_cache(n_params, verification=True)
    monkeypatch.setattr(jax_src, "compilation_cache", mod, raising=False)
    monkeypatch.setattr(jcc._STATE, "agreed", frozenset({"k1"}))

    jcc._install_agreement_patch()
    args = ["opts", "backend"] + (["devices"] if n_params == 4 else [])
    executable, compile_time = mod.get_executable_and_time("k1", *args)

    assert (executable, compile_time) == ("executable", 1.0)
    assert seen == [tuple(["k1"] + args)]


def test_agreement_hook_still_vetoes_an_unagreed_key(monkeypatch):
    """The shim must not accidentally widen what the agreement lets through."""
    mod, seen = _fake_compilation_cache(3, verification=False)
    monkeypatch.setattr(jax_src, "compilation_cache", mod, raising=False)
    monkeypatch.setattr(jcc._STATE, "agreed", frozenset({"k1"}))

    jcc._install_agreement_patch()
    assert mod.get_executable_and_time("other", "opts", "backend") == (None,
                                                                      None)
    assert seen == []


def test_agreement_announces_only_on_the_divergent_arity(monkeypatch, capsys):
    mod_09, _ = _fake_compilation_cache(4, verification=True)
    monkeypatch.setattr(jax_src, "compilation_cache", mod_09, raising=False)
    jcc._install_agreement_patch()
    assert "get_executable_and_time" not in capsys.readouterr().out

    jcc._COMPAT_SAID.clear()
    mod_05, _ = _fake_compilation_cache(3, verification=False)
    monkeypatch.setattr(jax_src, "compilation_cache", mod_05, raising=False)
    jcc._install_agreement_patch()
    out = capsys.readouterr().out
    assert "get_executable_and_time takes 3 parameters" in out


# ---------------------------------------------------------------------------
# shim 3 — content verification is a jax 0.9 feature
# ---------------------------------------------------------------------------
def _fake_config(*, check_contents):
    mod = types.SimpleNamespace(
        compilation_cache_max_size=types.SimpleNamespace(value=-1))
    if check_contents is not None:
        mod.compilation_cache_check_contents = types.SimpleNamespace(
            value=check_contents)
    return mod


@pytest.mark.parametrize("check_contents,verification,wraps", [
    (None, False, False),    # jax 0.5.3: neither symbol exists
    (False, True, False),    # jax 0.9, verification off
    (True, True, True),      # jax 0.9, verification on
])
def test_atomic_put_uses_verification_only_where_it_exists(
        monkeypatch, tmp_path, check_contents, verification, wraps):
    cc, _ = _fake_compilation_cache(4, verification=verification)
    monkeypatch.setattr(jax_src, "compilation_cache", cc, raising=False)
    monkeypatch.setattr(jax_src, "config", _fake_config(
        check_contents=check_contents), raising=False)

    jcc._install_atomic_put_patch()
    cache, path = cc.get_file_cache(str(tmp_path))

    assert path == str(tmp_path)
    from jax._src import lru_cache as _lru

    if wraps:
        assert cache[0] == "verified"
    else:
        assert isinstance(cache, _lru.LRUCache)


def test_atomic_put_announces_when_verification_is_absent(monkeypatch,
                                                          tmp_path, capsys):
    cc, _ = _fake_compilation_cache(4, verification=False)
    monkeypatch.setattr(jax_src, "compilation_cache", cc, raising=False)
    monkeypatch.setattr(jax_src, "config",
                        _fake_config(check_contents=None), raising=False)

    jcc._install_atomic_put_patch()
    out = capsys.readouterr().out
    assert "jax-compat" in out
    assert "VerificationCache" in out
    assert "compilation_cache_check_contents" in out


def test_atomic_put_writes_through_a_temp_file(monkeypatch, tmp_path):
    """The point of the patch — atomicity — must survive the shim."""
    cc, _ = _fake_compilation_cache(4, verification=False)
    monkeypatch.setattr(jax_src, "compilation_cache", cc, raising=False)
    monkeypatch.setattr(jax_src, "config",
                        _fake_config(check_contents=None), raising=False)

    jcc._install_atomic_put_patch()
    cache, _path = cc.get_file_cache(str(tmp_path))
    cache.put("somekey", b"payload")

    assert (tmp_path / "somekey-cache").read_bytes() == b"payload"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


# ---------------------------------------------------------------------------
# shim 4 — where a real XLA compile is counted
# ---------------------------------------------------------------------------
def _fake_compiler(names):
    calls = {n: 0 for n in names}

    def make(n):
        def fn(*a, **k):
            calls[n] += 1
            return "compiled"
        return fn

    return types.SimpleNamespace(**{n: make(n) for n in names}), calls


def test_counter_prefers_backend_compile_and_load(monkeypatch, capsys):
    mod, calls = _fake_compiler(["backend_compile_and_load", "backend_compile"])
    monkeypatch.setattr(jax_src, "compiler", mod, raising=False)
    monkeypatch.setattr(jcc._STATE, "compiles", 0)

    jcc._install_compile_counter()
    mod.backend_compile_and_load("x")

    assert jcc._STATE.compiles == 1
    assert calls["backend_compile_and_load"] == 1
    # NOT both: on jax 0.9 the preferred entry point calls the other one, so
    # wrapping both would double-count every compile.
    mod.backend_compile("x")
    assert jcc._STATE.compiles == 1
    assert "jax-compat" not in capsys.readouterr().out


def test_counter_falls_back_to_backend_compile_and_says_so(monkeypatch,
                                                           capsys):
    mod, calls = _fake_compiler(["backend_compile"])
    monkeypatch.setattr(jax_src, "compiler", mod, raising=False)
    monkeypatch.setattr(jcc._STATE, "compiles", 0)

    jcc._install_compile_counter()
    mod.backend_compile("x")

    assert jcc._STATE.compiles == 1
    out = capsys.readouterr().out
    assert "jax-compat" in out
    assert "backend_compile()" in out


def test_counter_refuses_rather_than_reporting_a_confident_zero(monkeypatch):
    """Neither entry point exists ⇒ raise, so the caller can announce.

    The defect this replaces was a bare ``except: pass`` at the call site: the
    counter installed nowhere and every run reported ``xla_compiles=0``.
    """
    mod, _ = _fake_compiler([])
    monkeypatch.setattr(jax_src, "compiler", mod, raising=False)

    with pytest.raises(jcc._JaxSurfaceUnsupported) as excinfo:
        jcc._install_compile_counter()
    assert "backend_compile_and_load" in str(excinfo.value)


# ---------------------------------------------------------------------------
# shim 5 — the one difference no shim can bridge
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_params,supported", [(3, False), (4, True)])
def test_peer_rebind_is_decided_by_the_executable_devices_parameter(
        monkeypatch, n_params, supported):
    mod, _ = _fake_compilation_cache(n_params, verification=False)
    monkeypatch.setattr(jax_src, "compilation_cache", mod, raising=False)
    assert jcc._peer_rebind_supported() is supported


def test_peer_rebind_reads_jax_not_our_own_installed_shim(monkeypatch):
    """Memoized on first call, because the agreement patch rewrites the hook.

    ``_agreed_get`` is ``(cache_key, *passthrough)`` — 2 parameters — so a
    later read would conclude "no executable_devices" on a jax 0.9 that has
    it, and turn off a cache that works.
    """
    mod, _ = _fake_compilation_cache(4, verification=False)
    monkeypatch.setattr(jax_src, "compilation_cache", mod, raising=False)
    assert jcc._peer_rebind_supported() is True

    jcc._install_agreement_patch()
    assert len(inspect.signature(mod.get_executable_and_time).parameters) < 4
    assert jcc._peer_rebind_supported() is True


def test_peer_rebind_assumes_supported_when_it_cannot_introspect(monkeypatch):
    """A shape we cannot read is not evidence of a mismatch."""
    mod = types.SimpleNamespace(get_executable_and_time=print)
    monkeypatch.setattr(jax_src, "compilation_cache", mod, raising=False)
    assert jcc._peer_rebind_supported() is True


# ---------------------------------------------------------------------------
# the announcement discipline itself
# ---------------------------------------------------------------------------
def test_compat_announces_once_per_key(capsys):
    jcc._compat("k", "first")
    jcc._compat("k", "second")
    out = capsys.readouterr().out
    assert out.count("jax-compat") == 1
    assert "first" in out and "second" not in out


def test_compat_is_silent_off_rank_zero(monkeypatch, capsys):
    monkeypatch.setattr(jcc._STATE, "proc_idx", 1)
    jcc._compat("k", "message")
    assert capsys.readouterr().out == ""


def test_generation_is_read_from_version_info_not_the_display_string():
    """Container JAXes re-stamp ``__version__`` with the build date."""
    import jax.version as jv

    text = jcc._jax_generation()
    assert text.startswith(".".join(str(x) for x in jv.__version_info__[:3]))
    assert "dev2026" not in text
