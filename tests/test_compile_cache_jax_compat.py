"""Unit tests for the ``jax._src`` surface ``jax_compile_cache`` patches.

These need no devices and no ``jax.distributed``.  Every case CONSTRUCTS the
condition it is about — a fake ``jax._src`` submodule with the shape this tree
supports, or with a shape it does not — rather than asserting that today's
environment happens to be fine.  A test that only passes because the machine it
ran on was healthy is the kind of check this whole workstream exists to replace.

WHAT CHANGED 2026-08-06.  This file used to test five compatibility shims
numbered "COMPAT shim N of 5", which existed so one source ran on both jax
0.5.3 (the old Perlmutter GPU container) and jax 0.9.  The GPU leg moved to
jax 0.7.0 and 0.5.3 support was dropped, so four of the five are gone:

  1  cache_key._hash_accelerator_config          was 3 params on 0.5.3, 2 now
  2  compilation_cache.get_executable_and_time   was 3 params on 0.5.3, 4 now
  4  compiler.backend_compile_and_load           was absent on 0.5.3, present now
  5  the missing ``executable_devices`` argument — the P>1 degradation

Their tests are replaced by FALSIFICATIONS of the removal: each of the four
0.5.3 shapes must now produce a loud failure rather than a quiet
accommodation, which is what makes "the shim is gone" a measured statement
instead of an absence.

  3  VerificationCache / compilation_cache_check_contents

SURVIVES, and its tests survive with it, because it was never a 0.5.3 shim:
both symbols are absent from every NVIDIA JAX container at every tag (ten
probed, 0.5.3 through 0.9.1 — CLAIMS 112) and present only in the released
wheel.  Removing its guard would restore the CLAIMS 114 defect on the new
image.
"""
from __future__ import annotations

import types

import jax  # noqa: F401  -- populates the jax._src submodule attributes
import jax._src as jax_src
import pytest

from common import jax_compile_cache as jcc


@pytest.fixture(autouse=True)
def _clean_compat_state(monkeypatch):
    """Each test starts with no announcement memo and rank 0 speaking."""
    jcc._COMPAT_SAID.clear()
    old_idx, jcc._STATE.proc_idx = jcc._STATE.proc_idx, 0
    # The module is process-global, just like JAX's cache hooks.  Isolate the
    # write receipt so a successful-put test cannot make a later read-only
    # test look like it wrote an entry.  monkeypatch restores the prior live
    # process state after this focused test module's fixture unwinds.
    for name, value in (
            ("write_metrics_available", False),
            ("local_writes", 0),
            ("local_write_bytes", 0),
            ("local_write_secs", 0.0)):
        monkeypatch.setattr(jcc._STATE, name, value)
    yield
    jcc._COMPAT_SAID.clear()
    jcc._STATE.proc_idx = old_idx


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
# the accelerator-config hook — 2 parameters, and only 2
# ---------------------------------------------------------------------------
def test_invariant_key_hook_hashes_the_devices_at_the_supported_arity(
        monkeypatch):
    """The canonical string is built from the device array alone."""
    mod, calls = _fake_cache_key(2)
    monkeypatch.setattr(jax_src, "cache_key", mod, raising=False)
    jcc._install_invariant_key_patch()
    mod._hash_accelerator_config(
        object(), _Devices([_device("A100"), _device("A100")]))

    assert len(calls) == 1
    assert calls[0].startswith("lorrax-canon:gpu:2:A100,A100:")


def test_invariant_key_hook_refuses_the_0_5_3_call_shape(monkeypatch):
    """FALSIFICATION of shim 1's removal.

    jax 0.5.3 called this hook ``(hash_obj, devices, backend)``.  The shim
    ended in ``*_compat_tail`` and swallowed that third positional silently.
    With the shim gone the call must FAIL — a silently accepted argument on an
    unsupported jax is exactly the class of quiet accommodation the owner's
    ruling removes.  ``runtime.jax_support`` catches this at startup instead,
    by name, before anything compiles.
    """
    mod, _calls = _fake_cache_key(2)
    monkeypatch.setattr(jax_src, "cache_key", mod, raising=False)
    jcc._install_invariant_key_patch()

    with pytest.raises(TypeError):
        mod._hash_accelerator_config(object(), _Devices([_device()]),
                                     "backend-object")


def test_invariant_key_patch_is_silent_on_a_supported_jax(monkeypatch, capsys):
    mod, _ = _fake_cache_key(2)
    monkeypatch.setattr(jax_src, "cache_key", mod, raising=False)
    jcc._install_invariant_key_patch()
    assert "jax-compat" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the lookup hook — forwards without interpreting
# ---------------------------------------------------------------------------
def _fake_compilation_cache(n_get_params: int, *, verification: bool,
                            result=("executable", 1.0)):
    seen = []

    if n_get_params == 3:
        def _get(cache_key, compile_options, backend):
            seen.append((cache_key, compile_options, backend))
            return result
    else:
        def _get(cache_key, compile_options, backend, executable_devices):
            seen.append((cache_key, compile_options, backend,
                         executable_devices))
            return result

    def _in_cache(backend, cache_key):
        return True

    mod = types.SimpleNamespace(
        get_executable_and_time=_get, is_executable_in_cache=_in_cache)
    if verification:
        mod.VerificationCache = lambda base: ("verified", base)
    return mod, seen


@pytest.mark.parametrize("result,want_hits", [
    (("executable", 1.0), 1),
    ((None, None), 0),
])
def test_p1_observer_counts_lookups_without_changing_jaxs_answer(
        monkeypatch, result, want_hits):
    """The P=1 instrument observes both a hit and its red-twin miss.

    An empty agreed set is deliberate: if the observer accidentally inherits
    the P>1 veto, neither call reaches JAX and the ``seen`` assertion fails.
    The returned tuple is asserted verbatim so telemetry cannot become policy.
    """
    mod, seen = _fake_compilation_cache(
        4, verification=True, result=result)
    monkeypatch.setattr(jax_src, "compilation_cache", mod, raising=False)
    monkeypatch.setattr(jcc._STATE, "agreed", frozenset())
    monkeypatch.setattr(jcc._STATE, "probes", 0)
    monkeypatch.setattr(jcc._STATE, "hits", 0)
    monkeypatch.setattr(jcc._STATE, "blocked", 0)
    monkeypatch.setattr(jcc._STATE, "read_secs", 0.0)
    monkeypatch.setattr(jcc._STATE, "probe_keys", set())

    jcc._install_observation_patch()
    got = mod.get_executable_and_time(
        "p1-key", "opts", "backend", "devices")

    assert got == result
    assert seen == [("p1-key", "opts", "backend", "devices")]
    assert jcc._STATE.probes == 1
    assert jcc._STATE.hits == want_hits
    assert jcc._STATE.blocked == 0
    assert jcc._STATE.probe_keys == {"p1-key"}


def test_p1_observer_preserves_a_cache_read_exception(monkeypatch):
    """RED arm: observation must not demote a corrupt read to a miss."""
    mod, _seen = _fake_compilation_cache(4, verification=True)

    def _raises(*_args):
        raise OSError("constructed corrupt cache entry")

    mod.get_executable_and_time = _raises
    monkeypatch.setattr(jax_src, "compilation_cache", mod, raising=False)
    monkeypatch.setattr(jcc._STATE, "probes", 0)
    monkeypatch.setattr(jcc._STATE, "hits", 0)
    monkeypatch.setattr(jcc._STATE, "read_secs", 0.0)
    monkeypatch.setattr(jcc._STATE, "probe_keys", set())

    jcc._install_observation_patch()
    with pytest.raises(OSError, match="constructed corrupt cache entry"):
        mod.get_executable_and_time(
            "broken", "opts", "backend", "devices")
    assert jcc._STATE.probes == 1
    assert jcc._STATE.hits == 0


def test_p1_cache_setup_arms_the_observer_before_return(monkeypatch, tmp_path):
    """The P=1 early return cannot bypass the observation hook again."""
    calls = []
    config_updates = []
    monkeypatch.setattr(jcc, "_COMPILATION_CACHE_READY", False)
    for name in ("enabled", "dir", "n_proc", "proc_idx"):
        monkeypatch.setattr(jcc._STATE, name, getattr(jcc._STATE, name))
    monkeypatch.setattr(jax, "process_count", lambda: 1)
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(jcc, "_install_compile_counter", lambda: None)
    monkeypatch.setattr(jcc.atexit, "register", lambda _fn: None)
    monkeypatch.setenv("ISDF_JAX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        jax.config, "update",
        lambda *args: config_updates.append(args))
    monkeypatch.setattr(jcc, "_install_atomic_put_patch", lambda: None)
    monkeypatch.setattr(jcc, "bound_cache_dir", lambda: "")
    monkeypatch.setattr(
        jcc, "_install_observation_patch",
        lambda: calls.append("observation"))
    monkeypatch.setattr(
        jcc, "_install_agreement_patch",
        lambda: pytest.fail("P=1 installed the agreement policy"))

    jcc.ensure_jax_compile_cache()

    assert calls == ["observation"]
    assert jcc._STATE.n_proc == 1
    assert jcc._STATE.enabled is True
    assert ("jax_compilation_cache_dir",
            str(tmp_path / "cache" / "np1")) in config_updates
    assert not [name for name, _value in config_updates
                if name == "jax_persistent_cache_min_compile_time_secs"]


def test_cache_default_prefers_workflow_over_global_fallback(monkeypatch,
                                                             tmp_path):
    """A named workflow takes precedence over the legacy global fallback."""
    monkeypatch.delenv("ISDF_JAX_CACHE_DIR", raising=False)
    monkeypatch.delenv("LORRAX_RUN_DIR", raising=False)
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("SCRATCH", str(scratch))
    # The module no longer exports this second owner; if a user supplies it,
    # LORRAX still resolves only its own documented policy.
    monkeypatch.setenv(
        "JAX_COMPILATION_CACHE_DIR", str(tmp_path / "native-jax-cache"))

    assert jcc._resolve_cache_base_dir() == (
        str(scratch / "lorrax_jax_cache"), "$SCRATCH fallback")

    run_dir = tmp_path / "this-workflow"
    monkeypatch.setenv("LORRAX_RUN_DIR", str(run_dir))
    assert jcc._resolve_cache_base_dir() == (
        str(run_dir / ".lorrax_jax_cache"), "LORRAX_RUN_DIR")


def test_explicit_cache_control_wins_over_run_dir(monkeypatch, tmp_path):
    """The legacy expert knob retains both path override and empty opt-out."""
    monkeypatch.setenv("LORRAX_RUN_DIR", str(tmp_path / "workflow"))
    explicit = tmp_path / "restart-campaign"
    monkeypatch.setenv("ISDF_JAX_CACHE_DIR", f"  {explicit}  ")
    assert jcc._resolve_cache_base_dir() == (str(explicit), "explicit")

    monkeypatch.setenv("ISDF_JAX_CACHE_DIR", "   ")
    assert jcc._resolve_cache_base_dir() == ("", "explicit")


@pytest.mark.parametrize("n_params", [3, 4])
def test_agreement_hook_forwards_whatever_jax_hands_it(monkeypatch, n_params):
    """``*passthrough`` is a design choice, not a leftover 0.5.3 branch.

    The wrapper reads the cache key and NOTHING else, so it must not claim to
    know what the remaining arguments mean.  Both arities are exercised
    precisely to pin "untouched" — 4 is the shape both supported jaxes have,
    3 is present here only as the negative case that proves the forwarding
    interprets nothing.  Supporting a 3-parameter jax is a separate question
    and the answer is no; ``runtime.jax_support`` refuses it at startup.
    """
    mod, seen = _fake_compilation_cache(n_params, verification=True)
    monkeypatch.setattr(jax_src, "compilation_cache", mod, raising=False)
    monkeypatch.setattr(jcc._STATE, "agreed", frozenset({"k1"}))

    jcc._install_agreement_patch()
    args = ["opts", "backend"] + (["devices"] if n_params == 4 else [])
    executable, compile_time = mod.get_executable_and_time("k1", *args)

    assert (executable, compile_time) == ("executable", 1.0)
    assert seen == [tuple(["k1"] + args)]


def test_agreement_hook_still_vetoes_an_unagreed_key(monkeypatch):
    """The wrapper must not widen what the agreement lets through."""
    mod, seen = _fake_compilation_cache(4, verification=False)
    monkeypatch.setattr(jax_src, "compilation_cache", mod, raising=False)
    monkeypatch.setattr(jcc._STATE, "agreed", frozenset({"k1"}))

    jcc._install_agreement_patch()
    assert mod.get_executable_and_time(
        "other", "opts", "backend", "devices") == (None, None)
    assert seen == []


def test_agreement_patch_is_silent_on_a_supported_jax(monkeypatch, capsys):
    """FALSIFICATION of shim 2's removal: no arity probe, so no announcement.

    The shim announced whenever ``get_executable_and_time`` was not the
    4-parameter shape.  Nothing announces now on either arity, because there
    is no longer a compatibility path to report having taken.
    """
    for n in (4, 3):
        # A fresh namespace each time, so the install guard never short-circuits.
        mod, _ = _fake_compilation_cache(n, verification=True)
        monkeypatch.setattr(jax_src, "compilation_cache", mod, raising=False)
        jcc._install_agreement_patch()
        assert "get_executable_and_time" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# THE SURVIVING SHIM — content verification exists only in the released wheel
# ---------------------------------------------------------------------------
def _fake_config(*, check_contents):
    mod = types.SimpleNamespace(
        compilation_cache_max_size=types.SimpleNamespace(value=-1))
    if check_contents is not None:
        mod.compilation_cache_check_contents = types.SimpleNamespace(
            value=check_contents)
    return mod


@pytest.mark.parametrize("n_proc,max_size,enabled", [
    (1, -1, True),
    (4, -1, True),
    (1, 1024, True),
    (1, 0, False),
    (4, 0, False),
])
def test_cache_size_policy_keeps_only_safe_arms(n_proc, max_size, enabled):
    assert jcc._cache_size_policy(n_proc, max_size) is enabled


def test_cache_size_policy_refuses_live_lru_eviction_at_pgt1():
    with pytest.raises(jcc.UnsafeCachePolicy, match="unsafe at P=4"):
        jcc._cache_size_policy(4, 1024)


def test_cache_size_policy_refuses_invalid_negative_limit():
    with pytest.raises(jcc.UnsafeCachePolicy, match="must be -1"):
        jcc._cache_size_policy(1, -2)


@pytest.mark.parametrize("check_contents,verification,wraps", [
    (None, False, False),    # every NVIDIA container: neither symbol exists
    (False, True, False),    # released wheel, verification off
    (True, True, True),      # released wheel, verification on
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
    """This is the announcement that must NOT disappear with the other four.

    Absence here is the container's normal state, not an error, but it is a
    capability the log has to name: the alternative is the CLAIMS 114 silence,
    where ``enabled=True`` was printed over a cache writing zero entries.
    """
    cc, _ = _fake_compilation_cache(4, verification=False)
    monkeypatch.setattr(jax_src, "compilation_cache", cc, raising=False)
    monkeypatch.setattr(jax_src, "config",
                        _fake_config(check_contents=None), raising=False)

    jcc._install_atomic_put_patch()
    out = capsys.readouterr().out
    assert "jax-compat" in out
    assert "VerificationCache" in out
    assert "compilation_cache_check_contents" in out


def test_atomic_put_survives_a_half_present_verification_surface(
        monkeypatch, tmp_path):
    """One symbol without the other must not be treated as "verification on".

    Constructed because the two names come from DIFFERENT modules
    (``compilation_cache`` and ``config``), so a future jax can ship one
    first, and reading only one of them would put a ``None`` where a class
    belongs.
    """
    cc, _ = _fake_compilation_cache(4, verification=True)   # class present
    monkeypatch.setattr(jax_src, "compilation_cache", cc, raising=False)
    monkeypatch.setattr(jax_src, "config",
                        _fake_config(check_contents=None),  # flag absent
                        raising=False)

    jcc._install_atomic_put_patch()
    cache, _path = cc.get_file_cache(str(tmp_path))
    from jax._src import lru_cache as _lru
    assert isinstance(cache, _lru.LRUCache)


def test_atomic_put_reports_exact_successful_write_cost(monkeypatch, tmp_path):
    """A completed atomic write has an exact count, payload size and time."""
    cc, _ = _fake_compilation_cache(4, verification=False)
    monkeypatch.setattr(jax_src, "compilation_cache", cc, raising=False)
    monkeypatch.setattr(jax_src, "config",
                        _fake_config(check_contents=None), raising=False)

    jcc._install_atomic_put_patch()
    cache, _path = cc.get_file_cache(str(tmp_path))
    payload = b"post-compression-payload"
    cache.put("somekey", payload)

    assert (tmp_path / "somekey-cache").read_bytes() == payload
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]
    stats = jcc.compile_cache_stats()
    assert stats["write_metrics_available"] is True
    assert stats["local_writes"] == 1
    assert stats["local_write_bytes"] == len(payload)
    assert stats["local_write_secs"] >= 0.0
    assert stats["is_cache_writer"] is True
    assert "process-local" in stats["write_scope"]

    # JAX's existing-entry fast path is not another write.
    cache.put("somekey", b"replacement-that-must-not-land")
    again = jcc.compile_cache_stats()
    assert again["local_writes"] == 1
    assert again["local_write_bytes"] == len(payload)


def test_atomic_put_failure_preserves_exception_and_reports_no_write(
        monkeypatch, tmp_path):
    """A failed rename remains JAX's exception, not a successful receipt."""
    cc, _ = _fake_compilation_cache(4, verification=False)
    monkeypatch.setattr(jax_src, "compilation_cache", cc, raising=False)
    monkeypatch.setattr(jax_src, "config",
                        _fake_config(check_contents=None), raising=False)

    jcc._install_atomic_put_patch()
    cache, _path = cc.get_file_cache(str(tmp_path))
    original = OSError("constructed rename failure")

    def _fail_rename(_src, _dst):
        raise original

    monkeypatch.setattr(jcc.os, "replace", _fail_rename)
    with pytest.raises(OSError) as excinfo:
        cache.put("broken", b"payload")

    assert excinfo.value is original
    stats = jcc.compile_cache_stats()
    assert stats["write_metrics_available"] is True
    assert stats["local_writes"] == 0
    assert stats["local_write_bytes"] == 0
    assert stats["local_write_secs"] == 0.0
    assert not (tmp_path / "broken-cache").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_cache_hit_observation_does_not_count_as_a_write(monkeypatch,
                                                         tmp_path):
    """Read/probe telemetry and write telemetry are independent."""
    cc, seen = _fake_compilation_cache(
        4, verification=False, result=("cached-executable", 3.0))
    monkeypatch.setattr(jax_src, "compilation_cache", cc, raising=False)
    monkeypatch.setattr(jax_src, "config",
                        _fake_config(check_contents=None), raising=False)
    monkeypatch.setattr(jcc._STATE, "probes", 0)
    monkeypatch.setattr(jcc._STATE, "hits", 0)
    monkeypatch.setattr(jcc._STATE, "read_secs", 0.0)
    monkeypatch.setattr(jcc._STATE, "probe_keys", set())

    jcc._install_atomic_put_patch()
    cc.get_file_cache(str(tmp_path))  # arms exact unlimited-write telemetry
    jcc._install_observation_patch()
    got = cc.get_executable_and_time(
        "warm-key", "opts", "backend", "devices")

    assert got == ("cached-executable", 3.0)
    assert seen == [("warm-key", "opts", "backend", "devices")]
    stats = jcc.compile_cache_stats()
    assert stats["probes"] == 1
    assert stats["hits"] == 1
    assert stats["read_secs"] >= 0.0
    assert stats["local_writes"] == 0
    assert stats["local_write_bytes"] == 0
    assert stats["local_write_secs"] == 0.0


def test_receipt_separates_read_and_local_p0_write_time(monkeypatch):
    """The human receipt cannot fold filesystem writes into cache reads."""
    said = []
    monkeypatch.setattr(jcc, "bound_cache_dir", lambda: "")
    monkeypatch.setattr(
        jcc, "_debug_say", lambda message: said.append(message))
    monkeypatch.setattr(jcc._STATE, "n_proc", 4)
    monkeypatch.setattr(jcc._STATE, "proc_idx", 0)
    monkeypatch.setattr(jcc._STATE, "probes", 7)
    monkeypatch.setattr(jcc._STATE, "hits", 6)
    monkeypatch.setattr(jcc._STATE, "read_secs", 1.25)
    jcc._STATE.set_write_metrics_available(True)
    jcc._STATE.record_write(19, 2.5)

    jcc._report_impl()

    assert len(said) == 1
    assert "cache_probes=7 hits=6 (1.25s)" in said[0]
    assert "cache_writes_local=1 bytes=19 (2.50s; JAX p0-only)" in said[0]


def test_write_receipt_reset_is_deterministic():
    """Repeated resets produce the same exact zero state."""
    jcc._STATE.set_write_metrics_available(True)
    jcc._STATE.record_write(17, 2.5)

    for _ in range(2):
        jcc._STATE.reset_write_metrics()
        assert jcc._STATE.write_metrics() == {
            "write_metrics_available": True,
            "local_writes": 0,
            "local_write_bytes": 0,
            "local_write_secs": 0.0,
        }
        jcc._STATE.record_write(17, 2.5)


def test_capped_lru_write_cost_is_explicitly_unmeasured(monkeypatch,
                                                        tmp_path):
    """The delegated P=1 eviction path must not publish numeric zeros.

    This is the ONE cell in this file that reaches jax's real
    ``lru_cache.LRUCache``, so it is the one cell with a dependency the rest
    of the module does not have.  See the guard below.
    """
    # CANNOT-RUN, WHICH IS NOT A PASS.
    #
    # Setting ``compilation_cache_max_size`` to a finite 1024 is exactly what
    # this test is about: a finite cap is what puts ``_AtomicLRUCache`` on its
    # parent's eviction branch (``eviction_enabled = max_size != -1``), which
    # is the branch whose write cost the patch refuses to report as a numeric
    # zero.  But that same branch is the one jax guards with
    # ``if filelock is None: raise RuntimeError(...)``
    # (jax/_src/lru_cache.py) — so with ``filelock`` absent the constructor
    # raises before a single assertion below executes.
    #
    # ``filelock`` is not a declared dependency of this tree (pyproject lists
    # h5py, jax, jaxlib, matplotlib, numpy, scipy, xmlschema, xsdata) and jax
    # itself treats it as optional, so on a bare environment this cell was a
    # deterministic hard FAIL that said nothing whatever about the code under
    # test — the kind of standing red everybody learns to scroll past, which
    # is how a real regression gets to hide behind it.
    #
    # Skipping is therefore the honest report and NOT a pass: a skip here
    # means the delegated-eviction receipt went UNVERIFIED on this machine.
    # The fix is to install ``filelock`` and re-run, not to read the green.
    pytest.importorskip(
        "filelock",
        reason="CANNOT-RUN (not a pass): the `filelock` package is not "
               "installed.  This cell sets a finite "
               "jax_compilation_cache_max_size, which is what puts jax's "
               "LRUCache on its eviction branch, and that branch raises "
               "RuntimeError('Please install the `filelock` package to set "
               "`jax_compilation_cache_max_size`') before any assertion here "
               "runs.  So the delegated-eviction write-metrics receipt is "
               "UNVERIFIED on this machine, not verified-good.  Fix: install "
               "filelock (pip install filelock) and re-run this cell.")

    cc, _ = _fake_compilation_cache(4, verification=False)
    monkeypatch.setattr(jax_src, "compilation_cache", cc, raising=False)
    monkeypatch.setattr(
        jax_src, "config",
        types.SimpleNamespace(
            compilation_cache_max_size=types.SimpleNamespace(value=1024)),
        raising=False)

    jcc._install_atomic_put_patch()
    cc.get_file_cache(str(tmp_path))

    stats = jcc.compile_cache_stats()
    assert stats["write_metrics_available"] is False
    assert stats["local_writes"] is None
    assert stats["local_write_bytes"] is None
    assert stats["local_write_secs"] is None


# ---------------------------------------------------------------------------
# where a real XLA compile is counted — one entry point, no preference order
# ---------------------------------------------------------------------------
def _fake_compiler(names):
    calls = {n: 0 for n in names}

    def make(n):
        def fn(*a, **k):
            calls[n] += 1
            return "compiled"
        return fn

    return types.SimpleNamespace(**{n: make(n) for n in names}), calls


def test_counter_patches_backend_compile_and_load_only(monkeypatch, capsys):
    mod, calls = _fake_compiler(["backend_compile_and_load", "backend_compile"])
    monkeypatch.setattr(jax_src, "compiler", mod, raising=False)
    monkeypatch.setattr(jcc._STATE, "compiles", 0)

    jcc._install_compile_counter()
    mod.backend_compile_and_load("x")

    assert jcc._STATE.compiles == 1
    assert calls["backend_compile_and_load"] == 1
    # NOT both: the preferred entry point calls the other one, so wrapping
    # both would double-count every compile.
    mod.backend_compile("x")
    assert jcc._STATE.compiles == 1
    assert "jax-compat" not in capsys.readouterr().out


def test_counter_refuses_on_the_0_5_3_compiler_surface(monkeypatch):
    """FALSIFICATION of shim 4's removal.

    jax 0.5.3 had only ``backend_compile``, and the shim silently counted
    there instead.  That fallback is gone: this surface must now REFUSE, so
    the caller announces rather than reporting a confident ``xla_compiles=0``.
    """
    mod, _ = _fake_compiler(["backend_compile"])
    monkeypatch.setattr(jax_src, "compiler", mod, raising=False)

    with pytest.raises(jcc._JaxSurfaceUnsupported) as excinfo:
        jcc._install_compile_counter()
    assert "backend_compile_and_load" in str(excinfo.value)


def test_counter_refuses_rather_than_reporting_a_confident_zero(monkeypatch):
    """No entry point at all ⇒ raise, so the caller can announce.

    The defect this replaces was a bare ``except: pass`` at the call site: the
    counter installed nowhere and every run reported ``xla_compiles=0``.
    """
    mod, _ = _fake_compiler([])
    monkeypatch.setattr(jax_src, "compiler", mod, raising=False)

    with pytest.raises(jcc._JaxSurfaceUnsupported) as excinfo:
        jcc._install_compile_counter()
    assert "backend_compile_and_load" in str(excinfo.value)


# ---------------------------------------------------------------------------
# the P>1 degradation is gone — the module must not carry it any more
# ---------------------------------------------------------------------------
def test_peer_rebind_degradation_is_removed():
    """FALSIFICATION of shim 5's removal, at the only altitude available.

    Shim 5 turned the cache OFF at P>1 on a jax whose
    ``get_executable_and_time`` has no ``executable_devices``.  Both supported
    jaxes have it, so the branch was unreachable and was deleted rather than
    left as a permanent compatibility layer.  Its detection duty moved to
    ``runtime.jax_support``, which requires that hook to take 4 parameters.

    Asserting on the module's own surface is deliberate: the branch cannot be
    exercised through :func:`ensure_jax_compile_cache` without a real P>1
    launch, so the honest check is that neither the helper nor its memo is
    still here, plus that the requirement really is stated where it moved to.
    """
    assert not hasattr(jcc, "_peer_rebind_supported")
    assert not hasattr(jcc, "_PEER_REBIND")

    from runtime import jax_support
    assert jax_support.REQUIRED_PRIVATE_ARITY[
        ("jax._src.compilation_cache", "get_executable_and_time")] == 4


def test_only_one_compat_announcement_site_is_left():
    """Every ``_compat`` announcement site must be one somebody decided on.

    A ``_compat`` call appearing without a decision would be a compatibility
    layer growing back by accident, which is the thing being removed.  This
    used to be a COUNT (``== 1``, the one surviving jax 0.5.3-era shim).  It is
    now the SET OF KEYS, which is a strictly stronger guard for the same
    purpose: a bare number goes green again as soon as someone deletes one site
    and adds another, and it cannot say which sites are sanctioned.  Read from
    the AST, not from a substring, so a mention in a comment or a docstring
    cannot move it.

    AMENDED 2026-08-09 (fix/multislice-cachekey): the shard-slice patch added
    the two ``shard-slice-*`` keys.  Neither is a version shim — they announce
    a RUN-TIME fallback (the canonical slicer declining a shape it did not
    anticipate) and the absence of ``ArrayImpl._multi_slice`` on some future
    jax.  They are here because ``_compat`` is this file's once-per-run,
    rank-0-only announcer and the alternative was a second one just like it.
    """
    import ast
    import inspect as _inspect

    sanctioned = {
        # the one surviving jax 0.5.3-era shim: VerificationCache /
        # compilation_cache_check_contents, absent from every container
        "compilation_cache.verification",
        # the shard-slice canonicalization's two non-silent exits
        "shard-slice-fallback",
        "shard-slice-absent",
    }

    tree = ast.parse(_inspect.getsource(jcc))
    sites = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_compat"]
    keys = set()
    for n in sites:
        assert n.args and isinstance(n.args[0], ast.Constant), (
            "a _compat() site must name its key as a literal, so this test "
            "can see it")
        keys.add(n.args[0].value)

    assert keys == sanctioned, (
        f"unsanctioned _compat() announcement site(s): "
        f"{sorted(keys - sanctioned)}; missing: {sorted(sanctioned - keys)}.  "
        f"Adding one is a decision — record it in the docstring above and in "
        f"the module's 'jax._src surface this file patches' block.")
    assert len(sites) == len(sanctioned), (
        f"{len(sites)} _compat() sites for {len(sanctioned)} keys — a key is "
        f"announced from two places, which makes the memo do the deciding")


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
