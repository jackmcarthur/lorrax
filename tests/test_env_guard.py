"""GATE env-frozen (runtime.env_guard): after the runtime starts native helper threads, a LORRAX
write that ADDS a process-environment key refuses; overwrites and unsets pass; other code is
announced and performed.  Also: SlabIO's striping resolver and the BSE W-stage strict gate no
longer write the environment (the rc 139 start-up SIGSEGV of 2026-10-16: glibc getenv/setenv)."""
import os

import pytest


@pytest.fixture
def frozen():
    from runtime import env_guard
    env_guard.freeze_process_environment("test")
    yield env_guard
    env_guard.thaw_for_tests()


def _run_as(path, src, env):
    exec(compile(src, path, "exec"), {"os": os, "env": env})


def test_own_new_key_refuses(frozen, monkeypatch):
    monkeypatch.delenv("LRX_ENV_GUARD_PROBE", raising=False)
    own = str(frozen._CHECKOUT / "src" / "common" / "_probe.py")
    with pytest.raises(RuntimeError, match="GATE env-frozen"):
        _run_as(own, "os.environ['LRX_ENV_GUARD_PROBE'] = '1'", None)
    assert "LRX_ENV_GUARD_PROBE" not in os.environ


def test_overwrite_and_unset_pass(frozen, monkeypatch):
    monkeypatch.setenv("LRX_ENV_GUARD_PROBE", "a")          # added before the check below
    own = str(frozen._CHECKOUT / "src" / "common" / "_probe.py")
    _run_as(own, "os.environ['LRX_ENV_GUARD_PROBE'] = 'b'", None)
    assert os.environ["LRX_ENV_GUARD_PROBE"] == "b"
    _run_as(own, "del os.environ['LRX_ENV_GUARD_PROBE']", None)
    assert "LRX_ENV_GUARD_PROBE" not in os.environ


def test_other_code_is_announced(frozen, monkeypatch, capsys):
    monkeypatch.delenv("LRX_ENV_GUARD_PROBE", raising=False)
    _run_as("/opt/thirdparty/lib.py", "os.environ['LRX_ENV_GUARD_PROBE'] = '1'", None)
    assert os.environ.pop("LRX_ENV_GUARD_PROBE") == "1"
    assert "GATE env-frozen" in capsys.readouterr().err


def test_striping_resolver_writes_nothing(frozen, monkeypatch):
    for k in ("LORRAX_PHDF5_STRIPE_COUNT", "LORRAX_PHDF5_STRIPE_SIZE_FS", "LORRAX_PHDF5_STRIPE_SIZE"):
        monkeypatch.delenv(k, raising=False)
    from file_io import _slab_io_ffi as s
    count, unit = s._resolve_striping(8)
    assert count >= 1 and unit >= 1 << 20
    assert "LORRAX_PHDF5_STRIPE_COUNT" not in os.environ and "LORRAX_PHDF5_STRIPE_SIZE_FS" not in os.environ


def test_sanity_forced_level(monkeypatch):
    from common import sanity
    monkeypatch.setenv("LORRAX_SANITY", "0")
    assert not sanity.sanity_strict()
    with sanity.forced_level("strict"):
        assert sanity.sanity_strict()
    assert not sanity.sanity_strict() and os.environ["LORRAX_SANITY"] == "0"
