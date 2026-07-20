"""Unit tests for the single-sourced distributed bootstrap in ``runtime``.

``bse.exciton_bands`` (and gw.gw_jax, psp.run_nscf) rely on
``runtime.set_default_env`` / ``runtime.init_jax_distributed`` /
``runtime.fallback_to_cpu_if_no_gpu_backend`` to bring up a multi-node
one-process-per-GPU job.  These tests pin the pure-logic pieces and the
single-process no-op / idempotency contract that keeps the 1-GPU and CI
paths byte-unchanged (they do NOT spawn real processes).
"""
import os

import pytest

import runtime
from runtime import (set_default_env, init_jax_distributed,
                     _resolve_proc_count, _resolve_proc_id,
                     _resolve_coordinator_address)

_ENV_KEYS = ("JAX_PROCESS_COUNT", "JAX_NUM_PROCESSES", "SLURM_NTASKS",
             "JAX_PROCESS_INDEX", "SLURM_PROCID", "JAX_COORDINATOR_ADDRESS",
             "SLURM_NODELIST", runtime._DISTRIBUTED_SENTINEL)


@pytest.fixture
def clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_resolve_proc_count_precedence(clean_env):
    assert _resolve_proc_count() == 1                       # nothing set
    clean_env.setenv("SLURM_NTASKS", "16")
    assert _resolve_proc_count() == 16
    clean_env.setenv("JAX_NUM_PROCESSES", "8")              # overrides SLURM
    assert _resolve_proc_count() == 8
    clean_env.setenv("JAX_PROCESS_COUNT", "4")              # highest precedence
    assert _resolve_proc_count() == 4


def test_resolve_proc_id_precedence(clean_env):
    assert _resolve_proc_id() == 0
    clean_env.setenv("SLURM_PROCID", "7")
    assert _resolve_proc_id() == 7
    clean_env.setenv("JAX_PROCESS_INDEX", "3")              # overrides SLURM
    assert _resolve_proc_id() == 3


def test_coordinator_address_env_override(clean_env):
    clean_env.setenv("JAX_COORDINATOR_ADDRESS", "myhost:1234")
    assert _resolve_coordinator_address() == "myhost:1234"


def test_coordinator_address_has_port(clean_env):
    # No SLURM_NODELIST → falls back to a hostname:port; must carry a port.
    addr = _resolve_coordinator_address()
    assert ":" in addr and addr.rsplit(":", 1)[1].isdigit()


def test_set_default_env_defaults(clean_env):
    clean_env.delenv("JAX_ENABLE_X64", raising=False)
    clean_env.delenv("JAX_PLATFORMS", raising=False)
    set_default_env()
    assert os.environ["JAX_ENABLE_X64"] == "1"
    assert os.environ["JAX_PLATFORMS"] == "cuda,cpu"


def test_set_default_env_respects_override(clean_env):
    clean_env.setenv("JAX_PLATFORMS", "cpu")               # caller wins (setdefault)
    set_default_env()
    assert os.environ["JAX_PLATFORMS"] == "cpu"


def test_init_distributed_single_process_is_noop(clean_env):
    """proc_count<=1 ⇒ NO jax.distributed.initialize(); just sets sentinel.

    This is the 1-GPU / CI path — importing bse.exciton_bands must not try to
    stand up a coordinator when run on one device.
    """
    import jax
    called = {"n": 0}
    orig = jax.distributed.initialize
    jax.distributed.initialize = lambda *a, **k: called.__setitem__("n", called["n"] + 1)
    try:
        init_jax_distributed()                              # proc_count == 1
    finally:
        jax.distributed.initialize = orig
    assert called["n"] == 0
    assert os.environ.get(runtime._DISTRIBUTED_SENTINEL) == "1"


def test_init_distributed_idempotent(clean_env):
    """Sentinel guard: a second call is a no-op even if proc_count>1 now."""
    import jax
    init_jax_distributed()                                  # sets sentinel (proc=1)
    clean_env.setenv("SLURM_NTASKS", "16")                  # pretend multi-proc
    called = {"n": 0}
    orig = jax.distributed.initialize
    jax.distributed.initialize = lambda *a, **k: called.__setitem__("n", called["n"] + 1)
    try:
        init_jax_distributed()                              # sentinel already set
    finally:
        jax.distributed.initialize = orig
    assert called["n"] == 0                                 # guarded out
