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
             "SLURM_NODELIST", "SLURM_STEP_NODELIST", "SLURM_STEP_ID",
             runtime._DISTRIBUTED_SENTINEL,
             # GPU allocator knobs: cleared so the allocator tests are
             # independent of BOTH test order and the ambient shell.  A
             # developer who has sourced config/frontera/ffi_env.sh has
             # XLA_PYTHON_CLIENT_ALLOCATOR exported, which would otherwise
             # make the "stays unset" assertion fail for the wrong reason.
             "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_ALLOCATOR",
             "TF_GPU_ALLOCATOR")


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


def test_coordinator_address_prefers_step_host_and_step_unique_port(
        clean_env, monkeypatch):
    clean_env.setenv("SLURM_NODELIST", "allocation[01-04]")
    clean_env.setenv("SLURM_STEP_NODELIST", "step04")
    clean_env.setenv("SLURM_STEP_ID", "97")

    class Result:
        stdout = "step04\n"

    seen = []

    def fake_run(argv, **kwargs):
        seen.append((argv, kwargs))
        return Result()

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    assert _resolve_coordinator_address() == "step04:22097"
    assert seen[0][0][-1] == "step04"


def test_set_default_env_defaults(clean_env):
    """``set_default_env`` asks for ``cuda,cpu``; the CPU demotion may win.

    NODE-DEPENDENT BY DESIGN, and the reason is worth stating because this
    assertion was previously written for a GPU node and failed on every CPU
    one.  ``set_default_env()`` does ``setdefault("JAX_PLATFORMS","cuda,cpu")``
    (``runtime/__init__.py:239``) and then calls
    ``skip_gpu_plugin_discovery()`` (``:246``).  On a node with no NVIDIA
    device that takes its arm-2 branch (``:344``) and deliberately OVERWRITES
    ``JAX_PLATFORMS="cpu"`` (``:410``) — the demotion argued at ``:404-408``,
    which is what saves ~77 s of dlopening a CUDA stack the run cannot use.

    So the invariant is not a fixed string: it is that the resolved platform
    agrees with whether a GPU is actually present.  Asserting either literal
    alone makes the suite pass on one queue and fail on the other.
    """
    from runtime import _gpu_is_present
    clean_env.delenv("JAX_ENABLE_X64", raising=False)
    clean_env.delenv("JAX_PLATFORMS", raising=False)
    set_default_env()
    assert os.environ["JAX_ENABLE_X64"] == "1"
    expected = "cuda,cpu" if _gpu_is_present() else "cpu"
    assert os.environ["JAX_PLATFORMS"] == expected, (
        f"resolved {os.environ['JAX_PLATFORMS']!r} but _gpu_is_present()="
        f"{_gpu_is_present()} implies {expected!r}")


def test_set_default_env_respects_override(clean_env):
    clean_env.setenv("JAX_PLATFORMS", "cpu")               # caller wins (setdefault)
    set_default_env()
    assert os.environ["JAX_PLATFORMS"] == "cpu"


def test_set_default_env_disables_gpu_preallocation(clean_env):
    """The canonical GPU allocator answer, pinned so it cannot drift back.

    Left unset, jaxlib omits the ``preallocate`` option and the PJRT GPU
    client preallocates 75% of the card.  LORRAX's FFI handlers allocate
    OUTSIDE XLA (the cuFFT arena in ``ffi/cufft``), so that hoard is taken
    straight out of their budget.  Measured on 8 Quadro RTX 5000 across 2
    nodes (jobs 7882442 / 7882447): with 6 GiB of live XLA arrays on a
    15.74 GB card, the largest cuFFT plan still creatable goes from 3.07 GB
    (preallocate unset) to 7.16 GB (``false``).

    This regression is exactly how the setting was lost once already: the
    value used to live in the drivers, and removing it from one of them
    silently changed GPU behaviour because nothing else supplied it.
    """
    clean_env.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    set_default_env()
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"

    # ALLOCATOR stays unset on purpose -> BFC, which is the only kind that
    # keeps memory_stats() populated for gw_init/gw_output/aot_memory.
    # `platform` reports bytes_limit=0 and peak_bytes_in_use=0 (job 7882447).
    assert "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ
    # TF_GPU_ALLOCATOR is a TensorFlow variable and is inert for JAX.
    assert "TF_GPU_ALLOCATOR" not in os.environ


def test_set_default_env_preallocate_override_wins(clean_env):
    """A deployment script's explicit export must beat our setdefault.

    ``config/frontera/ffi_env.sh`` exports the cuda_async pair together with
    the sm_75 command-buffer XLA_FLAGS it requires; that combination has to
    survive ``bootstrap()``.
    """
    clean_env.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    set_default_env()
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"


def test_set_default_env_cpu_does_not_touch_gpu_allocator(clean_env):
    """A CPU-forced run has no GPU client, so it gets no GPU allocator env."""
    clean_env.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    set_default_env(platform="cpu")
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ


def test_set_default_env_refuses_an_allocator_typo(clean_env):
    """A misspelled allocator must fail HERE, naming itself.

    jaxlib validates the value inside ``generate_pjrt_gpu_plugin_options()``,
    which runs during CUDA plugin discovery — where every exception is
    swallowed and re-reported as ``Backend 'cuda' is not in the list of
    known backends``.  That message names neither the variable nor the
    value and reads as missing hardware; with a CPU in ``JAX_PLATFORMS`` it
    is not even an error, just a silent fall back to CPU.

    Full behaviour matrix (incl. jaxlib's no-strip comparison) lives in
    ``tests/test_crossfile_requests.py``, which runs on the login node.
    """
    clean_env.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platfrom")
    with pytest.raises(ValueError, match="XLA_PYTHON_CLIENT_ALLOCATOR"):
        set_default_env(platform="cpu")


def test_set_default_env_removes_a_blank_allocator(clean_env):
    """Blank is UNSET in LORRAX's grammar — but jaxlib REJECTS ``''``, so it
    has to be removed rather than merely tolerated."""
    clean_env.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "")
    set_default_env(platform="cpu")
    assert "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ


def test_set_default_env_keeps_a_legal_allocator(clean_env):
    """``config/frontera/ffi_env.sh`` exports ``cuda_async``; it must survive."""
    clean_env.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    set_default_env(platform="cpu")
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"


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


def test_init_distributed_slurm_step_uses_explicit_unique_coordinator(
        clean_env, monkeypatch):
    clean_env.setenv("SLURM_NTASKS", "4")
    clean_env.setenv("SLURM_PROCID", "2")
    clean_env.setenv("SLURM_STEP_NODELIST", "step04")
    clean_env.setenv("SLURM_STEP_ID", "97")
    clean_env.setenv("CUDA_VISIBLE_DEVICES", "2")
    calls = []

    class Distributed:
        @staticmethod
        def initialize(**kwargs):
            calls.append(kwargs)

    class FakeJax:
        distributed = Distributed()

    class Result:
        stdout = "step04\n"

    monkeypatch.setattr(runtime, "skip_gpu_plugin_discovery", lambda: None)
    monkeypatch.setattr(runtime, "_import_jax", lambda: FakeJax())
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: Result())
    init_jax_distributed()
    assert calls == [{
        "coordinator_address": "step04:22097",
        "num_processes": 4,
        "process_id": 2,
        "local_device_ids": [0],
    }]


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
