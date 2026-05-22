"""Unit tests for runtime.init_jax_distributed().

Pure CPU, no JAX execution.  All JAX and subprocess calls are mocked.

Four parameterized cases, per the consensus Blitz #4 spec:

  1. Single-rank:   proc_count == 1  → jax.distributed.initialize never called;
                    sentinel set.
  2. Multi-rank fast path: CUDA_VISIBLE_DEVICES set to a single device id;
                    initialize() called with local_device_ids; sentinel set.
  3. Multi-rank fallback: first initialize() raises; second call uses explicit
                    (coordinator_address, num_processes, process_id) form.
  4. Re-entry: sentinel already set → noop (initialize never called a second time
                    even when proc_count > 1).

The module under test (``runtime``) is importable without JAX because it
only ``import jax`` inside the body of ``init_jax_distributed()`` at the
multi-rank branch.  We patch ``jax.distributed`` at the point where the
runtime module resolves it.
"""
from __future__ import annotations

import importlib
import os
import sys
import types
import unittest.mock as mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"


def _reload_runtime():
    """Return a freshly imported runtime module, bypassing the module cache."""
    if "runtime" in sys.modules:
        del sys.modules["runtime"]
    import runtime as rt
    return rt


def _make_jax_stub(initialize_side_effect=None):
    """Build a minimal jax stub with a mock jax.distributed.initialize."""
    jax_stub = types.ModuleType("jax")
    dist_stub = types.ModuleType("jax.distributed")
    init_mock = mock.MagicMock(side_effect=initialize_side_effect)
    dist_stub.initialize = init_mock
    jax_stub.distributed = dist_stub
    return jax_stub, init_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitJaxDistributed:
    """Parameterized cases for runtime.init_jax_distributed()."""

    @pytest.fixture(autouse=True)
    def clean_sentinel(self, monkeypatch):
        """Ensure the sentinel env var is absent before each test."""
        monkeypatch.delenv(_SENTINEL, raising=False)

    # ------------------------------------------------------------------
    # Case 1: Single-rank — no init called, sentinel set.
    # ------------------------------------------------------------------
    def test_single_rank_no_init_called(self, monkeypatch):
        """proc_count == 1: jax.distributed.initialize must NOT be called."""
        monkeypatch.setenv("SLURM_NTASKS", "1")
        monkeypatch.delenv("JAX_PROCESS_COUNT", raising=False)
        monkeypatch.delenv("JAX_NUM_PROCESSES", raising=False)

        jax_stub, init_mock = _make_jax_stub()

        rt = _reload_runtime()
        with mock.patch.dict(sys.modules, {"jax": jax_stub}):
            rt.init_jax_distributed()

        init_mock.assert_not_called()
        assert os.environ.get(_SENTINEL) == "1"

    # ------------------------------------------------------------------
    # Case 2: Multi-rank fast path — CUDA_VISIBLE_DEVICES set to one id.
    # ------------------------------------------------------------------
    def test_multi_rank_fast_path_local_device_ids(self, monkeypatch):
        """Multi-rank with CUDA_VISIBLE_DEVICES='0': local_device_ids=[0] passed."""
        monkeypatch.setenv("SLURM_NTASKS", "4")
        monkeypatch.delenv("JAX_PROCESS_COUNT", raising=False)
        monkeypatch.delenv("JAX_NUM_PROCESSES", raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

        jax_stub, init_mock = _make_jax_stub(initialize_side_effect=None)

        rt = _reload_runtime()
        with mock.patch.dict(sys.modules, {"jax": jax_stub}):
            rt.init_jax_distributed()

        init_mock.assert_called_once_with(local_device_ids=[0])
        assert os.environ.get(_SENTINEL) == "1"

    @pytest.mark.parametrize("cvd", ["0,1", "0,1,2,3", ""])
    def test_multi_rank_no_local_device_ids_when_multiple_or_empty(
        self, monkeypatch, cvd
    ):
        """Multi-rank with multiple or absent CUDA_VISIBLE_DEVICES: no local_device_ids."""
        monkeypatch.setenv("SLURM_NTASKS", "4")
        monkeypatch.delenv("JAX_PROCESS_COUNT", raising=False)
        monkeypatch.delenv("JAX_NUM_PROCESSES", raising=False)
        if cvd:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
        else:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        jax_stub, init_mock = _make_jax_stub(initialize_side_effect=None)

        rt = _reload_runtime()
        with mock.patch.dict(sys.modules, {"jax": jax_stub}):
            rt.init_jax_distributed()

        # First call should be with no local_device_ids (empty kwargs).
        first_call_kwargs = init_mock.call_args_list[0].kwargs
        assert "local_device_ids" not in first_call_kwargs or first_call_kwargs == {}

    # ------------------------------------------------------------------
    # Case 3: Multi-rank fallback — first initialize raises; second uses
    #         explicit coordinator_address / num_processes / process_id.
    # ------------------------------------------------------------------
    def test_multi_rank_fallback_uses_explicit_coordinator(self, monkeypatch):
        """First initialize raises → fallback with explicit coordinator args."""
        monkeypatch.setenv("SLURM_NTASKS", "4")
        monkeypatch.setenv("SLURM_PROCID", "2")
        monkeypatch.delenv("JAX_PROCESS_COUNT", raising=False)
        monkeypatch.delenv("JAX_NUM_PROCESSES", raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        # Provide a stable coordinator address via env (skip scontrol path).
        monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "nid00001:12355")

        # First call raises; second call succeeds.
        first = True

        def _side_effect(**kwargs):
            nonlocal first
            if first:
                first = False
                raise RuntimeError("simulated first-call failure")

        jax_stub, init_mock = _make_jax_stub(initialize_side_effect=_side_effect)

        rt = _reload_runtime()
        with mock.patch.dict(sys.modules, {"jax": jax_stub}):
            rt.init_jax_distributed()

        assert init_mock.call_count == 2
        # Second call must include coordinator_address, num_processes, process_id.
        second_kwargs = init_mock.call_args_list[1].kwargs
        assert second_kwargs["coordinator_address"] == "nid00001:12355"
        assert second_kwargs["num_processes"] == 4
        assert second_kwargs["process_id"] == 2
        assert os.environ.get(_SENTINEL) == "1"

    def test_multi_rank_fallback_scontrol_path(self, monkeypatch):
        """Fallback resolves coordinator via scontrol when SLURM_NODELIST is set."""
        monkeypatch.setenv("SLURM_NTASKS", "4")
        monkeypatch.setenv("SLURM_PROCID", "0")
        monkeypatch.setenv("SLURM_NODELIST", "nid[001-004]")
        monkeypatch.delenv("JAX_COORDINATOR_ADDRESS", raising=False)
        monkeypatch.delenv("JAX_PROCESS_COUNT", raising=False)
        monkeypatch.delenv("JAX_NUM_PROCESSES", raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        first = True

        def _side_effect(**kwargs):
            nonlocal first
            if first:
                first = False
                raise RuntimeError("simulated first-call failure")

        jax_stub, init_mock = _make_jax_stub(initialize_side_effect=_side_effect)

        scontrol_result = mock.MagicMock()
        scontrol_result.stdout = "nid001\nnid002\nnid003\nnid004\n"

        rt = _reload_runtime()
        with (
            mock.patch.dict(sys.modules, {"jax": jax_stub}),
            mock.patch("subprocess.run", return_value=scontrol_result) as sp_mock,
        ):
            rt.init_jax_distributed()

        # scontrol was called for the nodelist expansion.
        sp_mock.assert_called_once()
        second_kwargs = init_mock.call_args_list[1].kwargs
        assert second_kwargs["coordinator_address"] == "nid001:12355"
        assert os.environ.get(_SENTINEL) == "1"

    # ------------------------------------------------------------------
    # Case 4: Re-entry — sentinel already set → noop.
    # ------------------------------------------------------------------
    def test_reentry_noop(self, monkeypatch):
        """Sentinel already present → init_jax_distributed() is a no-op."""
        monkeypatch.setenv(_SENTINEL, "1")
        monkeypatch.setenv("SLURM_NTASKS", "4")

        jax_stub, init_mock = _make_jax_stub()

        rt = _reload_runtime()
        with mock.patch.dict(sys.modules, {"jax": jax_stub}):
            rt.init_jax_distributed()
            rt.init_jax_distributed()  # call twice to be sure

        init_mock.assert_not_called()

    # ------------------------------------------------------------------
    # Additional: JAX_PROCESS_COUNT / JAX_NUM_PROCESSES override SLURM_NTASKS.
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("env_var", ["JAX_PROCESS_COUNT", "JAX_NUM_PROCESSES"])
    def test_proc_count_env_override(self, monkeypatch, env_var):
        """JAX_PROCESS_COUNT / JAX_NUM_PROCESSES take precedence over SLURM_NTASKS."""
        monkeypatch.setenv(env_var, "2")
        monkeypatch.setenv("SLURM_NTASKS", "1")  # would mean single-rank if read
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        # Clear the other override env var.
        other = "JAX_NUM_PROCESSES" if env_var == "JAX_PROCESS_COUNT" else "JAX_PROCESS_COUNT"
        monkeypatch.delenv(other, raising=False)

        jax_stub, init_mock = _make_jax_stub(initialize_side_effect=None)

        rt = _reload_runtime()
        with mock.patch.dict(sys.modules, {"jax": jax_stub}):
            rt.init_jax_distributed()

        # proc_count was 2, so initialize should have been called.
        assert init_mock.call_count >= 1
