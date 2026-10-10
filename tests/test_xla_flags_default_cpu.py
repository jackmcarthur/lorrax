"""The runtime's XLA_FLAGS owner: the two GPU compile defaults are merged, a caller's value wins.

Plain Python, no jax: ``runtime.set_default_xla_gpu_autotune`` only edits ``os.environ``.
"""
import os

import pytest

from runtime import (_XLA_FLAGS_ENV, _XLA_GPU_AUTOTUNE_FLAG, _XLA_GPU_LLVM_PARALLEL_FLAG,
                     set_default_xla_gpu_autotune)

LLVM = f"{_XLA_GPU_LLVM_PARALLEL_FLAG}=true"
AUTOTUNE = f"{_XLA_GPU_AUTOTUNE_FLAG}=0"


def test_both_defaults_are_appended_once(monkeypatch):
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    receipt = set_default_xla_gpu_autotune(platform="gpu")
    assert os.environ[_XLA_FLAGS_ENV] == f"{AUTOTUNE} {LLVM}"
    assert receipt["value"] == "0" and receipt["llvm_parallel"] == "true"
    assert receipt["provenance"].startswith("LORRAX default")
    assert receipt["llvm_parallel_provenance"].startswith("LORRAX default")
    assert receipt["applicable"] is True and receipt["xla_flags"] == os.environ[_XLA_FLAGS_ENV]
    set_default_xla_gpu_autotune(platform="gpu")                 # idempotent
    assert os.environ[_XLA_FLAGS_ENV].split().count(LLVM) == 1


def test_a_caller_value_wins_flag_by_flag(monkeypatch):
    given = f"--xla_foo=1 {_XLA_GPU_LLVM_PARALLEL_FLAG}=false"
    monkeypatch.setenv(_XLA_FLAGS_ENV, given)
    receipt = set_default_xla_gpu_autotune(platform="gpu")
    assert os.environ[_XLA_FLAGS_ENV] == f"{given} {AUTOTUNE}"     # llvm kept, autotune appended
    assert receipt["llvm_parallel"] == "false"
    assert receipt["llvm_parallel_provenance"] == "caller-supplied XLA_FLAGS"
    assert receipt["value"] == "0" and receipt["provenance"].startswith("LORRAX default")

    both = f"{_XLA_GPU_AUTOTUNE_FLAG} 2 {_XLA_GPU_LLVM_PARALLEL_FLAG} false"
    monkeypatch.setenv(_XLA_FLAGS_ENV, both)
    receipt = set_default_xla_gpu_autotune(platform="gpu")
    assert os.environ[_XLA_FLAGS_ENV] == both                     # nothing rewritten
    assert (receipt["value"], receipt["llvm_parallel"]) == ("2", "false")


def test_a_cpu_startup_leaves_xla_flags_alone(monkeypatch):
    monkeypatch.setenv(_XLA_FLAGS_ENV, "--xla_foo=1")
    receipt = set_default_xla_gpu_autotune(platform="cpu")
    assert os.environ[_XLA_FLAGS_ENV] == "--xla_foo=1"
    assert receipt["applicable"] is False
    assert receipt["llvm_parallel"] is None and "forced CPU startup" in receipt["llvm_parallel_provenance"]
    with pytest.raises(ValueError):
        set_default_xla_gpu_autotune(platform="tpu")
