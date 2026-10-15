"""The runtime's XLA_FLAGS owner: each GPU leg's declared compile defaults are merged, a caller's value wins.

Plain Python, no jax: ``runtime.set_default_xla_gpu_autotune`` only edits ``os.environ``.
The leg is pinned through ``JAX_PLATFORMS`` (its first entry selects the leg).
"""
import os

import pytest

from lxkit import LEGS
from runtime import (_XLA_FLAGS_ENV, _XLA_GPU_AUTOTUNE_FLAG, _XLA_GPU_LLVM_PARALLEL_FLAG,
                     set_default_xla_gpu_autotune)

LLVM = f"{_XLA_GPU_LLVM_PARALLEL_FLAG}=true"
AUTOTUNE = f"{_XLA_GPU_AUTOTUNE_FLAG}=0"


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")


def test_both_defaults_are_appended_once(monkeypatch, cuda):
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    receipt = set_default_xla_gpu_autotune(platform="gpu")
    assert os.environ[_XLA_FLAGS_ENV] == f"{AUTOTUNE} {LLVM}"
    assert receipt["value"] == "0" and receipt["llvm_parallel"] == "true"
    assert receipt["provenance"].startswith("LORRAX default")
    assert receipt["llvm_parallel_provenance"].startswith("LORRAX default")
    assert receipt["applicable"] is True and receipt["xla_flags"] == os.environ[_XLA_FLAGS_ENV]
    assert receipt["leg"] == "CUDA"
    set_default_xla_gpu_autotune(platform="gpu")                 # idempotent
    assert os.environ[_XLA_FLAGS_ENV].split().count(LLVM) == 1


@pytest.mark.parametrize("plat,key", [("cuda", "CUDA"), ("rocm", "ROCM")])
def test_each_leg_gets_exactly_its_declared_defaults(monkeypatch, plat, key):
    monkeypatch.setenv("JAX_PLATFORMS", f"{plat},cpu")
    monkeypatch.delenv(_XLA_FLAGS_ENV, raising=False)
    receipt = set_default_xla_gpu_autotune(platform="gpu")
    want = [f"{f}={v}" for f, v in LEGS[key].xla_flag_defaults]
    assert os.environ.get(_XLA_FLAGS_ENV, "").split() == want
    assert receipt["leg"] == key
    for f, v in LEGS[key].xla_flag_defaults:
        assert receipt["flags"][f]["value"] == v


def test_a_caller_value_wins_flag_by_flag(monkeypatch, cuda):
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


def test_pin_first_visible_device_uses_the_legs_variable(monkeypatch):
    from runtime import pin_first_visible_device
    env = {"ROCR_VISIBLE_DEVICES": "4,5,6,7"}
    monkeypatch.setenv("JAX_PLATFORMS", "rocm,cpu")
    assert pin_first_visible_device(env) == ("ROCR_VISIBLE_DEVICES", "4")
    env = {"ROCR_VISIBLE_DEVICES": "4,5", "HIP_VISIBLE_DEVICES": "1,0"}
    assert pin_first_visible_device(env) == ("HIP_VISIBLE_DEVICES", "1")   # inner one narrows
    assert env["ROCR_VISIBLE_DEVICES"] == "4,5"
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    env = {"CUDA_VISIBLE_DEVICES": "2,3"}
    assert pin_first_visible_device(env) == ("CUDA_VISIBLE_DEVICES", "2")
    env = {}
    assert pin_first_visible_device(env) == ("CUDA_VISIBLE_DEVICES", "0")


def test_an_allocator_the_leg_cannot_build_refuses_by_name(monkeypatch):
    from runtime import _check_allocator_env
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    monkeypatch.setenv("JAX_PLATFORMS", "rocm,cpu")
    with pytest.raises(ValueError, match="GATE xla_allocator_leg"):
        _check_allocator_env()
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    _check_allocator_env()                                       # legal on CUDA
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    _check_allocator_env()                                       # CPU: not checked
    monkeypatch.setenv("JAX_PLATFORMS", "rocm,cpu")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc")
    _check_allocator_env()
