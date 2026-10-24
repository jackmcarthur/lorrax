"""The per-platform JAX window (owner decision 2026-10-24): the 0.12 series on
every leg, and exactly 0.9.1 also on ROCm, keyed by the leg the startup
configures (``lxkit.startup_gpu_leg``).  Stdlib only: no JAX is imported."""
from __future__ import annotations

from lxkit import LEGS, startup_gpu_leg
from runtime import jax_support as js


def test_window_table_per_leg():
    assert set(js.SUPPORTED_WINDOWS) == set(LEGS)
    for leg in LEGS:
        assert js.in_window((0, 12, 0), leg) and js.in_window((0, 12, 7), leg)
        for refused in ((0, 9, 0), (0, 9, 2), (0, 10, 0), (0, 11, 9), (0, 13, 0)):
            assert not js.in_window(refused, leg), (leg, refused)
    assert js.in_window((0, 9, 1), "ROCM")
    assert not js.in_window((0, 9, 1), "CUDA") and not js.in_window((0, 9, 1), "cpu")
    # 0.9.1's compile-cache hook has no host_callbacks; 0.12's does.
    key = ("jax._src.compilation_cache", "get_executable_and_time")
    assert js.required_private_params((0, 9, 1))[key][0][-1] == "executable_devices"
    assert js.required_private_params((0, 12, 0))[key][0][-1] == "host_callbacks"


def test_window_leg_is_the_startup_leg(tmp_path, monkeypatch):
    (tmp_path / "jax_plugins" / "xla_rocm7").mkdir(parents=True)
    no_dev = dict(exists=lambda pattern: False)
    # JAX_PLATFORMS naming a device leg wins; a forced-CPU process keys the
    # installed plugin's leg; with neither there is no device leg ("cpu").
    assert startup_gpu_leg(path=[], environ={"JAX_PLATFORMS": "cuda,cpu"}, **no_dev).key == "CUDA"
    assert startup_gpu_leg(path=[str(tmp_path)], environ={"JAX_PLATFORMS": "cpu"},
                           **no_dev).key == "ROCM"
    assert startup_gpu_leg(path=[], environ={"JAX_PLATFORMS": "cpu"}, **no_dev) is None
    monkeypatch.setenv("JAX_PLATFORMS", "rocm")
    assert js.window_leg() == "ROCM"
