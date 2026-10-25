"""The per-platform JAX window (owner decision 2026-10-24): the 0.12 series on
every leg, and exactly 0.9.1 also on ROCm, keyed by the leg the startup
configures (``lxkit.startup_gpu_leg``).  Stdlib only: no JAX is imported."""
from __future__ import annotations

from importlib import metadata, util
from pathlib import Path
from types import SimpleNamespace

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


def test_final_releases_in_preflight_and_runtime(monkeypatch, capsys):
    """Both guards reject prereleases even when their numeric tuple is supported."""
    path = Path(__file__).resolve().parents[1] / "tools" / "require_jax.py"
    spec = util.spec_from_file_location("require_jax_release_fixture", path)
    preflight = util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    cpu = SimpleNamespace(installed_plugin_names=lambda: (),
                          startup_gpu_leg=lambda **kwargs: None)
    monkeypatch.setattr(preflight, "_load",
                        lambda name, path: js if path == preflight._JAX_SUPPORT else cpu)
    versions = {"jax": "0.12.0.1", "jaxlib": "0.12.0.1"}
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(js, "describe", lambda: dict(
        version=versions["jax"], version_info=(0, 12, 0, 1), file="jax",
        jaxlib_version=versions["jaxlib"], jaxlib_version_info=(0, 12, 0, 1),
        jaxlib_file="jaxlib", is_dev_build=False))
    for version in ("0.12.0.1", "0.12.0.1.post1", "0.12.0.1+vendor.1",
                    "0.12.0.1.post1+vendor.1"):
        versions.update(jax=version, jaxlib=version)
        assert js.release_version_info(version) == (0, 12, 0, 1)
        assert not js.check_version("cpu", plugin_names=())
        assert preflight.main() == 0
    assert js.release_version_info("0.9.1+rocm7.14.1") == (0, 9, 1)
    for version in ("0.12.1.dev20261023", "0.12.1a1", "0.12.1b1", "0.12.1rc1"):
        assert not js.release_version_info(version)
        for package in versions:
            versions.update(jax="0.12.0.1", jaxlib="0.12.0.1")
            versions[package] = version
            assert any("not a final public release" in p
                       for p in js.check_version("cpu", plugin_names=()))
            assert preflight.main() == 86
            assert "JAX_ENV_REFUSED" in capsys.readouterr().err
    versions.update(jax="0.12.0.1", jaxlib="0.12.0.1")
    info = js.describe()
    info["is_dev_build"] = True
    monkeypatch.setattr(js, "describe", lambda: info)
    assert any("declares a development build" in p
               for p in js.check_version("cpu", plugin_names=()))


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


def test_gpu_wheel_public_versions(monkeypatch):
    def installed(packages):
        def version(name):
            if name not in packages:
                raise metadata.PackageNotFoundError(name)
            return packages[name]
        monkeypatch.setattr(metadata, "version", version)

    for prefix, version, names, packages in (
        ("", "0.12.0.1", (), {}),
        ("xla_cuda", "0.12.0.1", ("xla_cuda13",), {"jax-cuda13-plugin": "0.12.0.1", "jax-cuda13-pjrt": "0.12.0.1"}),
        ("xla_rocm", "0.9.1", ("xla_rocm7",), {"jax-rocm7-plugin": "0.9.1+rocm7.14.1", "jax-rocm7-pjrt": "0.9.1+rocm7.14.1"}),
        ("xla_rocm", "0.12.0", ("xla_rocm7",), {"jax-rocm7-plugin": "0.12.0", "jax-rocm7-pjrt": "0.12.0"}),
    ):
        installed(packages)
        assert not js.check_gpu_packages(version, prefix, names)

    for packages in (
        {"jax-rocm7-plugin": "0.12.0", "jax-rocm7-pjrt": "0.12.0"},
        {"jax-rocm7-plugin": "0.12.0.1"},
        {"jax-rocm7-pjrt": "0.12.0.1"},
        {"jax-rocm7-plugin": "unknown", "jax-rocm7-pjrt": "0.12.0.1"},
        {"jax-rocm7-plugin": "0.12.0.1.post1", "jax-rocm7-pjrt": "0.12.0.1"},
        {},
    ):
        installed(packages)
        assert js.check_gpu_packages("0.12.0.1", "xla_rocm", ("xla_rocm7",)), packages
    assert js.check_gpu_packages("0.12.0.1", "xla_rocm", ())
    installed({"jax-cuda13-plugin": "0.12.0.1", "jax-cuda13-pjrt": "0.12.0.1",
               "jax-cuda12-plugin": "0.12.0", "jax-cuda12-pjrt": "0.12.0"})
    assert js.check_gpu_packages("0.12.0.1", "xla_cuda", ("xla_cuda12", "xla_cuda13"))
    assert js.check_gpu_packages("unknown", "xla_cuda", ("xla_cuda13",))


def test_gpu_wheel_gate_preflight_and_forced_cpu(tmp_path, monkeypatch, capsys):
    """The metadata preflight and in-process gate reject the same stale ROCm pair."""
    (tmp_path / "jax_plugins" / "xla_rocm7").mkdir(parents=True)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    versions = {"jax": "0.12.0.1", "jaxlib": "0.12.0.1",
                "jax-rocm7-plugin": "0.12.0", "jax-rocm7-pjrt": "0.12.0"}
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(js, "describe", lambda: dict(
        version=versions["jax"], version_info=(0, 12, 0, 1), file="jax",
        jaxlib_version=versions["jaxlib"], jaxlib_version_info=(0, 12, 0), jaxlib_file="jaxlib"))
    assert js.window_leg() == "ROCM"
    assert len(js.check_version()) == 2
    path = Path(__file__).resolve().parents[1] / "tools" / "require_jax.py"
    spec = util.spec_from_file_location("require_jax_fixture", path)
    preflight = util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    assert preflight.main() == 86
    assert "jax-rocm7-plugin" in capsys.readouterr().err
    versions.update({"jax-rocm7-plugin": "0.12.0.1", "jax-rocm7-pjrt": "0.12.0.1"})
    assert not js.check_version()
    assert preflight.main() == 0
