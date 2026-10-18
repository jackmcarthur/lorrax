"""Loader-level refusal of a host-leg SLATE handler in a device process: neither
loader registers it, both probes report the GATE, and distrib_la's SLATE call
sites raise it by name (lxkit.native_provider.refuse_if_borrowed)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("jax")
from lxkit import native_provider as npv  # noqa: E402

GATE = "GATE native-closure: host-leg lorrax_slate_eigh needs SLATE-CPU but the device leg's SLATE serves this process"


@pytest.fixture
def device_process(monkeypatch, tmp_path):
    rocm_leg, host_leg = tmp_path / "rocm.so", tmp_path / "host.so"
    hip, cpu = tmp_path / "hip/libslate.so.2", tmp_path / "cpu/libslate.so.2"
    host_rec = SimpleNamespace(entries=(SimpleNamespace(family="libslate", path=cpu),))
    monkeypatch.setattr(npv, "_ENFORCED", {
        rocm_leg: SimpleNamespace(verified={"libslate": (hip, "x")}, served_elsewhere={},
                                  record=SimpleNamespace(entries=())),
        host_leg: SimpleNamespace(verified={}, served_elsewhere={"libslate": rocm_leg},
                                  record=host_rec)})
    monkeypatch.setattr(npv, "_PLATFORM_LEG", {"ROCM": rocm_leg, "cpu": host_leg})
    return hip, cpu


class _Lib:
    """Exports every handler symbol a loader's cpu table names."""
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return object()


def _capture_registrations(monkeypatch):
    import jax.ffi
    got = []
    monkeypatch.setattr(jax.ffi, "register_ffi_target",
                        lambda name, capsule, platform: got.append((name, platform)))
    monkeypatch.setattr(jax.ffi, "pycapsule", lambda fn: fn)
    return got


def test_distrib_la_loader_registers_no_borrowed_handler(device_process, monkeypatch):
    from distrib_la import loader
    got = _capture_registrations(monkeypatch)
    loader._register_ffi_targets(_Lib(), "cpu")
    names = {n for n, _ in got}
    assert not {n for n in names if n.startswith("lorrax_slate_")}
    assert "lorrax_scalapack_eigh" in names


def test_ffi_loader_registers_no_borrowed_handler(device_process, monkeypatch):
    from ffi.common import ffi_loader
    got = _capture_registrations(monkeypatch)
    ffi_loader._register_ffi_targets(_Lib(), "cpu")
    names = {n for n, _ in got}
    assert not {n for n in names if n.startswith("lorrax_slate_")}
    assert "lorrax_phdf5_read" in names


def test_both_probes_report_the_gate(device_process, monkeypatch):
    from distrib_la import loader
    from ffi.common import ffi_loader
    monkeypatch.setattr(loader, "get_lib", lambda platform: _Lib())
    monkeypatch.setattr(ffi_loader, "get_lib", lambda platform=None: _Lib())
    for probe in (loader.probe_target, ffi_loader.probe_target):
        ok, reason = probe("lorrax_slate_eigh", "cpu")
        assert not ok and reason.startswith(GATE)
        assert probe("lorrax_scalapack_eigh", "cpu")[0]


def test_distrib_la_slate_call_site_refuses_by_name(device_process, monkeypatch):
    from distrib_la import _slate, loader
    monkeypatch.setattr(loader, "get_lib", lambda platform: _Lib())
    monkeypatch.setattr(_slate, "_mesh_platform", lambda mesh: "cpu")
    with pytest.raises(loader.LibraryUnusable, match="^" + GATE.replace("(", r"\(")):
        _slate.ensure_registered(None, _slate._EIGH_FFI_TARGET)
    _slate.ensure_registered(None)                  # no target: loads only
    monkeypatch.setattr(_slate, "_mesh_platform", lambda mesh: "ROCM")
    _slate.ensure_registered(None, _slate._EIGH_FFI_TARGET)    # the device leg's own SLATE
