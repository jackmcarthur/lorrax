"""Parent convolution owns its startup gate and cross-rank dial."""
from types import SimpleNamespace
import numpy as np
import pytest
from ffi import fft
from ffi.gate import Gate, reset_gate_state


def test_parent_dial_is_independent_and_registered(monkeypatch):
    from ffi import FFI_DIAL_ENV
    from common.jax_compile_cache import RANK_FINGERPRINT_ENV
    from runtime import _ffi_dial_facts
    monkeypatch.setenv("LORRAX_CONV_KPAIR_FFI", "off")
    monkeypatch.delenv("LORRAX_CONV_KPARENT_FFI", raising=False)
    assert fft.CONV_KPARENT_GATE.mode() == "auto"
    assert fft.CONV_KPARENT_GATE.env in FFI_DIAL_ENV
    assert fft.CONV_KPARENT_GATE.env in RANK_FINGERPRINT_ENV
    assert any(x["env"] == fft.CONV_KPARENT_GATE.env for x in _ffi_dial_facts())


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
def test_parent_on_is_enforced_at_startup(monkeypatch, platform):
    from runtime import _enforce_required_ffi
    monkeypatch.setenv("LORRAX_CONV_KPARENT_FFI", "on")
    original = Gate.enforce
    monkeypatch.setattr(Gate, "enforce", lambda self, mesh, **kw:
        original(self, mesh, **kw) if self is fft.CONV_KPARENT_GATE else None)
    from ffi.common import ffi_loader
    monkeypatch.setattr(ffi_loader, "probe_target", lambda *a: (False, "missing parent target"))
    mesh = SimpleNamespace(devices=np.array([SimpleNamespace(platform=platform)]))
    with pytest.raises(RuntimeError, match="LORRAX_CONV_KPARENT_FFI"):
        _enforce_required_ffi(mesh)


def test_opt_out_announcements_do_not_collide(monkeypatch, capsys):
    reset_gate_state()
    for gate in (fft.CONV_KPAIR_GATE, fft.CONV_KPARENT_GATE):
        monkeypatch.setenv(gate.env, "off")
        gate.enforce(None)
    out = capsys.readouterr().out
    assert "LORRAX_CONV_KPAIR_FFI" in out and "[conv_kparent] off" in out
