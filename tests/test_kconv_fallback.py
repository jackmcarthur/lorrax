"""The k-convolution router's XLA fallback on CUDA: the decision, from shapes and device attributes.

CPU, no devices: a fake NVIDIA mesh (the strings a device client reports) and the opt-in
shared memory per block passed explicitly (A100 166912 B, sm_86/89 101376 B).  The numerical
match of the fallback against the XLA reference is ``tests/test_kconv_xla_gate.py``.
"""
import types
import warnings

A100, SM86 = 166912, 101376


def _cuda_mesh():
    client = types.SimpleNamespace(platform="cuda", platform_version="cuda 13020")
    dev = types.SimpleNamespace(platform="gpu", client=client, device_kind="NVIDIA A100-SXM4-80GB")
    return types.SimpleNamespace(devices=types.SimpleNamespace(flat=[dev]))


def test_refusal_rules():
    from ffi.fft import mathdx_refusal as why
    assert "axis above 40" in why((48, 1, 1), optin=A100)
    assert "axis above 40" in why((6, 41, 1), optin=A100)
    assert why((40, 1, 1), optin=A100) == ""
    assert why((20, 20, 20), optin=A100) == ""                     # plane 107776 B, column 134416 B
    assert why((20, 20, 20), kminor=True, optin=A100) == ""
    assert "plane tile" in why((20, 20, 20), optin=SM86)            # 107776 > 101376
    assert "k-minor" in why((20, 20, 20), kminor=True, optin=SM86)  # 134416 > 101376
    assert "plane tile" in why((26, 26, 26), optin=A100)            # 179968 > 166912
    assert why((20, 15, 1), optin=SM86) == ""                       # monolayer grids stay on mathdx


def test_backend_decision_and_one_warning(monkeypatch):
    from ffi import fft as F
    from ffi.gate import reset_gate_state
    reset_gate_state()
    monkeypatch.setattr(F, "_optin_smem_bytes", lambda ordinal=0: SM86)
    mesh = _cuda_mesh()
    assert F.kconv_backend(mesh) == "mathdx"
    assert F.kconv_backend(mesh, (6, 6, 1)) == "mathdx"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert F.kconv_backend(mesh, (48, 1, 1)) == "xla"
        assert F.kconv_backend(mesh, (48, 1, 1)) == "xla"
        assert F.kconv_backend(mesh, (20, 20, 20)) == "xla"
    assert len([w for w in seen if "CUDA -> XLA" in str(w.message)]) == 2   # once per grid
    with F.xla_reference():
        assert F.kconv_backend(mesh, (6, 6, 1)) == "xla"
    monkeypatch.setattr(F, "_MATHDX_DOWN", ["GATE mathdx-probe: test"])
    assert F.kconv_backend(mesh) == "xla" and F.kconv_backend(mesh, (6, 6, 1)) == "xla"
