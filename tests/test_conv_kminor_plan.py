"""Device-free routing contract for the fused k-minor convolution."""


def test_auto_selects_a_row_below_the_portable_floor(monkeypatch):
    from ffi import fft

    mesh = object()
    kgrid = (12, 12, 12)
    assert fft.conv_kminor_row_fits(kgrid)
    monkeypatch.setenv("LORRAX_CONV_KMINOR_FFI", "auto")
    monkeypatch.setattr(
        fft, "conv_kminor_available",
        lambda candidate: (candidate is mesh, "CUDA"),
    )

    assert fft.conv_kminor_plan(mesh, kgrid) == (True, "auto")


def test_over_floor_auto_falls_back_but_on_delegates(monkeypatch):
    from ffi import fft

    mesh = object()
    kgrid = (16, 16, 16)
    assert not fft.conv_kminor_row_fits(kgrid)
    monkeypatch.setattr(
        fft, "conv_kminor_available",
        lambda candidate: (candidate is mesh, "CUDA"),
    )

    monkeypatch.setenv("LORRAX_CONV_KMINOR_FFI", "auto")
    use_kernel, reason = fft.conv_kminor_plan(mesh, kgrid)
    assert not use_kernel
    assert "49152 B" in reason

    required = []
    monkeypatch.setattr(
        fft, "require_conv_kminor",
        lambda candidate: required.append(candidate) or "CUDA",
    )
    monkeypatch.setenv("LORRAX_CONV_KMINOR_FFI", "on")
    assert fft.conv_kminor_plan(mesh, kgrid) == (
        True, "on; device handler derives the residency ceiling",
    )
    assert required == [mesh]
