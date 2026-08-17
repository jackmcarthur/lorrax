"""Static/grammar contract for Sigma's direct k-leading fused-conv member."""

from pathlib import Path

import pytest


def test_default_is_off_and_grammar_is_three_mode(monkeypatch):
    from ffi.fft import CONV_KLEAD_GATE, conv_klead_mode

    monkeypatch.delenv("LORRAX_CONV_KLEAD_FFI", raising=False)
    assert conv_klead_mode() == "off"
    assert CONV_KLEAD_GATE.modes == ("off", "auto", "on")
    assert CONV_KLEAD_GATE.auto_capability


def test_residency_mirror_and_axis_envelope():
    from ffi.fft import conv_klead_row_fits

    assert conv_klead_row_fits((4, 4, 4))
    assert conv_klead_row_fits((3, 3, 1))
    assert not conv_klead_row_fits((25, 1, 1))
    assert not conv_klead_row_fits((24, 24, 24))


def test_residency_mirror_prices_the_coalesced_resident_staging_row():
    from ffi.fft import conv_klead_row_fits

    # (1,1,1): the T staging destination/resident row and resident W row are
    # 32 B total, rings are 48 B, aligned metadata is 16 B: no second tile.
    assert conv_klead_row_fits((1, 1, 1), smem_bytes=96)
    assert not conv_klead_row_fits((1, 1, 1), smem_bytes=95)


@pytest.mark.parametrize(
    "norm,nk,want",
    [
        ("ortho", 64, 64.0 ** -1.5),
        (None, 64, 64.0 ** -2.0),
        ("forward", 64, 64.0 ** -1.0),
    ],
)
def test_folded_scale_has_two_inverse_factors(norm, nk, want):
    from ffi.fft import conv_klead_scale

    assert conv_klead_scale(norm, nk) == pytest.approx(want, rel=0, abs=0)


def test_loader_and_isolated_build_target_are_registered():
    from ffi.common import ffi_loader

    assert (ffi_loader._CUDA_TARGET_SYMBOLS["lorrax_cufft_conv_klead"]
            == "CufftConvKLeadCudaFfi")
    root = Path(__file__).resolve().parents[1]
    cmake = (root / "src/ffi/cpp/CMakeLists.txt").read_text()
    assert "add_library(build_conv_klead_cuda OBJECT" in cmake
    assert "$<TARGET_OBJECTS:build_conv_klead_cuda>" in cmake


def test_zero_transpose_path_passes_native_kleading_t_directly():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/ffi/fft.py").read_text()
    body = text.split("def make_conv_klead_ffi(", 1)[1].split(
        "# ===========================================================================", 1)[0]
    assert "jnp.moveaxis(" not in body
    assert "t_local, w_local, **attrs" in body
    assert "input_output_aliases" not in body


def test_dial_is_in_both_cross_rank_registries():
    from common.jax_compile_cache import RANK_FINGERPRINT_ENV
    from ffi import FFI_DIAL_ENV

    assert "LORRAX_CONV_KLEAD_FFI" in FFI_DIAL_ENV
    assert "LORRAX_CONV_KLEAD_FFI" in RANK_FINGERPRINT_ENV
