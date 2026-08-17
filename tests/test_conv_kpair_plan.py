"""Device-free routing and registration contract for ISDF conv_kpair."""

from pathlib import Path

import pytest


def _available(monkeypatch, fft):
    monkeypatch.setattr(fft, "conv_kpair_available", lambda mesh: (True, "CUDA"))


def test_default_is_auto_and_grammar_is_three_mode(monkeypatch):
    from ffi import fft

    monkeypatch.delenv("LORRAX_CONV_KPAIR_FFI", raising=False)
    assert fft.conv_kpair_mode() == "auto"
    assert fft.CONV_KPAIR_GATE.modes == ("off", "auto", "on")


@pytest.mark.parametrize("ns,banks,want", [(1, 2, 80), (2, 3, 96), (4, 3, 96)])
def test_residency_mirror_prices_odd_banks_and_rings(ns, banks, want):
    from ffi.fft import conv_kpair_resident_bytes

    assert conv_kpair_resident_bytes((1, 1, 1), ns) == want
    assert conv_kpair_resident_bytes((24, 24, 24), ns) == (
        16 * (banks * (24**3 | 1) + 72))
    assert conv_kpair_resident_bytes((25, 1, 1), ns) == -1


@pytest.mark.parametrize("kgrid", [(3, 3, 3), (4, 4, 4), (2, 3, 5), (5, 5, 5)])
def test_auto_selects_resident_for_representative_fit_shapes(monkeypatch, kgrid):
    from ffi import fft

    monkeypatch.setenv("LORRAX_CONV_KPAIR_FFI", "auto")
    _available(monkeypatch, fft)
    arm, reason = fft.conv_kpair_plan(object(), kgrid, 2, (16, 16))
    assert arm == "resident"
    assert "resident minimum=" in reason


@pytest.mark.parametrize("kgrid", [(16, 16, 16), (24, 24, 24), (23, 19, 17)])
def test_auto_retains_xla_when_coverage_arm_is_slower(monkeypatch, kgrid):
    from ffi import fft

    monkeypatch.setenv("LORRAX_CONV_KPAIR_FFI", "auto")
    _available(monkeypatch, fft)
    arm, reason = fft.conv_kpair_plan(object(), kgrid, 2, (4, 4))
    assert arm == "xla"
    assert "measured XLA-fast region" in reason


def test_auto_crossover_and_axis_refusal(monkeypatch):
    from ffi import fft

    monkeypatch.setenv("LORRAX_CONV_KPAIR_FFI", "auto")
    _available(monkeypatch, fft)
    assert fft.conv_kpair_plan(object(), (1, 1, 1), 4, (128, 128))[0] == "resident"
    assert fft.conv_kpair_plan(object(), (3, 3, 3), 1, (2, 2))[0] == "resident"
    assert fft.conv_kpair_plan(object(), (14, 14, 14), 1, (8, 8))[0] == "device"
    assert fft.conv_kpair_plan(object(), (15, 15, 15), 1, (1, 64))[0] == "xla"
    assert fft.conv_kpair_plan(object(), (15, 15, 15), 1, (1, 1024))[0] == "device"
    assert fft.conv_kpair_plan(object(), (12, 12, 12), 2, (1, 64))[0] == "xla"
    assert fft.conv_kpair_plan(object(), (12, 12, 12), 2, (1, 1024))[0] == "device"
    assert fft.conv_kpair_plan(object(), (15, 15, 15), 2, (1, 64))[0] == "xla"
    assert fft.conv_kpair_plan(object(), (15, 15, 15), 2, (1, 1024))[0] == "device"
    arm, reason = fft.conv_kpair_plan(object(), (25, 1, 1), 2, (64, 64))
    assert arm == "xla"
    assert "outside [1,24]" in reason


def test_on_delegates_all_final_shape_and_residency_checks(monkeypatch):
    from ffi import fft

    required = []
    monkeypatch.setenv("LORRAX_CONV_KPAIR_FFI", "on")
    monkeypatch.setattr(
        type(fft.CONV_KPAIR_GATE), "require",
        lambda self, mesh, target=None: required.append((mesh, target)) or "CUDA",
    )
    mesh = object()
    assert fft.conv_kpair_plan(mesh, (25, 24, 24), 4, (1, 1)) == (
        "device", "on; C++ derives the final residency/refusal verdict")
    assert required == [(mesh, fft.CONV_KPAIR_TARGET)]


def test_registration_build_target_and_cross_rank_dials():
    from common.jax_compile_cache import RANK_FINGERPRINT_ENV
    from ffi import FFI_DIAL_ENV
    from ffi.common import ffi_loader

    assert (ffi_loader._CUDA_TARGET_SYMBOLS["lorrax_cufft_conv_kpair"]
            == "CufftConvKPairCudaFfi")
    assert "LORRAX_CONV_KPAIR_FFI" in FFI_DIAL_ENV
    assert "LORRAX_CONV_KPAIR_FFI" in RANK_FINGERPRINT_ENV
    cmake = (Path(__file__).resolve().parents[1]
             / "src/ffi/cpp/CMakeLists.txt").read_text()
    assert "add_library(build_conv_kpair_cuda OBJECT" in cmake
    assert "$<TARGET_OBJECTS:build_conv_kpair_cuda>" in cmake


def test_native_contract_names_fallback_and_rejects_aliasing():
    source = (Path(__file__).resolve().parents[1]
              / "src/ffi/cpp/cufft/conv_kpair_cuda_ffi.cc").read_text()
    assert "fallback=XLA reference chain" in source
    assert "U overlaps A or B; no in-place form exists" in source
    assert "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES" in source
    assert "kAxisMax = 24" in source


def test_rchunk_gamma_attributes_are_captured_before_jit_trace():
    source = (Path(__file__).resolve().parents[1]
              / "src/isdf/core.py").read_text()
    assert (
        "gamma_static = None if is_charge else "
        "_gamma_perm_phase_mu(vertex_mu_L)" in source)
    assert "gamma_mu = gamma_static" in source
    assert "int(vertex_mu_L)," in source
