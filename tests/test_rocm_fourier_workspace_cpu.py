"""The ROCm plan query and explicit-result bridge, without a device provider."""
import ctypes
from types import SimpleNamespace

import runtime  # noqa: F401
import jax
import numpy as np
import pytest


def _attrs():
    return dict(n=(25, 25, 25), kin=(13, 13, 13), kout=(25, 25, 25),
                in_idx=np.tile(np.arange(-6, 7) % 25, 3),
                out_idx=np.tile(np.arange(25), 3), sup_in=(1, 1, 1),
                sup_out=(0, 0, 0), gemm=(1, 1, 1), scale=(1/25,) * 3,
                order=(2, 1, 0), sign=1)


def test_native_query_sizes_the_actual_explicit_result_and_preserves_other_legs(monkeypatch):
    from ffi import fft as owner
    from ffi.common import ffi_loader
    calls, queries = [], []
    # Nonzero native value deliberately differs from any closed-form guess.
    exact = 25_293_312

    def query(d, batch, *args):
        n, kin, kout, src, dst, sin, sout, gemm, scale, order, count, sign, out, error, cap = args
        assert d == 3 and batch == 128 and count == 3 and sign == 1
        assert [n[i] for i in range(d)] == [25] * 3
        assert [kin[i] for i in range(d)] == [13] * 3
        assert [kout[i] for i in range(d)] == [25] * 3
        assert [order[i] for i in range(count)] == [2, 1, 0]
        assert [src[i] for i in range(39)] == _attrs()['in_idx'].tolist()
        assert [dst[i] for i in range(75)] == _attrs()['out_idx'].tolist()
        np.testing.assert_array_equal([scale[i] for i in range(d)], [1/25] * 3)
        queries.append((d, batch))
        ctypes.cast(out, ctypes.POINTER(ctypes.c_uint64))[0] = exact
        return 0

    monkeypatch.setattr(ffi_loader, 'get_lib', lambda leg: SimpleNamespace(
        lrx_fourier_plan_workspace_bytes=query))
    result = object()

    def ffi_call(target, outputs):
        def invoke(x, **attrs):
            calls.append((target, outputs, x, attrs))
            return (result, object()) if isinstance(outputs, tuple) else result
        return invoke

    monkeypatch.setattr(jax.ffi, 'ffi_call', ffi_call)
    x = jax.ShapeDtypeStruct((128, 13, 13, 13), np.complex128)
    strided = {}
    for name, value in _attrs().items():
        if name == 'sign':
            strided[name] = value
            continue
        value = np.asarray(value)
        bank = np.full(value.size * 2, 9999, dtype=value.dtype)
        bank[::2] = value
        strided[name] = bank[::2]
    for attrs_in in (_attrs(), strided):
        assert owner.fourier_plan_ffi(x, **attrs_in, leg='ROCM') is result
    target, outputs, operand, attrs = calls[-1]
    assert target == 'lorrax_fourier_plan_rocm' and operand is x
    assert outputs[0].shape == (128, 25, 25, 25) and outputs[0].dtype == np.complex128
    assert outputs[1].shape == (exact,) and outputs[1].dtype == np.uint8
    assert set(attrs) == set(_attrs()) | {'cubin_dir'}
    # The CUDA target/signature/attributes retain their original contract.
    assert owner.fourier_plan_ffi(x, **_attrs(), leg='CUDA') is result
    assert calls[-1][0] == 'lorrax_fourier_plan_mathdx'
    assert calls[-1][1].shape == (128, 25, 25, 25)
    assert set(calls[-1][3]) == set(_attrs()) | {'cubin_dir', 'mathdx_root'}
    assert len(queries) == 2


def test_native_query_failure_and_missing_provider_never_guess_zero(monkeypatch):
    from ffi import fft as owner
    from ffi.common import ffi_loader
    attrs = {k: np.asarray(v, np.float64 if k == 'scale' else np.int64)
             if k != 'sign' else np.int64(v) for k, v in _attrs().items()}
    monkeypatch.setattr(ffi_loader, 'get_lib', lambda leg: SimpleNamespace())
    with pytest.raises(RuntimeError, match='rebuild the whole native pair'):
        owner._fourier_plan_workspace_bytes((128, 13, 13, 13), attrs)

    def query(*args):
        ctypes.memmove(args[-2], b'vendor size query refused\0', 26)
        return 1

    monkeypatch.setattr(ffi_loader, 'get_lib', lambda leg: SimpleNamespace(
        lrx_fourier_plan_workspace_bytes=query))
    with pytest.raises(RuntimeError, match='vendor size query refused'):
        owner._fourier_plan_workspace_bytes((128, 13, 13, 13), attrs)
