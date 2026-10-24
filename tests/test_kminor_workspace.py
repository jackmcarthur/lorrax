"""Exact native k-minor allocation pricing, including compiled large-grid HLO.

The CPU FFI stub is compiled but never executed. It supplies real typed-FFI
HLO without allocating the multi-GiB test tensors or requiring a GPU. Its
size query is the actual provider header, built as the ROCm leg.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from ffi import fft
from ffi.common import ffi_loader
from runtime import aot_memory

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def provider(tmp_path_factory):
    directory = tmp_path_factory.mktemp("kminor_query")
    root = Path(__file__).resolve().parents[1]
    source = directory / "query.cc"
    source.write_text(r'''
#include <xla/ffi/api/ffi.h>
#include "common/c_abi.h"
#include "kdft/kdft_kminor_workspace.h"
namespace ffi = xla::ffi;
extern "C" std::int64_t LRX_C_ENTRY(lrx_kminor_workspace_bytes)(
 int nx,int ny,int nz,int mode,int out,std::int64_t rows,std::int64_t m,int cb) {
 return kdft::kminor_workspace_bytes(nx,ny,nz,mode,out,rows,m,cb);
}
ffi::Error stub(ffi::AnyBuffer, ffi::AnyBuffer, ffi::Result<ffi::AnyBuffer>,
 int64_t,int64_t,int64_t,int64_t) { return ffi::Error::Success(); }
XLA_FFI_DEFINE_HANDLER_SYMBOL(KminorWorkspaceStub, stub,
 ffi::Ffi::Bind().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
 .Attr<int64_t>("nkx").Attr<int64_t>("nky").Attr<int64_t>("nkz").Attr<int64_t>("out_layout"));
''')
    library = directory / "query.so"
    subprocess.run(["c++", "-shared", "-fPIC", "-std=c++17", "-O2",
        "-DLORRAX_FFI_ROCM=1", "-I" + jax.ffi.include_dir(),
        "-I" + str(root / "src/ffi/cpp"), str(source), "-o", str(library)],
        check=True, capture_output=True)
    lib = ctypes.CDLL(str(library))
    ffi_loader._bind_c_abi(lib, "ROCM")
    jax.ffi.register_ffi_target("lorrax_kdft_kconv_kminor",
        jax.ffi.pycapsule(lib.KminorWorkspaceStub), platform="cpu")
    return lib


def _compiled(shape, grid, *, layout=1, dtype=np.complex128):
    out = fft.kconv_kminor_out_shape(shape, layout)

    def call(x, k):
        return jax.ffi.ffi_call("lorrax_kdft_kconv_kminor",
            jax.ShapeDtypeStruct(out, x.dtype))(x, k, nkx=np.int64(grid[0]),
            nky=np.int64(grid[1]), nkz=np.int64(grid[2]), out_layout=np.int64(layout))

    return jax.jit(call).lower(jax.ShapeDtypeStruct(shape, dtype),
        jax.ShapeDtypeStruct((shape[1], shape[2], shape[-1]), dtype)).compile()


def test_exact_plan_query_and_refusals(provider, monkeypatch):
    monkeypatch.setattr(ffi_loader, "get_lib", lambda platform: provider)
    # Small single-pass, both split output layouts, both actual native modes,
    # both precisions and differing store strides: use the provider's plan.
    for dtype in (np.complex64, np.complex128):
        for rows in ((1, 16, 16, 2, 2), (3, 15, 17, 2, 2)):
            for grid in ((3, 3, 1), (20, 20, 20)):
                shape = (*rows, int(np.prod(grid)))
                for layout in (0, 1):
                    expected = (int(np.prod(shape))*np.dtype(dtype).itemsize
                                if grid == (20, 20, 20) and layout == 1 else 0)
                    assert fft.kminor_workspace_bytes(grid, shape, dtype,
                        out_layout=layout) == expected
                assert fft.kminor_workspace_bytes(grid, shape, dtype, mode=5,
                                                  out_layout=0) == 0
    assert fft.kminor_workspace_bytes((20, 20, 20), (0, 3, 2, 2, 2, 8000),
                                     np.complex128) == 0
    for mode in (0, 1):
        with pytest.raises(ValueError, match="provider refused"):
            fft.kminor_workspace_bytes((3, 3, 1), (1, 2, 3, 1, 1, 9),
                                       np.complex128, mode=mode)
    with pytest.raises(ValueError, match="provider refused"):
        fft.kminor_workspace_bytes((129, 1, 1), (1, 2, 3, 1, 1, 129), np.complex128)
    with pytest.raises(ValueError, match="provider refused"):
        fft.kminor_workspace_bytes((20, 20, 20), (2**30, 1, 1, 1, 1, 8000), np.complex128)
    with pytest.raises(ValueError, match="invalid local shape"):
        fft.kminor_workspace_bytes((3, 3, 1), (1, 2, 3, 1, 1, 8), np.complex128)
    monkeypatch.setattr(ffi_loader, "get_lib", lambda platform: SimpleNamespace())
    with pytest.raises(RuntimeError, match="rebuild.*new build directory"):
        fft.kminor_workspace_bytes((20, 20, 20), (1, 2, 3, 1, 1, 8000), np.complex128)


def test_compiled_local_hlo_and_large_abstract_batch(provider, monkeypatch):
    monkeypatch.setattr(ffi_loader, "get_lib", lambda platform: provider)
    for shape, dtype in (((1, 16, 16, 2, 2, 8000), np.complex128),
                         ((16, 32, 48, 2, 2, 8000), np.complex128),
                         ((2, 15, 17, 2, 2, 8000), np.complex64)):
        executable = _compiled(shape, (20, 20, 20), dtype=dtype)
        expected = int(np.prod(shape))*np.dtype(dtype).itemsize
        price = aot_memory.aot_kernel_peak_bytes(executable, platform="cpu")
        assert price.native_scratch == expected
        assert price.total == price.compiled_peak + expected
        assert price.resident_increment == executable.memory_analysis().output_size_in_bytes + expected
        # Repeated calls in a serialized executable need the maximum scratch,
        # not another allocation per call or a whole global-shape estimate.
        assert aot_memory.native_fft_workspace_bytes(executable.as_text()*2) == expected
        for bad in (executable.as_text().replace("out_layout = 1 : i64", "out_layout = ?"),
                    executable.as_text().replace("c64[", "f64[").replace("c128[", "f64[")):
            with pytest.raises(aot_memory.NativeFftQueryError, match="cannot parse"):
                aot_memory.native_fft_workspace_bytes(bad)
    for layout in (0, 1):
        ex = _compiled((1, 2, 3, 1, 1, 9), (3, 3, 1), layout=layout)
        assert aot_memory.aot_kernel_peak_bytes(ex, platform="cpu").native_scratch == 0
    ex = _compiled((1, 16, 16, 2, 2, 8000), (20, 20, 20), layout=0)
    assert aot_memory.aot_kernel_peak_bytes(ex, platform="cpu").native_scratch == 0
    # CUDA mathdx and CPU/XLA calls have no native k-minor arena allocation.
    assert aot_memory.native_fft_workspace_bytes('custom_call_target="lorrax_mathdx_kconv_kminor"') == 0
    assert aot_memory.native_fft_workspace_bytes('ROOT %x = c128[8] parameter(0)') == 0
    fft_hlo = '%fft = c128[2,20,20,20]{3,2,1,0} fft(%x), fft_type=FFT, fft_length={20,20,20}'
    ex = _compiled((1, 16, 16, 2, 2, 8000), (20, 20, 20))
    fake = SimpleNamespace(memory_analysis=lambda: SimpleNamespace(
        temp_size_in_bytes=100, argument_size_in_bytes=200, output_size_in_bytes=300,
        alias_size_in_bytes=0), as_text=lambda: ex.as_text() + fft_hlo)
    monkeypatch.setattr(aot_memory, "_fft_plan_api", lambda platform: "cufft")
    monkeypatch.setattr(aot_memory, "query_cufft_workspace_bytes", lambda specs, api: 200_000_000)
    price = aot_memory.aot_kernel_peak_bytes(fake, platform="cuda")
    assert price.cufft_scratch == 200_000_000
    assert price.native_scratch == 131_072_000
    assert price.total == 600 + 200_000_000
