"""CUDA handlers outside the k-convolution router refuse at startup, by name.

``ffi_loader.require_cuda_handlers`` (called by
``runtime.initialize_communicator_stack``) checks the contour accumulator and
the spin rotation symbols on a CUDA mesh; ``ffi.contour`` refuses
through ``probe_target``.  Each check is run against a library that lacks the
symbol (it must refuse, naming the symbol and the door) and against one that
has it (it must pass), so a check that can never fire cannot pass here.
"""
import types

import pytest


class _Lib:
    def __init__(self, symbols):
        for s in symbols:
            setattr(self, s, object())


def _cuda_mesh():
    dev = types.SimpleNamespace(platform="gpu")
    return types.SimpleNamespace(devices=types.SimpleNamespace(flat=[dev]))


def _cpu_mesh():
    dev = types.SimpleNamespace(platform="cpu")
    return types.SimpleNamespace(devices=types.SimpleNamespace(flat=[dev]))


def test_startup_check_refuses_each_missing_symbol_and_passes_a_complete_library(monkeypatch):
    from ffi.common import ffi_loader as L

    needed = {sym for sym, _door in L._CUDA_STARTUP_HANDLERS.values()}
    assert needed == {"ContourAccumulateFfi", "SpinRotateCentroidCudaFfi"}
    monkeypatch.setattr(L, "loaded_lib_path", lambda platform: "/fake/liblorrax_ffi.so")
    for missing in sorted(needed):
        monkeypatch.setattr(L, "get_lib", lambda platform, m=missing: _Lib(needed - {m}))
        with pytest.raises(RuntimeError, match=f"GATE ffi-handler: .* without {missing}"):
            L.require_cuda_handlers(_cuda_mesh())
    monkeypatch.setattr(L, "get_lib", lambda platform: _Lib(needed))
    L.require_cuda_handlers(_cuda_mesh())


def test_startup_check_is_a_no_op_off_cuda(monkeypatch):
    from ffi.common import ffi_loader as L

    def boom(platform):
        raise AssertionError("a cpu mesh must not open the CUDA library")
    monkeypatch.setattr(L, "get_lib", boom)
    L.require_cuda_handlers(_cpu_mesh())


def test_contour_door_refuses_an_unusable_target_and_names_the_probe(monkeypatch):
    import ffi.contour as C

    C._require.cache_clear()
    monkeypatch.setattr(C, "probe_target",
                        lambda target, platform: (False, f"loaded /x.so but it does not export {target}"))
    with pytest.raises(RuntimeError, match="GATE ffi-handler: got no usable lorrax_contour_accumulate"):
        C._require()
    C._require.cache_clear()
    monkeypatch.setattr(C, "probe_target", lambda target, platform: (True, "available"))
    C._require()
    C._require.cache_clear()


def test_contour_target_is_a_row_of_the_cuda_table():
    from ffi.common import ffi_loader as L
    import ffi.contour as C

    assert L._CUDA_TARGET_SYMBOLS[C.TARGET] == "ContourAccumulateFfi"
    assert L._CUDA_STARTUP_HANDLERS[C.TARGET][0] == "ContourAccumulateFfi"
