"""Opening a sharded HDF5 file must name its mode: a default write mode truncated reads.

``SlabIO(path, mesh=...)`` and ``ffi.io.open_file(path, mesh=...)`` used to
default to ``mode="w"``, so a read that forgot the keyword took the replace
path (rank-0 unlink + H5Fcreate TRUNC) and destroyed the file it meant to read
(KNOWN_LORRAX_ISSUES 2026-09-15 TRREF, SlabIO API row). Binding is checked
without opening anything, so the test needs no native library.
"""
import inspect

import pytest


def _binds(fn, *args, **kwargs):
    inspect.signature(fn).bind(*args, **kwargs)


def test_slab_io_refuses_a_missing_mode():
    from file_io.slab_io import SlabIO

    parameter = inspect.signature(SlabIO).parameters["mode"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="mode"):
        _binds(SlabIO, "file.h5", mesh=object())
    for mode in ("r", "a", "w"):
        _binds(SlabIO, "file.h5", mode=mode, mesh=object())


def test_open_file_refuses_a_missing_mode():
    from ffi import io as ffi_io

    parameter = inspect.signature(ffi_io.open_file).parameters["mode"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="mode"):
        _binds(ffi_io.open_file, "file.h5", mesh=object())
    _binds(ffi_io.open_file, "file.h5", mode="r", mesh=object())


@pytest.mark.parametrize(
    "module,name",
    (("file_io._slab_io_ffi", "_FfiBackend"),
     ("file_io._slab_io_serial", "_SerialBackend")),
)
def test_each_transport_backend_refuses_a_missing_mode(module, name):
    """The same door one layer down.

    ``SlabIO`` always passes ``mode=``, so these defaults were unreachable
    — but an unreachable truncating default is the defect waiting for its
    next direct caller, and both backends are constructed directly by
    ``tests/test_slab_io_emulated_mesh.py``.  Binding only; nothing opens.
    """
    import importlib

    backend = getattr(importlib.import_module(module), name)
    parameter = inspect.signature(backend).parameters["mode"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="mode"):
        _binds(backend, "file.h5", mesh=object())
    for mode in ("r", "a", "w"):
        _binds(backend, "file.h5", mesh=object(), mode=mode)
