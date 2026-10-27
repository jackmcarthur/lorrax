"""The native EIGH query/explicit arena contract without a GPU provider."""
import ctypes
from types import SimpleNamespace

import runtime  # noqa: F401
import jax
import numpy as np
import pytest


def test_native_query_sizes_the_actual_result_for_both_drivers_and_dtypes(monkeypatch):
    from distrib_la import local_eigh as owner, loader
    exact, requests, calls = 1_234_432, [], []

    def query(kind, n, batch, lower, jacobi, tile, out, error, capacity):
        requests.append((kind, n, batch, lower, jacobi, tile))
        ctypes.cast(out, ctypes.POINTER(ctypes.c_uint64))[0] = exact
        return 0

    monkeypatch.setattr(loader, 'get_lib', lambda _leg: SimpleNamespace(
        lrx_rocsolver_eigh_workspace_bytes=query))
    results = {}

    def ffi_call(target, outputs, **kwargs):
        def invoke(a, **attrs):
            calls.append((target, outputs, a, attrs))
            results['v'] = jax.numpy.zeros(a.shape, a.dtype)
            results['w'] = jax.numpy.zeros(a.shape[:-1], np.float64)
            results['info'] = jax.numpy.zeros(a.shape[:-2], np.int32)
            return results['v'], results['w'], results['info'], object()
        return invoke

    monkeypatch.setattr(jax.ffi, 'ffi_call', ffi_call)
    for dtype, kind, driver in ((np.float64, 1, 'syevd'), (np.complex128, 3, 'heevd'),
                                (np.float64, 1, 'syevdj'), (np.complex128, 3, 'heevdj')):
        a = jax.ShapeDtypeStruct((2, 3, 40, 40), dtype)
        w, v, info = owner._batched_call('lorrax_rocsolver_batched_eigh', a, driver, 1 << 30)
        target, outputs, operand, attrs = calls[-1]
        assert operand is a and target == 'lorrax_rocsolver_batched_eigh'
        assert len(outputs) == 4 and outputs[-1].shape == (exact,) and outputs[-1].dtype == np.uint8
        assert outputs[0].shape == a.shape and outputs[0].dtype == np.dtype(dtype)
        assert outputs[1].shape == (2,3,40) and outputs[2].shape == (2,3)
        assert requests[-1] == (kind, 40, 6, 1, int(driver.endswith('j')), 1 << 30)
        assert attrs == dict(lower=1, jacobi=int(driver.endswith('j')), tile_bytes=1 << 30)
        assert w is results['w'] and info is results['info'] and v.shape == a.shape
    assert len(requests) == len(calls) == 4


def test_missing_or_failed_native_query_never_becomes_a_zero_guess(monkeypatch):
    from distrib_la import local_eigh as owner, loader
    monkeypatch.setattr(loader, 'get_lib', lambda _leg: SimpleNamespace())
    with pytest.raises(RuntimeError, match='rebuild the whole native pair'):
        owner._workspace_bytes((3,128,128), np.complex128, 'heevd', 1 << 30)

    def query(*args):
        ctypes.memmove(args[-2], b'actual size query failed\0', 25)
        return 1

    monkeypatch.setattr(loader, 'get_lib', lambda _leg: SimpleNamespace(
        lrx_rocsolver_eigh_workspace_bytes=query))
    with pytest.raises(RuntimeError, match='actual size query failed'):
        owner._workspace_bytes((3,128,128), np.complex128, 'heevd', 1 << 30)


def test_external_scratch_keeps_legacy_and_scalar_jacobi_allowances(monkeypatch):
    from distrib_la import local_eigh as owner, workspace
    from jax.sharding import Mesh
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1,1), ('x','y'))
    monkeypatch.setattr(workspace, 'mesh_platform', lambda _mesh: 'ROCM')
    workspace.local_eigh_workspace.cache_clear()
    try:
        for dtype in (np.float64, np.complex128):
            win = owner.BATCHED_EIGH_ROUTE['ROCM'].windows[np.dtype(dtype).name]
            for share in (1,3):
                for n in (72,128,3121):
                    row = owner.batched_eigh_row((share,n,n), dtype, 'ROCM')
                    legacy = win.workspace_tiles*n*n*np.dtype(dtype).itemsize+n*8
                    assert workspace.local_eigh_workspace(mesh, share, n, dtype, batched=True) == (
                        0 if row is not None else legacy)
                    assert workspace.local_eigh_workspace(mesh, share, n, dtype) == (
                        0 if row is not None and dtype == np.float64 else legacy)
        # Native arena sizing is a separate query, never an external allowance.
        monkeypatch.setattr(owner, '_workspace_bytes', lambda *_args: 123_456)
        assert owner.batched_workspace_bytes((3,128,128), np.complex128, 'ROCM') == 123_456
        for leg in ('cpu','CUDA'):
            assert owner.batched_workspace_bytes((3,128,128), np.complex128, leg) == 0
    finally:
        workspace.local_eigh_workspace.cache_clear()
