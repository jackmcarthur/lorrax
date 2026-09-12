"""Workspace queries follow the GEMM provider and route, independently of eigh."""
from types import SimpleNamespace as NS
import importlib
import numpy as np
import pytest


@pytest.mark.parametrize('route,expected_context', [('auto', 701), ('batch_reshard', None)])
def test_eager_gemm_query_uses_its_own_context(monkeypatch, route, expected_context):
    ws = importlib.import_module('distrib_la.workspace')
    mm = importlib.import_module('distrib_la.matmul')
    ctx = importlib.import_module('distrib_la._cusolvermp')
    mesh = NS(shape={'x':2,'y':2}, devices=np.array([NS(platform='gpu')]*4))
    calls = []
    monkeypatch.setattr(mm, 'resolve_matmul_backend',
        lambda requested, mesh, **kwargs: 'cublasmp')
    def context(mesh, **kwargs):
        assert kwargs == {'col_major':False}
        return 701
    monkeypatch.setattr(ctx, 'get_or_init_context', context)
    monkeypatch.setattr(ws, '_vendor_query',
        lambda handle, op, sizes, dtype: (calls.append(handle) or 123, 0))
    monkeypatch.setattr(ws, '_local_gemm_temp', lambda *args: 456)
    size = ws.matmul_workspace_bytes_per_rank(mesh, ((4,8,4),(4,4,6)),
                                              np.complex128, batched_route=route)
    assert size == (123 if route == 'auto' else 456)
    assert calls == ([expected_context] if expected_context is not None else [])
