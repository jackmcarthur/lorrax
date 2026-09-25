"""One mesh shape on CPU and CUDA needs one context in each native library."""

from types import SimpleNamespace

import jax
import numpy as np

from distrib_la import _slate
from distrib_la.loader import context_key as hash_context


def _mesh(platform):
    device = SimpleNamespace(platform=platform)
    return SimpleNamespace(
        axis_names=("x", "y"), shape={"x": 2, "y": 2},
        devices=np.full((2, 2), device, dtype=object))


def test_platforms_bind_and_destroy_their_own_contexts(monkeypatch):
    monkeypatch.setattr(jax, "process_count", lambda: 4)
    monkeypatch.setattr(_slate, "_CACHE", {})
    monkeypatch.setattr(_slate, "_SUBROW_CACHE", {})
    monkeypatch.setattr(_slate, "_KEYS", {})
    made = []
    bound = []
    destroyed = []
    next_handle = iter(range(101, 105))

    def make(mesh, kind):
        handle = next(next_handle)
        made.append((kind, mesh.devices.flat[0].platform, handle))
        return handle

    def bind(platform, config, handle):
        bound.append((platform, config, handle))
        return hash_context(config)

    monkeypatch.setattr(_slate, "_make_ctx", lambda mesh: make(mesh, "world"))
    monkeypatch.setattr(_slate, "_make_subrow_ctx", lambda mesh: make(mesh, "subrow"))
    monkeypatch.setattr(_slate.loader, "bind_context", bind)
    monkeypatch.setattr(_slate.loader, "unbind_context",
                        lambda platform, key: destroyed.append((platform, key)))
    monkeypatch.setattr(_slate.loader, "destroy_slate_context",
                        lambda handle, *, platform: destroyed.append((platform, handle)))

    cpu, gpu = _mesh("cpu"), _mesh("gpu")
    for kind, key_fn, get_fn in (
            ("world", _slate.context_key, _slate.get_or_init_context),
            ("subrow", _slate._subrow_context_key, _slate._get_or_init_subrow_context)):
        host_handle = get_fn(cpu)
        cuda_handle = get_fn(gpu)
        assert host_handle != cuda_handle, kind
        assert get_fn(cpu) == host_handle
        assert get_fn(gpu) == cuda_handle
        assert key_fn(cpu) != key_fn(gpu)

    assert len(made) == len(bound) == 4
    assert {platform for platform, _, _ in bound} == {"cpu", "CUDA"}
    _slate._atexit_teardown()
    assert len(destroyed) == 8
    assert _slate._CACHE == _slate._SUBROW_CACHE == _slate._KEYS == {}
