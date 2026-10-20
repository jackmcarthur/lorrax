"""Metadata-only digest ownership: ephemeral faces must not be retained."""
import gc
from types import SimpleNamespace
import weakref

import numpy as np


def test_ephemeral_digest_preserves_checksum_without_retaining_face(monkeypatch):
    import gw.response_bank as owner

    class Face:
        shape = (64, 6000, 752)
        sharding = SimpleNamespace(mesh="actual-mesh")

    cache, calls = {}, []
    monkeypatch.setattr(owner, "_OPERATOR_DIGESTS", cache)

    def program(mesh, shape):
        def checksum(values):
            calls.append((mesh, shape))
            return np.uint64(123456789)
        return checksum

    monkeypatch.setattr(owner, "_operator_digest_program", program)
    face = Face()
    ref = weakref.ref(face)
    ephemeral = owner.operator_digest(face, None, cache=False)
    assert cache == {}
    assert calls == [("actual-mesh", face.shape)]
    cached = owner.operator_digest(face, None)
    assert ephemeral == cached
    assert owner.operator_digest(face, None) == cached
    assert len(calls) == 2  # The default static-object cache still hits.
    assert cache[id(face)][0] is face
    assert owner.operator_digest(face, None, cache=False) == cached
    assert len(calls) == 3  # Opt-out bypasses both lookup and retention.
    cache.clear()
    del face
    gc.collect()
    assert ref() is None


def test_many_ephemeral_snapshot_faces_leave_no_cached_device_owner(monkeypatch):
    import gw.response_bank as owner

    class Face:
        shape = (64, 6000, 752)
        sharding = SimpleNamespace(mesh="actual-mesh")

    cache = {}
    monkeypatch.setattr(owner, "_OPERATOR_DIGESTS", cache)
    monkeypatch.setattr(owner, "_operator_digest_program",
                        lambda mesh, shape: lambda values: np.uint64(987654321))
    refs, digests = [], []
    for _ in range(20):
        face = Face()
        refs.append(weakref.ref(face))
        digests.append(owner.operator_digest(face, None, cache=False))
    del face
    gc.collect()
    assert not cache
    assert all(ref() is None for ref in refs)
    assert len(set(digests)) == 1
