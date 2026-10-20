"""The shared-pole constructor receipt reads every native query a face-route capacity records
(run30k -f: a key of another shape crashed spole.receipts after the constructor, on all ranks)."""
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

pytestmark = pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices")


def test_face_capacity_native_queries_are_receipt_rows():
    from gw.shared_pole_capacity import ConstructorCapacity
    from gw.shared_pole_recipe import native_query_rows
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    capacity = SimpleNamespace(batch_width=2, native_queries={}, _mesh_xy=mesh, execution="face",
                               _resolution=SimpleNamespace(eigh_backend="auto"))
    for extent in (64, 32):
        ConstructorCapacity.eigh_bytes(capacity, extent)
    rows = native_query_rows(capacity.native_queries)
    assert [r["op"] for r in rows] == ["eigh", "eigh"]
    assert [r["shapes"] for r in rows] == [((2, 64, 64),), ((2, 32, 32),)]
    assert all(r["bytes_per_rank"] >= 0 for r in rows)
