"""Resident (in-memory) shared-pole bank: the deferred finite check of
``write_slab(require_finite=True)``.

The file route checks the writer's host staging bytes; a resident write has no
host copy, so each checked write ANDs a device flag and the flags are read once,
where the masks publish (``write_attr("header_json")``).  A non-finite field
refuses there, before the header (and with it the masks) changes.
"""
import numpy as np
import pytest

import runtime  # noqa: F401  (runtime before jax: the repo's bootstrap)
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from file_io import shared_pole_store as sps


def _mesh():
    devs = np.asarray(jax.devices())
    n = 4 if devs.size >= 4 else 1
    px = 2 if n == 4 else 1
    return Mesh(devs[:n].reshape(px, n // px), ("x", "y"))


def _field(mesh, value, shape=(2, 4, 4)):
    host = np.full(shape, 1.0 + 0.5j, np.complex128)
    host.flat[3] = value
    return jax.device_put(host, NamedSharding(mesh, P(None, "x", "y")))


@pytest.mark.parametrize("memory_kind", ["device", "host"])
def test_resident_require_finite_is_deferred_to_the_mask_publish(memory_kind, tmp_path):
    mesh = _mesh()
    bank = sps.ResidentBankPayload(mesh, carrier=4, label="t", memory_kind=memory_kind,
                                   root=str(tmp_path))
    bank.create_dataset("M0", shape=(2, 4, 4), dtype=np.complex128)
    bank.create_dataset("M2", shape=(2, 4, 4), dtype=np.complex128)
    bank.header_json = "{}"
    # Finite writes publish.
    bank.write_slab("M0", _field(mesh, 2.0), offset=(0, 0, 0), require_finite=True)
    bank.write_attr("header_json", np.bytes_(b'{"masks": 1}'))
    assert bank.header_json == '{"masks": 1}'
    # A NaN write returns (no sync); the publish refuses it, by field, and the
    # header (the masks) does not change.
    bank.write_slab("M0", _field(mesh, 3.0), offset=(0, 0, 0), require_finite=True)
    bank.write_slab("M2", _field(mesh, np.nan), offset=(0, 0, 0), require_finite=True)
    with pytest.raises(Exception, match="scratch M2 contains nonfinite"):
        bank.write_attr("header_json", np.bytes_(b'{"masks": 2}'))
    assert bank.header_json == '{"masks": 1}'
    # The verdict was consumed: the next transaction starts clean.
    bank.write_attr("header_json", np.bytes_(b'{"masks": 3}'))
    assert bank.header_json == '{"masks": 3}'
    bank.release()


def test_resident_unchecked_write_is_not_checked():
    mesh = _mesh()
    bank = sps.ResidentBankPayload(mesh, carrier=4, label="t")
    bank.create_dataset("M0", shape=(2, 4, 4), dtype=np.complex128)
    bank.write_slab("M0", _field(mesh, np.inf), offset=(0, 0, 0))
    bank.write_attr("header_json", np.bytes_(b"{}"))      # no require_finite: no verdict
    assert bank.header_json == "{}"


def test_store_writer_passes_require_finite_to_both_payloads():
    """The bank writer's call (``_write_bank_payload``) is accepted by both
    handles it can be given: SlabIO and the resident payload."""
    import inspect
    from file_io.slab_io import SlabIO
    for handle in (SlabIO, sps.ResidentBankPayload):
        assert "require_finite" in inspect.signature(handle.write_slab).parameters, handle
