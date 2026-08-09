"""The sharded Sigma(omega) cube must survive the k_irr star-spread stamp.

RED TWIN.  ``sigma_omega_layout = sharded`` (the layout that exists because
``replicated`` HANGS at nb >= 1024) died at P=4 the first time anyone ran it
through ``extract_and_stamp_k_irr``: the star-spread statistic did a bare
``np.asarray`` on one omega slice of a GLOBALLY sharded ``jax.Array``, and a
slice of a sharded array is still sharded, so every rank raised

    RuntimeError: Fetching value for `jax.Array` that spans non-addressable
    (non process local) devices is not possible.

The k_irr extraction landed 2026-08-08 and the sharded layout 2026-07-28;
the two had never met, because nothing in the suite runs the sharded layout
at P > 1.  Measured on MoS2 6x6 (mu=1496, nb=200) at P=4:
``_reports/ab_shard.log`` is the crash, ``_reports/ab_shard2.log`` the same
deck green after the fix, with eqp0/eqp1/eqp_g0w0/sigma_diag byte-identical
to the replicated arm.

These cells are host-only stubs on purpose: the defect is a
*host-transfer policy* question ("is this operand fully addressable?"), not
a device question, so it is checkable without a multi-process job -- which
is exactly why it went unnoticed.
"""
from __future__ import annotations

import numpy as np
import pytest

from file_io.sigma_output import _slab_to_host


class _NotAddressable:
    """A stand-in for a globally sharded jax.Array on a multi-process run."""

    is_fully_addressable = False

    def __init__(self, value):
        self._value = np.asarray(value)
        self.shape = self._value.shape

    def __array__(self, dtype=None, **kwds):
        # THE FAILING CASE, verbatim in spirit: jax raises here rather than
        # silently returning this process own shard.
        raise RuntimeError(
            "Fetching value for `jax.Array` that spans non-addressable "
            "(non process local) devices is not possible.")


def test_bare_np_asarray_on_a_sharded_slab_raises():
    """The red: this is precisely what the old code did, and it dies."""
    stub = _NotAddressable(np.arange(6.0).reshape(2, 3))
    with pytest.raises(RuntimeError, match="non-addressable"):
        np.asarray(stub)


def test_slab_to_host_gathers_a_non_addressable_slab(monkeypatch):
    """The green: _slab_to_host reconstructs it through process_allgather."""
    payload = np.arange(6.0).reshape(2, 3)
    stub = _NotAddressable(payload)
    seen = {}

    def _fake_allgather(a, tiled=False):
        seen["tiled"] = tiled
        seen["arg_is_stub"] = a is stub
        return payload

    from jax.experimental import multihost_utils
    monkeypatch.setattr(multihost_utils, "process_allgather", _fake_allgather)

    out = _slab_to_host(stub)
    np.testing.assert_array_equal(out, payload)
    # tiled=True is load-bearing: process_allgather REFUSES a globally
    # non-fully-addressable operand with tiled=False.
    assert seen["tiled"] is True
    assert seen["arg_is_stub"] is True


def test_slab_to_host_leaves_the_replicated_path_alone(monkeypatch):
    """The default layout must not acquire a collective it never had."""
    from jax.experimental import multihost_utils

    def _explode(*a, **k):                       # pragma: no cover
        raise AssertionError(
            "_slab_to_host reached process_allgather for a fully "
            "addressable operand -- the replicated layout just grew a "
            "P-linear collective it did not have.")

    monkeypatch.setattr(multihost_utils, "process_allgather", _explode)
    payload = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(_slab_to_host(payload), payload)
