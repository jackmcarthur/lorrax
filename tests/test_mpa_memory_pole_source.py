"""GN/HL in-memory pole source: the executor's batches are bit-identical to the store round trip.

The Sigma executor consumes a pole source only through ``read(...)`` and the
ledger's ``n_p`` / ``ordered_residues``, so equal batches mean equal Sigma.
Run with 4 emulated CPU devices (the padded mu extent needs a 2x2 mesh):
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu pytest tests/test_mpa_memory_pole_source.py
"""
import os
import tempfile

import numpy as np
import pytest
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

pytest.importorskip("h5py")

import common.collectives as collectives  # noqa: E402
import file_io.slab_io as slab_io  # noqa: E402
from file_io import mpa_store  # noqa: E402
from file_io.restart_bundle import PoleReader  # noqa: E402
from gw.mpa.sigma import MemoryPoleSource  # noqa: E402
from tests._mpa_test_geometry import HostSlabIO  # noqa: E402


class _PaddedReadSlabIO(HostSlabIO):
    """HostSlabIO whose reads zero-fill past the dataset extent, as SlabIO's do."""

    def read_slab(self, name, *, shape, offset, valid_shape=None, **kw):
        return super().read_slab(
            name, shape=shape, offset=offset,
            valid_shape=self.file[name].shape if valid_shape is None else valid_shape,
            **kw)


@pytest.mark.mesh(4)
@pytest.mark.parametrize("ordered", [True, False])
def test_memory_source_batches_equal_the_store_round_trip(monkeypatch, ordered):
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    monkeypatch.setattr(slab_io, "SlabIO", _PaddedReadSlabIO)
    monkeypatch.setattr(collectives, "process_rank", lambda: 0)
    monkeypatch.setattr(collectives, "barrier", lambda _name: False)
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    n_mu, n_pad, n_q = 3, 4, 2           # padded_mu_extent(3, 2x2 mesh) = 4
    rng = np.random.default_rng(3)
    Omega = rng.uniform(0.5, 2.0, (1, n_q, n_pad, n_pad)) - 1e-3j
    B = rng.standard_normal((1, n_q, n_pad, n_pad)) + 1j * rng.standard_normal((1, n_q, n_pad, n_pad))
    D = 0.1 * (rng.standard_normal((1, n_q, n_pad, n_pad)) + 0j) if ordered else None
    Omega[0, 0, 0, 1] = 0.0              # a dormant pole ...
    B[0, 0, 0, 1] = 0.0                  # ... is legal in both paths
    if D is not None:
        D[0, 0, 0, 1] = 0.0
    # Nonzero padding: the store drops it and reads it back as zero.
    Omega[..., n_mu:, :] = Omega[..., :, n_mu:] = 2.0
    B[..., n_mu:, :] = B[..., :, n_mu:] = 7.0
    put = lambda x: None if x is None else jax.device_put(x, sharding)  # noqa: E731

    path = os.path.join(tempfile.mkdtemp(prefix="memory_poles_"), "fit.h5")
    mpa_store.write_complete_pole_store_collective(
        path, put(Omega), put(B), B_odd_p=put(D), mesh_xy=mesh,
        n_mu_logical=n_mu, energy_unit="Ry",
        provenance={"fit_protocol": "two_point_ppm", "screening_diagrams": "w_rpa"},
        certification={"condition_max_allowed": 1.0, "backward_error_max_allowed": 1.0})
    with PoleReader(path, mesh_xy=mesh) as reader:
        want = reader.read(slice(0, 1), unfold=True, return_sharded=True,
                           to_unit="Ry", include_odd=True)
        want_ledger = reader.ledger

    source = MemoryPoleSource(
        put(Omega), put(B), put(D), n_mu_logical=n_mu, mesh_xy=mesh,
        provenance={"screening_diagrams": "w_rpa"})
    got = source.read(slice(0, 1), unfold=True, return_sharded=True,
                      to_unit="Ry", include_odd=True)

    assert source.ledger["n_p"] == want_ledger["n_p"] == 1
    assert source.ledger["ordered_residues"] is want_ledger["ordered_residues"] is ordered
    for g, w in zip(got, want):
        if w is None:
            assert g is None
            continue
        assert g.sharding.spec == P(None, None, "x", "y")
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def test_memory_source_refuses_a_live_noncausal_pole():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    Omega = np.full((1, 1, 2, 2), 0.8 + 0.1j)          # Im Omega > 0
    B = np.ones((1, 1, 2, 2), np.complex128)
    with pytest.raises(ValueError, match="live poles.*Im Omega > 0"):
        MemoryPoleSource(jax.numpy.asarray(Omega), jax.numpy.asarray(B),
                         n_mu_logical=2, mesh_xy=mesh, provenance={})
