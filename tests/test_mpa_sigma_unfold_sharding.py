"""The post-sweep Sigma(omega) unfold keeps the band axes sharded and matches the old path.

Run with 4 emulated CPU devices:
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu pytest tests/test_mpa_sigma_unfold_sharding.py
"""
from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from gw.mpa.sigma import _unfold_sigma_cube  # noqa: E402
from symmetry_maps import unfold_file_wedge_band_operator  # noqa: E402


class _Sym(SimpleNamespace):
    """A symmetry-table stand-in that hashes by identity, like ``SymMaps``:
    the unfold executable is cached per table (``_unfold_sigma_cube_fn``)."""
    __hash__ = object.__hash__


@pytest.mark.mesh(4)
@pytest.mark.parametrize("bracketed", [False, True])
def test_unfold_sigma_cube_is_sharded_and_matches_unpinned(bracketed):
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    nirr, nk, nss, nw, nb = 3, 8, 2, 5, 6
    rng = np.random.default_rng(7)
    sym = _Sym(
        irr_idx_k=rng.integers(0, nirr, nk).astype(np.int32),
        sym_idx_k=rng.integers(0, 2 * nss, nk).astype(np.int32),
        sym_mats_k=np.zeros((2 * nss, 3, 3)),
        nk_red=nirr)
    lead = (3, nw) if bracketed else (nw,)
    spec = P(*([None] * (len(lead) + 1)), "x", "y")
    sharding = NamedSharding(mesh, spec)
    host = (rng.standard_normal(lead + (nirr, nb, nb))
            + 1j * rng.standard_normal(lead + (nirr, nb, nb)))
    sigma = jax.device_put(host, sharding)
    k_axis = len(lead)

    got = _unfold_sigma_cube(sigma, sym, k_axis=k_axis, sharding=sharding)
    old = jax.jit(lambda v: jnp.moveaxis(unfold_file_wedge_band_operator(
        sym, jnp.moveaxis(v, k_axis, 0), trs_rule="transpose"), 0, k_axis))(sigma)

    assert got.shape == lead + (nk, nb, nb)
    assert got.sharding.spec == spec
    np.testing.assert_array_equal(np.asarray(got), np.asarray(old))
