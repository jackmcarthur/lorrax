"""BSE V(q) on the mesh's centroid carrier: a logical N_mu = 4k+1 that does not tile the 2x2 mesh is
padded to padded_mu_extent (inert zero centroids) so every (mu, nu) face tiles; the logical block
equals the unpadded result and the stencil stays on the face. P = 4 host devices."""
import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

NQ, NMU, NGK, NG = 4, 33, 40, 12          # NMU = 4*8 + 1


@pytest.fixture
def mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def test_zeta_tiles_pad_to_the_carrier(mesh):
    from file_io.restart_bundle import _ZetaGTiles
    zg = np.arange(2 * NMU * NGK, dtype=complex).reshape(2, NMU, NGK)
    loader = types.SimpleNamespace(zeta_layout="G_flat", n_q_on_disk=2, n_rmu_disk=NMU, n_G_sph_disk=NGK,
                                   read_zeta_G_local=lambda key: zg[key])
    tiles = _ZetaGTiles(loader, path="x", distributed=False, mu_carrier=36)
    assert tiles.shape == (2, 36, NGK)
    np.testing.assert_array_equal(tiles[1][:NMU], zg[1]); assert not np.any(tiles[1][NMU:])
    np.testing.assert_array_equal(tiles[0:2][:, :NMU], zg)
    slab = tiles.read_q_slab(0, 2, sharding=NamedSharding(mesh, P(None, "x", "y")))
    np.testing.assert_array_equal(np.asarray(slab)[:, :NMU], zg)


def test_padded_carrier_matches_the_logical_prep(mesh, monkeypatch):
    from bse import vq_interp as vq
    from common import gpu_utils
    from runtime.padding import padded_mu_extent
    carrier = padded_mu_extent(NMU, mesh)
    assert carrier == 36
    monkeypatch.setattr(gpu_utils, "shared_device_room", lambda stage: 1 << 34)
    rng = np.random.default_rng(13)
    zg = rng.standard_normal((NQ, NMU, NGK)) + 1j * rng.standard_normal((NQ, NMU, NGK))
    a = rng.standard_normal((NQ, NMU, 2 * NMU)) + 1j * rng.standard_normal((NQ, NMU, 2 * NMU))
    c = a @ np.conj(np.swapaxes(a, 1, 2))
    qfr = np.c_[rng.uniform(-.5, .5, (NQ, 2)), np.zeros(NQ)]
    rmu = rng.uniform(0, 1, (NMU, 3))
    gs = np.r_[rng.integers(-2, 3, (2, NG)), np.zeros((1, NG), int)]
    slots = rng.integers(-1, NGK, (NQ, NG))
    v = {None: rng.uniform(.5, 2, (NQ, NGK)), "slab_lr": rng.uniform(.1, .4, (NQ, NGK))}
    monkeypatch.setattr(vq, "lr_gset", lambda zx, alpha, degrees=None: gs)
    monkeypatch.setattr(vq, "v_sphere_padded", lambda zx, kind=None, alpha=None: v[kind])
    monkeypatch.setattr(vq, "_sphere_slot", lambda zx, q, GS: slots[q])
    monkeypatch.setattr(vq, "v_slab_on_set", lambda zx, q, GS, kind=None, alpha=None: np.ones(NG))

    def zx_at(n, m):
        pad = lambda x, axes: np.pad(x, [(0, n - NMU) if i in axes else (0, 0) for i in range(x.ndim)])
        z = pad(zg, (1,))
        reader = types.SimpleNamespace(read_q_slab=lambda s, nb, sharding: jax.device_put(z[s:s + nb], sharding))
        return dict(nq=NQ, n_mu=n, ngkmax=NGK, ZG=reader, qfr=qfr, rmu_frac=pad(rmu, (0,)), zeta_cutoff=10.0), \
            pad(c, (1, 2))
    one = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    kw = dict(degrees={0: 1}, keep_host_mirrors=True, q_chunk=4)
    zx, cq = zx_at(NMU, one)
    logical = vq.prepare_coarse(zx, cq, one, **kw)
    zx, cq = zx_at(carrier, mesh)
    padded = vq.prepare_coarse(zx, jax.device_put(cq, NamedSharding(mesh, P(None, "x", "y"))), mesh, **kw)
    assert padded["V_SRc"].sharding.is_equivalent_to(NamedSharding(mesh, P(None, "x", "y")), 3)
    for key, block in (("S", np.s_[:, :NMU, :NMU]), ("V_SRc_np", np.s_[:, :NMU, :NMU]), ("Fch", np.s_[:, :NMU])):
        ref = logical[key]
        np.testing.assert_allclose(padded[key][block], ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max())
        rest = padded[key].copy(); rest[block] = 0
        assert np.abs(rest).max() <= 1e-12 * np.abs(ref).max()          # the pad centroids are inert
