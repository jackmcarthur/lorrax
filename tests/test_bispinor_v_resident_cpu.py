"""``write_restart_tensors = false`` keeps the four-current V in the process (CPU, 2x2 emulated mesh).

The bispinor V file used to be written whatever the key said; for NiO 4^3 its 4.6 GB
charge tile is refused on CFS (``GATE collective_write_on_dvs``), although the key's
contract is that ``tmp/`` stays empty (``gw.gw_output.restart_tensor_writes_enabled``).
The writer, given the same seven tiles, now either writes the file (key ``true``) or
holds a ``ResidentBispinorV`` (key ``false``) that writes nothing, and every consumer
reading through ``BispinorVqReader`` gets the same tiles bit for bit, with the same
sharding, from either.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

N_Q, KGRID = 2, (2, 1, 1)
N_C, N_T = 5, 3                     # logical extents; the 2x2 carrier pads to 8 and 4
C_C, C_T = 8, 4


def _mesh():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices("cpu")
    if len(devices) < 4:
        pytest.skip("needs 4 host CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    from jax.sharding import Mesh
    return Mesh(np.array(devices[:4]).reshape(2, 2), ("x", "y"))


class _Parked:
    """Stands in for ``ParkedVTiles``: fresh device copies of fixed host tiles."""

    def __init__(self, tiles, mesh):
        self.tiles, self.mesh = tiles, mesh

    def restore(self):
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P
        put = lambda a, spec: None if a is None else jax.device_put(a, NamedSharding(self.mesh, spec))
        return [(put(V, P(None, "x", "y")), put(g0, P(*([None] * (g0.ndim - 1)), "x")) if g0 is not None else None)
                for V, g0 in self.tiles]


def _fixture(monkeypatch, mesh, *, seed=7):
    from gw import v_q_bispinor as vqb
    from gw import v_q_g_flat
    from symmetry_maps import verify_centroid_orbit_closure

    rng = np.random.default_rng(seed)
    cplx = lambda *s: rng.normal(size=s) + 1j * rng.normal(size=s)
    # Junk in the carrier pad: the file clips it on write and zero-fills on read.
    cc = [(cplx(N_Q, C_C, C_C), cplx(1, C_C))]
    tt = [(cplx(N_Q, C_T, C_T), cplx(3, 1, C_T) if mu == nu else None)
          for mu, nu in vqb.UNIQUE_TILES[1:]]
    verdict = {}
    for n in (N_C, N_T):
        frac = (np.arange(3 * n).reshape(n, 3) % 4) / 4.0
        verdict[n] = verify_centroid_orbit_closure(frac, np.eye(3)[None], tnp=np.zeros((1, 3)))

    def resolve(*, centroid_indices, **_kw):
        n = len(centroid_indices)
        pad = C_C if n == N_C else C_T
        q_irr_frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        return (None, q_irr_frac, np.arange(N_Q), None,
                np.arange(pad)[None, :], np.zeros((1, pad, 3), np.int32), True,
                SimpleNamespace(verdict=verdict[n]))

    monkeypatch.setattr(v_q_g_flat, "_resolve_ibz_q_list", resolve)
    monkeypatch.setattr(v_q_g_flat, "v_head_fn_in_V", lambda **_kw: None)
    monkeypatch.setattr(vqb, "_bispinor_qgrid_policy",
                        lambda **_kw: SimpleNamespace(unfold_sym_idx=np.zeros(N_Q, np.int64),
                                                      n_sym_spatial=1))
    monkeypatch.setattr(vqb, "_bispinor_tile_spec",
                        lambda mu, nu, **_kw: {"use_ibz": True, "is_charge_cc": mu == nu == 0})
    return cc, tt


def _build(cc, tt, mesh, path, *, resident):
    from gw.v_q_bispinor import compute_V_q_bispinor_g_flat_to_h5
    return compute_V_q_bispinor_g_flat_to_h5(
        zeta_C_loader=None, zeta_T_loaders=(None, None, None), output_h5_path=path,
        mesh_xy=mesh, kgrid=KGRID, fft_grid=(4, 4, 4), bvec=np.eye(3), cell_volume=1.0,
        sys_dim=3, n_rmu_C=N_C, n_rmu_T=N_T, sym=object(),
        centroid_C_idx=np.arange(N_C), centroid_T_idx=np.arange(N_T), use_ibz=True,
        cc_tile=_Parked(cc, mesh), tt_tiles=_Parked(tt, mesh), mc_average_vcoul_body=False,
        policy={"coulomb": "fixture", "tt_head": False}, resident=resident)


def test_key_false_writes_no_file_and_reads_the_file_tiles_bit_for_bit(monkeypatch, tmp_path):
    mesh = _mesh()
    from file_io.restart_bundle import BispinorVqReader, read_photon_charge
    from gw.v_q_bispinor import ResidentBispinorV, bispinor_v_source
    cc, tt = _fixture(monkeypatch, mesh)

    (tmp_path / "true").mkdir()
    (tmp_path / "false").mkdir()
    file_path = str(tmp_path / "true" / "v_q_bispinor.h5")
    label = str(tmp_path / "false" / "v_q_bispinor.h5")
    written, g0_file = _build(cc, tt, mesh, file_path, resident=False)
    held, g0_held = _build(cc, tt, mesh, label, resident=True)

    # Key false: nothing under tmp/, and the run's consumers resolve the held V.
    assert os.listdir(tmp_path / "false") == []
    assert isinstance(held, ResidentBispinorV) and str(held) == label
    assert bispinor_v_source(label) is held
    assert bispinor_v_source(file_path) == file_path
    assert os.path.exists(file_path)

    for a, b in zip(g0_file, g0_held):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with BispinorVqReader(written, mesh) as rf, BispinorVqReader(held, mesh) as rh:
        assert (rf.kgrid, rf.n_rmu_C, rf.n_rmu_T, rf.n_q_total) == (
            rh.kgrid, rh.n_rmu_C, rh.n_rmu_T, rh.n_q_total)
        for family in (0, 1):
            assert rf.q_tables[family].digest() == rh.q_tables[family].digest()
            assert rf.q_headers[family].centroid_hash == rh.q_headers[family].centroid_hash
        for mu in range(4):
            for nu in range(4):
                x, y = rf.get_tile(mu, nu), rh.get_tile(mu, nu)
                assert x.shape == y.shape and x.sharding == y.sharding, (mu, nu)
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=str((mu, nu)))
    np.testing.assert_array_equal(
        np.asarray(getattr(read_photon_charge(written, mesh, retain_wedge=True), "values",
                           read_photon_charge(written, mesh, retain_wedge=True))),
        np.asarray(getattr(read_photon_charge(held, mesh, retain_wedge=True), "values",
                           read_photon_charge(held, mesh, retain_wedge=True))))
    # A second read of the held V is the same (host copy, not consumed).
    with BispinorVqReader(held, mesh) as rh:
        np.testing.assert_array_equal(np.asarray(rh.get_tile(1, 2)), np.asarray(rh.get_tile(1, 2)))


def test_resident_identity_is_content_bound(monkeypatch, tmp_path):
    mesh = _mesh()
    from gw.response_bank import authenticate_coulomb
    from gw.sigma_x_bispinor import _packed_key
    from gw.v_q_bispinor import forget_resident_bispinor_v
    cc, tt = _fixture(monkeypatch, mesh)
    a, _ = _build(cc, tt, mesh, str(tmp_path / "a" / "v.h5"), resident=True)
    b, _ = _build(cc, tt, mesh, str(tmp_path / "b" / "v.h5"), resident=True)
    assert a.digest() == b.digest()                     # deterministic for the same tiles
    tt[3] = (tt[3][0] * (1 + 1e-15), tt[3][1])
    c, _ = _build(cc, tt, mesh, str(tmp_path / "c" / "v.h5"), resident=True)
    assert c.digest() != a.digest()

    resource = dict(path=str(a), basis="photon", q_irr_full_idx=[0, 1], sha256=a.digest(),
                    resident=True)
    authenticate_coulomb({"coulomb": resource}, [0, 1])
    with pytest.raises(ValueError, match="resident photon V"):
        authenticate_coulomb({"coulomb": dict(resource, sha256=c.digest())}, [0, 1])
    forget_resident_bispinor_v(str(a))
    with pytest.raises(ValueError, match="resident photon V"):
        authenticate_coulomb({"coulomb": resource}, [0, 1])

    plan = SimpleNamespace(sym_perm=np.arange(3), L_table=np.zeros((1, 3, 3)))
    basis = SimpleNamespace(canonical_indices=np.arange(3))
    layout = SimpleNamespace(packed_extent=12)
    assert _packed_key(a, plan, basis, layout, mesh) != _packed_key(b, plan, basis, layout, mesh)
