"""Independent bosonic coarse-quadrature oracle, not zero-embedding interpolation."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _direct(g, w, kgrid, qgrid):
    """Literal (1/Nq) sum_q G(k-q) W(q) on nested physical grids."""
    k = np.indices(kgrid).reshape(3, -1).T
    q = np.indices(qgrid).reshape(3, -1).T * (np.array(kgrid) // qgrid)
    result = np.zeros_like(g)
    for iq, delta in enumerate(q):
        kmq = np.ravel_multi_index(((k - delta) % kgrid).T, kgrid)
        result += g[kmq] * w[iq]
    return result / len(q)


@pytest.mark.parametrize("kgrid,qgrid", [((24, 24, 1), (6, 6, 1)),
    ((8, 6, 1), (2, 3, 1)), ((4, 3, 2), (2, 1, 1))])
@pytest.mark.parametrize("case", ["random", "constant_w", "constant_g"])
def test_periodic_bosonic_lattice_tile_equals_literal_coarse_q_sum(kgrid, qgrid, case):
    from gw.ppm_tau_kernel import sigma_lattice_tile
    rng = np.random.default_rng(739)
    nk, nq = int(np.prod(kgrid)), int(np.prod(qgrid))
    g = rng.normal(size=(nk, 2, 3)) + 1j*rng.normal(size=(nk, 2, 3))
    w = rng.normal(size=(nq, 2, 3)) + 1j*rng.normal(size=(nq, 2, 3))
    if case == "constant_w":
        w[:] = 0.4 - 0.7j
    if case == "constant_g":
        g[:] = 0.8 + 0.3j
    axes = (0, 1, 2)
    gr = np.fft.fftn(g.reshape(*kgrid, 2, 3), axes=axes, norm="ortho").reshape(nk, 2, 3)
    wr = np.fft.fftn(w.reshape(*qgrid, 2, 3), axes=axes, norm="ortho").reshape(nq, 2, 3)
    tiled = np.asarray(jax.jit(lambda a: sigma_lattice_tile(a, kgrid, qgrid))(wr))
    actual = np.fft.ifftn((gr*tiled/np.sqrt(nk)).reshape(*kgrid, 2, 3),
                         axes=axes, norm="ortho").reshape(nk, 2, 3)
    expected = _direct(g, w, kgrid, qgrid)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    # Zero embedding is a DIFFERENT fine-q interpolant/quadrature, not this API.
    embedded = np.zeros_like(gr)
    slots = np.ravel_multi_index(np.indices(qgrid).reshape(3, -1), kgrid)
    embedded[slots] = wr * np.sqrt(nk/nq)
    wrong = np.fft.ifftn((gr*embedded/np.sqrt(nk)).reshape(*kgrid, 2, 3),
                        axes=axes, norm="ortho").reshape(nk, 2, 3)
    if case != "constant_g":
        assert np.linalg.norm(wrong-expected) > 0.1


def test_bosonic_lattice_tile_default_is_identity_and_invalid_grids_refuse():
    from gw.ppm_tau_kernel import sigma_lattice_tile
    a = jnp.arange(12).reshape(4, 3)
    assert sigma_lattice_tile(a, (2, 2, 1), (2, 2, 1)) is a
    for k, q in [((6, 6, 1), (4, 3, 1)), ((4, 4, 1), (0, 2, 1)),
                 ((4, 4, 1), (2, 2))]:
        with pytest.raises(ValueError, match="sigma_qgrid"):
            sigma_lattice_tile(a, k, q)
    with pytest.raises(ValueError, match="W lattice axis"):
        sigma_lattice_tile(a, (6, 6, 1), (3, 3, 1))


def test_store_authenticates_separate_electronic_and_bosonic_grids():
    from file_io import shared_pole_store as store
    from symmetry_maps import QirrTables
    mesh = SimpleNamespace(shape={"x": 1, "y": 1})
    basis = SimpleNamespace(n_logical=2, canonical_indices=np.array([[0, 0, 0], [1, 0, 0]]),
                            mesh_xy=mesh)
    geometry = dict(nq=4, nspinor=2, nmu=2, px=1, py=1)
    meta = SimpleNamespace(nkx=4, nky=4, nkz=1, nk_tot=16, qgrid=(2, 2, 1),
        nspinor=2, mu_basis=basis, fft_grid=(2, 2, 2),
        shared_pole_capacity=SimpleNamespace(geometry=geometry, live_stages=()))
    qt = QirrTables(irr_idx_q=np.arange(4, dtype=np.int32), sym_idx_q=np.zeros(4, np.int32),
        sym_perm=np.arange(2, dtype=np.int32)[None], L_table=np.zeros((1, 2, 3)),
        q_irr_frac=np.indices((2, 2, 1)).reshape(3, -1).T/np.array([2, 2, 1]), n_sym_spatial=1)
    sym = SimpleNamespace(trs_allowed=True, active_symmetry_rows=np.array([0]),
        operation_rows=lambda rows: (np.eye(3, dtype=int)[None], np.zeros((1, 3)), np.array([False])),
        spinor_action=lambda rows, nspinor: np.ones((1, 1, 1)), operation_typing_source="oracle")
    header = store._metadata(meta, dict(qirr=qt, q_irr_full_idx=np.arange(4), sym=sym),
        {"gate_version": "oracle"}, {name: "two-grid-oracle" for name in store._IDENTITY_KEYS})
    assert header["grid"] == [2, 2, 1] and header["electronic_grid"] == [4, 4, 1]
    assert header["n_q_full"] == 4
    assert store._capacity(meta) is meta.shared_pole_capacity
    store._check_basis(meta, header)
    store._check_io_capacity(meta.shared_pole_capacity, mesh, header)
    with pytest.raises(ValueError, match="grid changed"):
        store._check_basis(meta, {**header, "electronic_grid": [8, 4, 1]})
    with pytest.raises(ValueError, match="grid changed"):
        store._check_basis(meta, {**header, "grid": [4, 1, 1]})
    # Legacy stores mean identical electronic/bosonic grids, not guessed grids.
    meta.qgrid = None
    with pytest.raises(ValueError, match="grid changed"):
        store._check_basis(meta, {k: v for k, v in header.items() if k != "electronic_grid"})
    meta.nkx, meta.nky, meta.nk_tot = 2, 2, 4
    store._check_basis(meta, {k: v for k, v in header.items() if k != "electronic_grid"})
    assert store._capacity(meta) is meta.shared_pole_capacity


def test_native_sigma_node_coarse_q_fft_tile_and_projection_equal_literal_sum(monkeypatch):
    """Actual mode9 -> tiled real lattice -> mode7 -> destination projection."""
    import warnings
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from gw.ppm_tau_kernel import (_get_sigma_kij_kernel, sigma_subtile_operands,
                                    sigma_kconv_tables)
    from gw.wavefunction_bundle import parent_sigma_operands, sigma_face_kernel_kwargs
    from symmetry_maps import QirrOperator
    from ffi import fft as fft_owner
    from tests import test_tdagw_dynsig_sigma as fixture

    devices = np.asarray(jax.devices("cpu"))
    mesh = Mesh(devices[:4].reshape(2, 2) if len(devices) >= 4 else devices[:1].reshape(1, 1), ("x", "y"))
    kgrid, qgrid = (4, 2, 1), (2, 1, 1)
    for name, value in dict(KGRID=kgrid, NBF=3, NB_PAD=4, NS=2, NMU=4, NOCC=2).items():
        monkeypatch.setattr(fixture, name, value)
    rng = np.random.default_rng(517)
    nk, nq, nb, ns, mu = 8, 2, 3, 2, 4
    psi = .2*(rng.normal(size=(nk, nb, ns, mu)) + 1j*rng.normal(size=(nk, nb, ns, mu)))
    energy = rng.uniform(-.8, .9, (nk, nb))
    occ = rng.uniform(.1, .9, (nk, nb))
    kf = np.indices(kgrid).reshape(3, -1).T / np.array(kgrid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plan, perm = fixture._plan(mesh, kf, rng)
    psi = psi[..., perm]
    wfns, _ = fixture._gw_bundle(mesh, psi, energy, occ, plan)
    operands = parent_sigma_operands(wfns)
    spatial = sigma_subtile_operands(*operands[:4], mesh_xy=mesh)
    w = .15*(rng.normal(size=(nq, mu, mu)) + 1j*rng.normal(size=(nq, mu, mu)))
    wdev = device_put_process_local(w, NamedSharding(mesh, P(None, "x", "y")))
    wedge = QirrOperator.whole_zone(wdev)
    t, ref = .21-.04j, .13
    with fft_owner.xla_reference():
        kernel = _get_sigma_kij_kernel(mesh_xy=mesh, kgrid=kgrid, qgrid=qgrid,
            q_wedge=wedge, **sigma_face_kernel_kwargs(wfns))
        actual = np.asarray(kernel(*spatial, operands[4], operands[5], ref, t,
            wdev, None, wedge.with_load(mesh).load, sigma_kconv_tables(mesh, plan)))[:, :nb, :nb]
    ks = np.indices(kgrid).reshape(3, -1).T
    qs = np.indices(qgrid).reshape(3, -1).T*(np.array(kgrid)//qgrid)
    expected = np.zeros((nk, nb, nb), complex)
    for iq, q in enumerate(qs):
        partner = np.ravel_multi_index(((ks-q) % kgrid).T, kgrid)
        vertex = np.einsum("kisr,kbsr->kibr", psi.conj(), psi[partner])
        weight = occ[partner]*np.exp(-1j*(energy[partner]-ref)*t)
        expected -= np.einsum("kibr,rs,kjbs,kb->kij", vertex, w[iq], vertex.conj(), weight)/nq
    np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=3e-12)
    # Metadata/factory contract only: verify the host-plan adapters use the
    # FFT owner's 3D vocabulary before any unavailable native host handler
    # is planned. No numerical native leg is replaced or claimed here.
    from common import fft_helpers
    from ffi.fft import validate_flat_spec
    seen = []

    class FactoriesChecked(Exception):
        pass

    def inverse_factory(mesh_arg, grid_arg, spec_arg, **kw):
        assert tuple(validate_flat_spec(spec_arg, "two-grid host W")) == (None, "x", "y")
        seen.append(("inverse", tuple(grid_arg)))
        return lambda value: value

    def forward_factory(mesh_arg, grid_arg, spec_arg, **kw):
        assert tuple(validate_flat_spec(spec_arg, "two-grid host W")) == (None, "x", "y")
        seen.append(("forward", tuple(grid_arg)))
        raise FactoriesChecked

    with monkeypatch.context() as contract:
        import gw.ppm_tau_kernel as sigma_owner
        contract.setattr(sigma_owner, "_sigma_kij_kernel_cache", {})
        contract.setattr(fft_owner, "kconv_backend", lambda *a, **kw: "plan")
        contract.setattr(fft_helpers, "make_flat_k_ifftn", inverse_factory)
        contract.setattr(fft_helpers, "make_flat_k_fftn", forward_factory)
        with pytest.raises(FactoriesChecked):
            _get_sigma_kij_kernel(mesh_xy=mesh, kgrid=kgrid, qgrid=qgrid,
                q_wedge=wedge, **sigma_face_kernel_kwargs(wfns))
    assert seen == [("inverse", qgrid), ("forward", kgrid)]


def test_screening_tables_and_coulomb_authenticate_bosonic_not_electronic_grid(monkeypatch):
    from gw.shared_pole_screening import _shared_pole_tables, _coulomb_resource
    from gw import response_bank
    from symmetry_maps import QirrOperator
    meta = SimpleNamespace(nkx=4, nky=4, nkz=1, nk_tot=16,
        qgrid=(2, 2, 1), fft_grid=(2, 2, 2),
        mu_basis=SimpleNamespace(n_packed=2))
    coords = np.indices(meta.qgrid).reshape(3, -1).T
    sym = SimpleNamespace(sym_matrices=np.eye(3, dtype=int)[None],
        translations=np.zeros((1, 3)), irr_idx_q=np.arange(4),
        sym_idx_q=np.zeros(4, int), q_irr_kgrid_int=coords,
        q_irr_full_idx=np.arange(4))
    tables = _shared_pole_tables(meta, sym, np.array([[0, 0, 0], [1, 0, 0]]))
    np.testing.assert_allclose(tables["qirr"].q_irr_frac,
        coords/np.array(meta.qgrid))
    assert tables["qirr"].n_q_full == 4
    monkeypatch.setattr(response_bank, "operator_digest", lambda *a: "coarse-V")
    monkeypatch.setattr(response_bank, "_operator_token", lambda *a: "coarse-token")
    resource = _coulomb_resource(QirrOperator.whole_zone(jnp.zeros((4, 2, 2))),
        meta, sym, None)
    assert resource["q_irr_full_idx"] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="packed bare V"):
        _coulomb_resource(QirrOperator.whole_zone(jnp.zeros((16, 2, 2))),
            meta, sym, None)
