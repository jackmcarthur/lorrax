"""Frontend ownership of separate bosonic and electronic q/k domains."""
from types import SimpleNamespace

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _sym(grid=(4, 4, 1), order=None):
    from symmetry_maps import SymMaps
    sym = object.__new__(SymMaps)
    coords = np.indices(grid).reshape(3, -1).T.astype(np.int32)
    if order is not None:
        coords = coords[order]
    sym.kvecs_asints = coords
    sym.nk_tot = len(coords)
    sym.nk_red = len(coords)
    sym.sym_matrices = np.array([np.eye(3, dtype=int), -np.eye(3, dtype=int)])
    sym.translations = np.zeros((2, 3))
    sym.irr_idx_k = np.arange(len(coords), dtype=np.int32)
    sym.sym_idx_k = np.zeros(len(coords), np.int32)
    sym.q_irr_full_idx = np.arange(len(coords), dtype=np.int32)
    sym.irr_idx_q = sym.q_irr_full_idx.copy()
    sym.sym_idx_q = np.zeros(len(coords), np.int32)
    sym.q_irr_kgrid_int = coords.copy()
    sym.active_symmetry_rows = np.array([0, 1], np.int32)
    return sym


@pytest.mark.parametrize("fine,coarse", [((24, 24, 1), (6, 6, 1)),
                                         ((4, 6, 2), (2, 3, 1))])
@pytest.mark.parametrize("permuted", [False, True])
def test_independent_q_domain_preserves_electrons_and_nested_physical_points(fine, coarse, permuted):
    n = int(np.prod(fine))
    order = np.random.default_rng(41).permutation(n) if permuted else None
    sym = _sym(fine, order)
    view = sym.qgrid_view(coarse, kgrid=fine)
    for name in ("kvecs_asints", "irr_idx_k", "sym_idx_k", "sym_matrices", "translations"):
        assert getattr(view, name) is getattr(sym, name)
    q = np.indices(coarse).reshape(3, -1).T
    np.testing.assert_array_equal(view.q_irr_kgrid_int, q)
    np.testing.assert_array_equal(sym.kvecs_asints[view.q_to_fine_full],
                                  q*(np.array(fine)//coarse))
    np.testing.assert_array_equal(view.q_irr_full_idx, np.arange(len(q)))
    np.testing.assert_array_equal(view.irr_idx_q, np.arange(len(q)))
    np.testing.assert_array_equal(view.sym_idx_q, 0)
    np.testing.assert_array_equal(view.active_symmetry_rows, [0])
    assert view.q_irr_is_full_identity
    assert sym.qgrid_view(fine, kgrid=fine) is sym


@pytest.mark.parametrize("grid", [(3, 2, 1), (0, 2, 1), (2, 2),
                                  (2., 2, 1), (True, 2, 1), (8, 2, 1)])
def test_q_domain_invalid_grid_refuses(grid):
    with pytest.raises(ValueError, match="qgrid_domain"):
        _sym().qgrid_view(grid, kgrid=(4, 4, 1))


def _config(**values):
    from gw.gw_config import ComputeMode, QPSolver
    defaults = dict(qgrid=(2, 2, 1), bispinor=False, compute_mode=ComputeMode.MPA,
        sigma=SimpleNamespace(w_model="shared_pole"),
        screening=SimpleNamespace(diagrams="w_rpa"), sys_dim=2,
        head=SimpleNamespace(mc_average_vcoul_body=True, wcoul0_source="s_tensor",
            uses_bgw_metal_q0shift=False, vhead=None, whead_0freq=None, whead_imfreq=None),
        restart=False, write_restart_tensors=False,
        debug=SimpleNamespace(write_wfn_h5=False),
        qp_solver=QPSolver.SELF_CONSISTENT, qp_rotations_k_storage="full",
        occ_smearing_width_ry=None, do_G0=True)
    defaults.update(values)
    return SimpleNamespace(**defaults)


def test_default_equal_and_supported_scalar_routes():
    from gw.gw_init import validate_qgrid_route
    # No new attribute requirements at all on unchanged routes.
    validate_qgrid_route(SimpleNamespace(), (4, 4, 1))
    validate_qgrid_route(SimpleNamespace(qgrid=(4, 4, 1)), (4, 4, 1))
    validate_qgrid_route(_config(), (4, 4, 1))
    cfg = _config(sys_dim=3)
    cfg.head.mc_average_vcoul_body = False
    validate_qgrid_route(cfg, (4, 4, 1))


@pytest.mark.parametrize("change,gate", [
    ({"bispinor": True}, "qgrid_route"),
    ({"write_restart_tensors": True}, "qgrid_restart_artifact"),
    ({"restart": True}, "qgrid_restart_input"),
    ({"qp_rotations_k_storage": "auto"}, "qgrid_qp_rotations"),
    ({"debug": SimpleNamespace(write_wfn_h5=True)}, "qgrid_wfn_artifact"),
    ({"occ_smearing_width_ry": .001}, "qgrid_metal_response"),
    ({"sys_dim": 3}, "qgrid_coulomb_cell"),
    ({"sigma": SimpleNamespace(w_model="mpa")}, "qgrid_route"),
    ({"screening": SimpleNamespace(diagrams="w_bse")}, "qgrid_route"),
])
def test_unsupported_mixed_grid_options_refuse_before_isdf(change, gate):
    from gw.gw_init import validate_qgrid_route
    with pytest.raises(ValueError, match=gate):
        validate_qgrid_route(_config(**change), (4, 4, 1))


@pytest.mark.parametrize("key,value,gate", [
    ("wcoul0_source", "epshead", "qgrid_epshead"),
    ("uses_bgw_metal_q0shift", True, "qgrid_metal_shift"),
    ("vhead", 0., "qgrid_head_override"),
    ("whead_0freq", 1., "qgrid_head_override"),
    ("whead_imfreq", 2., "qgrid_head_override"),
])
def test_unmeasured_head_cells_refuse(key, value, gate):
    from gw.gw_init import validate_qgrid_route
    cfg = _config()
    setattr(cfg.head, key, value)
    with pytest.raises(ValueError, match=gate):
        validate_qgrid_route(cfg, (4, 4, 1))
    if key == "wcoul0_source":
        cfg.do_G0 = False
        validate_qgrid_route(cfg, (4, 4, 1))


def _mesh():
    from jax.sharding import Mesh
    devices = np.asarray(jax.devices("cpu"))
    return Mesh(devices[:4].reshape(2, 2) if len(devices) >= 4 else devices[:1].reshape(1, 1),
                ("x", "y"))


def test_selected_reduced_operator_uses_bounded_owner_unfold(monkeypatch):
    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding, PartitionSpec as P
    from symmetry_maps import QirrOperator, unfold_isdf_operator
    import symmetry_maps.maps as owner
    mesh = _mesh()
    rng = np.random.default_rng(661)
    values = rng.normal(size=(2, 4, 4)) + 1j*rng.normal(size=(2, 4, 4))
    device = device_put_process_local(values, NamedSharding(mesh, P(None, "x", "y")))
    perm = np.array([[0, 1, 2, 3], [1, 0, 3, 2]]*2, np.int32)
    wraps = np.zeros((4, 4, 3)); wraps[[1, 3], :, 0] = [1, 0, -1, 0]
    op = QirrOperator(device, irr_idx=np.array([0, 1, 0, 1], np.int32),
        sym_idx=np.array([0, 0, 1, 3], np.int32), sym_perm=perm, L_table=wraps,
        q_irr_frac=np.array([[0., 0., 0.], [.25, .5, 0.]]),
        n_sym_spatial=2, full_rows=np.array([0, 1], np.int32))
    full = unfold_isdf_operator(op.values, irr_idx=op.irr_idx, sym_idx=op.sym_idx,
        sym_perm=op.sym_perm, L_table=op.L_table, q_irr_frac=op.q_irr_frac,
        mesh_xy=mesh, n_sym_spatial=op.n_sym_spatial)
    seen = []
    original = owner.unfold_isdf_operator
    def bounded(*args, **kwargs):
        seen.append(len(kwargs["irr_idx"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, "unfold_isdf_operator", bounded)
    actual = op.at_rows([3, 2], mesh_xy=mesh)
    np.testing.assert_allclose(actual, np.asarray(full)[[3, 2]], atol=1e-13)
    assert seen == [2]
    with pytest.raises(ValueError, match="mesh_xy"):
        op.at_rows([3, 2])


def test_coarse_coulomb_keeps_fine_bare_operator_and_uses_physical_rows():
    from gw.gw_init import screening_coulomb_view
    from symmetry_maps import QirrOperator
    values = np.arange(16*4).reshape(16, 2, 2).astype(complex)
    fine = QirrOperator.whole_zone(values)
    meta = SimpleNamespace(kgrid=(4, 4, 1), qgrid=(4, 4, 1))
    assert screening_coulomb_view(fine, meta=meta, mesh_xy=_mesh()) is fine
    meta.qgrid = (2, 2, 1)
    meta.q_sym = _sym().qgrid_view(meta.qgrid, kgrid=meta.kgrid)
    coarse = screening_coulomb_view(fine, meta=meta, mesh_xy=_mesh())
    np.testing.assert_array_equal(coarse.values, values[[0, 2, 8, 10]])
    assert coarse.n_full == 4 and fine.n_full == 16
    np.testing.assert_array_equal(fine.values, values)
    np.testing.assert_array_equal(coarse.q_irr_frac,
        np.indices(meta.qgrid).reshape(3, -1).T/np.array(meta.qgrid))


def test_restart_disabled_skips_static_model_evaluation_decision():
    from gw.screening import driver_persists_w0
    from gw.gw_config import ComputeMode
    cfg = _config()
    assert not driver_persists_w0(ComputeMode.MPA, cfg)
    cfg.write_restart_tensors = True
    assert driver_persists_w0(ComputeMode.MPA, cfg)


def test_source_observer_copies_fine_bare_and_coarse_screening_independently():
    from gw.ppm_pipeline import observe_window_sigma_source, publish_window_source
    from symmetry_maps import QirrOperator
    import jax.numpy as jnp
    bare = QirrOperator.whole_zone(jnp.ones((16, 2, 2), dtype=complex))
    screening = QirrOperator.whole_zone(jnp.full((4, 2, 2), 3+2j))
    with observe_window_sigma_source(screening=True) as captured:
        publish_window_source(V_q=bare, screening_inputs=dict(wfns=None, V_q=screening))
    assert captured["V_q"].values is not bare.values
    assert captured["screening_inputs"]["V_q"].values is not screening.values
    assert captured["V_q"].n_full == 16
    assert captured["screening_inputs"]["V_q"].n_full == 4
    np.testing.assert_array_equal(captured["V_q"].values, bare.values)
    np.testing.assert_array_equal(captured["screening_inputs"]["V_q"].values, screening.values)


def test_self_negative_uses_bosonic_grid_and_dyson_room_counts_coarse_faces():
    from gw.response_bank import _self_negative, _dyson_room
    ledger = SimpleNamespace(live_stages=(), room_bytes_per_rank=lambda **kwargs: 100000)
    meta = SimpleNamespace(nkx=4, nky=4, nkz=1, nk_tot=16,
        qgrid=(2, 2, 1), nq_tot=4, shared_pole_capacity=ledger)
    assert all(_self_negative(i, meta) for i in range(4))
    assert _dyson_room(meta, _mesh(), 2) == 100000 - 16*4*3*4//_mesh().size
