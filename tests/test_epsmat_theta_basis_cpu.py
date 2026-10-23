"""A fixed-q Fourier basis must match the producer's measured Θ realization."""
from types import SimpleNamespace as NS

import numpy as np
import pytest


def fixture(*, noise=1.e-6, trs=True, half=False):
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw import epsmat_export as ee
    jax.config.update("jax_enable_x64", True)
    mesh = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    q = np.array([[0., 0., 0.], [-.5, 0., 0.]]) if half else np.array(
        [[0., 0., 0.], [1./3, 0., 0.], [-1./3, 0., 0.]])
    gx = [0, 1, 2, -1] if half else [0, 1, -1, 2, -2]
    miller = np.column_stack([gx, np.zeros((len(gx), 2), int)])
    partner = np.array([1, 0, 3, 2] if half else [0, 2, 1, 4, 3])
    v = np.array([.7, .7, 1.3, 1.3] if half else [0., .7, .7, 1.3, 1.3])
    rng = np.random.default_rng(904)
    raw = rng.normal(size=(4, len(gx))) + 1j*rng.normal(size=(4, len(gx)))
    even = .5*(raw+raw[:, partner].conj())
    raw_noise = rng.normal(size=raw.shape) + 1j*rng.normal(size=raw.shape)
    Z = even + noise*.5*(raw_noise-raw_noise[:, partner].conj())
    gram = (Z.conj()*v[None])@Z.T
    V = gram.real if trs else gram
    chi = -np.eye(4)*(.2/np.linalg.eigvalsh(V).max())
    Wc = V@chi@V
    def operation_rows(rows):
        rows = np.asarray(rows)
        anti = rows.astype(bool)
        return (np.where(anti[..., None, None], -np.eye(3, dtype=int), np.eye(3, dtype=int)),
                np.zeros(rows.shape+(3,)), anti)
    sym = NS(trs_allowed=trs, q_irr_full_idx=np.arange(len(q)),
        active_symmetry_rows=np.arange(2 if trs else 1), operation_rows=operation_rows,
        reciprocal_phase=lambda _row, carriers: np.ones(len(carriers), complex))
    tables = NS(irr_idx_q=np.arange(len(q)), sym_idx_q=np.zeros(len(q), int),
                n_sym_spatial=1)
    src = NS(q_tables=tables, q_full=q, q_stored=q, kgrid=(len(q), 1, 1), sym=sym,
        zeta=NS(gvecs=lambda **_k: miller[None], ngk_valid=lambda **_k: np.array([len(gx)])),
        eigh=ee.eigh_plan(mesh, 4), n_mu=4)
    put = lambda a, spec: jax.device_put(np.asarray(a), NamedSharding(mesh, spec))
    arrays = (put(V[None].astype(complex), P(None, "x", "y")),
              put((V+Wc)[None].astype(complex), P(None, "x", "y")),
              put(Z[None], P(None, ("x", "y"), None)))
    return ee, src, mesh, arrays, even, v, Wc


def test_theta_even_basis_strict_metric_and_plane_wave_roundtrip(monkeypatch):
    ee, src, mesh, arrays, even, v, Wc = fixture()
    monkeypatch.setattr(ee, "read_q_rows", lambda *_a: arrays)
    monkeypatch.setattr(ee, "lorrax_v", lambda *_a: v)
    Z, _B, _K, chi, _V, _Wc, diag = ee.q_algebra(src, 0, mesh, rcond=1.e-8)
    np.testing.assert_allclose(np.asarray(Z)[0], even, atol=1.e-15, rtol=1.e-14)
    assert diag["err_v"] < ee.V_RECONSTRUCTION_TOL
    assert diag["zeta_anti_theta_removed"][0] > 1.e-8
    raw = np.asarray(arrays[2])[0]
    raw_gram = (raw.conj()*v[None])@raw.T
    assert np.linalg.norm(raw_gram-np.asarray(arrays[0])[0])/np.linalg.norm(raw_gram) > 1.e-8
    face = ee.zeta_face(Z, np.arange(len(v)), mesh=mesh, n_pad=len(v))
    block, _diagonal = ee.epsinv_block(face, chi, v, mesh=mesh, n_valid=len(v))
    back = np.asarray(ee.project_block(block, face, v, mesh=mesh,
                                     n_valid=len(v), drop_head=False))[0]
    np.testing.assert_allclose(back, Wc, atol=2.e-11, rtol=1.e-9)


def test_large_odd_basis_metric_refuses_at_unchanged_gate(monkeypatch):
    ee, src, mesh, arrays, _even, v, _Wc = fixture(noise=.1)
    monkeypatch.setattr(ee, "read_q_rows", lambda *_a: arrays)
    monkeypatch.setattr(ee, "lorrax_v", lambda *_a: v)
    with pytest.raises(ValueError, match="epsmat_v_reconstruction"):
        ee.q_algebra(src, 0, mesh, rcond=1.e-8)


def test_nonfixed_magnetic_and_independent_rows_keep_original_basis():
    ee, src, mesh, arrays, _even, _v, _Wc = fixture()
    original = arrays[2]
    def forbidden(*_args):
        raise AssertionError("No Θ image is allowed for this row")
    src.sym.operation_rows = forbidden
    assert ee.fixed_q_zeta_basis(src, 1, original, mesh=mesh) == (original, None)
    src.sym.trs_allowed = False
    src.sym.active_symmetry_rows = np.array([0])
    assert ee.fixed_q_zeta_basis(src, 0, original, mesh=mesh) == (original, None)
    src.q_tables = None
    assert ee.fixed_q_zeta_basis(src, 0, original, mesh=mesh) == (original, None)


def test_half_grid_fixed_q_uses_typed_umklapp_and_missing_partner_refuses():
    ee, src, mesh, arrays, even, _v, _Wc = fixture(half=True)
    result, removed = ee.fixed_q_zeta_basis(src, 1, arrays[2], mesh=mesh)
    np.testing.assert_allclose(np.asarray(result)[0], even, atol=1.e-15, rtol=1.e-14)
    assert np.asarray(removed)[0] > 1.e-8
    bad = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    src.zeta.gvecs = lambda **_k: bad[None]
    with pytest.raises(ValueError, match="epsmat_theta_sphere"):
        ee.fixed_q_zeta_basis(src, 1, arrays[2], mesh=mesh)
