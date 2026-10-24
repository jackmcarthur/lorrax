"""Stored ζ parents are q-star rows, not a prefix of a nonminimal WFN wedge."""
import runtime  # noqa: F401
from types import ModuleType, SimpleNamespace as NS
import sys

import h5py
import numpy as np
import pytest


def _fixture(tmp_path, monkeypatch):
    from symmetry_maps import QirrTables, write_qirr_tensor
    import file_io
    from file_io import restart_bundle as rb
    path = tmp_path/'restart.h5'
    parent = np.array([[0., 0., 0.], [0., 1/3, 0.], [1/3, 1/3, 0.]])
    raw = np.array([[0., 0., 0.], [0., 1/3, 0.], [0., 2/3, 0.], [1/3, 1/3, 0.]])
    full = np.stack(np.meshgrid(np.arange(3), np.arange(3), [0], indexing='ij'), -1).reshape(-1, 3)/[3, 3, 1]
    ops = []
    for swap in (False, True):
        for sx in (1, -1):
            for sy in (1, -1):
                s = np.eye(3, dtype=int)
                s[:2, :2] = np.array([[0, sx], [sy, 0]]) if swap else np.diag([sx, sy])
                ops.append(s)
    ops = np.asarray(ops)
    labels = []
    for q in full:
        candidates = [(iq, ir) for iq, p in enumerate(parent) for ir, s in enumerate(ops)
                      if np.allclose(s@p-q, np.rint(s@p-q), atol=1e-14)]
        labels.append(candidates[0])
    tables = QirrTables(irr_idx_q=np.array(labels)[:, 0], sym_idx_q=np.array(labels)[:, 1],
        q_irr_frac=parent, sym_perm=np.zeros((len(ops), 1), int),
        L_table=np.zeros((len(ops), 1, 3), int), n_sym_spatial=len(ops))
    write_qirr_tensor(path, 'V_qmunu', np.ones((3, 1, 1), complex), tables=tables,
                 centroids_frac=np.zeros((1, 3)), sym_matrices=ops, tau=np.zeros((len(ops), 3)))
    gx, gy = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing='ij')
    all_g = np.stack((gx.ravel(), gy.ravel(), np.zeros(gx.size, int)))
    spheres = [all_g[:, np.sum((q[:, None]+all_g)**2, axis=0) <= 1.] for q in parent]
    ng = np.array([g.shape[1] for g in spheres])
    gvec = np.zeros((3, 3, ng.max()), int)
    for i, g in enumerate(spheres):
        gvec[i, :, :g.shape[1]] = g
    loader = NS(gvec_components=gvec, ngk_per_q=ng, fft_grid=np.array([5, 5, 1]),
        kpoints=raw, adot=np.eye(3), blat=1., bvec=np.eye(3), cell_volume=1.,
        r_mu_fft_idx=np.zeros((1, 3), int), zeta_cutoff_ry=1., ifmax=np.ones((1, 4), int))
    monkeypatch.setattr(rb, 'read_coarse_interactions', lambda *_a, **_k:
        dict(psi=np.zeros((9, 1, 1, 1), complex), kgrid=np.array([3, 3, 1])))
    monkeypatch.setattr(rb, 'open_zeta', lambda *_a, **_k: loader)
    monkeypatch.setattr(rb, '_ZetaGTiles', lambda *_a, **_k: NS(shape=(3, 1, int(ng.max()))))
    monkeypatch.setattr(file_io, 'read_coulomb_policy_from_h5', lambda *_a: None)
    vq = ModuleType('bse.vq_interp')
    vq.assert_slab_scope = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, 'bse.vq_interp', vq)
    # The real metadata reader may read q tables, never the interaction payload.
    original = h5py.Dataset.__getitem__
    def metadata_only(ds, key):
        assert ds.name != '/V_qmunu', 'Metadata transport must not read the V payload'
        return original(ds, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', metadata_only)
    return rb, path, loader, parent


def test_refit_nonminimal_wfn_wedge_uses_authenticated_zeta_parent_labels(tmp_path, monkeypatch):
    rb, path, loader, parent = _fixture(tmp_path, monkeypatch)
    wrong = dict(nq=3, qfr_raw=loader.kpoints[:3], qfr=loader.kpoints[:3].copy(),
                 ngk=loader.ngk_per_q, gvec=loader.gvec_components, bvec=np.eye(3), zeta_cutoff=1.)
    with pytest.raises(AssertionError, match='q=2: no candidate wrap'):
        rb._fix_sphere_wrap(wrong)
    zx = rb.read_vq_payload(str(path), 'header-only-zeta',
        require_slab=False, require_full_bz_zeta=False, log_fn=lambda *_a: None)
    np.testing.assert_array_equal(zx['qfr'], parent)
    np.testing.assert_array_equal(zx['gvec'], loader.gvec_components)
    assert zx['nk'] == 9 and zx['nq'] == 3 and zx['zeta_cutoff'] == 1.
    with pytest.raises(ValueError, match='FULL-BZ zeta'):
        rb.read_vq_payload(str(path), 'header-only-zeta', require_slab=False)
    # Correct labels must still refuse a genuinely corrupt physical sphere.
    loader.gvec_components[2, :, 0] = [9, 9, 0]
    with pytest.raises(AssertionError, match='no candidate wrap'):
        rb.read_vq_payload(str(path), 'header-only-zeta',
            require_slab=False, require_full_bz_zeta=False, log_fn=lambda *_a: None)


def test_zeta_parent_metadata_refuses_wrong_extent_grid_or_torn_tables(tmp_path, monkeypatch):
    rb, path, _loader, _parent = _fixture(tmp_path, monkeypatch)
    for nq, grid in ((4, (3, 3, 1)), (3, (4, 4, 1))):
        with pytest.raises(ValueError, match='GATE vq_zeta_q_labels'):
            rb._zeta_parent_qpoints(str(path), nq, grid)
    with h5py.File(path, 'a') as f:
        f['V_qmunu__qirr/q_irr_frac'][2, 0] = .123
    with pytest.raises(ValueError, match='table hash mismatch'):
        rb._zeta_parent_qpoints(str(path), 3, (3, 3, 1))
    legacy = tmp_path/'legacy.h5'
    with h5py.File(legacy, 'w') as f:
        f.create_dataset('V_qmunu', shape=(9, 1, 1), dtype=complex)
    with pytest.raises(ValueError, match='GATE vq_zeta_q_labels'):
        rb._zeta_parent_qpoints(str(legacy), 3, (3, 3, 1))
