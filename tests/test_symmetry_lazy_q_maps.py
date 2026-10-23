"""Lazy q metadata agrees with literal lookups and preserves writable views."""
from types import SimpleNamespace
import warnings

import numpy as np
from ffi import _services
_services.ensure_on_path()
from symmetry_maps import SymMaps


def _system(identity):
    grid = np.asarray([4,3,2])
    coords = np.indices(tuple(grid)).reshape(3,-1).T
    if identity:
        shift = np.asarray([.5,.5,0.])
        coords = coords[np.random.default_rng(412).permutation(len(coords))]
        matrices = np.eye(3,dtype=np.int32)[None]
    else:
        shift = np.zeros(3)
        representatives = {min(tuple(k),tuple((-k)%grid)) for k in coords}
        coords = np.asarray(sorted(representatives))
        matrices = np.asarray([np.eye(3,dtype=np.int32),-np.eye(3,dtype=np.int32)])
    return SimpleNamespace(kgrid=grid,shift=shift,kpoints=(coords+shift)/grid,
        nkpts=len(coords),ntran=len(matrices),sym_matrices=matrices,
        translations=np.zeros((len(matrices),3)),trs_holds=True,
        avec=np.eye(3),bvec=np.eye(3),nspinor=1)


def _literal_map(full,qpoints):
    output = np.empty((len(full),len(qpoints)),dtype=np.int32)
    for i,k in enumerate(full):
        for j,q in enumerate(qpoints):
            delta = full-(k-q)
            delta -= np.rint(delta)
            distance = np.sum(np.abs(delta),axis=1)
            hit = np.flatnonzero(distance<1e-7)
            assert len(hit)==1
            output[i,j]=hit[0]
    return output


def test_lazy_q_tables_match_literal_and_preserve_view_mutation():
    for identity in (True,False):
        wfn = _system(identity)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            sym = SymMaps(wfn)
        assert bool(sym._q_map_identity)==identity
        lazy = ('kq_map','kqfull_map','all_unfolded_qpt_ids')
        assert all(name not in vars(sym) for name in lazy)
        assert sym.all_unfolded_qpts.ndim==2
        full = np.asarray(sym.unfolded_kpts)
        q_reduced = (sym.kvecs_asints/wfn.kgrid if identity else wfn.kpoints)
        expected_reduced = _literal_map(full,q_reduced)
        expected_full = _literal_map(full,sym.kvecs_asints/wfn.kgrid if identity else full)
        reduced = sym.kq_map
        np.testing.assert_array_equal(reduced,expected_reduced)
        assert reduced is sym.kq_map
        assert 'kqfull_map' not in vars(sym) and 'all_unfolded_qpt_ids' not in vars(sym)
        full_map = sym.kqfull_map
        np.testing.assert_array_equal(full_map,expected_full)
        assert full_map is sym.kqfull_map
        assert not np.shares_memory(reduced,full_map)
        assert 'all_unfolded_qpt_ids' not in vars(sym)
        pairs = sym.kvecs_asints[:,None,:]-sym.kvecs_asints[None,:,:]
        vectors = np.unique(pairs.reshape(-1,3),axis=0)
        lookup = {tuple(q):i for i,q in enumerate(vectors)}
        expected_labels = np.asarray([[lookup[tuple(q)] for q in row] for row in pairs],np.int32)
        np.testing.assert_array_equal(sym.all_unfolded_qpts,vectors)
        np.testing.assert_array_equal(sym.all_unfolded_qpt_ids,expected_labels)
        assert sym.all_unfolded_qpt_ids is sym.all_unfolded_qpt_ids

        # Build a fresh view before touching lazy arrays: it must retain the
        # historical shallow full/pair shares and independent reduced copy.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            parent = SymMaps(wfn)
        view = parent.trivial_view()
        assert view.kqfull_map is parent.kqfull_map
        assert view.all_unfolded_qpt_ids is parent.all_unfolded_qpt_ids
        assert view.all_unfolded_qpts is parent.all_unfolded_qpts
        np.testing.assert_array_equal(view.kq_map,parent.kqfull_map)
        assert not np.shares_memory(view.kq_map,parent.kqfull_map)
        old = int(parent.kqfull_map[0,0])
        view.kq_map[0,0]=-5
        assert parent.kqfull_map[0,0]==old
        view.kqfull_map[0,0]=-7
        assert parent.kqfull_map[0,0]==-7
        parent.kqfull_map[0,0]=old
        replacement = np.asarray([[92]],dtype=np.int32)
        view.kqfull_map=replacement
        assert view.kqfull_map is replacement and parent.kqfull_map.shape==expected_full.shape
        for name in lazy:
            setattr(sym,name,replacement)
            assert getattr(sym,name) is replacement

        # Old eager arrays represented construction-time geometry even when
        # a caller subsequently edited the writable coordinate metadata.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            frozen = SymMaps(wfn)
        old_full = frozen.unfolded_kpts.copy()
        old_coords = frozen.kvecs_asints.copy()
        qpoints = old_coords/wfn.kgrid if identity else wfn.kpoints.copy()
        frozen.unfolded_kpts[:] = 0
        frozen.kvecs_asints[:] = 0
        wfn.kpoints[:] = 0
        np.testing.assert_array_equal(frozen.kq_map,_literal_map(old_full,qpoints))
        np.testing.assert_array_equal(frozen.kqfull_map,
            _literal_map(old_full,old_coords/wfn.kgrid if identity else old_full))
        np.testing.assert_array_equal(frozen.all_unfolded_qpt_ids,expected_labels)
