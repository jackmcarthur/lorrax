"""Ordinary complete Qirr tensor reading authenticates physical coordinates.

These tiny CPU files use SlabIO for array payloads and the public Qirr stamp
owner for metadata. No reader/closure helper is mocked in the positive path.
"""
from types import SimpleNamespace
import shutil
import numpy as np
import pytest


def _fixture(tmp_path, kind):
    import h5py
    import jax
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from common.centroid_basis import PackedCentroidBasis
    from file_io.slab_io import SlabIO
    from gw.v_q_bispinor import UNIQUE_TILES,tile_dataset_name,_publish_unique_tile_inventory
    from symmetry_maps import QirrTables,stamp_qirr_tensor,verify_centroid_orbit_closure
    mesh=Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),('x','y'))
    grid=np.asarray([8,10,12],np.int32)
    points=(np.asarray([[1,2,3],[5,6,7]],np.int32) if kind=='fft_indices'
            else np.asarray([[.123456789,.234567891,.345678912],
                             [.623456789,.734567891,.845678912]],np.float64))
    sym=SimpleNamespace(sym_matrices=np.eye(3,dtype=np.int32)[None],
        translations=np.zeros((1,3)),irr_idx_q=np.asarray([0],np.int32),
        sym_idx_q=np.asarray([0],np.int32),q_irr_full_idx=np.asarray([0],np.int32),
        q_irr_kgrid_int=np.zeros((1,3),np.int32),trs_allowed=False,
        active_symmetry_rows=np.asarray([0],np.int32))
    basis=PackedCentroidBasis.build(points,sym,grid,mesh,coordinate_kind=kind)
    physical=points/grid if kind=='fft_indices' else points
    closure=verify_centroid_orbit_closure(physical,sym.sym_matrices,tnp=sym.translations)
    perm=np.tile(np.arange(2,dtype=np.int32),(2,1))
    tables=QirrTables(irr_idx_q=sym.irr_idx_q,sym_idx_q=sym.sym_idx_q,
        q_irr_frac=np.zeros((1,3)),sym_perm=perm,L_table=np.zeros((2,2,3),np.int64),
        n_sym_spatial=1)
    plan=SimpleNamespace(sym=sym,n_sym_spatial=1,fft_grid=grid,
        spatial_ops=sym.sym_matrices,translations=sym.translations,
        sym_perm=perm,L_table=tables.L_table)
    path=tmp_path/'complete_v.h5';expected={}
    with SlabIO(path,mode='w',mesh=mesh) as io:
        for i,pair in enumerate(UNIQUE_TILES):
            raw=np.asarray([[[2+i,.13+.07j], [.13-.07j,3+i]]],np.complex128)
            expected[pair]=raw
            name=tile_dataset_name(*pair)
            io.create_dataset(name,shape=raw.shape,dtype=np.complex128)
            io.write_slab(name,jax.device_put(raw,NamedSharding(mesh,P(None,'x','y'))))
            io.sync_writes()
    for pair in UNIQUE_TILES:
        stamp_qirr_tensor(path,tile_dataset_name(*pair),tables=tables,
                          closure_verdict=closure,n_rmu_logical=2)
    with h5py.File(path,'a') as f:
        for name,value in {'kgrid':np.asarray([1,1,1],np.int32),
                           'n_rmu_C':2,'n_rmu_T':2,'n_q_total':1}.items():
            f.create_dataset(name,data=value)
        _publish_unique_tile_inventory(f,filename=path,n_q_total=1,n_rmu_C=2,n_rmu_T=2)
    return path,mesh,basis,plan,expected


@pytest.mark.parametrize('kind',['fft_indices','fractional'])
def test_complete_reader_typed_coordinates_and_literal_payload(tmp_path,kind):
    from file_io.restart_bundle import BispinorVqReader
    from common.collectives import gather_to_host
    path,mesh,basis,plan,expected=_fixture(tmp_path,kind)
    with BispinorVqReader(path,mesh,mu_bases=(basis,basis),family_plans=(plan,plan)) as reader:
        for pair,raw in expected.items():
            np.testing.assert_array_equal(np.asarray(gather_to_host(reader.get_tile(*pair))),raw)
        np.testing.assert_array_equal(np.asarray(gather_to_host(reader.get_tile(2,1))),
                                     expected[1,2].conj().transpose(0,2,1))


@pytest.mark.parametrize('mutation',['moved_point','wrong_kind','wrong_hash'])
def test_fractional_reader_refuses_changed_metadata_before_collective_open(tmp_path,monkeypatch,mutation):
    import h5py
    from common.centroid_basis import PackedCentroidBasis
    from file_io import slab_io
    from file_io.restart_bundle import BispinorVqReader
    from gw.v_q_bispinor import UNIQUE_TILES,tile_dataset_name
    path,mesh,basis,plan,_=_fixture(tmp_path,'fractional')
    if mutation=='moved_point':
        points=basis.canonical_indices.copy();points[0,0]+=1e-3
        basis=PackedCentroidBasis.build(points,plan.sym,plan.fft_grid,mesh,coordinate_kind='fractional')
    elif mutation=='wrong_kind':
        basis=PackedCentroidBasis.build(basis.canonical_indices,plan.sym,plan.fft_grid,mesh,
                                       coordinate_kind='fft_indices')
    else:
        altered=tmp_path/'altered_hash.h5';shutil.copy2(path,altered);path=altered
        with h5py.File(path,'a') as f:
            for pair in UNIQUE_TILES:f[tile_dataset_name(*pair)].attrs['qirr_centroid_hash']='changed'
    def forbidden(*args,**kwargs):raise AssertionError('collective reader opened before coordinate refusal')
    monkeypatch.setattr(slab_io,'SlabIO',forbidden)
    with pytest.raises(ValueError,match='centroid set differs'):
        BispinorVqReader(path,mesh,mu_bases=(basis,basis),family_plans=(plan,plan))
