"""Admission happens before direct full-port verification allocates panels."""
import numpy as np
import pytest

from gw.transition_galerkin import transition_catalogue, transition_panel_capacity


def test_direct_panel_price_binds_source_and_retained_basis_without_a_rank_cut():
    energy=np.array([[0.,1.]])
    rho=np.array([np.diag([1.,0.])],complex)
    catalogue=transition_catalogue(energy,rho,np.array([0]),pair_tile=2)
    unknown=transition_panel_capacity(catalogue,978,retained_basis_rows=1000,
        max_block_size=32)
    assert unknown['working_bytes_per_rank'] is None
    assert not unknown['material_admission']
    assert not unknown['checked_Dyson_workspace_included']
    known=transition_panel_capacity(catalogue,978,retained_basis_rows=1000,
        max_block_size=32,source_resident_bytes_per_rank=4096)
    assert known['working_bytes_per_rank']==known['known_working_bytes_per_rank']+4096
    assert known['retained_basis_bytes_per_rank']==16*2*1000
    assert known['vertex_bytes_per_rank']==16*2*978
    assert known['full_ports']==978
    assert known['all_parent_retention_admitted'] is False


@pytest.mark.parametrize('source,rows',[(float('nan'),0),(-1,0),(0,-1),(0,1.5)])
def test_direct_panel_invalid_prices_refuse(source,rows):
    cat=transition_catalogue(np.array([[0.,1.]]),np.array([np.diag([1.,0.])]),np.array([0]),pair_tile=2)
    with pytest.raises(ValueError,match='transition_capacity'):
        transition_panel_capacity(cat,978,retained_basis_rows=rows,
            source_resident_bytes_per_rank=source)


def test_native_Pi_update_straddles_y_seam_without_matrix_collectives():
    import jax
    from jax.sharding import Mesh
    from gw.transition_galerkin import make_transition_bridge
    jax.config.update('jax_enable_x64',True)
    devices=jax.devices('cpu')
    if len(devices)<4:
        pytest.skip('four CPU devices exercise the native XY seam')
    mesh=Mesh(np.asarray(devices[:4]).reshape(2,2),('x','y'))
    bridge=make_transition_bridge(mesh,(8,2,1,6),2,normalization=1.,pair_tile=2)
    target=bridge['allocate_ports']()
    first=jax.device_put(np.arange(24).reshape(6,4).astype(complex),bridge['port_sharding'])
    last=jax.device_put((100+np.arange(12).reshape(6,2)).astype(complex),bridge['port_sharding'])
    start=jax.device_put(np.array(0),bridge['port_sharding'])
    compiled=bridge['update_ports'].lower(target,first,start).compile()
    hlo=compiled.as_text().lower()
    assert not any(word in hlo for word in ('all-gather','all_gather','all-to-all','collective-permute'))
    target=compiled(target,first,start)
    target=bridge['update_ports'](target,last,jax.device_put(np.array(4),bridge['port_sharding']))
    np.testing.assert_array_equal(np.asarray(target)[0],np.concatenate([np.asarray(first),np.asarray(last)],axis=1))
    assert all(shard.data.shape==(1,3,3) for shard in target.addressable_shards)


def test_direct_physical_Pi_and_existing_checked_Dyson_match_literal_full_rho():
    from types import SimpleNamespace
    import jax
    from gw import transition_galerkin as owner
    from gw.response_bank import response_algebra
    from gw.shared_pole_recipe import CapacityLedger
    from tests.test_native_transition_galerkin_bridge_cpu import _case,_mesh,_bridge
    from tests._full_microscopic_oracle import ordered_pr
    _,psi,energy,rho,partners=_case()
    mesh=_mesh();bridge,cat,t=_bridge(mesh,psi,energy,rho,partners[2])
    meta=SimpleNamespace(nk_tot=8,nspin=1,nspinor_wfnfile=2,nspinor=2,n_rmu=4)
    meta.shared_pole_capacity=CapacityLedger(meta,mesh_xy=mesh,device_budget_bytes=2**30)
    meta.shared_pole_capacity.live_stages=()
    config=SimpleNamespace(backend=SimpleNamespace(linalg='local'))
    dyson,_,_,algebra=response_algebra(meta,config,mesh_xy=mesh,n=4,ordered=True)
    rng=np.random.default_rng(817)
    root=.1*(np.eye(4)+.03*(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))))
    root=(root+root.conj().T)/2
    v=root@root
    for z in (.17+.23j,-.29+.18j):
        pi,receipt=owner.transition_polarization(bridge,t,cat,energy,rho,z,gamma=.047,
            max_block_size=3,source_resident_bytes_per_rank=4096)
        exact=ordered_pr(psi,energy,rho,partners[2],[z],.047,spin_weight=1.)[0]
        np.testing.assert_allclose(np.asarray(pi)[0],exact,rtol=4e-12,atol=4e-14)
        wc=owner.transition_dyson_value(pi,root,dyson,raw_prefactor=algebra['prefactor'])
        expected=np.linalg.solve(np.eye(4)-v@exact,v)-v
        np.testing.assert_allclose(np.asarray(wc)[0],expected,rtol=5e-11,atol=5e-14)
        assert receipt['physical_prefactor_in_T']
        assert receipt['extra_electronic_eta_ry']==0.
        assert receipt['panels']==2
        assert not receipt['material_admission']
    # The existing owner prefactor must be divided out, not applied twice.
    wrong=np.asarray(dyson.value(root[None],pi))[0]
    assert np.linalg.norm(wrong-np.asarray(wc)[0])>1e-6
    with pytest.raises(ValueError,match='transition_capacity.*source residency'):
        owner.transition_polarization(bridge,t,cat,energy,rho,.2+.3j,gamma=.047,
            max_device_bytes=2**30)
