"""Owned GN handoffs retire buffers; borrowed/observed sources keep their fields."""
from types import SimpleNamespace as NS
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest


def case(side, identity, odd):
    from common.centroid_basis import PackedCentroidBasis
    from common.grouped_layout import (build_square_grouped_shard_layout,
                                      identity_square_grouped_shard_layout)
    from runtime.padding import padded_mu_extent
    if jax.device_count() < side*side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.array(jax.devices()[:side*side]).reshape(side,side), ('x','y'))
    logical = 19
    canonical = padded_mu_extent(logical, mesh)
    layout = (identity_square_grouped_shard_layout(logical, canonical, (side,side))
              if identity else build_square_grouped_shard_layout(np.arange(logical)%5, (side,side)))
    basis = PackedCentroidBasis(mesh, layout, np.zeros((logical,3), np.int32), canonical)
    mu = basis.n_packed
    r = np.arange(mu)
    block = basis.active_mask[:,None] & basis.active_mask[None,:]
    wc = -.02/(1+(r[:,None]-r[None,:])**2)
    om = .6+(r[:,None]+r[None,:])/20
    wi = wc*om**2/(om**2+4)
    wi[:3,:3] = 2*wc[:3,:3]
    d = 1j*.001/(1+(r[:,None]-r[None,:])**2) if odd else 0
    v = .2*np.eye(mu)
    sh = NamedSharding(mesh,P(None,'x','y'))
    put = lambda a: jax.device_put(np.broadcast_to(np.where(block,a,0),(2,mu,mu)).astype(complex).copy(),sh)
    V, W0, Wp = put(v),put(v+wc),put(v+wi+d)
    return mesh,basis,V,W0,Wp


def fit(mesh,basis,V,W0,Wp,odd,consume=False):
    from gw.ppm_sigma import fit_ppm
    return fit_ppm(W0,Wp,V,2j,mesh,n_mu_logical=basis.n_logical,
        mu_active_mask=basis.active_mask,ordered_orientations=odd,
        coarsen_extreme_tails=False,consume_supports=consume)


@pytest.mark.parametrize('side,identity,odd',[(2,False,False),(2,True,True),(2,False,True),(3,False,True)])
def test_owned_and_borrowed_source_fields_and_static_readers(monkeypatch,side,identity,odd):
    from gw import ppm_sigma as S
    from gw.mpa import sigma as M
    source_calls=[]; static_calls=[]
    plan=NS(n_brackets=2,bounds=((0,2),(2,4)),counts=(2,4))
    monkeypatch.setattr(S,'_resolve_ppm_band_plan',lambda *a,**k:plan)

    def static(wfns,wc,invalid,meta,mesh,**kwargs):
        assert not wc.is_deleted()
        value=float(np.sum(np.abs(np.asarray(wc))*np.asarray(invalid)))
        static_calls.append(value)
        return np.array([value])
    def bracket(wfns,wc,invalid,meta,mesh,bounds,**kwargs):
        value=static(wfns,wc,invalid,meta,mesh,**kwargs)[0]
        return np.array([[value/3],[2*value/3]])
    monkeypatch.setattr(S,'_compute_invalid_static_sigma',static)
    monkeypatch.setattr(S,'_invalid_static_coh_by_bracket',bracket)
    monkeypatch.setattr(S,'_add_static_ppm_term',lambda body,term,*a:body+term.sum())

    def executor(wfns,poles,meta,mesh,**kwargs):
        fields=tuple(None if a is None else np.asarray(a).copy()
                     for a in (poles.Omega,poles.B,poles.B_odd))
        source_calls.append(fields)
        # Independent one-pole resolvent, sensitive to masked frequencies,
        # both ordered residues, canonical permutation and exact pad zeros.
        o,b,d=fields; d=0 if d is None else d
        z=.2+.7j
        body=np.array([np.sum((b+d)/(z-o)+(b-d)/(z+o))])
        return NS(sigma_c_kij=body,band_axis=None,sigma_c_odd_kij=None,
                  band_counts=(2,4),omega_ry=np.array([.2]),omega_ev=np.array([.2]),branches=None)
    monkeypatch.setattr(M,'compute_sigma_c_mpa_omega_grid',executor)
    results=[]
    for mode in ('zero','2ry','static_limit'):
        pair=[]
        for consume in (False,True):
            mesh,basis,V,W0,Wp=case(side,identity,odd)
            beforeV=np.asarray(V).copy()
            ppm=fit(mesh,basis,V,W0,Wp,odd,consume)
            model=tuple(None if a is None else np.asarray(a).copy() for a in
                        (ppm.Omega_q,ppm.B_q,ppm.valid_mask_q,ppm.B_odd_q,ppm.Wc0_q))
            wfns=NS(slices=NS(full=slice(0,4)),enk=np.array([[-1,-.6,.5,1],[-.9,-.5,.6,1.1]]),
                    occ=np.array([[1,1,0,0],[1,1,0,0]]))
            meta=NS(nk_tot=2,n_rmu=basis.n_logical,mu_basis=basis)
            out=S.compute_sigma_c_ppm_omega_grid(wfns,ppm,meta,mesh,
                ppm_cfg=NS(invalid_mode=mode),sigma_cfg=NS(regularization_ev=.1,
                    window_edge_factor=1.,quadrature_eps=1e-4,omega_step_ev=.1),
                mpa_cfg=NS(pole_batch_size=1),ansatz='gn_ppm',screening_diagrams='w_rpa',
                omega_grid_ry=np.array([.2]),plan=plan,consume_ppm=consume)
            np.testing.assert_array_equal(np.asarray(V),beforeV)
            inv=basis.layout.axis.canonical_to_packed
            o,b,valid,d,_=model
            live=(o>1e-14)&(valid|(mode=='2ry'))
            for got,raw in zip(source_calls[-1],(o,b,d)):
                if raw is None:
                    assert got is None
                    continue
                canonical=np.where(live,raw,0)[:,inv][:,:,inv]
                pad=basis.n_canonical-basis.n_logical
                expected=np.pad(canonical,((0,0),(0,pad),(0,pad)))[None]
                np.testing.assert_array_equal(got,expected)
            if consume:
                assert W0.is_deleted() and Wp.is_deleted()
                assert all(a is None or a.is_deleted() for a in
                           (ppm.Omega_q,ppm.B_q,ppm.valid_mask_q,ppm.B_odd_q,ppm.Wc0_q))
            else:
                assert not W0.is_deleted() and not Wp.is_deleted()
                for a,before in zip((ppm.Omega_q,ppm.B_q,ppm.valid_mask_q,ppm.B_odd_q,ppm.Wc0_q),model):
                    if a is not None:np.testing.assert_array_equal(np.asarray(a),before)
            pair.append(out)
        np.testing.assert_array_equal(pair[0].sigma_c_kij,pair[1].sigma_c_kij)
        if mode=='static_limit':
            np.testing.assert_array_equal(pair[0].static_coh_at_counts,pair[1].static_coh_at_counts)
            assert len(static_calls)==4 and static_calls[0]>0
        results.append(pair)


def test_owned_supports_refuse_aliases_before_touching_borrowed_v():
    from gw.ppm_sigma import _require_owned_ppm_supports
    mesh,basis,V,W0,Wp=case(2,False,True)
    for a,b,v in ((W0,W0,V),(W0,W0.reshape(W0.shape),V),(V,Wp,V),(W0,V.reshape(V.shape),V)):
        with pytest.raises(ValueError,match='ppm_support_ownership'):
            _require_owned_ppm_supports(a,b,v)
        assert all(not x.is_deleted() for x in (V,W0,Wp))


def test_real_pipeline_observers_disable_support_and_model_consumption(monkeypatch):
    from gw import ppm_pipeline as PP
    from gw.ppm_sigma import SigmaOmegaResult
    seen=[]
    plan=NS(enabled=False,n_brackets=1,bounds=((0,4),),counts=(4,))
    monkeypatch.setattr(PP,'plan_sigma_band_brackets',lambda *a,**k:plan)
    monkeypatch.setattr(PP,'band_count_point',lambda a,i:a[i])
    monkeypatch.setattr(PP,'_fit_head_correction',lambda *a,**k:None)
    monkeypatch.setattr(PP,'_compute_analytic_head_diag',lambda *a,**k:None)
    def executor(wfns,ppm,meta,mesh,**kwargs):
        seen.append(kwargs['consume_ppm'])
        assert not ppm.B_q.is_deleted() and not ppm.Wc0_q.is_deleted()
        return SigmaOmegaResult(np.zeros((1,1,1,1)),band_axis=None,efermi_ry=0.)
    monkeypatch.setattr(PP,'compute_sigma_c_ppm_omega_grid',executor)
    for observing in (None,False,True):
        mesh,basis,V,W0,Wp=case(2,False,True)
        meta=NS(kgrid=(2,1,1),n_rmu=basis.n_logical,mu_solve_extent=basis.n_packed,
                mu_active_mask=basis.active_mask)
        config=NS(do_screened=True,compute_mode=NS(ppm_model='gn',value='gn_ppm'),
            ppm=NS(omega_p=2.,fallback_omega=2.),
            debug=NS(sigma_omega_grid_debug=True,sigma_freq_debug_output=False),
            omega_grid_ry=np.array([.2]),sigma=None,mpa=None,screening=NS(diagrams='w_rpa'))
        import contextlib
        context=contextlib.nullcontext(None) if observing is None else PP.observe_window_sigma_source(screening=observing)
        with context as published:
            PP.compute_ppm_sigma_pipeline(wfns=NS(),V_q=V,W_static_q=W0,W_probe_q=Wp,
                quad=NS(node_count=3),config=config,meta=meta,mesh_xy=mesh,
                head_resolver=None,band_slices=None,wfn=NS(num_electrons=2),
                sym=NS(trs_allowed=False),consume_screening=True,print_fn=lambda *a:None)
        assert seen[-1] == (observing is None)
        assert W0.is_deleted() == (observing is None)
        assert Wp.is_deleted() == (observing is None)
        assert not V.is_deleted()
        if observing is not None:
            assert not published['poles'].B_q.is_deleted()
            np.testing.assert_array_equal(np.asarray(published['poles'].Wc0_q),np.asarray(W0-V))
            assert ('W_supports' in published)==observing


def test_owned_single_pole_axis_has_no_second_field_allocation():
    from gw.ppm_sigma import _ppm_single_pole_program
    mesh=Mesh(np.array(jax.devices()[:4]).reshape(2,2),('x','y'))
    sh=NamedSharding(mesh,P(None,'x','y'))
    host=np.arange(2*20*20).reshape(2,20,20).astype(complex)
    field=jax.device_put(host,sh)
    program=_ppm_single_pole_program()
    compiled=program.lower(field).compile()
    ma=compiled.memory_analysis()
    assert ma.alias_size_in_bytes==ma.output_size_in_bytes
    result=compiled(field)
    result.block_until_ready()
    assert field.is_deleted()
    np.testing.assert_array_equal(np.asarray(result),host[None])


@pytest.mark.parametrize('side,identity',[(2,False),(2,True),(3,False)])
def test_final_source_read_retires_canonical_fields_after_census(monkeypatch,side,identity):
    from gw.mpa import sigma as M
    mesh,basis,*_=case(side,identity,True)
    sh=NamedSharding(mesh,P(None,None,'x','y'))
    mu=basis.n_canonical
    host=(1+np.arange(2*mu*mu).reshape(1,2,mu,mu)/10000).astype(complex)
    p2c=basis.layout.axis.packed_to_canonical
    take=np.maximum(p2c,0)
    mask=(p2c>=0)[:,None]&(p2c>=0)[None,:]
    outputs=[]
    for consume in (False,True):
        arrays=[jax.device_put((host*v).copy(),sh) for v in (1,-.1,.01)]
        source=M.MemoryPoleSource(*arrays,n_mu_logical=basis.n_logical,
            mesh_xy=mesh,provenance={},consume=True)
        census=tuple(np.asarray(a).copy() for a in source.read(return_sharded=True,include_odd=True))
        for got in census:
            assert np.all(got[...,basis.n_logical:,:]==0)
        # Model the real reader generator: it keeps the canonical fields
        # while yielded. The pack transfer must retire those exact buffers.
        def execute(wfns,batches,n_poles,plan,request,meta,mesh,**kwargs):
            result=[]
            for lo,o,b,d in batches:
                packed=M._pack_pole_fields(meta.mu_basis,o,b,d,consume=kwargs['consume_poles'])
                for a,can in zip(packed,census):
                    expected=np.where(mask,can[:,:,take][:,:,:,take],0)
                    np.testing.assert_array_equal(np.asarray(a),expected)
                result.append(tuple(np.asarray(a).copy() for a in packed))
            return result
        monkeypatch.setattr(M,'_integrate_sigma_batches',execute)
        result=M.integrate_sigma_store(None,source,1,None,None,NS(mu_basis=basis),mesh,
                                       consume_source=consume)
        if consume and not identity:
            assert all(a.is_deleted() for a in (source.Omega,source.B,source.B_odd))
        else:
            for a,can in zip(source.read(return_sharded=True,include_odd=True),census):
                np.testing.assert_array_equal(np.asarray(a),can)
        outputs.append(result)
    for a,b in zip(outputs[0][0],outputs[1][0]):np.testing.assert_array_equal(a,b)

    # The generic batch mover also preserves every distinct leading batch,
    # and chunks the flattened pole*q axis INSIDE its jit.
    block=np.concatenate([host,2*host],axis=0)
    block[...,basis.n_logical:,:]=0
    block[...,:,basis.n_logical:]=0
    field=jax.device_put(block.copy(),sh)
    packed=basis._operator_kernel(P(None,None,'x','y'),False,
                                  q_tile_bytes=1,consume=True)(field)
    expected=np.where(mask,block[:,:,take][:,:,:,take],0)
    np.testing.assert_array_equal(np.asarray(packed),expected)


def test_final_read_contract_refuses_reread_and_nonmemory_owners():
    from gw.mpa import sigma as M
    mesh,basis,*_=case(2,False,True)
    sh=NamedSharding(mesh,P(None,None,'x','y'))
    make=lambda npoles:M.MemoryPoleSource(*[jax.device_put(np.full((npoles,2,basis.n_canonical,basis.n_canonical),v,complex),sh)
        for v in (1,-.1,.01)],n_mu_logical=basis.n_logical,mesh_xy=mesh,provenance={})
    for source,odd in ((make(1),True),(make(2),False),('store.h5',False)):
        with pytest.raises(ValueError,match='sigma_source_final_read'):
            M.compute_sigma_c_mpa_omega_grid(None,source,None,mesh,efermi_ry=0.,
                regularization_width_ry=.01,quadrature_eps=1e-4,omega_grid_step_ry=.1,
                omega_grid_ry=np.array([.2]),consume_source=True,odd_reference=odd)
    with pytest.raises(ValueError,match='sigma_source_final_read'):
        M.integrate_sigma_store(None,make(2),2,None,None,None,mesh,consume_source=True)
