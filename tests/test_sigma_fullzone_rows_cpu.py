"""Full-q scalar stores and typed identity wedges share the public row owner."""
from dataclasses import replace
from pathlib import Path
import runpy
from types import SimpleNamespace

import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('side,ns,ordered,layout,stored', [
    (2, 1, False, 'axis', False), (2, 2, True, 'face', False),
    (3, 1, True, 'face', False), (3, 2, False, 'axis', False),
    (2, 4, True, 'face', False), (2, 4, True, 'face', True),
    (2, 2, True, 'axis', True), (2, 4, True, 'face', 2),
    (2, 2, True, 'axis', 2)])
def test_untyped_full_q_matches_typed_rows_and_literal_sum(monkeypatch, side, ns, ordered, layout, stored):
    """Same complete public Sigma, ragged carriers and several orbit windows."""
    from common.centroid_basis import PackedCentroidBasis
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.wavefunction_bundle import BandSlices, ParentGreenCarrier, Wavefunctions
    from gw.mpa import sigma as M
    from gw import subtile_stream as S
    from gw.ppm_tau_kernel import sigma_pass_price
    from runtime import tiles
    from symmetry_maps import SymMaps, QirrOperator
    from ffi import fft as F
    from lxkit import memory
    helpers = runpy.run_path(str(Path(__file__).with_name('test_tdagw_dynsig_sigma.py')))
    _WfnStub, _dense_sigma, KGRID = (helpers[n] for n in ('_WfnStub','_dense_sigma','KGRID'))

    if jax.device_count() < side * side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:side*side]).reshape(side, side), ('x', 'y'))
    nk, logical_mu, logical_bands, nqp = int(np.prod(KGRID)), 73 if stored else 27, 11, 4
    nb = ((logical_bands + side - 1) // side) * side
    kfrac = np.stack(np.unravel_index(np.arange(nk), KGRID), 1) / np.asarray(KGRID)
    sym = SymMaps(_WfnStub(kfrac))
    fft_grid = (73, 1, 1) if stored else (3, 3, 3)
    cidx = np.stack(np.unravel_index(np.arange(logical_mu), fft_grid), 1).astype(np.int32)
    basis = PackedCentroidBasis.build(cidx, sym, fft_grid, mesh)
    plan = build_centroid_k_unfold_plan(sym, cidx, fft_grid, mesh, nspinor=ns,
        parent_k_frac=kfrac, layout=basis.layout)
    rng = np.random.default_rng(41271 + side + ns)
    psi = (rng.normal(size=(nk,logical_bands,ns,logical_mu))
           + 1j*rng.normal(size=(nk,logical_bands,ns,logical_mu))) / np.sqrt(logical_mu*logical_bands)
    energy = np.tile(np.r_[np.linspace(-.8,-.5,3),np.linspace(.3,1.2,logical_bands-3)],(nk,1))
    occupation = np.zeros_like(energy); occupation[:,:3] = 1
    packed = basis.pack_host(psi, axis=3)
    padded = np.pad(packed, ((0,0),(0,nb-logical_bands),(0,0),(0,0)))
    e = np.pad(energy, ((0,0),(0,nb-logical_bands)), constant_values=3.)
    f = np.pad(occupation, ((0,0),(0,nb-logical_bands)))
    put = lambda x,s: jax.device_put(x, NamedSharding(mesh,s))
    nmu = put(padded,P(None,'x',None,'y'))
    mun = put(padded.transpose(0,2,3,1),P(None,None,'x','y'))
    enk,occ = put(e,P()),put(f,P())
    parent = ParentGreenCarrier(nmu,mun,enk,occ,plan)
    slices = BandSlices.from_band_edges(0,0,3,nqp,nb,b4_logical=logical_bands)
    wfns = Wavefunctions(enk,occ,slices,green_parent=parent)
    meta = SimpleNamespace(nkx=KGRID[0],nky=KGRID[1],nkz=KGRID[2],mu_basis=basis)
    u = rng.normal(size=(nk,logical_mu)) + 1j*rng.normal(size=(nk,logical_mu))
    B = .1*u[:,:,None]*u[:,None,:].conj()/logical_mu
    D = .08*B if ordered else np.zeros_like(B)
    Omega = np.broadcast_to((1.3 + .02*np.arange(nk))[:,None,None],B.shape).copy()
    field = lambda x: put(np.pad(x[None],((0,0),(0,0),(0,basis.n_canonical-logical_mu),
        (0,basis.n_canonical-logical_mu))),P(None,None,'x','y'))
    fields = field(Omega.astype(complex)),field(B),field(D) if ordered else None
    identity = replace(QirrOperator.whole_zone(jax.ShapeDtypeStruct(
        (nk,basis.n_packed,basis.n_packed),jnp.complex128)),values=None)
    omega = np.array([-.2,.25])
    E = np.broadcast_to(omega,(nk,nqp,len(omega)))
    reference = np.moveaxis(_dense_sigma(psi,energy,occupation,psi[:,:nqp],E,
        (Omega,B,D,np.ones_like(B,bool),np.zeros_like(B)),.08),0,1)
    monkeypatch.setattr(tiles,'TILE_BYTES',1024)
    monkeypatch.setattr(memory,'room',lambda **kw: 1<<40)
    if stored:
        # Exercise the native stored-output contract through the independent
        # reference implementation on a tiny, conservatively indivisible
        # input window. This changes neither values nor the literal oracle.
        from common import fft_helpers
        real_conv = fft_helpers.make_kconv_klead_unfold
        def bounded_conv(*args, **kwargs):
            conv = real_conv(*args, **kwargs)
            conv.stores_rows_in_kernel = True
            return conv
        monkeypatch.setattr(fft_helpers,'make_kconv_klead_unfold',bounded_conv)
        # Two unequal orbit spans give the final input window a nonzero
        # live-row offset, independently of its strided stored-output blocks.
        cuts = (3*(basis.n_packed//side)//5,) if stored == 2 else ()
        monkeypatch.setattr(S,'orbit_cuts',lambda *a,**kw: cuts)
    original = S.prepare_green_operands
    calls = []
    def prepare(operands,mesh,build,**kwargs):
        def inspect(chosen,abstract):
            context,programs,native = build(chosen,abstract)
            price = sigma_pass_price(context)
            calls.append((chosen,price['passes'],price['output_blocks']))
            return context,programs,native
        return original(operands,mesh,inspect,**kwargs,_force_layout=layout)
    monkeypatch.setattr(S,'prepare_green_operands',prepare)
    values=[]
    with F.xla_reference():
        for wedge in (None,identity):
            source=M.MemoryPoleSource(*fields,n_mu_logical=logical_mu,mesh_xy=mesh,
                provenance={'screening_diagrams':'w_rpa'},q_wedge=wedge)
            result=M.compute_sigma_c_mpa_omega_grid(wfns,source,meta,mesh,
                omega_grid_ry=omega,efermi_ry=0.,regularization_width_ry=.08,
                quadrature_eps=1e-6,omega_grid_step_ry=.05,pole_batch_size=1,
                odd_reference=False,print_fn=lambda *a,**k: None)
            value=np.asarray(result.sigma_c_kij)[...,:nqp,:nqp]
            assert np.isfinite(value).all()
            np.testing.assert_allclose(value,reference,rtol=3e-5,atol=1e-10)
            values.append(value)
            assert source.q_wedge is wedge
            assert source.ledger['q_storage']==('full' if wedge is None else 'ibz')
            assert not source.B.is_deleted()
    np.testing.assert_array_equal(values[0],values[1])
    assert len(calls)==2 and all(chosen==layout for chosen,_,_ in calls)
    if stored:
        assert all(passes==(2 if stored == 2 else 1) and blocks>1 for _,passes,blocks in calls)
    else:
        assert all(passes>1 and blocks==1 for _,passes,blocks in calls)


def test_stored_rows_live_mask_uses_stored_coordinates_with_complex_partner():
    """A nonzero stored block, padding and a complex antiunitary partner."""
    from common import fft_helpers
    from ffi import fft as F
    if jax.device_count() < 4:
        pytest.skip('requires four virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2,2),('x','y'))
    helpers = runpy.run_path(str(Path(__file__).with_name('test_minimax_response_rows_cpu.py')))
    grid,plan,_,_,_,_ = helpers['fixture'](mesh,4,True)
    nr,nk,mu,ns = plan.n_parent,plan.n_full,plan.layout.axis.n_padded,4
    rng = np.random.default_rng(620491)
    complex_array = lambda shape: (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(mu)
    g = complex_array((nr,mu,ns,mu,ns))
    gt = complex_array(g.shape)  # deliberately not conj(g)
    w = complex_array((nk,mu,mu))
    put = lambda x,s: jax.device_put(x,NamedSharding(mesh,s))
    fields = (put(g,P(None,'x',None,'y',None)),put(gt,P(None,'x',None,'y',None)),
              put(w,P(None,'x','y')))
    full=[]
    for k in range(nk):
        parent,op = int(plan.irr_idx[k]),int(plan.sym_idx[k])
        idx = plan.sym_perm[op]; good=idx>=0; idx=np.maximum(idx,0)
        anti=op>=plan.n_sym_spatial
        value=(gt if anti else g)[parent][idx][:,:,idx]
        phase=np.exp(2j*np.pi*(plan.L_table[op]@plan.k_parent_frac[parent]))
        if anti:phase=phase.conj()
        value=value*phase[:,None,None,None]*phase.conj()[None,None,:,None]
        value=np.where(good[:,None,None,None]&good[None,None,:,None],value,0)
        spin=plan.spin_action_full[k]
        full.append(np.einsum('ab,mbnc,dc->mand',spin,value,spin.conj()))
    real=np.fft.ifftn(np.asarray(full).reshape(grid+(mu,ns,mu,ns)),axes=(0,1,2),norm='ortho')
    product=real*w.reshape(grid+(mu,1,mu,1))
    expected=np.fft.fftn(product,axes=(0,1,2),norm='ortho').reshape(nk,mu,ns,mu,ns)
    expected=(-1/np.sqrt(nk))*expected.transpose(0,2,1,4,3)
    mx=mu//2
    # The last piece deliberately extends beyond the local input. Live rows
    # exclude the first/last valid pieces in the STORED output positions.
    block=(1,2,max(3,mx//2-1),3)
    source=fft_helpers.x_block_rows(block)
    live=np.array([1,len(source)-1],np.int32)
    stored=np.arange(len(source))
    valid=(source<mx)&(stored>=live[0])&(stored<live[1])
    expected=np.concatenate([np.where(valid[None,None,:,None,None],
        np.take(expected,x*mx+np.minimum(source,mx-1),axis=2),0) for x in range(2)],axis=2)
    with F.xla_reference():
        conv=fft_helpers.make_kconv_klead_unfold(mesh,grid,plan.unfold_load_tables(),
            store_rows=tuple(range(nk)),norm='ortho',mult=-1/np.sqrt(nk))
        assert not conv.stores_rows_in_kernel
        actual=np.asarray(conv(*fields,rows=block,live=put(live,P())))
    np.testing.assert_allclose(actual,expected,rtol=5e-12,atol=2e-11)
