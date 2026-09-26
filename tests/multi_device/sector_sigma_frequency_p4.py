"""Production sector frequency Sigma against a tiny explicit spectral oracle.

Unequal C/T centroid extents, independent CC/TT/CT poles, nonreciprocal q,
fractional occupations and nonzero W_infinity-V. The direct band/q/pole
sum is confined to this small harness. ``resident`` hands Sigma the same
models as device-resident ResidentSectorModel objects; main() requires that
Sigma to equal the file route's bit for bit.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import sys
import time


def check(mesh, root, layout, resident=False):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.centroid_basis import PackedCentroidBasis
    from common.units import RYD_TO_EV
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.wavefunction_bundle import BandSlices, Wavefunctions, ParentGreenCarrier
    from common.wfn_layout import psi_specs
    from gw.shared_pole_recipe import CapacityLedger
    from gw.photon_layout import PhotonBasisLayout, pack_photon_operator
    from gw.mpa.sector_sigma import compute_sector_sigma
    import gw.mpa.sector_sigma as production
    assert Path(production.__file__).resolve().parents[2] == Path(__file__).resolve().parents[2]/'src'
    print('sector Sigma source:', production.__file__,flush=True)
    from file_io import shared_pole_store as store
    from symmetry_maps import SymMaps, QirrTables, centroid_source_map_and_wrap
    assert mesh.size == 4
    rng=np.random.default_rng(1717)
    nmu_spec,mun_spec=psi_specs(layout)
    nk,nb,ns=4,4,4
    grid=(nk,1,1); fft=(8,1,1)
    k=np.array(list(np.ndindex(grid)))/np.array(grid)
    sym=SymMaps(SimpleNamespace(kpoints=k,kgrid=np.array(grid),shift=np.zeros(3),nkpts=nk,
        ntran=1,sym_matrices=np.eye(3,dtype=int)[None],translations=np.zeros((1,3)),
        avec=np.eye(3),atom_types=np.array([1]),atom_crys=np.zeros((1,3)),trs_holds=False))
    energy=np.linspace(.1,.8,nk*nb).reshape(nk,nb)
    f=1/(1+np.exp((energy-.45)/.15)); eta=.15
    occ=SimpleNamespace(f_kn=f,mu_ry=.45,n_electrons=f.sum()/nk)
    slices=BandSlices.from_band_edges(0,0,0,nb,nb)
    def put(value,spec=P()):
        value=np.asarray(value)
        return jax.make_array_from_callback(value.shape,NamedSharding(mesh,spec),lambda ix:value[ix])
    bases=[];families=[];bare=[];tables=[]
    for mu in (4,8):
        points=np.column_stack((np.arange(mu),np.zeros((mu,2),int)))
        basis=PackedCentroidBasis.build(points,sym,fft,mesh)
        plan=build_centroid_k_unfold_plan(sym,points,fft,mesh,nspinor=ns,parent_k_frac=k,layout=basis.layout)
        psi=.12*(rng.normal(size=(nk,nb,ns,mu))+1j*rng.normal(size=(nk,nb,ns,mu)))
        packed=basis.pack_host(psi,axis=3)
        en,oc=put(energy),put(f)
        families.append(Wavefunctions(enk=en,occ=oc,slices=slices,layout=layout,
            green_parent=ParentGreenCarrier(put(packed,nmu_spec),
                put(packed.transpose(0,2,3,1),mun_spec),en,oc,plan)))
        perm,wraps=centroid_source_map_and_wrap(points,sym.sym_matrices,sym.translations,
                                              np.array(fft),extend_trs=True)
        qt=QirrTables(irr_idx_q=np.arange(nk,dtype=np.int32),sym_idx_q=np.zeros(nk,np.int32),
            q_irr_frac=k,sym_perm=perm,L_table=wraps,n_sym_spatial=1)
        tables.append(dict(qirr=qt,q_irr_full_idx=np.arange(nk,dtype=np.int64),sym=sym))
        bases.append(basis);bare.append(psi)
    meta=SimpleNamespace(mu_basis=bases[0],nspin=1,nspinor=4,nkx=nk,nky=1,nkz=1,kgrid=grid,
        fft_grid=fft,nk_tot=nk,n_rmu=4,nb_sigma=nb,nelec=2,b_id_0=0,b_id_3=nb,b_id_4_user=nb,
        cell_volume=1.,shared_pole_recipe=dict(eta_ev=eta*RYD_TO_EV,sigma_tolerance=1e-4))
    meta.shared_pole_capacity=CapacityLedger(meta,mesh_xy=mesh,device_budget_bytes=1<<30)
    meta.shared_pole_capacity.reserve('fixture.wavefunctions',resident_bytes_per_rank=
        sum(2*x.nbytes//mesh.size for x in bare),workspace_bytes_per_rank=0)
    meta.shared_pole_capacity.live_stages=('fixture.wavefunctions',)
    identity={key:'planted-'+key for key in store._IDENTITY_KEYS}
    recipe=dict(version='shared_real_pole_v1_r3b',gate_version='shared_real_pole_gates_v1_r3b')
    factors={}; poles={}; models={}
    # CC tiles the 2x2 mesh; TT and CT deliberately have odd physical K.
    for name,which,om in (('CC',0,[1.0,1.4]),('TT',1,[1.2,1.6,2.0]),
                         ('CT_C',0,[1.1,1.5,1.9]),('CT_T',1,[1.1,1.5,1.9])):
        nc=3 if which else 1;mu=bases[which].n_logical
        count=len(om)
        factor=.2*(rng.normal(size=(nk,mu,nc,count))+1j*rng.normal(size=(nk,mu,nc,count)))
        # q-specific pole offsets discriminate the ordered -q pole census.
        p=np.broadcast_to(om,(nk,count))+np.arange(nk)[:,None]*.02
        carrier=(count+1)//2*2
        stored_factor=np.pad(factor,((0,0),(0,0),(0,0),(0,carrier-count)))
        stored_p=np.pad(p*p,((0,0),(0,carrier-count)),constant_values=1.0)
        path=(store.ResidentSectorModel(mesh,label=str(root/f'{name}.h5')) if resident
              else root/f'{name}.h5')
        header=store.write_shared_pole_model(path,
            put(bases[which].pack_host(stored_factor,axis=1),P(None,'x',None,'y')),
            put(stored_p),np.full(nk,count,np.int64),q_span=(0,nk),meta=meta,
            tables=tables[which],recipe=recipe,receipts=dict(identity=identity),
            sector=name,ordered=True,basis=bases[which])
        models[name]=(path,header);factors[name]=factor;poles[name]=p
    photon_layout=PhotonBasisLayout.from_centroid_extents(4,8,mesh)
    constant={}
    for A in range(4):
        for B in range(4):
            ma,mb=(bases[bool(i)].n_logical for i in (A,B))
            constant[A,B]=.02*(rng.normal(size=(nk,ma,mb))+1j*rng.normal(size=(nk,ma,mb)))
    # Hermitian exchange operator at each q, while q and -q remain different.
    for A in range(4):
        for B in range(A,4):
            value=(constant[A,B]+constant[B,A].conj().swapaxes(-1,-2))/2
            constant[A,B]=value;constant[B,A]=value.conj().swapaxes(-1,-2)
    packed=pack_photon_operator(lambda A,B:put(constant[A,B],P(None,'x','y')),nk,photon_layout,mesh)
    bank=root/'bank.h5'
    bank_recipe=dict(recipe,role_codes=dict(line=0,imaginary=1,infinity=2,held_line=3,held_imaginary=4),
        z_ry=np.array([.3+.2j,.7+.4j]),role=np.array([0,3],np.int8),
        distinct_id=np.array([0,1],np.int64),held=np.array([False,True]),
        support_pair=np.array([[-1,-1],[0,1]],np.int64),fit_ids=np.array([0],np.int64),held_ids=np.array([1],np.int64))
    store.initialize_shared_pole_bank(bank,meta=meta,tables=tables[0],recipe=bank_recipe,
        identity=identity,mesh_xy=mesh,photon_layout=photon_layout,mu_bases=tuple(bases))
    # Sigma reads only the bank's constant; the line sample (id 0) stores empty panels.
    zero=jnp.zeros_like(packed);samples=zero[:,None]
    panel=lambda rows,fields:put(np.zeros((nk,fields,rows,2),np.complex128),P(None,None,'x','y'))
    rows=dict(C=bases[0].n_packed,T=3*bases[1].n_packed)
    store.write_shared_pole_bank(bank,q_span=(0,nk),line=dict(sample=0,
        panels={f:panel(rows[f],9) for f in rows},cross={f:panel(rows['T' if f=='C' else 'C'],8) for f in rows},
        counts={f:np.zeros(nk,np.int64) for f in rows}),meta=meta,expected_identity=identity,mesh_xy=mesh)
    store.write_shared_pole_bank(bank,q_span=(0,nk),sample_span=(1,2),Wc=samples,dWc_ds=samples,
        M0=zero,M1=zero,M2=zero,M3=zero,constant=packed,meta=meta,expected_identity=identity,mesh_xy=mesh)
    handle=store.write_shared_pole_sector_manifest(root/'manifest.json',models=models,
        bank=dict(path=bank),identity=identity,receipts=dict(scope='synthetic oracle'),mesh_xy=mesh)
    omega=np.array([-.3,.0,.35])
    started=time.perf_counter()
    result=compute_sector_sigma(handle,tuple(families),tuple(bases),meta,mesh,
        omega_grid_ry=omega,efermi_ry=occ.mu_ry,occupation_state=occ,
        regularization_width_ry=eta,quadrature_eps=1e-4,quadrature_cache_dir=str(root/'rules'),
        omega_grid_step_ry=.3,print_fn=print)
    result.sigma_c_kij.block_until_ready()
    consumer_wall_s=time.perf_counter()-started
    # The four sector calls share one Sigma-rule request scope (union census).
    assert len([p for p in (root/'rules').iterdir() if p.name.startswith('request_')])==1
    pauli=(np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1]))
    gamma=[np.eye(4)]+[np.block([[np.zeros((2,2)),a],[a,np.zeros((2,2))]]) for a in pauli]
    expected=np.zeros((len(omega),nk,nb,nb),complex)
    instantaneous=np.zeros((nk,nb,nb),complex)
    halves=[np.zeros_like(expected),np.zeros_like(expected)]
    for ik in range(nk):
        for q in range(nk):
            km=(ik-q)%nk;qm=(-q)%nk
            for A in range(4):
                for B in range(4):
                    va=np.einsum('ism,st,ltm->ilm',bare[bool(A)][ik].conj(),gamma[A],bare[bool(A)][km])
                    vb=np.einsum('lsn,st,jtn->ljn',bare[bool(B)][km].conj(),gamma[B],bare[bool(B)][ik])
                    instantaneous[ik]-=np.einsum('ilm,l,mn,ljn->ij',va,f[km],constant[A,B][q],vb)/nk
                    names=(('CC','CC') if A==B==0 else ('TT','TT') if A and B else
                           ('CT_T','CT_C') if A else ('CT_C','CT_T'))
                    ca,cb=(factors[name] for name in names)
                    for h in (0,1):
                        qq=qm if h else q
                        l=ca[qq,:,A-1 if A else 0,:];r=cb[qq,:,B-1 if B else 0,:]
                        residue=np.einsum('mp,np->pmn',l.conj() if h else l,r if h else r.conj())
                        om=poles[names[0]][qq]
                        for iw,w in enumerate(omega):
                            denom=(w-(energy[km]-.45))[:,None]+(om[None,:]-1j*eta if h else -om[None,:]+1j*eta)
                            coeff=(f[km] if h else 1-f[km])[:,None]/(2*om[None,:]*denom)
                            halves[h][iw,ik]+=np.einsum('ilm,lp,pmn,ljn->ij',va,coeff,residue,vb)/nk
    expected=halves[0]+halves[1]+instantaneous[None]
    from jax.experimental.multihost_utils import process_allgather
    got=np.asarray(process_allgather(result.sigma_c_kij,tiled=True))[:,:,:nb,:nb]
    error=float(np.max(abs(got-expected)))
    scale=float(np.max(abs(expected)))
    assert error < 3e-4*scale+1e-9,(error,scale)
    if resident:
        assert all(model[0].header_json is None for model in models.values()),'Sigma releases resident models'
    return got,dict(status='PASS',layout=layout,resident=resident,consumer_wall_s=consumer_wall_s,
        max_absolute_error_ry=error,reference_max_ry=scale,
        constant_max_ry=float(abs(instantaneous).max()),
        frequency_half_max_ry=[float(abs(x).max()) for x in halves],
        capacity_estimates=[{key:row[key] for key in
            ('stage','status','aggregate_bytes_per_rank','resident_bytes_per_rank','workspace_bytes_per_rank')}
            for row in meta.shared_pole_capacity.entries if row['stage'].startswith('sigma.sector')],
        scope='production sector Sigma entry, frequency integrated, fractional f, nonzero constant, unequal centroid families; no real material')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layouts',choices=('face','axis','both'),default='face');args=p.parse_args()
    from runtime import initialize_communicator_stack,finalize_process
    initialize_communicator_stack()
    from common.collectives import resolve_mesh
    import jax
    mesh=resolve_mesh()
    assert jax.process_count() == 4
    layouts=('face','axis') if args.layouts=='both' else (args.layouts,)
    results={}
    for layout in layouts:
        root=args.output.parent/layout if len(layouts)>1 else args.output.parent
        (root/'resident').mkdir(parents=True,exist_ok=True)
        got,results[layout]=check(mesh,root,layout)
        held,row=check(mesh,root/'resident',layout,resident=True)
        import numpy as np
        assert np.array_equal(got,held),'resident sector models must give the file route Sigma bit for bit'
        results[layout+'_resident']=dict(row,bitwise_equal_to_files=True)
    result=dict(status='PASS',layouts=results,job=os.environ.get('SLURM_JOB_ID'),
                step=os.environ.get('SLURM_STEP_ID'))
    if jax.process_index()==0:
        encoded=json.dumps(result,indent=2,default=str)+'\n'
        args.output.write_text(encoded)
        print(encoded,flush=True)
    finalize_process()

if __name__=='__main__':main()
