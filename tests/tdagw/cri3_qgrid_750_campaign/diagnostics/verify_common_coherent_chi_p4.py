"""Native P4 common coherent/equilibrium chi owner proof; software only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback


def literal(psi, energy, left, right, coordinates, grid, times, moment_power=None):
    """Independent transition-index sum, including spin and both density legs."""
    import numpy as np
    lm=np.einsum('kasr,kab->kbsr',psi,left)
    rm=np.einsum('kasr,kab->kbsr',psi,right)
    result=[]
    for q in coordinates:
        partner=np.ravel_multi_index(((coordinates-q)%grid).T,grid)
        amplitude=np.einsum('kbsr,kdsr->kbdr',lm,rm[partner].conj())
        vertex=np.einsum('kbsr,kdsr->kbdr',psi,psi[partner].conj())
        gap=energy[:,:,None]-energy[partner,None,:]
        result.append(np.array([np.einsum('kbdm,kbdn,kbd->mn',amplitude,vertex.conj(),
            np.exp(-gap*t) if moment_power is None else gap**moment_power)
            /np.sqrt(len(psi)) for t in times]))
    return np.asarray(result)


def response_stage(c):
    import jax
    import numpy as np
    from types import SimpleNamespace
    from jax.sharding import PartitionSpec as P
    from common.collectives import gather_to_host
    from gw.greens_function_kernel import density_left_endpoint
    from gw.gw_config import LorraxConfig
    from gw.ppm_sigma import fresh_shared_pole_capacity
    from gw.response_bank import _bank_execution, full_density_bare_moments
    from gw.w_isdf import full_density_response_stream, _get_chi_fractional_contour_kernel_face
    from verify_fejer_p4_38dbea1 import literal_crop

    mesh,grid,coarse=(c[k] for k in ('mesh','grid','coarse'))
    nk,nb,mu,ns=(c[k] for k in ('nk','nb','mu','ns'))
    times=np.array([0.,.13+.11j,-.09+.17j,.04-.07j])
    weights=np.zeros((2,8,4),complex)
    weights[0,:4]=np.eye(4);weights[1,4:]=np.eye(4)
    origin=.193
    center=np.zeros((nk,nb,nb),complex);center[:,1:3,1:3]=np.eye(2)
    filled=np.zeros_like(center);filled[:,0,0]=1
    empty=np.zeros_like(center);empty[:,3,3]=1
    rng=np.random.default_rng(423)
    u,_=np.linalg.qr(rng.normal(size=(nk,2,2))+1j*rng.normal(size=(nk,2,2)))
    rho=np.zeros_like(center);rho[:,0,0]=1
    rho[:,1:3,1:3]=(u*np.array([.21,.79])[None,None])@u.conj().swapaxes(-1,-2)
    endpoint=density_left_endpoint(mesh,c['pm'].shape)
    host=lambda a: np.asarray(gather_to_host(a))
    maximum=0.; calls=0; executions=[]; moment_error=0.
    for qgrid,window in ((grid,'sharp'),(coarse,'fejer')):
        nq=int(np.prod(qgrid))
        meta=SimpleNamespace(nkx=8,nky=8,nkz=1,nk_tot=nk,nq_tot=nq,nspinor=ns,n_rmu=mu,
                             qgrid=qgrid,qgrid_response_window=window)
        meta=fresh_shared_pole_capacity(meta,mesh)
        meta.shared_pole_capacity.at_phase()
        row=meta.shared_pole_capacity.reserve('software_carry',
            resident_bytes_per_rank=16*nq*8*mu*mu//mesh.size,workspace_bytes_per_rank=0)
        meta.shared_pole_capacity.live_stages=(row['stage'],)
        execution=dict(seconds={},memory=[],compiled=[])
        execute=_bank_execution(meta,mesh,execution,LorraxConfig())
        for current in range(2):
            energy=c['energy']+.03*current*np.sin(np.arange(nk*nb).reshape(nk,nb))
            ed=c['put'](energy,P())
            density=rho.copy()
            if current: density[:,1:3,1:3]=np.eye(2)-density[:,1:3,1:3]
            af=density*center; au=center-af
            pairs=dict(full=(np.eye(nb)[None]-density,density),FF=(empty,filled),
                       FA=(au,filled),AF=(empty,af),AA=(au,af))
            for name,(left,right) in pairs.items():
                support=[np.any(a!=0,axis=1) for a in (left,right)]
                bounds=tuple(v for mask in support for v in (energy[mask].min(),energy[mask].max()))
                bands=np.arange(nb)[None]
                ranges=np.array([[np.min(np.where(mask,bands,nb),axis=1),
                                  np.max(np.where(mask,bands+1,0),axis=1)] for mask in support])
                stream=full_density_response_stream(mesh,grid,(nk,nb,mu,ns),q_ids=range(nq),
                    n_outputs=8,k_unfold_plan=c['plan'],execute=execute,qgrid=qgrid,
                    response_window=window,rule_reference_ry=origin,pair_bounds_ry=bounds,
                    pair_band_ranges=ranges)
                actual=host(stream(c['put'](times,P()),c['put'](weights,P()),
                    endpoint(c['pm'],c['put'](left,P())),c['pn'],
                    endpoint(c['pm'],c['put'](right,P())),ed))
                forward=literal(c['psi'],energy,left,right,c['coords'],grid,times)
                reverse=literal(c['psi'],energy,right,left,c['coords'],grid,-times.conj())
                expected=np.concatenate([forward*np.exp(origin*times)[None,:,None,None],
                    reverse*np.exp(origin*times.conj())[None,:,None,None]],axis=1)
                if qgrid!=grid: expected=literal_crop(expected,grid,qgrid)
                np.testing.assert_allclose(actual,expected,rtol=4e-11,atol=3e-13,err_msg=name)
                maximum=max(maximum,float(np.max(abs(actual-expected))));calls+=1
            ph=endpoint(c['pm'],c['put'](density,P()))
            operands=(c['pm'],c['pn'],ph,ed)
            native=full_density_bare_moments(operands,meta,mesh_xy=mesh,q_ids=range(nq),
                                             k_unfold_plan=c['plan'],execute=execute)
            # Exact transition powers, not finite-frequency/finite-difference
            # moments or reverse(a,b) incorrectly treated as (a,b).
            identity=np.broadcast_to(np.eye(nb),density.shape)
            exact=tuple((literal(c['psi'],energy,identity,density,c['coords'],grid,[0.],p)
                         -literal(c['psi'],energy,density,identity,c['coords'],grid,[0.],p))[:,0]
                        for p in (0,1))
            for got,want in zip(native,exact):
                if qgrid!=grid: want=literal_crop(want,grid,qgrid)
                got=host(got)
                np.testing.assert_allclose(got,want,rtol=4e-11,atol=3e-13)
                moment_error=max(moment_error,float(np.max(abs(got-want))))
        executions.append(execution)
    # Typed-parent overflow witness: the deep RIGHT column is structurally
    # absent, so no exp(+2000*100) is allowed before applying its zero mask.
    energy=c['energy'].copy();energy[:,0]=-2000.
    keep=np.zeros_like(rho);keep[:,2,2]=1
    left=endpoint(c['pm'],c['put'](keep,P()))
    ranges=np.broadcast_to(np.array([2,3])[None,:,None],(2,2,nk)).copy()
    bounds=(energy[:,2].min(),energy[:,2].max())*2
    stream=full_density_response_stream(mesh,grid,(nk,nb,mu,ns),q_ids=range(16),n_outputs=1,
        k_unfold_plan=c['plan'],execute=execute,qgrid=coarse,response_window='fejer',
        pair_bounds_ry=bounds,pair_band_ranges=ranges)
    got=host(stream(c['put'](np.array([100.+.1j]),P()),c['put'](np.array([[[1.]],[[0.]]],complex),P()),
                    left,c['pn'],left,c['put'](energy,P())))
    assert np.isfinite(got).all()
    # Ordinary equilibrium graph: omitted endpoint flag is unchanged.
    options=dict(k_unfold_plan=c['plan'],selected_q=tuple(range(nk)),pair_mode='direct',ordered=True)
    a=_get_chi_fractional_contour_kernel_face(mesh,grid,1,(nk,nb,mu,ns),**options)
    b=_get_chi_fractional_contour_kernel_face(mesh,grid,1,(nk,nb,mu,ns),endpoint_pair=False,**options)
    args=(c['put'](times,P()),c['put'](weights[:,:1],P()),c['pm'],c['pn'],c['ed'],
          c['fd'],1-c['fd'],c['put'](np.array([0.,0.]),P()))
    np.testing.assert_array_equal(host(a(*args)).view(np.uint64),host(b(*args)).view(np.uint64))
    assert all(r['compiled'] and r['memory'] for r in executions)
    return dict(max_literal_error=maximum,moment_error=moment_error,current_inputs=2,
        paired_calls=calls,panels=['full','FF','FA','AF','AA'],typed_identity_parent=True,
        native_mode11=True,inactive_overflow_finite=True,default_vs_explicit_false_bits=True,
        execution=executions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--source-revision',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--max-wall-seconds',type=float,default=180.)
    args=parser.parse_args()
    if not 0<args.max_wall_seconds<=180 or os.environ.get('LEG_HEAD')!=args.source_revision:
        raise ValueError('exact source revision and <=180 second cap required')
    watchdog=threading.Timer(args.max_wall_seconds,lambda:os._exit(124))
    watchdog.daemon=True;watchdog.start()
    import gw.gw_jax as driver
    import runtime
    import jax
    import numpy as np
    import gw.w_isdf, gw.response_bank
    from common.collectives import rank0_transaction
    from ffi.fft import kconv_backend
    import verify_fejer_p4_38dbea1 as fixture_owner
    root=Path(args.source_root).resolve()
    modules=(driver,gw.w_isdf,gw.response_bank,fixture_owner)
    origins={m.__name__:str(Path(m.__file__).resolve()) for m in modules}
    if any(root not in Path(p).parents for p in origins.values()):
        raise ValueError('source module origin mismatch')
    if (jax.process_count()!=4 or jax.local_device_count()!=1 or driver.RUNTIME.mesh.devices.shape!=(2,2)
            or any(d.platform!='gpu' for d in jax.devices()) or not jax.config.x64_enabled
            or kconv_backend(driver.RUNTIME.mesh,(8,8,1))!='mathdx'):
        raise ValueError('requires native CUDA P4 2x2 FP64; no reference override')
    out=Path(args.out).resolve()
    rank0_transaction(out,stage='common_chi/new',write=lambda:out.mkdir(parents=True,exist_ok=False))
    started=time.monotonic()
    try:
        result=response_stage(fixture_owner.fixture(driver.RUNTIME.mesh))
    except BaseException:
        print(f'Common chi rank {jax.process_index()} failed:\n'+traceback.format_exc(),flush=True)
        raise
    report=dict(status='PASS_SOFTWARE_ONLY_NOT_MATERIAL',source_revision=args.source_revision,
        module_origins=origins,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_archive_sha256=os.environ.get('LEG_SOURCE_ARCHIVE_SHA256'),
        elapsed_seconds=time.monotonic()-started,max_wall_seconds=args.max_wall_seconds,
        active_GPUh_cap=4*args.max_wall_seconds/3600,result=result)
    def write():
        tmp=out/'receipt.tmp.json';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        os.replace(tmp,out/'receipt.json')
    rank0_transaction(out,stage='common_chi/pass',write=write)
    watchdog.cancel();runtime.finalize_process(0)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
