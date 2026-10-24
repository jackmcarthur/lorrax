"""Explicit spin adaptation and interleaved optical/charge operator identities."""
import os
from pathlib import Path
import subprocess
import sys


def _spin_embedding(f):
    """Embed each scalar orbital into independent up/down spin orbitals."""
    import numpy as np
    spinor = dict(f, ns=2, nc=2*f['nc'], nv=2*f['nv'])
    for key in ('psi_c', 'psi_v'):
        x=f[key][:,:,0]
        out=np.zeros((f['nk'],2*x.shape[1],2,f['nmu']),complex)
        out[:,0::2,0]=x;out[:,1::2,1]=x
        spinor[key]=out
    for key in ('eps_c','eps_v'):
        spinor[key]=np.repeat(f[key],2,axis=1)
    # Columns are normalized S,T0,T+,T- in electron/hole spin-pair space.
    spin=np.array([[1,1,0,0],[0,0,1,0],[0,0,0,1],[1,-1,0,0]],float)
    spin[:,:2]/=np.sqrt(2)
    n=2*f['nc']*f['nv']*f['nk']
    projectors=[]
    for column in spin.T:
        u=np.zeros((4*n,n),complex)
        for b in range(2):
            for c in range(f['nc']):
                for v in range(f['nv']):
                    for k in range(f['nk']):
                        j=((b*f['nc']+c)*f['nv']+v)*f['nk']+k
                        for t,(sc,sv) in enumerate(((0,0),(0,1),(1,0),(1,1))):
                            if b:
                                sc,sv=sv,sc  # A=(v,c): physical spin legs reverse
                            i=((b*spinor['nc']+2*c+sc)*spinor['nv']+2*v+sv)*f['nk']+k
                            u[i,j]=column[t]
        projectors.append(u)
    return spinor,projectors


def _vertices(f,data,sh,mesh):
    """Execute the production two-leg seed and readout on unit columns."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from bse.bse_ring_comm import (build_realspace_random_transition_generator,
        build_density_snapshot_operator)
    from bse.bse_w_exact import build_probe_rhs,_project_screening_actions
    nc,nv=int(data['n_cond_pad']),int(data['n_val_pad'])
    mu=int(data['V_q0'].shape[0]);n=2*f['nc']*f['nv']*f['nk']
    gen=build_realspace_random_transition_generator(mesh,f['nk'],1,1,nc,nv)
    snapshot=build_density_snapshot_operator(mesh,f['nk'],1,1,scatter_nu_on_y=True,
                                              exchange_channel='singlet')
    rhs=np.asarray(build_probe_rhs(np.eye(mu),data,gen,sh))
    B=rhs[:,:,:f['nc'],:f['nv']].transpose(0,2,3,4,1).reshape(n,mu)[:,:f['nmu']]
    trial=np.zeros((2,n,nc,nv,f['nk']),complex)
    labels=np.indices((2,f['nc'],f['nv'],f['nk'])).reshape(4,-1)
    trial[tuple([labels[0],np.arange(n),*labels[1:]])]=1
    C=np.asarray(_project_screening_actions({'value':jax.device_put(jnp.asarray(trial),sh.X_full)},
        data,snapshot)['value'])[:f['nmu']]
    return B,C


def _check(p):
    from unittest.mock import patch
    import runpy
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from ffi.fft import xla_reference
    from bse.bse_ring_comm import build_bse_ring_matvec_full
    from bse.bse_stack_matvec import build_bse_stack_matvec,build_bse_stack_matvec_full
    from bse.bse_davidson_helpers import build_bse_exact_diagonal
    from bse.bse_feast import matvec_operands
    from bse.head_resolvent import build_head_operator
    from bse.w_ladder import build_ladder_resolvent
    jax.config.update('jax_enable_x64',True)
    h=runpy.run_path(str(Path(__file__).with_name('test_ladder_physical_branches_cpu.py')))
    f=h['fixture'](1);two,U=_spin_embedding(f)
    mesh=Mesh(np.array(jax.devices('cpu')[:p*p]).reshape(p,p),('x','y'))
    with xla_reference(),patch('common.contract_bands.bands_gemm_ffi_enabled',lambda:False):
        data,sh=h['_production_payload'](f,mesh)
        doubled,sh2=h['_production_payload'](two,mesh)
        for q in range(f['nk']):
            hs,dq=h['_materialize_physical'](f,data,sh,mesh,q,True)
            h2,dq2=h['_materialize_physical'](two,doubled,sh2,mesh,q,True)
            triplet=build_bse_ring_matvec_full(mesh,f['nk'],1,1,
                include_W=True,physical_branches=True,exchange_channel='triplet')
            ht,_=h['_materialize_physical'](f,data,sh,mesh,q,True,triplet)
            assert np.linalg.norm(hs-ht)>1e-3
            for i,u in enumerate(U):
                np.testing.assert_allclose(u.conj().T@h2@u,hs if i==0 else ht,atol=4e-13,rtol=3e-12)
                for v in U[i+1:]:
                    np.testing.assert_allclose(u.conj().T@h2@v,0,atol=4e-13)
            B,C=_vertices(f,dq,sh,mesh);B2,C2=_vertices(two,dq2,sh2,mesh)
            np.testing.assert_allclose(U[0].conj().T@B2,np.sqrt(2)*B,atol=4e-14,rtol=3e-12)
            np.testing.assert_allclose(C2@U[0],C/np.sqrt(2),atol=4e-14,rtol=3e-12)
            for u in U[1:]:
                np.testing.assert_allclose(u.conj().T@B2,0,atol=4e-14)
                np.testing.assert_allclose(C2@u,0,atol=4e-14)
            z=.31+.42j
            np.testing.assert_allclose(C@np.linalg.solve(z*np.eye(len(hs))-hs,B),
                C2@np.linalg.solve(z*np.eye(len(h2))-h2,B2),atol=4e-13,rtol=3e-12)

        # SAME arrays, S -> T -> S; no JAX/memo/head-cache clearing.
        nc,nv=int(data['n_cond_pad']),int(data['n_val_pad']);n=f['nc']*f['nv']*f['nk']
        trial=np.zeros((n,nc,nv,f['nk']),complex)
        cvk=np.indices((f['nc'],f['nv'],f['nk'])).reshape(3,-1)
        trial[tuple([np.arange(n),*cvk])]=1
        unit=jax.device_put(jnp.asarray(trial),sh.X)
        args=matvec_operands(data);w0=data['W_q'][:,:,0,0,0]
        diag_args=(data['eps_c'],data['eps_v'],data['psi_c_X'],data['psi_v_Y'],w0,data['M'],data['V_q0'],f['nk'])
        matrices={};diagonals={}
        for channel in ('singlet','triplet','singlet'):
            apply=build_bse_stack_matvec(mesh,f['nk'],1,1,exchange_channel=channel)
            out=np.asarray(apply(unit,*args))[:,:f['nc'],:f['nv']].reshape(n,n).T
            diag=build_bse_exact_diagonal(*diag_args,sharding=NamedSharding(mesh,P('x','y',None)),
                                          exchange_channel=channel)
            logical=np.asarray(diag)[:f['nc'],:f['nv']].reshape(-1)
            np.testing.assert_allclose(logical,np.diag(out).real,atol=4e-13,rtol=3e-12)
            if channel in matrices:
                np.testing.assert_array_equal(out,matrices[channel])
                np.testing.assert_array_equal(logical,diagonals[channel])
            matrices[channel]=out;diagonals[channel]=logical
            # The optical full wrapper must forward the same static choice.
            full=build_bse_stack_matvec_full(mesh,f['nk'],1,1,exchange_channel=channel)
            block=jnp.stack((unit,jnp.zeros_like(unit)))
            top=np.asarray(full(block,*args))[0,:,:f['nc'],:f['nv']].reshape(n,n).T
            np.testing.assert_allclose(top,out,atol=4e-13,rtol=3e-12)
        assert np.linalg.norm(matrices['singlet']-matrices['triplet'])>1e-3
        # Revisiting the same explicit S metadata may hit the one-slot memo.
        assert build_bse_exact_diagonal(*diag_args,sharding=NamedSharding(mesh,P('x','y',None)),
                                       exchange_channel='singlet') is diag

        borrowed=dict(data,exchange_channel='triplet')
        op=build_head_operator(mesh,borrowed,include_w=True)
        same=build_head_operator(mesh,dict(data,exchange_channel='singlet'),include_w=True)
        assert op.key==same.key and op.matvec is same.matvec
        charge,dq=h['_materialize_physical'](f,borrowed,sh,mesh,0,True,op.matvec)
        expected=h['dense'](f,0)['H']
        np.testing.assert_allclose(charge,expected,atol=4e-13,rtol=3e-12)
        head_diag=np.asarray(op.diag_h)[:,:, :f['nc'],:f['nv']].reshape(-1)
        np.testing.assert_allclose(head_diag,np.diag(charge),atol=4e-13,rtol=3e-12)
        fresh,diag,*_=build_ladder_resolvent(mesh,borrowed,include_w=True)
        charge,_=h['_materialize_physical'](f,borrowed,sh,mesh,0,True,fresh)
        np.testing.assert_allclose(charge,expected,atol=4e-13,rtol=3e-12)
        assert borrowed['exchange_channel']=='triplet'


def test_explicit_spin_blocks_and_interleaved_channel_ownership():
    env=dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    flags=[x for x in env.get('XLA_FLAGS','').split() if not x.startswith('--xla_force_host_platform_device_count')]
    env['XLA_FLAGS']=' '.join(flags+['--xla_force_host_platform_device_count=4'])
    root=Path(__file__).resolve().parents[1]
    env['PYTHONPATH']=str(root/'src')+os.pathsep+env.get('PYTHONPATH','')
    worker="import runtime,runpy,sys; from runtime.source_closure import ensure_source_closure; ensure_source_closure(); x=runpy.run_path(sys.argv[1]); [x['_check'](p) for p in (1,2)]; print('SPIN_CHANNELS_P1_P4_PASS')"
    result=subprocess.run([sys.executable,'-c',worker,str(Path(__file__).resolve())],
        cwd=root,env=env,text=True,capture_output=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'SPIN_CHANNELS_P1_P4_PASS' in result.stdout
