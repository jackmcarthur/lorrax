"""Q-local held/passivity diagnostics against independent dense identities."""
from pathlib import Path
import json,os
import numpy as np


def check(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la
    from gw.shared_pole_local import local_model_checks
    b,n,k,s=4,8,8,2
    rng=np.random.default_rng(81)
    c=(rng.normal(size=(b,n,k))+1j*rng.normal(size=(b,n,k)))*.01
    poles=np.broadcast_to(np.arange(1,k+1,dtype=float),(b,k)).copy()
    mask=np.ones((b,k),bool);mask[1,-2:]=False;c[1,:,-2:]=0;poles[1,-2:]=1
    supports=np.array([-1.,2+3j]);eta=.25
    w=np.where(mask[:,None,:],1/(supports[None,:,None]-poles[:,None,:]),0)
    wc=(c[:,None]*w[:,:,None,:]) @ c.conj().swapaxes(-1,-2)[:,None]
    dw=(c[:,None]*(-w*w)[:,:,None,:]) @ c.conj().swapaxes(-1,-2)[:,None]
    inv=np.broadcast_to(np.eye(n,dtype=complex),(b,n,n)).copy()
    def put(a,spec):
        sh=NamedSharding(mesh,spec)
        return jax.make_array_from_callback(a.shape,sh,lambda i:a[i])
    fn=local_model_checks(mesh,distrib_la.plan('eigh',mesh,n=n,backend='off',batched_route='batch_reshard').native_fn)
    args=(tuple(put(a,P(None,'x','y') if i==0 else P()) for i,a in enumerate((c,poles,mask))),put(inv,P(None,'x','y')),put(wc,P(None,None,'x','y')),put(dw,P(None,None,'x','y')),put(supports,P()),put(np.asarray(eta),P()))
    passive,errors,reciprocity=fn(*args)
    assert np.all(np.asarray(reciprocity['passed']))
    assert not np.any(np.asarray(reciprocity['applicable']))
    assert float(jnp.max(errors))<1e-12
    expected=np.linalg.eigvalsh((c/(poles+eta**2)[:,None,:]) @ c.conj().swapaxes(-1,-2))
    np.testing.assert_allclose(np.asarray(passive['passivity_max']),expected[:,-1],rtol=1e-12,atol=1e-15)
    assert np.all(np.asarray(passive['passivity']))
    changed=list(args);changed[2]=args[2]*1.01
    _,red,_=fn(*changed)
    np.testing.assert_allclose(np.asarray(red)[:,0],.01/1.01,rtol=1e-12)
    return dict(status='PASS',max_held_relative=float(jnp.max(errors)),scope='P4 q-local held W/dW, passivity dense identity, changed-input red')

if __name__=='__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    def main():
        import sys,jax
        from common.collectives import resolve_mesh,barrier
        result=check(resolve_mesh());result['job_step']=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
        if jax.process_index()==0:Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
        barrier('local-checks-test')
    run_main_and_finalize(main)
