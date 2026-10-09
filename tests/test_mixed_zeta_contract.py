"""P4 streamed mixed Coulomb metric, physical heads and written ζ oracle."""
from pathlib import Path
import json
import os


def check_mixed_contract(runtime, *, compensated=False):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from isdf.augmentation import onsite_coulomb_tile
    from isdf.zeta_mubatch import ZStore, ZetaG, contract_v_group
    from runtime.padding import padded_axis

    mesh = runtime.mesh
    assert int(mesh.size) == 4
    Q, Qpad, mu, ng, gpad, gt, na = 3, 4, 8, 6, 8, 4, 3
    qowner = NamedSharding(mesh,P(('x','y'),None,None))
    rng = np.random.default_rng(19301)
    complex_random = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
    Z = complex_random((Qpad,mu,gpad)) / 3.
    Z[Q:] = 0.
    Z[...,ng:] = 0.
    B = np.stack([np.linalg.qr(complex_random((mu,mu)))[0]
                  / np.sqrt(np.arange(1,mu+1))[None] for _ in range(Qpad)])
    B[Q:] = 0.
    coeff_rhs = complex_random((Qpad,mu,na)) / 7.
    coeff_rhs[Q:] = 1e6  # q-padding must be inert even with a poisoned RHS pad
    df, gf = complex_random((Qpad,na,gpad))/5., complex_random((Qpad,na,gpad))/6.
    kd, kg = np.diag([5.,4.,3.]).astype(complex), np.diag([.7,.6,.5]).astype(complex)
    ngk = np.array([5,3,6],np.int32)
    mask = np.arange(gpad)[None] < np.pad(ngk,(0,1))[:,None]
    v = rng.uniform(.2,1.1,size=(Qpad,gpad)) * mask
    keep = np.array([[0,2],[0,1],[0,5]],np.int32)
    put = lambda a: device_put_process_local(np.asarray(a),qowner)
    solve_host = lambda a: B @ (B.conj().transpose(0,2,1) @ a)
    smooth, coeff = solve_host(Z), solve_host(coeff_rhs)
    delta, comp = coeff @ df, coeff @ gf
    smooth, delta, comp = (a*mask[:,None] for a in (smooth,delta,comp))
    product = lambda a,b: np.einsum('qmg,qg,qng->qmn',a.conj(),v,b)
    onsite = np.einsum('qmi,ij,qnj->qmn',coeff.conj(),kd-kg,coeff)
    expected = product(smooth,smooth)+product(smooth,delta)+product(delta,smooth)+product(comp,comp)+onsite
    if compensated:
        expected = product(smooth+comp, smooth+comp)+onsite
    no_cross = product(smooth,smooth)+product(comp,comp)+onsite
    physical = smooth+delta
    expected_shell = np.take_along_axis(physical[:Q],keep[:,None],axis=2)
    st = ZStore(mesh=mesh,q_axis=padded_axis(Q,4,name='mixed test q'),mu_pad=mu,
                g_axis=padded_axis(ng,gt,name='mixed test G'),b=4,placement='host')
    for beta in range(2):
        rows = device_put_process_local(Z[:Q,beta*4:(beta+1)*4],
                                        NamedSharding(mesh,P(None,('x','y'),None)))
        st.write_batch(beta,rows)
    zeta = ZetaG(st,mesh=mesh,L_q=put(B),lu_piv=None,solver_kind='rank_truncate',
                  batched_route='batch_reshard',n_rmu_solve=mu,n_rmu=mu,
                  mu_basis=None,ngk_per_q=ngk,gvec_components=np.zeros((Q,3,ng),np.int32),
                  path='synthetic',print_fn=lambda line:None)
    @jax.jit
    def fourier(c,f):
        return jnp.einsum('qma,qag->qmg',c,f)
    onsite_apply = jax.jit(onsite_coulomb_tile,out_shardings=qowner)
    dfd, gfd = put(df),put(gf)
    kddev,kgdev = jnp.asarray(kd),jnp.asarray(kg)
    counts = dict(fourier=0,onsite=0)
    def fourier_tile(t,c):
        counts['fourier'] += 1
        return (fourier(c,dfd[...,t*gt:(t+1)*gt]),
                fourier(c,gfd[...,t*gt:(t+1)*gt]))
    def onsite_callback(c):
        counts['onsite'] += 1
        return onsite_apply(c,kddev,kgdev)
    provider=dict(rhs=put(coeff_rhs),fourier_tile=fourier_tile,onsite=onsite_callback)
    if compensated:
        provider['body_metric'] = 'compensated'
    got=np.asarray(gather_to_host(zeta.contract_v(v[:Q,:ng],keep=keep,local_augmentation=provider)))
    shell=np.asarray(gather_to_host(zeta.shell))
    relative_error=float(np.linalg.norm(got-expected[:Q])/np.linalg.norm(expected[:Q]))
    shell_error=float(np.max(np.abs(shell-expected_shell)))
    assert relative_error < 2e-13,(relative_error,shell_error)
    assert shell_error < 2e-13
    assert counts == dict(fourier=2,onsite=1),counts
    cross_signal=float(np.linalg.norm(expected[:Q]-no_cross[:Q])/np.linalg.norm(expected[:Q]))
    assert cross_signal > .01,cross_signal
    # Exact zero correction must preserve the incumbent route's final V.
    baseline=np.asarray(gather_to_host(zeta.contract_v(v[:Q,:ng],keep=keep)))
    zero_provider=dict(provider,rhs=put(np.zeros_like(coeff_rhs)))
    zero=np.asarray(gather_to_host(zeta.contract_v(v[:Q,:ng],keep=keep,local_augmentation=zero_provider)))
    assert np.array_equal(zero,baseline)
    # File-only pass uses physical s+delta, not the compensation density.
    class Capture:
        def __init__(self):
            self.tiles=[]
        def write_slab(self,name,tile,*,offset):
            self.tiles.append((offset[-1],np.asarray(gather_to_host(tile))))
    zeta.local_augmentation=provider
    capture=Capture()
    zeta.write_file(capture)
    written=np.concatenate([tile for _,tile in capture.tiles],axis=2)
    file_error=float(np.max(np.abs(written-physical[:Q])))
    assert file_error < 2e-13,file_error
    invalid = dict(provider, body_metric='unknown')
    try:
        zeta.contract_v(v[:Q,:ng],keep=keep,local_augmentation=invalid)
    except ValueError as exc:
        assert 'unknown local Coulomb body metric' in str(exc)
    else:
        raise AssertionError('unknown Coulomb body metric was accepted')
    try:
        contract_v_group([zeta],[(0,0)],[v[:Q,:ng]],keep=keep)
    except ValueError as exc:
        assert 'augmented_current_coulomb' in str(exc)
    else:
        raise AssertionError('grouped current route silently accepted scalar augmentation')
    result=dict(P=4,q_logical=Q,q_carrier=Qpad,mu=mu,G_logical=ng,G_carrier=gpad,
                G_tile=gt,local_basis=na,relative_metric_error=relative_error,
                max_shell_error=shell_error,max_written_zeta_error=file_error,
                omitted_cross_relative_error=cross_signal,zero_exact=True,
                callback_counts_after_v=dict(fourier=2,onsite=1),
                grouped_current_refused=True,compensated_body=compensated)
    if jax.process_index()==0:
        print(json.dumps(result),flush=True)
        out=os.environ.get('MIXED_CONTRACT_REPORT')
        if out:
            Path(out).write_text(json.dumps(result,indent=2)+'\n')
    zeta.close()


if __name__=='__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def main():
        check_mixed_contract(runtime)
        return 0
    run_main_and_finalize(main)
