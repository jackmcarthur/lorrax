"""Bounded real latent-measure counterexample for conjugate port choices.

Run on a compute node with JAX_PLATFORMS=cpu. The independent explicit
Krylov projection is compared with the production data-only pencil.
"""
def main():
    import argparse
    import json
    import os
    from pathlib import Path
    import runtime
    runtime.bootstrap(platform='cpu')
    import numpy as np
    import jax.numpy as jnp
    from gw.shared_pole_constructor import assemble_shared_pole_pencil, reduce_shared_pole_pencil
    from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as gates

    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(918)
    c = rng.normal(size=(8,64))*.04
    poles = np.linspace(.1,4.,64)
    adj = lambda a: a.conj().swapaxes(-1,-2)
    m1 = c@c.T/2
    m3 = (c*poles)@c.T/2
    qi = np.linalg.eigh(m1)[1][:,-2:]
    def sample(s):
        return (c/(s-poles))@c.T,(-c/(s-poles)**2)@c.T
    def mm(a,b,transa='N',transb='N'):
        return (adj(a) if transa=='C' else a)@(adj(b) if transb=='C' else b)
    batch = lambda a: jnp.asarray(a[None],dtype=jnp.complex128)
    rows = []
    for mode in ('same_direction','conjugated_direction'):
        states,latent = [],[]
        for point in (.5+.4j,1.9+.6j):
            q = adj(np.linalg.svd(sample(point)[0])[2])[:,:2]
            for conjugate in (False,True):
                s = point.conjugate() if conjugate else point
                port = q.conj() if conjugate and mode=='conjugated_direction' else q
                w,dw = sample(s)
                states.append((s,batch(port),batch(w@port),batch(dw@port)))
                latent.append((c.T@port)/(s-poles)[:,None])
        infinity = tuple(batch(a) for a in (qi,m1@qi,m3@qi))
        latent.append(c.T@qi)
        x = np.concatenate(latent,axis=-1)
        exact_g = adj(x)@x
        pencil = assemble_shared_pole_pencil(states,infinity,matmul=mm)
        model,diag,_ = reduce_shared_pole_pencil(pencil,jnp.ones((1,x.shape[1]),bool),
                              eigh=jnp.linalg.eigh,matmul=mm,gates=gates)
        b,t,active = [np.asarray(a)[0] for a in model]
        basis = np.linalg.qr(x)[0]
        t_ref = adj(basis)@(poles[:,None]*basis)
        out_ref = c@basis
        point = -.37
        got = (b/(point-t))@adj(b)
        expected = out_ref@np.linalg.solve(point*np.eye(t_ref.shape[0])-t_ref,adj(out_ref))
        row = dict(mode=mode,gram_relative=float(np.linalg.norm(np.asarray(pencil[0])[0]-exact_g)/np.linalg.norm(exact_g)),
                   projection_relative=float(np.linalg.norm(got-expected)/np.linalg.norm(expected)),
                   imaginary_relative=float(np.linalg.norm(got.imag)/np.linalg.norm(got)),
                   symmetry_relative=float(np.linalg.norm(got-got.T)/np.linalg.norm(got)),
                   gram_min_relative=float(np.asarray(diag['gram_min_relative'])[0]),
                   retained_rank=int(active.sum()),minimum_pole=float(t[active].min()))
        assert row['gram_relative']<1e-11 and row['projection_relative']<1e-10
        assert row['minimum_pole']>0
        rows.append(row)
    assert rows[0]['imaginary_relative']>1e-4
    assert rows[1]['imaginary_relative']<1e-11
    result = dict(status='PASS',rows=rows,job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
                  scope='tiny planted real latent measure, no Si source-attribution or QP accuracy claim')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
