"""Nonzero-frequency CT joint-projection plants, one parent per GPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def run_checks(mesh):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import distrib_la
    from common.shard_map import shard_map
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as gates
    from gw.shared_pole_sectors import (cross_pencil_block,
        joint_sector_pencil, reduce_sector_pencil)

    assert jax.process_count() == 4 and mesh.size == 4
    batch = mesh.size
    spec = P(('x', 'y'))
    layout = NamedSharding(mesh, spec)
    adj = lambda a: np.swapaxes(a.conj(), -1, -2)

    def put(a):
        a = np.broadcast_to(np.asarray(a), (batch,) + np.shape(a)).copy()
        return jax.make_array_from_callback(a.shape, layout, lambda ix: a[ix])

    def mm(a, b, *, transa='N', transb='N'):
        def op(x, t):
            return x if t == 'N' else jnp.swapaxes(x.conj(), -1, -2)
        return op(a, transa) @ op(b, transb)

    native = distrib_la.plan('eigh', mesh, n=2, backend='off').native_fn

    def reduce(c, t, cross):
        return reduce_sector_pencil(joint_sector_pencil(c, t, cross, matmul=mm),
                                   eigh=native, matmul=mm, gates=gates)

    kernel = jax.jit(shard_map(reduce, mesh=mesh,
        in_specs=(spec, spec, spec), out_specs=(spec, spec), check_vma=False))
    rows = []
    # The two kept spans together span the exact two-state latent space.
    # Each diagonal sector alone has one Ritz pole; CT needs two different poles.
    yc = np.array([[1.], [0.]], complex)
    yt = np.array([[.3], [np.sqrt(.91)]], complex)
    cc = np.array([[1., .2j], [.1, .4]], complex)
    tt = np.array([[.3j, 1.], [.6, .2j], [.2, -.1j]], complex)
    metric = adj(yc) @ yt
    for ordered in (False, True):
        value = np.array([[.6, .15j], [-.15j, -.4]], complex) if ordered else np.array(
            [[2., .4j], [-.4j, 5.]], complex)
        vc, vt = (adj(yc) @ value @ yc).real.diagonal(), (adj(yt) @ value @ yt).real.diagonal()
        charge = (put(np.eye(1, dtype=complex)), put(vc), put(cc @ yc), put(tt @ yc))
        trans = (put(np.eye(1, dtype=complex)), put(vt), put(tt @ yt), put(cc @ yt))
        model, diag = kernel(charge, trans, (put(metric), put(adj(yc) @ value @ yt)))
        assert bool(jnp.all(diag['gram_valid']))
        assert bool(jnp.all(diag['retained_metric_positive']))
        c, t, lam, active = model
        errors, twins = [], []
        for z in (.7 + .2j, 1.3 + .6j, 2j):
            coordinate = z if ordered else z*z
            weights = 1 / (coordinate * lam - 1) if ordered else 1 / (coordinate-lam)
            evaluate = jax.jit(shard_map(lambda a,b,w: (a*w[:,None,:]) @ jnp.swapaxes(b.conj(),-1,-2),
                mesh=mesh, in_specs=(spec,spec,spec), out_specs=spec, check_vma=False))
            got = evaluate(c,t,weights*active)
            exact = cc @ np.linalg.solve(coordinate*value-np.eye(2) if ordered else
                                         coordinate*np.eye(2)-value, adj(tt))
            error = float(jnp.linalg.norm(got-put(exact))/jnp.linalg.norm(put(exact)))
            assert error < 1e-12, error
            errors.append(error)
            # Reuse the CC pole and its own span: a deliberately wrong twin.
            wrong = (cc @ yc) @ adj(tt @ yc) / (coordinate*vc[0]-1 if ordered else coordinate-vc[0])
            twins.append(float(np.linalg.norm(wrong-exact)/np.linalg.norm(exact)))
        assert min(twins) > .1
        rows.append(dict(name='ordered' if ordered else 'even', held_W_relative=errors,
                         CC_pole_reuse_relative=twins,
                         gram_min_relative=float(jnp.min(diag['gram_min_relative'])),
                         retained_rank=int(jnp.min(diag['retained_rank']))))

    # A physical particle-hole plant: CT is odd under z -> -z on TRS
    # data although both diagonal sectors are even. Both signed sector
    # spans have two states; their union recovers four exact signed poles.
    mu = np.array([.5, 1.25, -.5, -1.25])
    value = np.diag(mu).astype(complex)
    v = np.array([1.,1.])/np.sqrt(2)
    w = np.array([.3,np.sqrt(.91)])
    xc=np.zeros((4,2),complex); xt=np.zeros((4,2),complex)
    xc[:2,0]=v; xc[2:,1]=v
    xt[:2,0]=w; xt[2:,1]=w
    vc=np.diag(adj(xc)@value@xc).real
    vt=np.diag(adj(xt)@value@xt).real
    for broken in (False,True):
        cp=np.array([[.8,.3]],complex) + (np.array([[.1j,-.2j]]) if broken else 0)
        tp=np.array([[.2j,.7j],[.6j,-.1j],[.3j,.4j]])
        if broken:
            tp=tp+np.array([[.1,.2],[-.2,.1],[.3,-.1]])
        cfull=np.concatenate((cp,cp.conj()),axis=-1)
        tfull=np.concatenate((tp,tp.conj()),axis=-1)
        charge=(put(np.eye(2,dtype=complex)),put(vc),put(cfull@xc),put(tfull@xc))
        trans=(put(np.eye(2,dtype=complex)),put(vt),put(tfull@xt),put(cfull@xt))
        model,diag=kernel(charge,trans,(put(adj(xc)@xt),put(adj(xc)@value@xt)))
        c,t,lam,active=model
        assert bool(jnp.all(diag['gram_valid'])) and bool(jnp.all(diag['retained_metric_positive']))
        errors=[]
        for z in (.7+.2j,1.3+.6j,2j):
            exact=(cfull/(z*mu-1))@adj(tfull)
            got=evaluate(c,t,active/(z*lam-1))
            error=float(jnp.linalg.norm(got-put(exact))/jnp.linalg.norm(put(exact)))
            assert error<1e-12,error
            errors.append(error)
            if not broken:
                mirror=(cfull/(-z*mu-1))@adj(tfull)
                assert np.linalg.norm(exact+mirror)<1e-12
        # Neither exact positive pole is in the retained CC pole set.
        assert np.min(abs(1/mu[:2,None]-1/vc[None,:]))>.1
        rows.append(dict(name='physical_broken_CT' if broken else 'physical_TRS_odd_CT',
                         held_W_relative=errors,retained_rank=int(jnp.min(diag['retained_rank']))))

    # Cross Loewner entries and the confluent derivative are checked against
    # explicit latent vectors, including complex directions and rectangular CT.
    poles = np.array([.4, 1.7])
    qc = np.array([[1.], [.2j]])
    qt = np.array([[.3j], [1.], [.2]])
    def w(s):
        return (cc/(s-poles)) @ adj(tt)
    for a,b in ((.3+.4j,.8+.2j),(.3+.4j,.3-.4j)):
        derivative = -(cc/(b-poles)**2) @ adj(tt)
        def assemble(qc,qt,wa,wb,d):
            return cross_pencil_block((a,qc),(b,qt),(wa,wb,d),matmul=mm)
        block = jax.jit(shard_map(assemble,mesh=mesh,in_specs=(spec,)*5,
            out_specs=(spec,spec),check_vma=False))
        g,h = block(put(qc),put(qt),put(w(a.conjugate())),put(w(b)),put(derivative))
        xc=(adj(cc)@qc)/(a-poles)[:,None]
        xt=(adj(tt)@qt)/(b-poles)[:,None]
        error=max(float(jnp.max(jnp.abs(g-put(adj(xc)@xt)))),
                  float(jnp.max(jnp.abs(h-put(adj(xc)@(poles[:,None]*xt))))))
        assert error < 1e-12,error
        rows.append(dict(name='confluent' if b==a.conjugate() else 'cross_loewner',absolute_error=error))
    # Public endpoint components through the actual shared-pole Sigma
    # contraction, checked against an independent flattened band sum.
    from gw.mpa.sigma import synthesize_shared_pole_parents
    rng = np.random.default_rng(471)
    poles2 = np.array([[.4, .8, 1.2, 1.9]])**2
    for left_components,right_components in ((1,1),(1,3),(3,3)):
        bx = rng.normal(size=(1,2,left_components,4)) + 1j*rng.normal(size=(1,2,left_components,4))
        by = rng.normal(size=(1,2,right_components,4)) + 1j*rng.normal(size=(1,2,right_components,4))
        def face(a,spec):
            return jax.make_array_from_callback(a.shape,NamedSharding(mesh,spec),lambda ix:a[ix])
        gemm = distrib_la.gemm_plan(mesh,m=2*left_components,n=2*right_components,k=4,
                                   nq=1,dtype=np.complex128)
        def synthesize(x,y,p,i,e,t):
            return synthesize_shared_pole_parents(x,y,p,i,e,t,mesh_xy=mesh,gemm=gemm)
        synthesize = jax.jit(synthesize)
        tau=.4-.2j
        got,partner = synthesize(face(bx,P(None,'x',None,'y')),
            face(by,P(None,'y',None,'x')),jnp.asarray(poles2),
            jnp.asarray([[0,4]],jnp.int32),jnp.asarray(.1),jnp.asarray(tau))
        omega=np.sqrt(poles2)
        weights=np.exp(-1j*(omega-.1)*tau)/(2*omega)
        exact=(bx.reshape(1,-1,4)*weights[:,None]) @ adj(by.reshape(1,-1,4))
        error=float(jnp.max(jnp.abs(got-face(exact,P(None,'x','y')))))
        partner_error=float(jnp.max(jnp.abs(partner-face(exact.swapaxes(-1,-2),P(None,'x','y')))))
        assert max(error,partner_error)<1e-12,(error,partner_error)
        rows.append(dict(name=f'consumer_components_{left_components}_{right_components}',
                         absolute_error=error,partner_error=partner_error))
    assert len(rows)==9
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from runtime import initialize_communicator_stack, finalize_process
    initialize_communicator_stack()
    import jax
    from common.collectives import resolve_mesh
    mesh=resolve_mesh()
    rows=run_checks(mesh)
    rows.extend(run_store_checks(mesh,args.output.parent))
    result=dict(status='PASS',checks=rows,expected_checks=14,
                job=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                scope='P4 sector algebra, factor consumer and bitwise storage; no complete Sigma or production deck')
    if jax.process_index()==0:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finalize_process()


def run_store_checks(mesh,root):
    import importlib.util
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_recipe import CapacityLedger
    from file_io import shared_pole_store as store
    from file_io.slab_io import SlabIO
    spec=importlib.util.spec_from_file_location('sector_store_fixture',
        Path(__file__).resolve().parents[1]/'test_shared_pole_store.py')
    fixture=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    suffix=os.environ['SLURM_STEP_ID']
    fixture.check_roundtrip(mesh,root/f'scalar_{suffix}.h5')
    rows=[dict(name='scalar_store_roundtrip',bitwise=True)]
    meta,tables,recipe,identity=fixture._fixture(mesh)
    meta.nspinor=4
    meta.shared_pole_capacity=CapacityLedger(meta,mesh_xy=mesh)
    meta.shared_pole_capacity.reserve('fixture_live_bound',resident_bytes_per_rank=32768,
                                      workspace_bytes_per_rank=0)
    meta.shared_pole_capacity.live_stages=('fixture_live_bound',)
    canonical,_,poles,counts=fixture._model(meta)
    for sector in ('CC','TT','CT_C','CT_T'):
        components=3 if sector in ('TT','CT_T') else 1
        factor=np.concatenate([canonical*(1+1j*i) for i in range(components)],axis=2)
        packed=meta.mu_basis.pack_host(factor,axis=1)
        path=root/f'{sector}_{suffix}.h5'
        header=store.write_shared_pole_model(path,
            fixture._device(packed,mesh,P(None,'x',None,'y')),
            fixture._device(poles,mesh,P()),counts,q_span=(0,3),meta=meta,
            tables=tables,recipe=recipe,receipts=dict(identity=identity),sector=sector)
        store.validate_shared_pole_model(path,expected_identity=identity,mesh_xy=mesh,
                                        capacity=meta.shared_pole_capacity)
        with SlabIO(path,mode='r',mesh=mesh) as io:
            x,y,p,k=store.read_shared_pole_faces(io,(0,3),meta=meta,header=header)
            fixture._assert_local(x,packed)
            fixture._assert_local(y,packed)
            np.testing.assert_array_equal(np.asarray(p),poles)
            np.testing.assert_array_equal(np.asarray(k),counts)
        rows.append(dict(name=f'{sector}_store_roundtrip',bitwise=True,digest=header['digest']))
    return rows


if __name__=='__main__':
    main()
