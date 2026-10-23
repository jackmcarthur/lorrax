"""Compute-node gate: tiled Gram tails versus independent transition features."""

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh-x',type=int)
    args=parser.parse_args()
    import json
    import math
    import os
    from pathlib import Path
    import re
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host, device_put_process_local
    from isdf.core import gram_q0_tiled_from_psi_sm, _gram_q0_tiled_from_psi_kernel
    from common.pivoted_cholesky import pivoted_cholesky_select

    mesh = stack.mesh
    if args.mesh_x is not None:
        mesh=Mesh(np.asarray(mesh.devices).reshape(args.mesh_x,-1),('x','y'))
    div = math.lcm(int(mesh.shape['x']), int(mesh.shape['y']))
    n = 5 * div
    rng = np.random.default_rng(5621)
    weights = np.asarray([.31, .69])
    sigma = [np.array([[0,1],[1,0]], complex),
             np.array([[0,-1j],[1j,0]], complex),
             np.array([[1,0],[0,-1]], complex)]
    alpha = [np.block([[np.zeros((2,2)),s],[s,np.zeros((2,2))]]) for s in sigma]
    errors = {}
    hlos = []
    for ns, mode in ((1,'charge'), (2,'charge'), (4,'transverse')):
        psi = (rng.standard_normal((2,6,ns,n)) + 1j*rng.standard_normal((2,6,ns,n))) / 4
        for label, left, right, nested in (
                ('equal',(0,4),(0,4),True),
                ('nested',(1,4),(0,6),True),
                ('disjoint',(0,2),(3,6),False)):
            l, r = psi[:,left[0]:left[1]], psi[:,right[0]:right[1]]
            features = []
            for k in range(2):
                if mode == 'charge':
                    z = np.einsum('cam,vam->mcv', r[k].conj(), l[k]).reshape(n,-1)
                else:
                    z = np.concatenate([np.einsum('cam,ab,vbm->mcv',r[k].conj(),a,l[k]).reshape(n,-1) for a in alpha],axis=1)
                features.append(z)
            ref = sum(w * (z @ z.conj().T) for w,z in zip(weights,features))
            def put(a,spec):
                return device_put_process_local(a,NamedSharding(mesh,spec))
            rx = put(r.conj().transpose(0,3,1,2),P(None,'x',None,None))
            ry = put(r,P(None,None,None,'y'))
            if nested:
                lx = ly = None
                band_slice = (left[0]-right[0],left[1]-right[0])
            else:
                lx = put(l.conj().transpose(0,3,1,2),P(None,'x',None,None))
                ly = put(l,P(None,None,None,'y'))
                band_slice = None
            kw = put(weights,P())
            for tile in (2*div,5*div,7*div):
                g = put(np.full((n,n),complex(723,-19)),P('x','y'))
                value = gram_q0_tiled_from_psi_sm(g,lx,ly,rx,ry,kw,mesh_xy=mesh,
                    tile_width=tile,gamma_mode=mode,left_bands=band_slice)
                actual = np.asarray(gather_to_host(value))
                rel = float(np.max(abs(actual-ref))/np.max(abs(ref)))
                errors[f'{mode}{ns}/{label}/tile{tile}'] = rel
                assert rel < 3e-12, (mode,label,tile,rel)
                assert np.max(abs(actual-actual.conj().T)) < 3e-12*np.max(abs(ref))
                # Same released selector sees exactly the same first pivots/rank.
                if tile == 2*div:
                    got = pivoted_cholesky_select(jnp.asarray(actual),min(5,n),tol_rel=1e-12)
                    want = pivoted_cholesky_select(jnp.asarray(ref),min(5,n),tol_rel=1e-12)
                    assert int(got[2]) == int(want[2])
                    np.testing.assert_array_equal(np.asarray(got[0]),np.asarray(want[0]))
                if label == 'equal' and tile == 2*div:
                    from common.gamma_matrices import gammas_perm,gammas_phase
                    perm = gammas_perm if mode=='transverse' else jnp.arange(ns,dtype=jnp.int32)
                    phase = gammas_phase if mode=='transverse' else jnp.ones(ns,dtype=jnp.complex128)
                    kernel = _gram_q0_tiled_from_psi_kernel(mesh,2,n,4,4,ns,tile,gamma_mode=mode,left_bands=(0,4))
                    shapes=[jax.ShapeDtypeStruct((n,n),jnp.complex128),
                            jax.ShapeDtypeStruct(rx.shape,rx.dtype),jax.ShapeDtypeStruct(ry.shape,ry.dtype),
                            jax.ShapeDtypeStruct(kw.shape,kw.dtype)]
                    executable = kernel.lower(*shapes,perm,phase,perm,phase).compile()
                    hlo = executable.as_text()
                    local=(n//int(mesh.shape['x']),n//int(mesh.shape['y']))
                    pattern=rf'c128\[{local[0]},{local[1]}\].*=.*\bcopy\('
                    forbidden=[line for line in hlo.splitlines() if re.search(pattern,line)]
                    assert not forbidden, forbidden
                    out=Path(os.environ.get('GRAM_ALIAS_EVIDENCE','.'))/f'gram_tail_{mode}{ns}.hlo'
                    if jax.process_index()==0:
                        out.write_text(hlo)
                    hlos.append(str(out))
    if jax.process_index()==0:
        print('GRAM_TILED_ALIAS '+json.dumps(dict(errors=errors,max_relative=max(errors.values()),mesh=list(mesh.devices.shape),hlo=hlos)),flush=True)
        print('GRAM_TILED_ALIAS PASS',flush=True)
    runtime.finalize_process(0)

if __name__=='__main__':
    main()
