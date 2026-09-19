"""P4 direct-operator mirror actions for diagonal and rectangular photon sectors."""


def main():
    import argparse
    import json
    import os
    from pathlib import Path
    from types import SimpleNamespace
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host, resolve_mesh
    from gw.shared_pole_directions import _round_kernels
    from gw.shared_pole_sectors import cross_round_actions
    from runtime import initialize_communicator_stack, finalize_process

    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    initialize_communicator_stack()
    mesh=resolve_mesh()
    assert jax.process_count()==4 and mesh.size==4
    rng=np.random.default_rng(477)
    z=.73+.41j
    nodes=(z,z.conjugate(),-z,-z.conjugate())
    mu=np.array([.4,.9,-.5,-1.2])
    left=rng.normal(size=(2,4))+1j*rng.normal(size=(2,4))
    right=rng.normal(size=(3,4))+1j*rng.normal(size=(3,4))
    q=rng.normal(size=(4,2,3,2))+1j*rng.normal(size=(4,2,3,2))
    q/=np.linalg.norm(q,axis=2,keepdims=True)
    adj=lambda a:np.swapaxes(a.conj(),-1,-2)
    def value(node,l,r):
        weight=1/(node*mu-1)
        slope=-mu/(node*mu-1)**2/(2*node)
        return (l*weight)@adj(r),(l*slope)@adj(r)
    direct=value(z,left,right)
    reverse=value(z,right,left)
    mirror=value(-z.conjugate(),left,right)
    mirror_reverse=value(-z.conjugate(),right,left)
    sharding=NamedSharding(mesh,P(('x','y')))
    def put(a):
        a=np.asarray(a)
        a=np.broadcast_to(a,(4,)+a.shape).copy()
        return jax.make_array_from_callback(a.shape,sharding,lambda idx:a[idx])
    rep=lambda a:jax.device_put(np.asarray(a),NamedSharding(mesh,P()))
    def stack(a):return put(a[None])
    states=[(node,put(q[0,k%2])) for k,node in enumerate(nodes)]
    roles=[[dict(sample_id=0,conjugate=k%2==1,mirror=k>=2) for k in range(4)] for _ in range(4)]
    recipe=dict(z_ry=np.array([z]),distinct_id=np.array([0]))
    def action(n):
        indices=np.tile(np.arange(n,dtype=np.int32),(4,1))
        return put(indices[0]),put(indices[0]),put(np.ones(n,np.complex128)),None
    # cross_round_actions expects direct LR, direct RL, their derivatives,
    # then mirror LR, mirror RL, their derivatives.
    panels=(stack(direct[0]),stack(reverse[0]),stack(direct[1]),stack(reverse[1]),
            stack(mirror[0]),stack(mirror_reverse[0]),stack(mirror[1]),stack(mirror_reverse[1]))
    result=cross_round_actions(panels,states,roles,recipe,sample_lo=0,mesh_xy=mesh,
        partner_slots=[0,1,2,3],endpoint_actions=(action(2),action(3)))
    errors=[]
    for k,node in enumerate(nodes):
        exact,derivative=value(node,left,right)
        direction=q[0,k%2]
        got=tuple(np.asarray(gather_to_host(a))[0] for a in result[k])
        errors.extend((float(np.max(np.abs(got[0]-exact@direction))),
                       float(np.max(np.abs(got[1]-(2*node*derivative)@direction)))))
    assert max(errors)<1e-12,errors

    # Diagonal mirror uses its stored -conj(z) value, adjointed only for
    # the original state at -z. Wrongly using the direct +z value is visible.
    square=value(-z.conjugate(),right,right)
    xs=put(q[0])
    scales=rep(np.array([-2*z,-2*z.conjugate()],np.complex128))
    kernel=_round_kernels(mesh).literal_mirrors((False,True))
    actual=kernel(stack(square[0]),stack(square[1]),xs,rep(np.int32(0)),scales)
    diagonal=[]
    for k,node in enumerate(nodes[2:]):
        exact,derivative=value(node,right,right)
        direction=q[0,k]
        got=tuple(np.asarray(gather_to_host(a))[0] for a in actual[2*k:2*k+2])
        diagonal.extend((float(np.max(np.abs(got[0]-exact@direction))),
                         float(np.max(np.abs(got[1]-(2*node*derivative)@direction)))))
    assert max(diagonal)<1e-12,diagonal
    wrong=kernel(stack(value(z,right,right)[0]),stack(value(z,right,right)[1]),
                 xs,rep(np.int32(0)),scales)
    red=float(np.linalg.norm(np.asarray(gather_to_host(wrong[0]))[0]
        -np.asarray(gather_to_host(actual[0]))[0]))
    assert red>1e-3,red
    result=dict(status='PASS',job=os.environ.get('SLURM_JOB_ID'),
        step=os.environ.get('SLURM_STEP_ID'),direct_cross_max=max(errors),
        direct_diagonal_max=max(diagonal),wrong_positive_sample_red=red,
        scope='P4 literal mirror actions at nonzero complex frequency; no bank producer, moments or material model')
    if jax.process_index()==0:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finalize_process()


if __name__=='__main__':
    main()
