"""P4 device carry extent and coded value/derivative parity for unequal groups.

The production group wrapper, weights, zero allocator and donation run here.
A coded linear stream replaces only the physics kernel; physical chi/W/Sigma
parity belongs to the separate fresh Na gate.
"""
def main():
    import argparse,json
    from pathlib import Path
    from types import SimpleNamespace
    import runtime
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    mesh=runtime.initialize_communicator_stack().mesh
    import jax
    import jax.numpy as jnp
    import numpy as np
    import gw.response_bank as owner
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local
    a.out.mkdir(parents=True,exist_ok=True)
    nq,n=3,8;shape=(nq,n,n)
    q,r,c=np.indices(shape)
    coded=(1+q*16+r+c/16+1j*(q/8-r/32+c/64)).astype(np.complex128)
    face=device_put_process_local(coded,NamedSharding(mesh,P(None,'x','y')))
    forward=np.array([1,2,4,8],complex);reverse=2j*forward
    groups=[]
    for members in ((4,1),(0,3,2)):
        value=np.empty((len(members),2,4),complex);derivative=np.empty_like(value)
        for row,sample in enumerate(members):
            for side in range(2):
                for node in range(4):
                    value[row,side,node]=(sample+1)*2**(-node)+1j*(side+1)/8
                    derivative[row,side,node]=(side+1)*2**(-node-2)+1j*(sample+1)/16
        groups.append(dict(members=list(members),count=4,t=np.array([.125,.25,.5,1.],complex),
                           value=value,derivative=derivative))
    meta=SimpleNamespace(mu_basis=SimpleNamespace(n_packed=n));allocated=[];compiled={}
    real_stream,real_weights,real_scratch,real_zeros=(owner.response_stream,owner.stream_weights,
                                                    owner._stream_scratch,owner._group_zeros)
    def zeros(mesh,shape):
        out=real_zeros(mesh,shape)
        allocated.append(dict(shape=list(shape),local_bytes=int(out.addressable_data(0).nbytes)))
        return out
    def stream(*args,n_outputs,**kwargs):
        if n_outputs not in compiled:
            def kernel(times,weights,face,f,u,refs,raw):
                amplitude=jnp.sum(weights[0]*jnp.asarray(forward)+weights[1]*jnp.asarray(reverse),axis=-1)
                return raw+amplitude[:,None,None,None]*face[None]
            compiled[n_outputs]=jax.jit(kernel,donate_argnums=(6,))
        return compiled[n_outputs],(face,)
    owner.response_stream,owner.stream_weights,owner._stream_scratch,owner._group_zeros=(
        stream,lambda *args:jnp.asarray([1.]),lambda *args:0,zeros)
    stats=[];outputs={}
    def execute(kernel,args,stage,**kwargs):
        executable=kernel.lower(*args).compile();m=executable.memory_analysis()
        stats.append(dict(stage=stage,outputs=int(m.output_size_in_bytes),aliases=int(m.alias_size_in_bytes)))
        return executable(*args)
    try:
        for slots in (5,3):
            rules=dict(plan=dict(reference=0.),refs=np.array([0.,0.]),slots=slots,f=None,u=None,band_ranges=())
            receipt=dict(correlation_count=0)
            for ig,group in enumerate(groups):
                got=owner.integrate_response_group(None,meta,mesh,rules,group,q_ids=(0,1,2),
                    execute=execute,receipt=receipt)
                index=got.sharding.devices_indices_map(got.shape)[jax.local_devices()[0]]
                expected=np.zeros(got.shape,complex)
                for row in range(len(group['members'])):
                    for field,name in enumerate(('value','derivative')):
                        coeff=group[name][row]
                        amplitude=-sum(coeff[0]*forward+coeff[1]*reverse)
                        expected[2*row+field]=amplitude*coded
                local=np.asarray(got.addressable_data(0));np.testing.assert_array_equal(local,expected[index])
                outputs[slots,ig]=local.copy()
                del got
            assert receipt['correlation_count']==8
        for ig,group in enumerate(groups):
            count=2*len(group['members'])
            np.testing.assert_array_equal(outputs[5,ig][:count],outputs[3,ig][:count])
        assert [r['shape'][0] for r in allocated]==[10,10,6,6]
        assert allocated[2]['local_bytes']*5==allocated[0]['local_bytes']*3
    finally:
        owner.response_stream,owner.stream_weights,owner._stream_scratch,owner._group_zeros=(
            real_stream,real_weights,real_scratch,real_zeros)
    result=dict(verdict='PASS',group_members=[g['members']for g in groups],planned_slots=5,compact_slots=3,
        allocations=allocated,compiled=stats,active_value_derivative_exact=True,unused_rows_zero=True,
        scope='Production group carry/weight/donation transport with coded stream; not physical response or whole-program peak')
    (a.out/f'rank{jax.process_index()}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESPONSE_GROUP_SLOTS PASS',jax.process_index(),flush=True)

if __name__=='__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
