"""One P4 exact photon spin-trace parity and compiled-peak acceptance."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    from runtime import initialize_communicator_stack,finalize_process
    initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import resolve_mesh,gather_to_host
    from gw.w_isdf import _get_chi_fractional_contour_kernel_face as candidate
    mesh=resolve_mesh()
    assert mesh.size==4
    path=args.run/'reference_w_isdf.py'
    spec=importlib.util.spec_from_file_location('gw._spin_trace_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reference=module._get_chi_fractional_contour_kernel_face
    def put(a,spec):
        a=np.asarray(a)
        return jax.make_array_from_callback(a.shape,NamedSharding(mesh,spec),lambda ix:a[ix])
    def endpoints(nk,nb,n,ns):
        # Each callback constructs its local tile only; no global wavefunction
        # exists on the host, including the Fe-geometry synthetic case.
        def carrier(shape,spec,seed):
            def local(ix):
                dims=tuple((s.stop or size)-(s.start or 0) for s,size in zip(ix,shape))
                offsets=tuple(s.start or 0 for s in ix)
                rng=np.random.default_rng(seed+sum((i+1)*v for i,v in enumerate(offsets)))
                return (rng.normal(size=dims)+1j*rng.normal(size=dims))/np.sqrt(nb)
            return jax.make_array_from_callback(shape,NamedSharding(mesh,spec),local)
        mun=tuple(carrier((nk,ns,n,nb),P(None,None,'x','y'),seed) for seed in (477,478))
        nmu=tuple(carrier((nk,nb,ns,n),P(None,'x',None,'y'),seed) for seed in (479,480))
        energy=np.broadcast_to(np.linspace(-1,2,nb),(nk,nb)).copy()
        f=1/(1+np.exp(energy/.3))
        return mun,nmu,put(energy,P()),f
    rows=[]
    geometries=[('tiny',(2,2,2),8,16),('fe_shape',(4,4,4),36,1032)]
    for label,grid,nb,n in geometries:
        nk=int(np.prod(grid));ns=4
        mun,nmu,energy,f=endpoints(nk,nb,n,ns)
        cases=('moment','retarded','laplace_ordered','kms_static') if label=='tiny' else ('moment',)
        for case in cases:
            mode='retarded' if case in ('moment','retarded') else case
            times=np.array([0.]) if case=='moment' else np.array([.1,.3,.5])
            nout=2 if case=='laplace_ordered' else 1
            projection_rows=2*nout if case=='laplace_ordered' else nout
            projection=np.arange(1,projection_rows*len(times)+1,dtype=float).reshape(projection_rows,-1)*(.1+.2j)
            lower=f;upper=-1j*(1-f) if case=='moment' else 1-f
            ref=np.array(0.)
            if case=='laplace_ordered':
                lower=np.stack((f,.4*f));upper=np.stack((1-f,.7*(1-f)));ref=np.array([-1.,2.])
            if case=='kms_static':
                ref=np.array([2.,0.])
            inputs=(put(times,P()),put(projection,P()),mun,nmu,energy,
                    put(lower.astype(complex),P()),put(upper.astype(complex),P()),put(ref,P()))
            selected=(0,3,7) if label=='tiny' else tuple(range(13))
            compiled=[];memory=[];seconds=[];outputs=[]
            for name,builder in (('reference',reference),('candidate',candidate)):
                kernel=builder(mesh,grid,nout,(nk,nb,n,ns),selected_q=selected,
                               ordered=True,vertex=True,pair_mode=mode)
                executable=kernel.lower(*inputs).compile()
                stats=executable.memory_analysis()
                memory.append(dict(arguments=stats.argument_size_in_bytes,outputs=stats.output_size_in_bytes,
                                   temporaries=stats.temp_size_in_bytes,aliases=stats.alias_size_in_bytes))
                value=executable(*inputs);jax.block_until_ready(value)
                started=time.monotonic()
                measured=executable(*inputs);jax.block_until_ready(measured)
                seconds.append(time.monotonic()-started)
                del measured
                outputs.append(value);compiled.append(executable)
                if jax.process_index()==0:
                    (args.run/f'{label}_{case}_{name}.hlo').write_text(executable.as_text())
            relative=float(gather_to_host(jnp.linalg.norm(outputs[1]-outputs[0])/
                                          jnp.maximum(jnp.linalg.norm(outputs[0]),1e-300)))
            assert relative<2e-12,(label,case,relative)
            row=dict(geometry=label,case=case,relative=relative,memory=dict(zip(('reference','candidate'),memory)),
                     seconds=dict(zip(('reference','candidate'),seconds)))
            rows.append(row)
            if jax.process_index()==0:
                print(json.dumps(row),flush=True)
            del outputs,compiled,inputs
        del mun,nmu,energy
    large=rows[-1]
    assert large['memory']['candidate']['temporaries']<large['memory']['reference']['temporaries']/4
    result=dict(schema='lorrax.photon-spin-trace-gate.v1',status='PASS',rows=rows,
        reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        job=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
        scope='P4 synthetic exact spin trace across moment/retarded/Laplace/KMS; Fe-shaped one-correlation peak/timing, not material bank/map peak')
    if jax.process_index()==0:
        (args.run/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    finalize_process()

if __name__=='__main__':
    main()
