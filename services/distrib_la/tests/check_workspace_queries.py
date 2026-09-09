"""Shape-only dense workspace queries on the actual P4/P16 mesh."""
from pathlib import Path
import json
import os
import sys
import numpy as np


def main():
    import jax
    from jax.sharding import Mesh
    import distrib_la as D
    from distrib_la.workspace import _workspace_details
    root=Path(sys.argv[1]);parts=int(np.sqrt(jax.device_count()))
    assert parts*parts==jax.device_count()==jax.process_count()
    mesh=Mesh(np.asarray(jax.devices()).reshape(parts,parts),('x','y'))
    rows=[]
    for label,backend,route in [('local','off','batch_reshard'),('distributed','distributed','auto')]:
        for n in (896,1792,2576):
            p=D.plan('eigh',mesh,backend=backend,n=n,batched_route=route)
            for op,shapes in [('eigh',((1,n,n),)),('gemm',((1,n,n),(1,n,n)))]:
                details=_workspace_details(p,op,shapes,np.complex128)
                count=D.workspace_bytes_per_rank(p,op,shapes,np.complex128)
                assert type(count) is int and count>=0 and count==details['device_bytes']
                assert D.workspace_bytes_per_rank(p,op,shapes,np.complex128)==count
                rows.append(dict(plan=label,op=op,shapes=shapes,dtype='complex128',**details))
                print(label,op,n,details,flush=True)
        p=D.plan('eigh',mesh,backend=backend,n=896,batched_route=route)
        shapes=((1,224,896),(1,896,896))
        rows.append(dict(plan=label,op='gemm',shapes=shapes,dtype='complex128',
                         **_workspace_details(p,'gemm',shapes,np.complex128)))
        for bad_op,bad_shapes,bad_dtype in [('solve',((896,896),),np.complex128),
                                            ('eigh',((895,895),),np.complex128),
                                            ('eigh',((896,896),),np.complex64)]:
            try:D.workspace_bytes_per_rank(p,bad_op,bad_shapes,bad_dtype)
            except (ValueError,TypeError):pass
            else:raise AssertionError('missing workspace refusal')
    # Single calls honor the distributed backend even when the same plan's
    # batched route stages locally. Native auto batches get a conservative
    # per-member vendor bound, not the serial staged-path bound.
    mixed=D.plan('eigh',mesh,backend='distributed',n=896,batched_route='batch_reshard')
    single=_workspace_details(mixed,'eigh',((896,896),),np.complex128)
    staged=_workspace_details(mixed,'eigh',((1,896,896),),np.complex128)
    assert not single['local'] and staged['local']
    native_auto=D.plan('eigh',mesh,backend='off',n=896,batched_route='auto')
    native_batch=D.workspace_bytes_per_rank(native_auto,'eigh',((3,896,896),),np.complex128)
    assert native_batch==3*staged['device_bytes']
    receipt=dict(status='PASS' ,job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
                 rank=jax.process_index(),geometry=[parts,parts],rows=rows,
                 shape_policy_controls=dict(single_distributed=single,batched_local=staged,native_batch3=native_batch),
                 scope='workspace query only; no dense operands/results/queried workspace allocated; context and compiler setup excluded')
    (root/f'query_rank{jax.process_index()}.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
    return 0


if __name__=='__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    run_main_and_finalize(main)
