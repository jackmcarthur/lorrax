"""One public local/distributed CC/TT/CT/store known-answer acceptance."""

def main():
    import argparse
    import json
    import os
    from pathlib import Path
    import numpy as np
    from runtime import initialize_communicator_stack, finalize_process
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    initialize_communicator_stack()
    import jax
    from common.collectives import resolve_mesh
    from shared_pole_sector_constructor_p4 import check_sector_constructor
    mesh=resolve_mesh()
    assert mesh.size==4
    rows=[];values=[]
    for mode in ('local','distributed'):
        receipt,observables=check_sector_constructor(mesh,args.output.parent,
            linalg=mode,parents=4,return_observables=True)
        rows.append(receipt);values.append(observables)
    relative={name:float(np.linalg.norm(values[0][name]-values[1][name])/
                        np.linalg.norm(values[0][name])) for name in values[0]}
    assert all(np.isfinite(v) and v<1e-7 for v in relative.values()),relative
    result=dict(status='PASS',checks=rows,local_distributed_relative=relative,
        job=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
        scope='P4 configured local/distributed public CC/TT/CT and four stores; four tiny planted parents; no large-parent or material scalability measurement')
    if jax.process_index()==0:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finalize_process()

if __name__=='__main__':
    main()
