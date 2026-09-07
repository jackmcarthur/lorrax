"""Audit whether the requested W realization admits positive-real balancing.

For lossless Y(s)=-s W(i s), A is skew-Hermitian, C=B†, D=0.
The KYP storage X=I is exact. With distinct state frequencies and nonzero
input rows it is unique: trace(A†X+XA)=0 forces its negative-semidefinite
KYP block to zero, hence [X,A]=0 and XB=B fixes every diagonal entry.
Thus lossless PR characteristic values are all one, not a damping-order
curve. This audits the singular Riccati/Lure preconditions; it does not
claim to have run a regularized Riccati reduction.
"""
from pathlib import Path
import argparse
import json
import os
import numpy as np
from balance import LOCATOR,sha,EV,ETA


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    assert os.getenv('SLURM_JOB_ID')
    from runtime import initialize_communicator_stack
    initialize_communicator_stack()
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from file_io.slab_io import SlabIO
    from common.collectives import single_device_mesh
    pin=json.loads(LOCATOR.read_text())['reference']['green_function_binding']['parent_binding']
    assert sha(pin['path'])==pin['sha256']
    parent=json.loads(Path(pin['path']).read_text());model=parent['model'];ds=model['datasets']
    assert sha(model['path'])==model['sha256']
    mesh=single_device_mesh();rows=[]
    with SlabIO(model['path'],mode='r',mesh=mesh) as io:
        width=model['factor_shape'][2]
        for q in (0,1,14):
            c=io.read_slab(ds['factor'],shape=(1,896,width),offset=(q,0,0),partition_spec=P(None,'x','y'))[0]
            lam=io.read_slab(ds['poles2'],shape=(1,width),offset=(q,0),partition_spec=P(None,'y'))[0]
            mask=io.read_slab(ds['factor_mask'],shape=(1,width),offset=(q,0),partition_spec=P(None,'y'))[0]
            ids=np.flatnonzero(np.asarray(mask)>0);om=jnp.sqrt(lam[ids]);c=c[:,ids]
            mass=jnp.sum(jnp.abs(c)**2,axis=0)
            trace_static=float(jnp.real(jnp.sum(mass/((1j*ETA)**2-om**2))))
            trace_high=float(jnp.real(jnp.sum(mass/((180/EV+1j*ETA)**2-om**2))))
            # Y has B=[C†/sqrt2; C†/sqrt2], C_y=B†.
            a=np.r_[-1j*np.asarray(om),1j*np.asarray(om)]
            assert trace_static<0 and trace_high>0
            rows.append(dict(q=q,q_full_row=parent['q_parent_full_rows'][q],
                 K=len(ids),state_dimension=2*len(ids),
                 H_zero_real_trace=trace_static,minus_H_high_frequency_real_trace=-trace_high,
                 H_positive_real=False,minus_H_positive_real=False,
                 lossless_Y_KYP_A_adjoint_plus_A_max=float(np.max(abs(a+a.conj()))),
                 lossless_Y_KYP_XB_minus_Cadjoint=0.,
                 lossless_Y_input_row_norm2_min=float(jnp.min(mass/2)),
                 distinct_state_frequencies=len(np.unique(a)),
                 unique_identity_KYP_storage=bool(len(np.unique(a))==len(a) and jnp.min(mass)>0),
                 feedthrough_rank=0,ordinary_Riccati_inverse_D_plus_Dadjoint='undefined',
                 regularized_Riccati_reduction_executed=False))
    args.out.write_text(json.dumps(dict(jobid=os.getenv('SLURM_JOB_ID'),stepid=os.getenv('SLURM_STEP_ID'),
         source_sha256=sha(__file__),model_sha256=model['sha256'],rows=rows,
         scope='three-q KYP/Riccati precondition audit, not a truncated-model comparison'),indent=2)+'\n')

if __name__=='__main__':main()
