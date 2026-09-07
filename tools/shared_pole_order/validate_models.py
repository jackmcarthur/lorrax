"""Stream Run216 and Run300 physical W diagnostics for signed modal exports."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import numpy as np
from balance import S,F,EV,sha,LOCATOR


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--q',type=int,nargs='+',required=True);args=p.parse_args()
    assert os.getenv('SLURM_JOB_ID')
    from runtime import initialize_communicator_stack
    initialize_communicator_stack()
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from file_io.slab_io import SlabIO
    from common.collectives import single_device_mesh
    mesh=single_device_mesh()
    ownerpath=F/'299_shared_residue_ls_20260907/anchor_input.py'
    spec=importlib.util.spec_from_file_location('order_anchor_owner',ownerpath)
    anchor=importlib.util.module_from_spec(spec);spec.loader.exec_module(anchor)
    assert sha(anchor.OLD)==anchor.OLD_SHA and sha(anchor.COULOMB)==anchor.COULOMB_SHA
    low=json.loads(anchor.OLD.read_text())
    highpath=F/'300_na_shared_ls_inputs_20260907/01_broad_a/physical_receipt.json'
    high=json.loads(highpath.read_text())
    assert high['coordinate']=='canonical_physical_Wc'
    args.out.mkdir(parents=True,exist_ok=True)
    for q in args.q:
        with SlabIO(anchor.COULOMB,mode='r',mesh=mesh) as io:
            v=io.read_slab('V_canonical_qwedge',shape=(1,896,896),offset=(q,0,0),partition_spec=P(None,'x','y'))[0]
        ve,vu=jnp.linalg.eigh((v+v.conj().T)/2)
        assert float(ve[0])>0
        vh=(vu*jnp.sqrt(ve)[None,:])@vu.conj().T
        models={}
        for k in ('eps1e-02','eps1e-03','eps1e-04','K224','K448','K896','K1792'):
            path=args.models/f'q{q:02d}'/f'{k}_model.npz'
            if not path.exists():continue
            with np.load(path) as f:
                assert int(f['q_parent'])==low['parent_qrows'][q]
                models[k]=(jnp.asarray(f['poles_ry']),jnp.asarray(f['residue_left']),jnp.asarray(f['residue_right']))
        # Parent baseline distinguishes truncation loss from parent/bank mismatch.
        pin=json.loads(LOCATOR.read_text())['reference']['green_function_binding']['parent_binding']
        assert sha(pin['path'])==pin['sha256']
        parent=json.loads(Path(pin['path']).read_text())['model'];ds=parent['datasets'];width=parent['factor_shape'][2]
        assert sha(parent['path'])==parent['sha256']
        with SlabIO(parent['path'],mode='r',mesh=mesh) as io:
            pc=io.read_slab(ds['factor'],shape=(1,896,width),offset=(q,0,0),partition_spec=P(None,'x','y'))[0]
            pl=io.read_slab(ds['poles2'],shape=(1,width),offset=(q,0),partition_spec=P(None,'y'))[0]
            pm=io.read_slab(ds['factor_mask'],shape=(1,width),offset=(q,0),partition_spec=P(None,'y'))[0]
        ids=np.flatnonzero(np.asarray(pm)>0);pom=jnp.sqrt(pl[ids]);pb=pc[:,ids]/jnp.sqrt(2*pom)[None,:]
        models['R896_parent']=(jnp.concatenate([pom,-pom]),jnp.concatenate([pb,-pb],axis=1),jnp.tile(pb.conj().T,(2,1)))
        results={k:{} for k in models}
        for label,bank,eta in [('low',low,.25),('broadA',high,high['sampling_eta_ev'])]:
            item=bank['q_receipts'][q];path=Path(item['artifact']);assert sha(path)==item['artifact_sha256']
            with SlabIO(path,mode='r',mesh=mesh) as io:
                for split,dataset in [('construction','construction_value_cubic'),('held','held_value_cubic')]:
                    freqs=bank['schedule'][split+'_ev']
                    sums={k:[0.,0.] for k in models}
                    errors={k:[] for k in models}
                    for i,x in enumerate(freqs):
                        w=io.read_slab(dataset,shape=(1,896,896),offset=(i,0,0),partition_spec=P(None,'x','y'))[0]
                        if label=='low':w=vh@w@vh
                        norm=float(jnp.linalg.norm(w)**2)
                        z=(x+1j*eta)/EV
                        for k,(poles,left,right) in models.items():
                            fit=(left/(z-poles)[None,:])@right
                            error=float(jnp.linalg.norm(fit-w)**2)
                            sums[k][0]+=error;sums[k][1]+=norm;errors[k].append(float(np.sqrt(error/norm)))
                    for k in models:
                        results[k][label+'_'+split]=dict(z_real_ev=freqs,eta_ev=eta,
                            per_point_relative_frobenius=errors[k],stacked_relative_frobenius=float(np.sqrt(sums[k][0]/sums[k][1])),
                            artifact=str(path),artifact_sha256=item['artifact_sha256'])
        (args.out/f'q{q:02d}.json').write_text(json.dumps(dict(q=q,jobid=os.getenv('SLURM_JOB_ID'),stepid=os.getenv('SLURM_STEP_ID'),
            models=results,script_sha256=sha(__file__),coulomb_sha256=anchor.COULOMB_SHA,
            low_activity_floor=1e-14,broad_activity_floor=0,
            scope='Actual physical W bank diagnostics, no Sigma accuracy assertion'),indent=2)+'\n')

if __name__=='__main__':main()
