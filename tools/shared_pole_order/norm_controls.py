"""Euclidean positive-residue column controls through Run289's canonical exporter."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
from types import SimpleNamespace
import numpy as np
from balance import S, F, LOCATOR, EV, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    assert os.getenv('SLURM_JOB_ID')
    from runtime import initialize_communicator_stack
    initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from file_io.slab_io import SlabIO
    from common.collectives import single_device_mesh
    assert jax.process_count()==1
    mesh=single_device_mesh()
    ref=json.loads(LOCATOR.read_text())['reference']
    pin=ref['green_function_binding']['parent_binding']
    assert sha(pin['path'])==pin['sha256']
    parent=json.loads(Path(pin['path']).read_text());model=parent['model'];ds=model['datasets']
    assert sha(model['path'])==model['sha256']
    receiptpath=F/'300_na_shared_ls_inputs_20260907/01_broad_a/physical_receipt.json'
    bank=json.loads(receiptpath.read_text())
    parents=np.array(parent['q_parent_full_rows'])
    assert bank['parent_qrows']==parents.tolist()
    # This is an input adapter for the existing exporter, not a new symmetry
    # implementation: pass its one consumed table from the authenticated owner.
    sym=SimpleNamespace(q_irr_full_idx=parents)
    exporterpath=F/'289_na_rpa_reduction_sigma_20260906/export_store.py'
    spec=importlib.util.spec_from_file_location('order_export_owner',exporterpath)
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    assert sha(exporterpath)=='ffaa9396246af6fe38321b82ed2fdb248c7447cc964780f91b9fa8da7d86b429'
    # Read the existing canonical negative table from the parent model itself.
    import h5py
    with h5py.File(model['path'],'r') as f:
        negatives=np.asarray(f['q_negative_full_rows']) if 'q_negative_full_rows' in f else None
    if negatives is None:
        from common.sanity import neg_q_index
        negatives=neg_q_index((8,8,8))[parents]
    records={k:[] for k in (224,448,896,1792)}
    models={k:[] for k in records}
    args.out.mkdir(parents=True,exist_ok=True)
    with SlabIO(model['path'],mode='r',mesh=mesh) as io:
        width=model['factor_shape'][2]
        for q in range(29):
            c=io.read_slab(ds['factor'],shape=(1,896,width),offset=(q,0,0),partition_spec=P(None,'x','y'))[0]
            lam=io.read_slab(ds['poles2'],shape=(1,width),offset=(q,0),partition_spec=P(None,'y'))[0]
            mask=io.read_slab(ds['factor_mask'],shape=(1,width),offset=(q,0),partition_spec=P(None,'y'))[0]
            active=np.flatnonzero(np.asarray(mask)>0)
            # R=b b† with b=C/sqrt(2Omega); rank columns by ||b||².
            mass=np.asarray(jnp.sum(jnp.abs(c[:,active])**2,axis=0)/(2*jnp.sqrt(lam[active])))
            ranked=active[np.argsort(-mass,kind='stable')]
            for k in models:
                ids=np.sort(ranked[:k])
                ck,lk=c[:,ids],lam[ids]
                models[k].append((ck,lk))
                records[k].append(dict(qslot=q,q_full_row=int(parents[q]),status='COMPLETE',
                    K=k,J=len(np.unique(np.asarray(lk))),port_width=896,
                    source_column_ids=ids.tolist(),criterion='largest physical ||b_j||², b=C/sqrt(2Omega_Ry)',
                    damping_fraction=0,cond=None,storage_bytes=int(ck.nbytes+lk.nbytes),
                    jobid=os.getenv('SLURM_JOB_ID'),stepid=os.getenv('SLURM_STEP_ID')))
    chain={str(Path(__file__).resolve()):sha(__file__),str(exporterpath):sha(exporterpath),
           str(pin['path']):pin['sha256'],str(model['path']):model['sha256']}
    for k in models:
        exporter.export_allq(models[k],meta=None,sym=sym,mesh=mesh,destination=args.out/f'K{k}',
                            coordinate='canonical_physical_Wc',per_q_records=records[k],source_chain=chain,
                            occupation=dict(eta_ev=.25,kT_ry=.01,mu_ry=parent['occupation']['mu_ry']),
                            q_negative_full_rows=negatives)
    (args.out/'receipt.json').write_text(json.dumps(dict(status='COMPLETE',budgets=list(models),
        jobid=os.getenv('SLURM_JOB_ID'),stepid=os.getenv('SLURM_STEP_ID'),source_chain=chain),indent=2)+'\n')

if __name__=='__main__':
    main()
