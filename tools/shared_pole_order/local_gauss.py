"""Frequency-local PSD compression through the authenticated Run146 loop.

The historical owner inlines its cluster kernel. Extract that exact-hash loop,
retain its moment whitening/localizer/Gauss rotation, and adapt only complex
adjoints, variable cluster rank and single-GPU placement. The equivalent state-space
QR form avoids squaring the near-null moment conditioning. No Sigma owner changes.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import textwrap
from types import SimpleNamespace
import numpy as np
from balance import F, S, LOCATOR, EV, sha

OWNER=S/'tmp/worktrees/wt_run146_clustered_psd_factors_20260905/runs/frequency_integration_sandbox/146_na_clustered_psd_factor_compression_20260905/run_cluster_compression.py'
OWNER_SHA='afbaf51683021d1558ef4d383d6b0b6f9394e46422b3480e0334d8dde0b4641f'


def owner_blocks(factor, poles2, groups, mesh):
    """Reuse Run146: M0=C C†, M1=C t C†, and the whitened localizer.

    Parameters
    ----------
    factor : jax.Array, (896,K)
        Physical squared-pole factors, tiled on the single-GPU mesh.
    poles2 : numpy.ndarray, (K,)
        Positive t=Omega² in Ry².
    groups : tuple of numpy.ndarray
        Disjoint original column indices in frequency-local bins.
    mesh : jax.sharding.Mesh
        Assigned one-GPU mesh, axes x/y.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    assert sha(OWNER)==OWNER_SHA
    source=OWNER.read_text()
    start=source.index('    for ic, indices in enumerate(groups):')
    stop=source.index('    ball =',start)
    code=textwrap.dedent(source[start:stop])
    changes={
        'factor.T)':'factor.T.conj())',
        'if positive < LOCAL_RANK:\n        raise RuntimeError(\n            f"cluster {ic} has only {positive} directions above frozen floor")':'LOCAL_RANK = min(192, positive)\n    if LOCAL_RANK == 0:\n        raise RuntimeError(f"cluster {ic} has no resolved PSD direction")',
        'h_host = np.real(0.5 * (h_host_raw + h_host_raw.T.conj()))':'h_host = 0.5 * (h_host_raw + h_host_raw.T.conj())',
        'np.asarray(q, np.float64)':'np.asarray(q, np.complex128)',
        'b = jax.device_put(jnp.real(b), face)':'b = jax.device_put(b, face)',
        'q @ np.diag(theta) @ q.T - h_host':'q @ np.diag(theta) @ q.T.conj() - h_host',
        'h = mm(whiten.T.conj(), mm(r1, whiten))':
            'state = (factor.T.conj() @ whiten) * mask_dev[:,None]\n    state, _ = jnp.linalg.qr(state, mode="reduced")\n    h = state.T.conj() @ (poles2_dev[:,None] * state)',
        'b = right_multiply(u * sqrtlam[None, :], qdev)':
            'b = right_multiply(factor @ state, qdev)',
        'norms = np.real(np.diag(q.T @ np.diag(lam) @ q))':
            'norms = np.asarray(jax.device_get(jnp.sum(jnp.abs(b)**2,axis=0)))',
    }
    for old,new in changes.items():
        assert old in code,old
        code=code.replace(old,new)
    ns=dict(np=np,jax=jax,jnp=jnp,json=json,groups=groups,factor=factor,
            poles2=poles2,poles2_dev=jnp.asarray(poles2),K0=len(poles2),mesh=mesh,
            face=NamedSharding(mesh,P('x','y')),rep2=NamedSharding(mesh,P(None,None)),
            mm=lambda a,b:a@b,herm=lambda a:(a+a.T.conj())*.5,eop=jnp.linalg.eigh,
            gather_to_host=jax.device_get,replicate_to_mesh=lambda a,mesh:jnp.asarray(a),
            right_multiply=lambda a,q:a@q,RYD_TO_EV=EV,blocks=[],block_poles=[],block_norms=[],local=[])
    exec(compile(code,str(OWNER)+':cluster_loop','exec'),ns)
    return jnp.concatenate(ns['blocks'],axis=1),np.concatenate(ns['block_poles']),np.concatenate(ns['block_norms']),ns['local']


def main():
    """Construct all-q real-pole controls before opening any Sigma reference."""
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();assert os.getenv('SLURM_JOB_ID')
    from runtime import initialize_communicator_stack
    initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.collectives import single_device_mesh
    from file_io.slab_io import SlabIO
    import h5py
    assert jax.process_count()==1 and len(jax.local_devices())==1
    mesh=single_device_mesh();args.out.mkdir(parents=True,exist_ok=True)
    pin=json.loads(LOCATOR.read_text())['reference']['green_function_binding']['parent_binding']
    assert sha(pin['path'])==pin['sha256']
    parent=json.loads(Path(pin['path']).read_text());model=parent['model'];ds=model['datasets']
    assert sha(model['path'])==model['sha256']
    with h5py.File(model['path'],'r') as f:negatives=np.asarray(f['q_negative_full_rows'])
    weight_path=F/'304_order_weighted_bt_20260907/01_weight/weight.npz'
    with np.load(weight_path) as f:
        grid=f['omega_ry']*EV;weight=f['weight_ry_minus2']/EV**2
    # Equidistribute integral sqrt(w) over the parent's actual frequency range.
    # The bin choice uses only the published Sigma kernel weight, never errors.
    density=np.sqrt(weight)
    cumulative=np.r_[0,np.cumsum((density[:-1]+density[1:])*.5*np.diff(grid))]
    exporterpath=F/'289_na_rpa_reduction_sigma_20260906/export_store.py'
    assert sha(exporterpath)=='ffaa9396246af6fe38321b82ed2fdb248c7447cc964780f91b9fa8da7d86b429'
    spec=importlib.util.spec_from_file_location('order_gauss_export',exporterpath)
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    models={k:[] for k in (448,896,1792)};records={k:[] for k in models}
    schedule=json.loads((S/'tmp/worktrees/wt_run101_conjugation_closed_physical_20260905/runs/frequency_integration_sandbox/216_na_allq_cubic_high_precision_actions_20260906/physical_receipt.json').read_text())['schedule']
    z=(np.r_[schedule['construction_ev'],schedule['held_ev']]+.25j)/EV
    with SlabIO(model['path'],mode='r',mesh=mesh) as io:
        width=model['factor_shape'][2]
        for q in range(29):
            c=io.read_slab(ds['factor'],shape=(1,896,width),offset=(q,0,0),partition_spec=P(None,'x','y'))[0]
            t=io.read_slab(ds['poles2'],shape=(1,width),offset=(q,0),partition_spec=P(None,'y'))[0]
            mask=io.read_slab(ds['factor_mask'],shape=(1,width),offset=(q,0),partition_spec=P(None,'y'))[0]
            ids=np.flatnonzero(np.asarray(mask)>0);c=c[:,ids];t=np.asarray(t)[ids]
            om=np.sqrt(t)*EV
            endpoints=np.interp([om.min(),om.max()],grid,cumulative)
            edges=np.interp(np.linspace(*endpoints,25),cumulative,grid)
            bins=np.clip(np.searchsorted(edges,om,side='right')-1,0,23)
            groups=tuple(np.flatnonzero(bins==i) for i in range(24) if np.any(bins==i))
            ball,theta,norms,local=owner_blocks(c,t,groups,mesh)
            # Unchanged Run146 low-resolvent ranking, now on weighted bins.
            z2=((np.arange(0,9+.125,.25)+.25j)/EV)**2
            gain=np.max(np.abs(1/(z2[:,None]-theta[None,:])),axis=0)
            ranked=np.argsort(norms*gain)[::-1]
            assert len(theta)>=1792,('insufficient rotated rank',q,len(theta))
            truth=[(c/(zi**2-t)[None,:])@c.T.conj() for zi in z]
            for k in models:
                selected=np.sort(ranked[:k]);ck=ball[:,selected];tk=theta[selected]
                errors=[float(jnp.linalg.norm((ck/(zi**2-tk)[None,:])@ck.T.conj()-exact)/jnp.linalg.norm(exact)) for zi,exact in zip(z,truth)]
                row=dict(qslot=q,q_full_row=parent['q_parent_full_rows'][q],status='COMPLETE',K=k,J=len(np.unique(tk)),port_width=896,
                    damping_fraction=0,cond=max(x['r0_retained_condition'] for x in local),storage_bytes=int(ck.nbytes+tk.nbytes),
                    jobid=os.getenv('SLURM_JOB_ID'),stepid=os.getenv('SLURM_STEP_ID'),W16_parent_relative_errors=errors,
                    bin_edges_ev=edges.tolist(),clusters=local,rotated_K=len(theta),owner_sha256=OWNER_SHA,
                    complex_adaptation='Hermitian adjoints, complex localizer and algebraically equivalent state QR to resolve near-null moment conditioning',
                    criterion='Run146 low_resolvent selection after sqrt(weight) frequency-local Gauss rotation')
                models[k].append((ck,jnp.asarray(tk)));records[k].append(row)
                print(json.dumps(dict(q=q,K=k,W16_max=max(errors))),flush=True)
            (args.out/f'q{q:02d}.json').write_text(json.dumps({str(k):records[k][-1] for k in records},indent=2)+'\n')
    chain={str(OWNER):OWNER_SHA,str(Path(__file__).resolve()):sha(__file__),str(exporterpath):sha(exporterpath),str(weight_path):sha(weight_path),str(model['path']):model['sha256']}
    for k in models:
        exporter.export_allq(models[k],meta=None,sym=SimpleNamespace(q_irr_full_idx=np.array(parent['q_parent_full_rows'])),mesh=mesh,
            destination=args.out/f'K{k}',coordinate='canonical_physical_Wc',per_q_records=records[k],source_chain=chain,
            occupation=dict(eta_ev=.25,kT_ry=.01,mu_ry=parent['occupation']['mu_ry']),q_negative_full_rows=negatives)
    (args.out/'receipt.json').write_text(json.dumps(dict(status='COMPLETE',jobid=os.getenv('SLURM_JOB_ID'),stepid=os.getenv('SLURM_STEP_ID'),source_chain=chain),indent=2)+'\n')


if __name__=='__main__':main()
