"""Offline square-root balanced reduction of the authenticated Na R896 parent.

Dense state-space work is O((2K)^3) time / O((2K)^2) memory per q.
This diagnostic is explicitly single-GPU; it is not a distributed driver.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import numpy as np

S = Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14')
F = S / 'runs/frequency_integration_sandbox'
LOCATOR = F / '268_na_run110_sigma_reassessment_20260906/scoring/control/e0_binding.json'
EV = 13.605693122994
ETA = .25 / EV

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--weight', type=Path)
    parser.add_argument('--q', type=int, nargs='+', default=list(range(29)))
    parser.add_argument('--curves-only', action='store_true')
    args = parser.parse_args()
    assert os.getenv('SLURM_JOB_ID'), 'Compute only'
    from runtime import initialize_communicator_stack
    initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.collectives import single_device_mesh
    from file_io.slab_io import SlabIO
    assert jax.process_count() == 1 and len(jax.local_devices()) == 1
    mesh = single_device_mesh()
    adj = lambda a: a.conj().T
    ref = json.loads(LOCATOR.read_text())['reference']
    pin = ref['green_function_binding']['parent_binding']
    assert sha(pin['path']) == pin['sha256']
    parent = json.loads(Path(pin['path']).read_text())
    model = parent['model']
    assert sha(model['path']) == model['sha256']
    ds = model['datasets']
    lowpath = S/'tmp/worktrees/wt_run101_conjugation_closed_physical_20260905/runs/frequency_integration_sandbox/216_na_allq_cubic_high_precision_actions_20260906/physical_receipt.json'
    schedule = json.loads(lowpath.read_text())['schedule']
    z = (np.r_[schedule['construction_ev'], schedule['held_ev']]+.25j)/EV
    assert len(z) == 16
    args.out.mkdir(parents=True, exist_ok=True)
    gweight = None
    if args.weight:
        with np.load(args.weight) as f:
            om = f['omega_ry']; w = f['weight_ry_minus2']
        grid = np.r_[-om[:0:-1], om]
        weights = np.r_[w[:0:-1], w] * (om[1]-om[0]) / (2*np.pi)
        weights[[0,-1]] *= .5
        gweight = (jnp.asarray(grid), jnp.asarray(weights))
    for q in args.q:
        out = args.out/f'q{q:02d}'
        out.mkdir(exist_ok=False)
        t0 = time.monotonic()
        receipt = dict(q=q, q_full_row=parent['q_parent_full_rows'][q],
                       jobid=os.environ['SLURM_JOB_ID'], stepid=os.getenv('SLURM_STEP_ID'),
                       model_path=model['path'], model_sha256=model['sha256'],
                       script_sha256=sha(__file__), weight='unweighted' if args.weight is None else str(args.weight),
                       eta_ev=.25, status='STARTED')
        write(out/'receipt.json', receipt)
        print(json.dumps(receipt), flush=True)
        with SlabIO(model['path'], mode='r', mesh=mesh) as io:
            width = model['factor_shape'][2]
            factor = io.read_slab(ds['factor'], shape=(1,896,width), offset=(q,0,0), partition_spec=P(None,'x','y'))[0]
            poles2 = io.read_slab(ds['poles2'], shape=(1,width), offset=(q,0), partition_spec=P(None,'y'))[0]
            mask = io.read_slab(ds['factor_mask'], shape=(1,width), offset=(q,0), partition_spec=P(None,'y'))[0]
        active = np.flatnonzero(np.asarray(mask)>0)
        factor = factor[:,active]
        om = jnp.sqrt(poles2[active])
        assert bool(jnp.all(om>0))
        b = factor / jnp.sqrt(2*om)[None,:]
        k = len(active)
        a = jnp.concatenate([-ETA-1j*om, -ETA+1j*om])
        B = jnp.tile(adj(b), (2,1))
        C = jnp.concatenate([-1j*b, 1j*b], axis=1)
        def parent_w(zi):
            return (factor/(zi**2-om**2)[None,:])@adj(factor)
        def relative(x,y):
            return float(jnp.linalg.norm(x-y)/jnp.linalg.norm(y))
        roundtrip = [relative((C/(-1j*zi-ETA-a)[None,:])@B, parent_w(zi)) for zi in z]
        receipt.update(parent_K=k, parent_J=len(np.unique(np.asarray(om))), port_width=896,
                       state_dimension=2*k, parent_storage_bytes=int(factor.nbytes+om.nbytes),
                       realization_roundtrip_max=max(roundtrip))
        assert max(roundtrip)<=1e-10, receipt
        write(out/'receipt.json', receipt)
        kernel = -1/(a[:,None]+a.conj()[None,:])
        if gweight is not None:
            # Partial fractions avoid the [state,state,frequency] tensor exactly:
            # 1/(xy)=(1/x+1/y)/(x+y), x+y=-a_i-conj(a_j).
            grid, weights = gweight
            gs = []
            for start in range(0, len(a), 128):
                gs.append(jnp.sum(weights[None,:]/(1j*grid[None,:]-a[start:start+128,None]), axis=1))
            g = jnp.concatenate(gs)
            kernel = kernel*(g[:,None]+g.conj()[None,:])
        gram_b = B@adj(B)
        gram_c = adj(C)@C
        Pc = gram_b*kernel
        Qc = gram_c*kernel.T
        Pc = (Pc+adj(Pc))/2
        Qc = (Qc+adj(Qc))/2
        lp = jnp.linalg.cholesky(Pc)
        lq = jnp.linalg.cholesky(Qc)
        jax.block_until_ready((lp,lq))
        if not bool(jnp.all(jnp.isfinite(lp))) or not bool(jnp.all(jnp.isfinite(lq))):
            receipt.update(status='CHOLESKY_REFUSED', elapsed_seconds=time.monotonic()-t0)
            write(out/'receipt.json', receipt)
            raise RuntimeError('Semidefinite/indefinite Gramian; no regularization allowed without diagnosis')
        u, hsv, vh = jnp.linalg.svd(adj(lq)@lp, full_matrices=False)
        hsv.block_until_ready()
        h = np.asarray(hsv)
        np.save(out/'hsv.npy', h)
        ranks = {str(eps): int(np.count_nonzero(h>eps*h[0])) for eps in (1e-2,1e-3,1e-4)}
        tail = np.r_[np.cumsum(h[::-1])[::-1],0]
        receipt.update(hsv_relative_state_ranks=ranks,
                       hsv_relative_tail_state_ranks={str(eps):int(np.flatnonzero(tail<=eps*tail[0])[0]) for eps in (1e-2,1e-3,1e-4)},
                       hsv_condition=float(h[0]/h[-1]), elapsed_seconds=time.monotonic()-t0,
                       status='CURVES_COMPLETE', Sigma_order_mev=None)
        write(out/'receipt.json', receipt)
        print(json.dumps(receipt), flush=True)
        if args.curves_only:
            continue
        from reductions import reduce_models
        reduce_models(a, B, C, (lp,lq,u,hsv,vh),
                      dict(out=out,receipt=receipt,z=z,factor=factor,omega=om,mesh=mesh))
        receipt.update(status='REDUCTIONS_COMPLETE', elapsed_seconds=time.monotonic()-t0)
        write(out/'receipt.json', receipt)

if __name__ == '__main__':
    main()
