"""All-P weighted spectral-shape census, prior to any pole fit."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
import numpy as np
import banks


def spectrum(gram):
    """Numerical ranks by singular amplitude and relative Frobenius tail."""
    ev = np.linalg.eigvalsh((gram+gram.conj().T)/2)[::-1]
    if ev[-1] < -1e-10*ev[0]:
        raise ValueError('Gram is materially indefinite')
    lam = np.maximum(ev, 0)
    sigma = np.sqrt(lam/lam[0])
    tail = np.sqrt(np.maximum(0, np.r_[lam.sum(), lam.sum()-np.cumsum(lam)])/lam.sum())
    return dict(eigenvalues=ev.tolist(), singular_relative=sigma.tolist(),
                rank_amplitude={str(t): int(np.sum(sigma > t)) for t in (1e-2, 1e-3, 1e-4)},
                rank_frobenius={str(t): int(np.flatnonzero(tail <= t)[0]) for t in (1e-2, 1e-3, 1e-4)})


def main(out):
    assert os.getenv('SLURM_JOB_ID'), 'Compute only'
    from runtime import initialize_communicator_stack
    initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import resolve_mesh, barrier
    from distrib_la import plan, gemm_plan
    assert jax.process_count() == 4
    mesh = resolve_mesh()
    face = NamedSharding(mesh, P('x', 'y'))
    stack = NamedSharding(mesh, P(None, 'x', 'y'))
    rep = NamedSharding(mesh, P())
    eig = plan('eigh', mesh, backend='distributed', n=896, batched_route='auto')
    owner = banks.low_owner(mesh)
    low, broad, z, zh = banks.metadata()
    weights = banks.loss_weights(z)
    job = os.getenv('SLURM_JOB_ID')+'.'+os.getenv('SLURM_STEP_ID', '?')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    out.mkdir(parents=True, exist_ok=True)
    product = gemm_plan(mesh,m=896,k=896,n=896,nq=1,dtype=jnp.complex128,backend='distributed')
    mm = lambda a,b: product(a[None],b[None])[0]
    # Preserve Run299's physical conversion verbatim; its @ products are
    # traced with all-P input/output layouts. No matrix leaves the device.
    eig_jit = jax.jit(lambda v: eig((v+v.conj().T)/2), in_shardings=face, out_shardings=(rep,face))
    inverse_root = jax.jit(lambda u,e: mm(u/jnp.sqrt(e)[None,:],
        jax.lax.with_sharding_constraint(u.conj().T,face)),out_shardings=face)
    adj = lambda w: jnp.swapaxes(w.conj(), -1, -2)

    @jax.jit
    def channels(w):
        return jax.lax.with_sharding_constraint(jnp.concatenate(((w+adj(w))/2, (w-adj(w))/(2j))), stack)

    @jax.jit
    @jax.shard_map(mesh=mesh, in_specs=P(None,'x','y'), out_specs=P(), check_vma=False)
    def gram(w):
        flat = w.reshape(w.shape[0], -1)
        return jax.lax.psum(flat@flat.conj().T, ('x','y'))

    @jax.jit
    def sketch(w, v):
        return jnp.concatenate((jnp.trace(w, axis1=1, axis2=2)[:,None],
                                jnp.einsum('am,iab,bm->im',v.conj(),w,v)),axis=1)

    @jax.jit
    def congruence_rows(v, w):
        def one(_, wi):
            return None, mm(mm(v, wi), v)
        return jax.lax.scan(one, None, w, unroll=1)[1]

    rng = np.random.default_rng(306)
    probes = rng.normal(size=(896,4))+1j*rng.normal(size=(896,4))
    probes /= np.linalg.norm(probes,axis=0)
    vd = jnp.asarray(probes)
    records = []
    for q in range(29):
        start = time.perf_counter()
        print(f'GRAM q{q:02d} start job={job}',flush=True)
        w, wh, v, provenance = banks.load(q, mesh, eig_jit, owner, broad)
        assert w.shape == (36,896,896) and wh.shape == (8,896,896)
        ev, u = eig_jit(v)
        if float(ev[0]) <= 0:
            raise ValueError('Nonpositive Coulomb')
        vi = inverse_root(u,ev)
        ww = congruence_rows(vi,w)
        wwh = congruence_rows(vi,wh)
        gw = np.asarray(gram(ww))
        all_white = jnp.concatenate((ww,wwh))
        gc = np.asarray(gram(channels(all_white))).real
        gp = np.asarray(gram(channels(jnp.concatenate((w,wh))))).real
        sketches = np.asarray(sketch(ww,vd))
        record = dict(q=q, job_step=job, source=source, input_paths=provenance,
                      weight='trapezoid * 1/(1+(omega_eV/20)^2); each height normalized equally',
                      low_floor=1e-14, broad_floor=0., scopes={})
        for label, indices in [('low', np.arange(12)),('broad',np.arange(12,36)),('combined',np.arange(36))]:
            wi = np.sqrt(weights[indices])
            g = gw[np.ix_(indices,indices)]*wi[:,None]*wi[None,:]
            ci = np.r_[indices,indices+44]
            cw = np.r_[wi,wi]
            record['scopes'][label] = dict(complex=spectrum(g),
                hermitian_channels=spectrum(gc[np.ix_(ci,ci)]*cw[:,None]*cw[None,:]), rows=len(indices))
        record['seconds'] = time.perf_counter()-start
        record['allocator'] = os.getenv('XLA_PYTHON_CLIENT_ALLOCATOR','runtime default')
        record['peak_bank_bytes_per_rank'] = int(44*896*896*16/4)
        if jax.process_index()==0:
            with (out/f'q{q:02d}.npz').open('xb') as stream:
                np.savez(stream, z=z, zh=zh, weights=weights, complex_gram=gw,
                         channel_gram=gc, physical_channel_gram=gp, sketches=sketches)
            (out/f'q{q:02d}.json').write_text(json.dumps(record,indent=2))
        records.append(record)
        print(f'GRAM q{q:02d} complete {record["seconds"]:.3f}s ranks={record["scopes"]["combined"]["complex"]["rank_amplitude"]}',flush=True)
        del w, wh, ww, wwh, all_white, u, v, vi
    barrier('gram-complete')
    if jax.process_index()==0:
        (out/'receipt.json').write_text(json.dumps(dict(status='COMPLETE',job_step=job,source=source,records=records),indent=2))
        lines = ['# Weighted V-whitened Gram ranks', '', f'All 29 q; job.step {job}. Training only. Amplitude cutoff is sqrt(lambda/lambda_max); tail rank also in JSON. Equal line normalization, Sigma proxy stated in receipts.', '', '|q|low 1e-2/3/4|broad 1e-2/3/4|combined 1e-2/3/4|','|---|---|---|---|']
        for r in records:
            cells = ['/'.join(str(v) for v in r['scopes'][s]['complex']['rank_amplitude'].values()) for s in ('low','broad','combined')]
            lines.append('|'+str(r['q'])+'|'+'|'.join(cells)+'|')
        (out/'rank_table.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    main(args.out)
