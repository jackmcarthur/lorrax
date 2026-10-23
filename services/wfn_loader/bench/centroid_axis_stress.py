"""Bounded independent checks for the actual 521-k/20k-band stress WFN.

No file coefficients, WFN-transform helpers, or FFT implementation construct
reference faces. The reference is the fixture's declared dyadic formula and
literal sparse plane-wave sums, evaluated only in bounded local blocks.
"""


def main():
    import argparse
    import runtime
    stack = runtime.initialize_communicator_stack()
    import gc
    import json
    import resource
    import time
    import jax
    import numpy as np
    from common.meta import Meta
    from common.wfn_transforms import load_centroids_band_chunked
    from wfn_loader import WfnLoader, IBZRows

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wfn', required=True)
    p.add_argument('--mode', choices=('raw','kaxis','baxis','staged'), default='raw')
    p.add_argument('--stages', type=int, default=2)
    p.add_argument('--k-count', type=int, default=17)
    p.add_argument('--band-chunk', type=int, default=257)
    p.add_argument('--k-chunk', type=int, default=3)
    p.add_argument('--band-start', type=int)
    p.add_argument('--band-stop', type=int)
    a = p.parse_args()
    mesh = stack.mesh
    G = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    started = time.perf_counter()


    def cache_snapshot(wfn):
        """Evidence-only inspection of request-cache retention and object identity."""
        result = {}
        for name in ('_gvecs_cache','_ngk_valid_cache','_gvecs_dev_cache'):
            values = getattr(wfn,name)
            records = [dict(key=repr(key),identity=id(value),bytes=int(value.nbytes))
                       for key,value in values.items()]
            result[name] = dict(entries=sorted(records,key=lambda x:x['key']),
                retained_bytes=sum(x['bytes'] for x in records))
        return result


    def coefficients(bands, parents, ng):
        """Independent arithmetic form of the declared dyadic fixture data."""
        b = np.asarray(bands, dtype=np.int64)[None,:,None]
        k = np.asarray(parents, dtype=np.int64)[:,None,None]
        g = np.arange(ng, dtype=np.int64)[None,None,:]
        z = np.ldexp(((b+1)*8192+(k+1)*8+g+1).astype(float),-29)
        z = z+1j*np.ldexp(((17*b+131*k+7*g) & ((1<<20)-1))-(1<<19),-21)
        z = np.where(g < (3+2*(k % 3)),z,0)
        z = np.where(b == 0, (g == 0).astype(complex), z)
        return z


    def literal_faces(sym, rows, bands, mu_ids, coords, grid):
        key = (tuple(map(int,rows)),tuple(map(int,mu_ids)))
        if key in phase_cache:
            phase,parents,anti = phase_cache[key]
        else:
            active = np.asarray(mu_ids) < len(coords)
            phase = np.zeros((len(rows),7,len(mu_ids)),complex)
            parents = np.asarray(sym.irr_idx_k)[rows].astype(int)
            anti = []
            for i,(row,parent) in enumerate(zip(rows,parents)):
                R,tau,tr = sym.operation_rows(int(sym.sym_idx_k[row]))
                U = np.asarray(sym.spinor_action(int(sym.sym_idx_k[row]),nspinor=1))[0,0]
                rg = G@np.asarray(R).T
                full_k = np.asarray(sym.unfolded_kpts[row])
                raw_k = np.array([parent/int(sym.nk_tot),0,0])
                offset = np.rint(full_k-raw_k@np.asarray(R).T)
                physical = rg-offset+full_k
                phase[i,:,active] = (U*np.exp(-1j*(rg@np.asarray(tau)))[:,None]
                    * np.exp(2j*np.pi*(physical@(coords[np.asarray(mu_ids)[active]]/grid).T))
                    / np.sqrt(np.prod(grid))).T
                anti.append(bool(tr))
            phase_cache[key] = (phase,parents,anti)
        z = coefficients(bands, parents,7)
        z = np.where(np.asarray(anti)[:,None,None],z.conj(),z)
        return np.einsum('kbg,kgm->kbm',z,phase)


    with WfnLoader(a.wfn,mesh=mesh,backend='auto') as wfn:
        nk_stored, nb_stored = int(wfn.nkpts), int(wfn.nbands)
        assert int(wfn.nspinor) == 1 and int(wfn.ngkmax) == 7
        assert nk_stored > 137 and nk_stored % 137, 'k order stride must be coprime to nk'
        from math import gcd
        assert gcd(nk_stored,137) == 1
        assert nb_stored > 106
        sym = wfn.symmetry()
        order = (np.arange(nk_stored)*137) % nk_stored
        if a.mode == 'raw':
            before = time.perf_counter()
            lo = 3 if a.band_start is None else a.band_start
            hi = nb_stored-1 if a.band_stop is None else a.band_stop
            data = wfn.load(bands=(lo,hi),k=IBZRows(tuple(order)))
            jax.block_until_ready(data)
            io_seconds = time.perf_counter()-before
            for shard in data.addressable_shards:
                actual = np.asarray(shard.data)
                indices = [np.arange(n)[sl] for n,sl in zip(data.shape,shard.index)]
                k_ids,b_ids,_,g_ids = indices
                for k0 in range(0,len(k_ids),32):
                    k1 = min(k0+32,len(k_ids))
                    for b0 in range(0,len(b_ids),512):
                        b1 = min(b0+512,len(b_ids))
                        absolute_bands = b_ids[b0:b1]+lo
                        expected = coefficients(absolute_bands,order[k_ids[k0:k1]],data.shape[-1])
                        expected[:,absolute_bands>=hi,:] = 0
                        block = actual[k0:k1,b0:b1,0,:]
                        assert np.array_equal(block,expected), ('raw mismatch',jax.process_index(),k0,b0,np.max(np.abs(block-expected)))
            indices = np.asarray(wfn.box_index(k=IBZRows(tuple(order))))
            counts = 3+2*(order % 3)
            for i,count in enumerate(counts):
                assert np.all(indices[i,int(count):]>=512), ('G sentinel alias',i)
                assert len(np.unique(indices[i,int(count):])) == indices.shape[1]-int(count)
            del data
            print(json.dumps(dict(status='PASS',rank=jax.process_index(),P=jax.process_count(),mode=a.mode,
                read_seconds=io_seconds,nb=hi-lo,window=[lo,hi],nk=nk_stored,coeff_exact=True,zero_pad_exact=True,
                rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                total_seconds=time.perf_counter()-started)),flush=True)
        else:
            grid = np.asarray(wfn.fft_grid,dtype=int)
            coords = np.array(np.unravel_index(np.array([0,1,7,8,63,64,127,128,255,256,511,37,90,181,278,309,410,453,499]),grid)).T
            meta = Meta.from_system(wfn,sym,nval=min(int(wfn.nelec),103),
                ncond=max(0,nb_stored-int(wfn.nelec)),nband=nb_stored,n_rmu=len(coords),bispinor=False)
            meta.memory_per_device_gb = 8.
            requests = []
            if a.mode == 'kaxis':
                requests = [order]
                lo,hi = 3,106
            else:
                for start in range(0,nk_stored,a.k_count):
                    requests.append(order[start:start+a.k_count])
                    if len(requests) >= (1 if a.mode=='baxis' else a.stages):
                        break
                lo,hi = 3,nb_stored
            if a.band_start is not None:
                lo = a.band_start
            if a.band_stop is not None:
                hi = a.band_stop
            for stage,rows in enumerate(requests):
                phase_cache = {}
                before_cache = cache_snapshot(wfn)
                before = time.perf_counter()
                faces = load_centroids_band_chunked(wfn,sym,meta,coords,False,mesh,(lo,hi),
                    band_chunk_size=a.band_chunk,k_chunk_size=a.k_chunk,
                    k_domain='full_bz',full_k_rows=tuple(map(int,rows)))
                jax.block_until_ready(faces)
                sample_seconds = time.perf_counter()-before
                after_cache = cache_snapshot(wfn)
                assert before_cache == after_cache, ('retained request cache',stage,before_cache,after_cache)
                worst = 0.
                for which,value in enumerate(faces):
                    for shard in value.addressable_shards:
                        # Bounded expected blocks; no global reference face is allocated.
                        actual = np.asarray(shard.data)
                        ids = [np.arange(n)[sl] for n,sl in zip(value.shape,shard.index)]
                        k_ids = ids[0]; b_ids = ids[1 if which==0 else 2]
                        mu_ids = ids[3 if which==0 else 1]
                        for k0 in range(0,len(k_ids),32):
                            k1 = min(k0+32,len(k_ids))
                            for b0 in range(0,len(b_ids),512):
                                b1 = min(b0+512,len(b_ids))
                                expected = literal_faces(sym,rows[k_ids[k0:k1]],b_ids[b0:b1]+lo,mu_ids,coords,grid)
                                block = actual[k0:k1,b0:b1,0,:] if which==0 else actual[k0:k1,:,b0:b1,0].transpose(0,2,1).conj()
                                scale = max(1e-12,float(np.max(np.abs(expected),initial=0)))
                                error = float(np.max(np.abs(block-expected),initial=0))/scale
                                worst = max(worst,error)
                                assert error < 5e-12, ('face mismatch',which,shard.index,stage,k0,b0,error)
                                assert np.all(block[:,:,mu_ids>=len(coords)]==0), 'nonzero mu pad'
                print(json.dumps(dict(status='PASS',rank=jax.process_index(),P=jax.process_count(),mode=a.mode,
                    stage=stage,first_k=int(rows[0]),nk=len(rows),window=[lo,hi],band_chunk=a.band_chunk,
                    k_chunk=a.k_chunk,cache_before=before_cache,cache_after=after_cache,shapes=[list(x.shape) for x in faces],sample_seconds=sample_seconds,
                    max_relative=worst,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    total_seconds=time.perf_counter()-started)),flush=True)
                del faces, value, shard, actual, block, expected
                phase_cache.clear()
                gc.collect()
    runtime.finalize_process(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
