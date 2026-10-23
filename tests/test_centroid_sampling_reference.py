"""Shared centroid sampling versus raw-file sparse Fourier/Pauli sums.

Compute-node script for P4/P16 and larger meshes. The independent reference
reads raw HDF coefficients/G vectors and uses only canonical typed symmetry
metadata as physics inputs. No WFN coefficient loader, FFT, unfold kernel,
face reshard, or sampling schedule supplies reference values. Includes
nb<P, terminal k/b tiles, user/file band padding, retained canonical parents,
reused full-BZ G carriers and both kinetic-balance representations.
"""


def main():
    import argparse
    import runtime
    stack = runtime.initialize_communicator_stack()
    import json
    from pathlib import Path
    import time
    import h5py
    import jax
    import numpy as np
    from common.meta import Meta
    from common.wfn_transforms import load_centroids_band_chunked, load_psi_gflat_padded
    from wfn_loader import WfnLoader

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wfn', default=str(Path(__file__).parent / 'hsuite/fixture/WFN.h5'))
    p.add_argument('--case', choices=('all', 'narrow', 'multi', 'pad', 'raw', 'parents', 'bulk'), default='all')
    p.add_argument('--backend', choices=('auto', 'eager', 'phdf5'), default='auto')
    p.add_argument('--lift', choices=('none', 'raw', 'normalized_rkb'), default='none')
    args = p.parse_args()
    mesh = stack.mesh
    started = time.perf_counter()


    def raw_coeff(f, parent, lo, hi, counts):
        start = int(np.sum(counts[:parent])); stop = start+int(counts[parent])
        data = f['wfns/coeffs'][lo:hi, :, start:stop, :]
        return data[..., 0]+1j*data[..., 1], f['wfns/gvecs'][start:stop]


    def fourier_reference(wfn, sym, coords, lo, hi, user_end, *, raw, rows, mu_extent):
        """Raw HDF coefficients plus analytic plane-wave/Pauli sums.

        Full-zone cases treat the service's typed R/tau/U/TR action metadata as
        physics inputs. No loader coefficient, unfold kernel, FFT, face reshard,
        or batching implementation supplies the reference numerical values.
        """
        nb = hi-lo; ns = 4 if args.lift != 'none' else int(wfn.nspinor)
        expected = np.zeros((len(rows), nb, ns, mu_extent), complex)
        grid = np.asarray(wfn.fft_grid)
        positions = coords/grid
        with h5py.File(wfn.path, 'r') as f:
            counts = f['mf_header/kpoints/ngk'][:]
            k_table = f['mf_header/kpoints/rk'][:]
            real_hi = min(hi, user_end, int(wfn.nbands))
            if lo >= real_hi:
                return expected
            for out, row in enumerate(rows):
                parent = int(row) if raw else int(sym.irr_idx_k[int(row)])
                coeff, g = raw_coeff(f, parent, lo, real_hi, counts)
                if raw:
                    physical_K = g+k_table[parent]
                else:
                    op = int(sym.sym_idx_k[int(row)])
                    R, tau, tr = sym.operation_rows(op)
                    U = np.asarray(sym.spinor_action(op, nspinor=int(wfn.nspinor)))
                    rg = np.asarray(g)@np.asarray(R).T
                    coeff = np.conj(coeff) if bool(tr) else coeff
                    coeff = np.einsum('st,btg->bsg', U, coeff)
                    coeff *= np.exp(-1j*(rg@np.asarray(tau)))[None, None, :]
                    # Canonical full-k representatives retain the service's gauge,
                    # including finite-decimal WFN k-point input. Only the
                    # reciprocal-lattice offset is integer-valued.
                    full_k = np.asarray(sym.unfolded_kpts[int(row)])
                    rotated_k = k_table[parent]@np.asarray(R).T
                    umklapp = np.rint(full_k-rotated_k).astype(np.int64)
                    physical_K = rg-umklapp+full_k
                if args.lift != 'none':
                    momentum = physical_K@(float(wfn.blat)*np.asarray(wfn.bvec))
                    x, y, z = momentum.T
                    up, down = coeff[:, 0], coeff[:, 1]
                    small = .00364867628215*np.stack((
                        z*up+(x-1j*y)*down,
                        (x+1j*y)*up-z*down), axis=1)
                    coeff = np.concatenate((coeff, small), axis=1)
                    if args.lift == 'normalized_rkb':
                        coeff /= np.sqrt(1+.00364867628215**2*np.sum(momentum**2, axis=1))[None, None, :]
                phases = np.exp(2j*np.pi*(physical_K@positions.T))/np.sqrt(np.prod(grid))
                expected[out, :real_hi-lo, :, :len(coords)] = np.einsum('bsg,gm->bsm', coeff, phases)
        return expected


    with WfnLoader(args.wfn, mesh=mesh, backend=args.backend) as wfn:
        requested_case, requested_lift = args.case, args.lift
        cases = [(requested_case, requested_lift)]
        if requested_case == 'all':
            cases = [(name, 'none') for name in ('narrow', 'multi', 'pad', 'parents', 'bulk')]
            if int(wfn.nspinor) == 2:
                cases += [('narrow', name) for name in ('raw', 'normalized_rkb')]
        for args.case, args.lift in cases:
            if args.lift != 'none' and int(wfn.nspinor) != 2:
                raise ValueError('bispinor oracle requires a two-component fixture')
            sym = wfn.symmetry()
            grid = np.asarray(wfn.fft_grid, dtype=int)
            seed_points = np.array([[0,0,0], [grid[0]-1,0,0], [0,grid[1]-1,0],
                                   [0,0,grid[2]-1], grid-1])
            flat = list(dict.fromkeys(np.ravel_multi_index(seed_points.T, grid).tolist()))
            for value in np.random.default_rng(9517).permutation(int(np.prod(grid))):
                if int(value) not in flat:
                    flat.append(int(value))
                if len(flat) == 19:
                    break
            coords = np.asarray(np.unravel_index(flat, grid)).T
            user_end = int(wfn.nbands)-1 if args.case == 'pad' else int(wfn.nbands)
            meta = Meta.from_system(wfn, sym, nval=min(int(wfn.nelec), user_end),
                ncond=max(0, user_end-int(wfn.nelec)), nband=user_end,
                n_rmu=len(coords), bispinor=args.lift != 'none')
            meta.memory_per_device_gb = 8.
            lo, hi = (min(3, int(wfn.nbands)-2), min(8, int(wfn.nbands)))
            if args.case == 'multi':
                lo, hi = 1, int(wfn.nbands)
            if args.case == 'pad':
                lo, hi = max(0, int(wfn.nbands)-5), int(meta.b_id_4)
            raw = args.case == 'raw'
            full_rows = None
            if args.case not in ('raw', 'bulk', 'parents'):
                # Reversed, noncontiguous request order; include a late full-zone row.
                rows = list(dict.fromkeys([int(sym.nk_tot)-1, 0, int(sym.nk_tot)//2]))
                full_rows = tuple(rows)
            else:
                rows = list(range(int(wfn.nkpts) if raw else int(sym.nk_tot)))
            psi = None
            if args.case == 'bulk':
                psi = load_psi_gflat_padded(wfn, (lo, hi), mesh_xy=mesh,
                    bispinor=args.lift != 'none', k='full_bz', bispinor_lift=args.lift if args.lift != 'none' else 'raw')
            before = time.perf_counter()
            output = load_centroids_band_chunked(wfn, sym, meta, coords,
                args.lift != 'none', mesh, (lo, hi), band_chunk_size=3,
                k_chunk_size=None if args.case == 'bulk' else (1 if args.case == 'parents' else 2),
                k_domain='ibz' if raw else 'full_bz', full_k_rows=full_rows,
                return_ibz_parents=args.case == 'parents', psi_G_flat=psi,
                bispinor_lift=args.lift if args.lift != 'none' else 'raw')
            jax.block_until_ready(output)
            sample_seconds = time.perf_counter()-before
            reference = fourier_reference(wfn, sym, coords, lo, hi, user_end,
                raw=raw, rows=rows, mu_extent=int(meta.n_rmu_padded))
            refs = [reference, reference.conj().transpose(0,3,1,2)]
            if args.case == 'parents':
                parent_ref = fourier_reference(wfn, sym, coords, lo, hi, user_end,
                    raw=True, rows=list(range(int(wfn.nkpts))), mu_extent=int(meta.n_rmu_padded))
                # Only canonical representatives selected by the full-zone map
                # are retained by this request; surplus stored WFN rows are unused.
                canonical = np.unique(np.asarray(sym.irr_idx_k))
                unused = np.setdiff1d(np.arange(int(wfn.nkpts)), canonical)
                parent_ref[unused] = 0
                refs += [parent_ref, parent_ref.conj().transpose(0,3,1,2)]
            worst = 0.
            for value, expected in zip(output, refs):
                assert value.shape == expected.shape, (value.shape, expected.shape)
                for shard in value.addressable_shards:
                    actual = np.asarray(shard.data); ref = expected[shard.index]
                    scale = max(1e-14, float(np.max(np.abs(ref), initial=0)))
                    error = float(np.max(np.abs(actual-ref), initial=0))/scale
                    worst = max(worst, error)
                    
                    assert error < 2e-10, (args.case, args.lift, shard.index, error)
                    assert np.all(actual[ref == 0] == 0), 'a band/k/mu pad has nonzero data'
            print(json.dumps(dict(status='PASS', rank=jax.process_index(), P=jax.process_count(),
                case=args.case, lift=args.lift, backend=wfn.backend, window=[lo,hi],
                logical_mu=len(coords), carrier_mu=meta.n_rmu_padded, output_shapes=[list(x.shape) for x in output],
                full_k_rows=full_rows, sample_seconds=sample_seconds,
                max_relative=worst, total_seconds=time.perf_counter()-started)), flush=True)
    runtime.finalize_process(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
