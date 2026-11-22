"""Per-k WFN streams read k tiles: parallel-transport links and the QP-WFN rotation.

Compute-node script, any process count on one node (P4 is the verdict;
``srun -n 4``; files go to the node's ``$TMPDIR``); the pytest entry skips
without the launcher.  Both repository fixtures: H2- (25 full-zone centres
over 9 stored k, 2 forward steps, 2 spinors) and bcc Na (27 over 4, 6 steps,
1 spinor).

(a) Parallel-transport links (``file_io.parallel_transport._write_link_stage``).
    The reference is the per-k path, forced by ``_link_center_tile`` =
    ``(1, False)``: one centre held, each neighbour read on its own as it is
    used (``1 + n_dir`` reads per centre, the pre-tile pattern, also the
    fallback when one centre's spheres do not fit the tile).  The fixed
    tile (every centre: 2 reads) and a 7-centre tile (remainders) give the
    same links, singular values and attributes to 1e-12.  A grid with no
    forward step (``n_dir = 0``, a 1x1x1 grid) reads nothing and writes the
    empty tables.  The links are written once per tile of whole centres
    (once per centre on the per-k path), not once per edge.
(b) Little-group band representations (``file_io.qp_wfn``): every raw
    parent at once; identity-only groups need no read and the rest are one
    read for the fixed tile (one per parent at a tile of one), with the same
    representations to 1e-12.
(c) The QP-WFN rotation (``file_io.qp_wfn.write_qp_wfn_h5``): the fixed tile,
    windows of 4 and 3 k rows (a last window with ranks that hold no
    column), one row per window, and G windows narrower than a row; the
    coefficient reads are the windows the plan makes, and the rotated
    coefficients equal ``sum_m U[k, m, n] c[m]`` on the active bands and the
    source elsewhere, from the raw file, to 1e-12.

``--keep DIR`` keeps the output files (rank 0 copies them).
"""


def main():
    import argparse
    import os
    import shutil
    from pathlib import Path
    from types import SimpleNamespace
    import runtime
    stack = runtime.initialize_communicator_stack()
    import h5py
    import jax
    import numpy as np
    import runtime.tiles as tiles
    from common.collectives import barrier
    from distrib_la import ROUTE_BATCH_RESHARD
    from file_io import parallel_transport as pt
    from file_io import qp_wfn
    from file_io.slab_io import SlabIO
    from wfn_loader import WfnLoader

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--keep', default=None)
    args = p.parse_args()
    mesh = stack.mesh
    P = int(mesh.size)
    rank = jax.process_index()
    root = Path(os.environ.get('TMPDIR', '/tmp')) / (
        f"lorrax_per_k_reads_{os.environ.get('SLURM_JOB_ID', '0')}_"
        f"{os.environ.get('SLURM_STEP_ID', '0')}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    barrier('per_k_reads.mkdir')
    here = Path(__file__).parent / 'hsuite'
    fixtures = (('h2m', here / 'fixture/WFN.h5', (2, 6), (0, 9)),
                ('na', here / 'fixture_na/WFN.h5', (3, 9), (0, 1)))
    report = []

    def same(a_path, b_path, what):
        with h5py.File(a_path, 'r') as fa, h5py.File(b_path, 'r') as fb:
            names = []
            fa.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
            got = []
            fb.visititems(lambda n, o: got.append(n) if isinstance(o, h5py.Dataset) else None)
            assert sorted(names) == sorted(got), (what, names, got)
            for name in names:
                a, b = fa[name][...], fb[name][...]
                assert a.shape == b.shape, (what, name, a.shape, b.shape)
                if a.size:
                    scale = float(np.abs(a).max()) or 1.0
                    np.testing.assert_allclose(b, a, rtol=0, atol=1e-12 * scale,
                                               err_msg=f'{what} {name}')
            assert set(fa.attrs) == set(fb.attrs), (what, sorted(fa.attrs), sorted(fb.attrs))
            for key in fa.attrs:
                np.testing.assert_array_equal(fa.attrs[key], fb.attrs[key], err_msg=f'{what} {key}')

    for fx, path, (b0, b1), window in fixtures:
        # (a) links
        with WfnLoader(path, mesh=mesh) as wfn:
            sym = wfn.symmetry()
            nb = int(wfn.nbands)
            plan_polar, _, _ = pt._require_service_apis()
            polar = plan_polar(mesh, n=pt.band_storage_extent(mesh, nb), backend='auto',
                               rcond=1.0e-10, batched_route=ROUTE_BATCH_RESHARD)
            reduced = pt.link_symmetry_reduction_applies(
                sym, tuple(int(n) for n in np.asarray(wfn.kgrid).reshape(3)), wfn=wfn)
            nrk = int(np.asarray(sym.kirr_fullids).size) if reduced else int(sym.nk_tot)
            nd = int(pt.wfn_link_stencil(wfn).steps.shape[0])
            planner, stencil = pt._link_center_tile, pt.wfn_link_stencil

            def links(name, plan=None, no_steps=False):
                reads, load, writes, write_slab = [], wfn.load, [], SlabIO.write_slab

                def counted_write(self, dataset, *a, **kw):
                    if dataset == pt.LINKS_DATASET:
                        writes.append(kw.get('offset'))
                    return write_slab(self, dataset, *a, **kw)

                def counted(*a, **kw):
                    reads.append(kw.get('k'))
                    return load(*a, **kw)
                out = root / f'{fx}_{name}.h5'
                with SlabIO(str(out), mode='w', mesh=mesh):
                    pass
                if plan is not None:
                    pt._link_center_tile = lambda *a, **kw: plan
                if no_steps:
                    pt.wfn_link_stencil = lambda w: SimpleNamespace(
                        steps=np.zeros((0, 3), dtype=np.int32))
                wfn.load = counted
                SlabIO.write_slab = counted_write
                try:
                    pt._write_link_stage(str(out), wfn=wfn, sym=sym, mesh=mesh, nbands=nb,
                                         bispinor=False, polar_plan=polar)
                finally:
                    wfn.load = load
                    SlabIO.write_slab = write_slab
                    pt._link_center_tile, pt.wfn_link_stencil = planner, stencil
                return out, len(reads), len(writes)

            tile, hold = planner(mesh, nbands=nb, nspinor=int(wfn.nspinor),
                                 file_nspinor=int(wfn.nspinor), ngkmax=int(wfn.ngkmax),
                                 n_dir=nd, n_centers=nrk)
            per_k, reads_per_k, writes_per_k = links('per_k', plan=(1, False))
            fixed, reads_fixed, writes_fixed = links('fixed')
            seven, reads_seven, writes_seven = links('tile7', plan=(7, True))
            empty, reads_empty, writes_empty = links('no_steps', no_steps=True)
        assert hold and tile == nrk, (fx, tile, hold)
        assert reads_per_k == nrk * (1 + nd), (fx, reads_per_k, nrk, nd)
        assert reads_fixed == 2, (fx, reads_fixed)
        assert reads_seven == 2 * (-(-nrk // 7)), (fx, reads_seven)
        assert reads_empty == 0 and writes_empty == 0, (fx, reads_empty, writes_empty)
        # One hyperslab per tile of whole centres (per centre on the per-k path).
        assert (writes_per_k, writes_fixed, writes_seven) == (nrk, 1, -(-nrk // 7)), (
            fx, writes_per_k, writes_fixed, writes_seven)
        barrier(f'per_k_reads.{fx}.links')
        same(per_k, fixed, f'{fx} links fixed tile')
        same(per_k, seven, f'{fx} links 7-centre tile')
        with h5py.File(empty, 'r') as f:
            assert f[pt.SINGULAR_VALUES_DATASET].shape == (nrk, 0, nb), fx
            assert f['source_steps'].shape == (0, 3), fx
        report.append(f'{fx} links: centres={nrk} n_dir={nd} reads per-k={reads_per_k} '
                      f'fixed={reads_fixed} tile7={reads_seven} no-steps={reads_empty}; link writes per-k={writes_per_k} '
                      f'fixed={writes_fixed} tile7={writes_seven}')

        # (b) little-group band representations, every raw parent at once
        with WfnLoader(path, mesh=mesh) as wfn:
            parents = list(range(int(wfn.nkpts)))
            nontrivial = [q for q in parents if len(wfn.little_group_rows(q)) > 1]
            lg_tile = qp_wfn._little_group_tile

            def groups(tile_override=None):
                reads, load = [], wfn.load

                def counted(*a, **kw):
                    reads.append(kw.get('k'))
                    return load(*a, **kw)
                if tile_override is not None:
                    qp_wfn._little_group_tile = lambda *a, **kw: tile_override
                wfn.load = counted
                try:
                    reps = qp_wfn.little_group_band_representations(
                        wfn, mesh=mesh, parents=parents, band_start=window[0],
                        band_stop=window[1])
                finally:
                    wfn.load = load
                    qp_wfn._little_group_tile = lg_tile
                return reps, len(reads)

            reps_tile, reads_lg = groups()
            reps_one, reads_lg_one = groups(1)
        assert reads_lg == (1 if nontrivial else 0), (fx, reads_lg, len(nontrivial))
        assert reads_lg_one == len(nontrivial), (fx, reads_lg_one, len(nontrivial))
        for q in parents:
            a, b = reps_one[q], reps_tile[q]
            assert [r for r, _, _ in a] == [r for r, _, _ in b], (fx, q)
            for (_, Da, ua), (_, Db, ub) in zip(a, b):
                assert ua == ub, (fx, q)
                np.testing.assert_allclose(Db, Da, rtol=0, atol=1e-12, err_msg=f'{fx} parent {q}')
            if q not in nontrivial:
                np.testing.assert_array_equal(b[0][1], np.eye(window[1] - window[0]))
        report.append(f'{fx} little groups: parents={len(parents)} non-trivial={len(nontrivial)} '
                      f'reads tile={reads_lg} one per tile={reads_lg_one}')

        # (c) QP-WFN rotation
        with WfnLoader(path, mesh=mesh) as wfn:
            nk, nbf, ns = int(wfn.nkpts), int(wfn.nbands), int(wfn.nspinor)
            ngkmax, na = int(wfn.ngkmax), b1 - b0
            rng = np.random.default_rng(20261121)
            z = rng.standard_normal((nk, na, na)) + 1j * rng.standard_normal((nk, na, na))
            U = np.linalg.qr(z)[0]
            E = np.asarray(wfn.energies[0], dtype=np.float64)[:, b0:b1].copy()
            whole_k = -(-ngkmax // P)
            column = max(5 * nbf, 3 * nbf + 5 * na) * ns * 16
            row_bytes = column * whole_k + 16 * na * na
            cases = (('fixed', None), ('rows4', 4 * row_bytes), ('rows3', 3 * row_bytes),
                     ('row1', row_bytes), ('row_no_u', column * whole_k),
                     ('g_windows', column * max(1, whole_k // 3)))
            outs = []
            for name, tile_bytes in cases:
                saved = tiles.TILE_BYTES
                if tile_bytes is not None:
                    tiles.TILE_BYTES = int(tile_bytes)
                k_tile, width = qp_wfn._coefficient_window(
                    mesh, nbands=nbf, nspinor=ns, ngkmax=ngkmax, nb_active=na, nk=nk)
                ngk = np.asarray(wfn.ngk, dtype=np.int64)
                if width < P * whole_k:
                    want_reads = int(sum(-(-int(g) // width) for g in ngk))
                else:
                    want_reads = -(-nk // k_tile)
                coeff_reads, read_slab = [], SlabIO.read_slab

                def counted_read(self, name_, *a, **kw):
                    if name_ == 'wfns/coeffs':
                        coeff_reads.append(kw.get('offset'))
                    return read_slab(self, name_, *a, **kw)
                SlabIO.read_slab = counted_read
                out = root / f'{fx}_WFN_qp_{name}.h5'
                try:
                    qp_wfn.write_qp_wfn_h5(str(out), wfn, U, E, b0, b1, mesh=mesh)
                finally:
                    SlabIO.read_slab = read_slab
                    tiles.TILE_BYTES = saved
                assert len(coeff_reads) == want_reads, (fx, name, len(coeff_reads), want_reads)
                outs.append((name, out, k_tile, width, len(coeff_reads)))
        barrier(f'per_k_reads.{fx}.qp')
        with h5py.File(path, 'r') as fs:
            src = fs['wfns/coeffs'][...]
            ngk = fs['mf_header/kpoints/ngk'][...]
        c = src[..., 0] + 1j * src[..., 1]
        want = c.copy()
        starts = np.concatenate([[0], np.cumsum(ngk)[:-1]])
        for k in range(nk):
            g = slice(int(starts[k]), int(starts[k] + ngk[k]))
            want[b0:b1, :, g] = np.einsum('mn,msg->nsg', U[k], c[b0:b1, :, g])
        worst = 0.0
        for name, out, k_tile, width, n_reads in outs:
            with h5py.File(out, 'r') as fo:
                got = fo['wfns/coeffs'][...]
            have = got[..., 0] + 1j * got[..., 1]
            err = float(np.abs(have - want).max()) / float(np.abs(c).max())
            assert err <= 1e-12, (fx, name, err)
            worst = max(worst, err)
        report.append(f'{fx} qp_wfn: k={nk} ' + ' '.join(
            f'{name}(k_tile={k_tile},W={width},reads={n_reads})' for name, _, k_tile, width, n_reads in outs)
            + f' max|dc|/max|c|={worst:.1e}')

    if args.keep and rank == 0:
        keep = Path(args.keep)
        keep.mkdir(parents=True, exist_ok=True)
        for item in root.iterdir():
            shutil.copy2(item, keep / item.name)
    print(f'PER_K_READS PASS rank={rank} P={P} | ' + ' | '.join(report), flush=True)
    barrier('per_k_reads.done')
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip('run directly under the configured MPI launcher')


if __name__ == '__main__':
    main()
