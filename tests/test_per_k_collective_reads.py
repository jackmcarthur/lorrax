"""Per-k WFN streams read k tiles: parallel-transport links and the QP-WFN rotation.

Compute-node script for P4 on one node (``srun -n 4``; files go to the node's
``$TMPDIR``); the pytest entry skips without the launcher.

(a) Parallel-transport links (``file_io.parallel_transport._write_link_stage``)
    on the hsuite H2- fixture (25 full-zone centres, 2 forward steps): the
    fixed tile holds every centre, so the stage makes two collective WFN
    reads (the centres and the union of their neighbours), not ``1 + n_dir``
    per centre; with one centre per tile it makes two per centre, and the
    links and singular values agree with the one-tile ones to 1e-12.
(b) The QP-WFN rotation (``file_io.qp_wfn.write_qp_wfn_h5``) on the same
    fixture (9 stored k): one collective coefficient read for the whole file
    when the window admits every k row, and the rotated coefficients equal
    ``sum_m U[k, m, n] c[m]`` on the active bands and the source elsewhere,
    from the raw file, to 1e-12.

Before 2026-11-21 both streams made one collective read per k (links:
``1 + n_dir`` per centre), which on a slow CFS set the dipole step's wall.
``--keep DIR`` keeps the output files (rank 0 copies them).
"""


def main():
    import argparse
    import os
    import shutil
    from pathlib import Path
    import runtime
    stack = runtime.initialize_communicator_stack()
    import h5py
    import jax
    import numpy as np
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
    assert mesh.size == 4, mesh
    rank = jax.process_index()
    root = Path(os.environ.get('TMPDIR', '/tmp')) / (
        f"lorrax_per_k_reads_{os.environ.get('SLURM_JOB_ID', '0')}_"
        f"{os.environ.get('SLURM_STEP_ID', '0')}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    barrier('per_k_reads.mkdir')
    fixture = Path(__file__).parent / 'hsuite/fixture/WFN.h5'

    def link_stage(wfn, sym, name, polar):
        """Run the link stage into ``root/name``; return its path and its WFN reads."""
        reads, load = [], wfn.load

        def counted(*a, **kw):
            reads.append(kw.get('k'))
            return load(*a, **kw)
        path = root / name
        with SlabIO(str(path), mode='w', mesh=mesh):
            pass
        wfn.load = counted
        try:
            pt._write_link_stage(str(path), wfn=wfn, sym=sym, mesh=mesh,
                                 nbands=int(wfn.nbands), bispinor=False,
                                 polar_plan=polar)
        finally:
            wfn.load = load
        return path, len(reads)

    # (a) parallel-transport links, the fixed tile
    with WfnLoader(fixture, mesh=mesh) as wfn:
        sym = wfn.symmetry()
        plan_polar, _, _ = pt._require_service_apis()
        polar = plan_polar(mesh, n=pt.band_storage_extent(mesh, int(wfn.nbands)),
                           backend='auto', rcond=1.0e-10,
                           batched_route=ROUTE_BATCH_RESHARD)
        reduced = pt.link_symmetry_reduction_applies(
            sym, tuple(int(n) for n in np.asarray(wfn.kgrid).reshape(3)), wfn=wfn)
        n_centres = (int(np.asarray(sym.kirr_fullids).size) if reduced
                     else int(sym.nk_tot))
        n_dir = int(pt.wfn_link_stencil(wfn).steps.shape[0])
        one_tile, reads_one = link_stage(wfn, sym, 'links_tile.h5', polar)

    # (b) QP-WFN rotation
    b0, b1 = 2, 6
    with WfnLoader(fixture, mesh=mesh) as wfn:
        nk = int(wfn.nkpts)
        rng = np.random.default_rng(20261121)
        z = (rng.standard_normal((nk, b1 - b0, b1 - b0))
             + 1j * rng.standard_normal((nk, b1 - b0, b1 - b0)))
        U = np.linalg.qr(z)[0]
        E = np.asarray(wfn.energies[0], dtype=np.float64)[:, b0:b1].copy()
        coeff_reads, read_slab = [], SlabIO.read_slab

        def counted_read(self, name, *a, **kw):
            if name == 'wfns/coeffs':
                coeff_reads.append(kw.get('offset'))
            return read_slab(self, name, *a, **kw)
        SlabIO.read_slab = counted_read
        try:
            out = root / 'WFN_qp.h5'
            qp_wfn.write_qp_wfn_h5(str(out), wfn, U, E, b0, b1, mesh=mesh)
        finally:
            SlabIO.read_slab = read_slab

    # Reads: two for the links' one tile, one for the QP coefficients.
    assert (reads_one, len(coeff_reads)) == (2, 1), (
        f'links: {reads_one} reads for {n_centres} centres x (1 + {n_dir}); '
        f'qp_wfn: {len(coeff_reads)} coefficient reads for {nk} k')

    # (a) one centre per tile: two reads per centre, the same links.
    with WfnLoader(fixture, mesh=mesh) as wfn:
        sym = wfn.symmetry()
        planner = pt._link_center_tile
        pt._link_center_tile = lambda *a, **kw: 1
        try:
            per_centre, reads_per = link_stage(wfn, sym, 'links_per_centre.h5', polar)
        finally:
            pt._link_center_tile = planner
    assert reads_per == 2 * n_centres, (reads_per, n_centres)
    barrier('per_k_reads.written')
    with h5py.File(one_tile, 'r') as fa, h5py.File(per_centre, 'r') as fb:
        for name in (pt.LINKS_DATASET, pt.SINGULAR_VALUES_DATASET):
            a, b = fa[name][...], fb[name][...]
            scale = float(np.abs(a).max())
            assert scale > 0, name
            np.testing.assert_allclose(b, a, rtol=0, atol=1e-12 * scale, err_msg=name)
        assert set(fa.attrs) == set(fb.attrs), (sorted(fa.attrs), sorted(fb.attrs))
        for attr in fa.attrs:
            np.testing.assert_array_equal(fa.attrs[attr], fb.attrs[attr], err_msg=attr)

    # (b) the rotated coefficients against the raw file.
    with h5py.File(fixture, 'r') as fs, h5py.File(out, 'r') as fo:
        src = fs['wfns/coeffs'][...]
        got = fo['wfns/coeffs'][...]
        ngk = fs['mf_header/kpoints/ngk'][...]
    c = src[..., 0] + 1j * src[..., 1]
    want = c.copy()
    starts = np.concatenate([[0], np.cumsum(ngk)[:-1]])
    for k in range(nk):
        g = slice(int(starts[k]), int(starts[k] + ngk[k]))
        want[b0:b1, :, g] = np.einsum('mn,msg->nsg', U[k], c[b0:b1, :, g])
    have = got[..., 0] + 1j * got[..., 1]
    err = float(np.abs(have - want).max()) / float(np.abs(c).max())
    assert err <= 1e-12, err

    if args.keep and rank == 0:
        keep = Path(args.keep)
        keep.mkdir(parents=True, exist_ok=True)
        for path in (one_tile, per_centre, out):
            shutil.copy2(path, keep / path.name)
    print(f'PER_K_READS PASS rank={rank} links: centres={n_centres} n_dir={n_dir} '
          f'reads one tile={reads_one} one centre per tile={reads_per}; '
          f'qp_wfn: k={nk} coefficient reads={len(coeff_reads)} '
          f'max|dc|/max|c|={err:.1e}', flush=True)
    barrier('per_k_reads.done')
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip('run directly under the configured MPI launcher')


if __name__ == '__main__':
    main()
