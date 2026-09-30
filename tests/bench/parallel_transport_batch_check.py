"""P4 real-WFN link, artifact and bounded polar-batch regression."""
from runtime import initialize_communicator_stack, run_main_and_finalize
R = initialize_communicator_stack()
import argparse
from dataclasses import replace
from pathlib import Path
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import device_put_process_local, gather_to_host, rank0_transaction
from common.meta import Meta
from common.parallel_transport import (
    band_storage_extent, build_forward_neighbor_table, build_g_wrap_lookup,
    g_wrap_for_forward_step, make_cross_k_overlap)
from common.wfn_layout import band_sphere_spec
from distrib_la import plan_polar_factor, ROUTE_BATCH_RESHARD
from file_io import WFNReader
from file_io.slab_io import SlabIO
from gw.qsgw_head import load_parallel_transport_head, load_dft_velocity_head


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', required=True)
parser.add_argument('--spinor-wfn', required=True)
parser.add_argument('--perf-wfn')
parser.add_argument('--perf-bands', type=int, default=225)
args = parser.parse_args()


def host(value):
    return np.asarray(gather_to_host(value))


def raw_edges(wfn, nb, count, *, bispinor=False):
    """Use the canonical actual-WFN overlap and reciprocal-wrap owners."""
    sym = wfn.symmetry()
    plus = build_forward_neighbor_table(sym.kvecs_asints, wfn.kgrid)
    center_on_x, overlap = make_cross_k_overlap(R.mesh)
    centers = (0, int(sym.nk_tot)//2, int(sym.nk_tot)-1)
    rows, ids = [], []
    for center in centers:
        psi = wfn.load(bands=(0, nb), k=[center],
                       sharding=band_sphere_spec(), bispinor=bispinor)
        cx = center_on_x(psi)
        gc = wfn.gvecs(k=[center])[0]
        nc = int(wfn.ngk_valid(k=[center])[0])
        for direction in range(3):
            neighbor = int(plus[center, direction])
            wrap = g_wrap_for_forward_step(sym.unfolded_kpts, center,
                                          neighbor, direction, wfn.kgrid)
            gi, valid = build_g_wrap_lookup(
                wfn.gvecs(k=[neighbor])[0], gc, wrap,
                ngk_neighbor=int(wfn.ngk_valid(k=[neighbor])[0]), ngk_center=nc)
            pn = wfn.load(bands=(0, nb), k=[neighbor],
                          sharding=band_sphere_spec(), bispinor=bispinor)
            raw = overlap(cx, pn, gi, valid)
            raw.block_until_ready()
            rows.append(raw)
            ids.append((center, direction))
            del pn
            if len(rows) == count:
                return jnp.stack(rows), ids
        del psi, cx
    return jnp.stack(rows), ids


def compare(raw, *, rcond=1e-10):
    n = int(raw.shape[-1])
    legacy = plan_polar_factor(R.mesh, n=n, backend='distributed', rcond=rcond)
    batch = plan_polar_factor(R.mesh, n=n, backend='distributed', rcond=rcond,
                             budget_bytes=512*1024**2)
    assert batch.route_for(raw.shape, raw.dtype) == ROUTE_BATCH_RESHARD
    t0 = time.monotonic()
    reference = [legacy(raw[i]) for i in range(len(raw))]
    jax.block_until_ready(reference)
    distributed_wall = time.monotonic()-t0
    t0 = time.monotonic()
    link, values = batch.batched(raw)
    jax.block_until_ready((link, values))
    cold = time.monotonic()-t0
    t0 = time.monotonic()
    jax.block_until_ready(batch.batched(raw))
    warm = time.monotonic()-t0
    link_host, value_host = host(link), host(values)
    reference_link = np.stack([host(row[0]) for row in reference])
    reference_values = np.stack([host(row[1]) for row in reference])
    error = float(np.linalg.norm(link_host-reference_link)/max(np.linalg.norm(reference_link),1e-30))
    sv_error = float(np.max(np.abs(value_host-reference_values)))
    assert error < 5e-10, error
    assert sv_error < 5e-12, sv_error
    assert link.sharding.spec == P(None, 'x', 'y')
    tiny = plan_polar_factor(R.mesh, n=n, backend='distributed', rcond=rcond,
                            budget_bytes=1)
    assert tiny.route_for(raw.shape, raw.dtype) != ROUTE_BATCH_RESHARD
    fallback_l, fallback_s = tiny.batched(raw[:1])
    assert np.linalg.norm(host(fallback_l)[0]-reference_link[0]) < 5e-10
    assert np.max(np.abs(host(fallback_s)[0]-reference_values[0])) < 5e-12
    return dict(n=n, edges=int(raw.shape[0]), relative_link_error=error,
                max_singular_value_error=sv_error,
                distributed_wall_s=distributed_wall, batch_cold_wall_s=cold,
                batch_warm_wall_s=warm), link_host, value_host


def main():
    out = Path(args.out)
    receipts = []
    with R.mesh:
        # Closed singular clusters, a small retained direction, rejected
        # nonzero directions and exact null padding; compare the invariant
        # polar matrix, never the gauge-dependent eigenvectors.
        rng = np.random.default_rng(71)
        n = 8
        left = np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
        right = np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
        singular = np.array([1,.5,.5,.3,.05,.01,1e-5,0])
        matrix = (left*singular)@right.conj().T
        raw = device_put_process_local(np.stack([matrix, matrix.conj(), matrix.T]),
                                      NamedSharding(R.mesh, P(None,'x','y')))
        row, got, _ = compare(raw, rcond=1e-4)
        expected = (left*(singular>1e-4))@right.conj().T
        assert np.linalg.norm(got[0]-expected) < 5e-11
        row['kind'] = 'rank_deficient_clustered'
        receipts.append(row)

        # Genuine Ns=2 and native kinetic-balance Ns=4 WFN overlaps. Seven
        # edges exercise a leading-batch remainder and reciprocal wrapping.
        spinor = WFNReader(args.spinor_wfn, mesh=R.mesh)
        for bispinor in (False, True):
            raw, _ = raw_edges(spinor, 5, 7, bispinor=bispinor)
            row, _, _ = compare(raw)
            row.update(kind='actual_spinor_wfn', nspinor=4 if bispinor else 2)
            receipts.append(row)

        # Full genuine bcc fixture producer: 27 full-zone centers, 81 edges,
        # 14 logical / 16 carrier bands and a one-edge final batch at P4.
        from psp.get_dipole_mtxels import main as dipole_main
        dipole_main(['-i', str(out/'pt.in'), '--out', str(out/'dipole.h5'),
                     '--parallel-transport-out', str(out/'pt.h5'),
                     '--parallel-transport-bands', '14'])
        dipole_main(['-i', str(out/'pt.in'), '--out', str(out/'dipole_velocity.h5'),
                     '--parallel-transport-out', str(out/'velocity.h5'),
                     '--parallel-transport-velocity-only'])
        wfn = WFNReader(str(out/'WFN.h5'), mesh=R.mesh)
        sym = wfn.symmetry()
        meta = Meta.from_system(wfn,sym,nval=5,ncond=8,nband=13,n_rmu=56,bispinor=False)
        pt = load_parallel_transport_head(str(out/'pt.h5'), mesh=R.mesh,
                                         sym=sym,wfn=wfn,meta=meta)
        velocity = load_dft_velocity_head(str(out/'velocity.h5'),mesh=R.mesh,
                                         sym=sym,wfn=wfn,meta=meta)
        v_error = float(jax.device_get(jnp.max(jnp.abs(
            pt.velocity_dft_cart-velocity.velocity_dft_cart))))
        assert v_error < 5e-12, v_error
        assert pt.nb_links == 14
        raw, ids = raw_edges(wfn,14,7)
        row, reference_link, reference_values = compare(raw)
        links = host(pt.forward_links)
        sv = np.asarray(pt.singular_values)
        for i,(center,direction) in enumerate(ids):
            assert np.linalg.norm(links[direction,center]-reference_link[i]) < 5e-10
            assert np.max(np.abs(sv[center,direction]-reference_values[i,:14])) < 5e-12
        # A consumer needing an unjudged manifold must still refuse.
        try:
            load_parallel_transport_head(str(out/'pt.h5'),mesh=R.mesh,
                sym=sym,wfn=wfn,meta=replace(meta,b_id_4_user=15))
        except ValueError as exc:
            assert 'does not contain' in str(exc) or 'judged bands' in str(exc)
        else:
            raise AssertionError('PT reader accepted a mismatched manifold')
        try:
            load_parallel_transport_head(str(out/'pt.h5'),mesh=R.mesh,
                sym=sym,wfn=spinor,meta=meta)
        except ValueError as exc:
            assert 'WFN fingerprint differs' in str(exc)
        else:
            raise AssertionError('PT reader accepted a different source WFN')
        with SlabIO(str(out/'pt.h5'),mode='r',mesh=R.mesh) as io:
            assert int(io.read_small('connection_complete',dtype=np.int32)) == 1
            assert int(io.read_small('velocity_validation_complete',dtype=np.int32)) == 1
            assert int(io.read_small('links_symmetry_reduced',dtype=np.int32)) == 0
        row.update(kind='bcc_artifact',nk=int(meta.nk_tot),velocity_max_error=v_error,
                   outer_bands=14,head_bands=13,authenticated=True)
        receipts.append(row)

        if args.perf_wfn:
            wfn_perf = WFNReader(args.perf_wfn,mesh=R.mesh)
            raw, _ = raw_edges(wfn_perf,args.perf_bands,7)
            row,_,_ = compare(raw)
            row.update(kind='production_size_actual_wfn',logical_bands=args.perf_bands)
            receipts.append(row)
    rank0_transaction(lambda:(out/'parity.json').write_text(json.dumps(receipts,indent=2)+'\n'))
    print(json.dumps(receipts),flush=True)


run_main_and_finalize(main)
