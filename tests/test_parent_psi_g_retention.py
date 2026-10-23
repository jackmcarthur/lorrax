"""Compute-node P4 regression for retained fit bands and full centroid faces.

The fixture supplies index-coded raw coefficients through the loader protocol.
All transport, tiled direct Fourier sampling, face finishing and retention use
the production backend. The independent reference uses explicit Fourier sums.
"""


def main():
    import argparse
    import gc
    import json
    import os
    from pathlib import Path
    import resource
    import socket
    import time
    from types import SimpleNamespace
    import runtime
    stack = runtime.initialize_communicator_stack(platform="cpu")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding
    from common.psi_G_store import load_parent_psi_G
    from common.collectives import all_gather_processes
    from runtime.aot_memory import compiled_new_bytes
    from runtime.padding import padded_axis, spec_divisor

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--large', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    mesh = stack.mesh
    assert int(mesh.size) == 4
    pairs = set()
    for cpu in os.sched_getaffinity(0):
        p = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        pairs.add((p.joinpath('physical_package_id').read_text(), p.joinpath('core_id').read_text()))
    assert len(pairs) == 4 and os.environ.get('OMP_NUM_THREADS') == '2'
    assert len(jax.local_devices()) == 1
    assert os.environ.get('JAX_CPU_COLLECTIVES_IMPLEMENTATION') == 'mpi'
    args.out.mkdir(parents=True, exist_ok=True)

    grid = (16, 16, 16) if args.large else (8, 8, 8)
    ng, nk, ns = (4093, 1, 2) if args.large else (11, 2, 2)
    read = (0, 10000) if args.large else (3, 15)
    keep = (0, 128) if args.large else (5, 10)
    user_stop = 10000 if args.large else 12
    file_stop = 10000 if args.large else 13
    m = 64 if args.large else 5
    coords = np.stack(np.unravel_index(np.arange(m)*3, grid), axis=-1).astype(np.int32)
    meta = SimpleNamespace(nspinor=ns, nk_tot=nk, fft_grid=grid,
                           b_id_4_user=user_stop, memory_per_device_gb=8.0)
    nrtot = int(np.prod(grid))

    def coeff(k, b, s, g):
        value = (((17*k + 13*b + 7*s + 3*g) % 101) / 101.0
                 + 1j*((11*k + 5*b + 19*s + 23*g) % 103)/103.0)
        return np.where((b < file_stop) & (g < ng), value, 0).astype(np.complex128)

    class Loader:
        nbands, nkpts, nspinor, ngkmax = file_stop, nk, ns, ng

        def __init__(self):
            self.reads = []

        def box_index(self, *, k):
            return np.broadcast_to(np.arange(ng, dtype=np.int32), (nk, ng)).copy()

        def kvecs(self, *, k):
            return np.array([[i/4., 0., 0.] for i in range(nk)])

        def load(self, *, bands, k, sharding, **unused):
            self.reads.append(tuple(bands))
            lo, hi = bands
            width = padded_axis(hi-lo, spec_divisor(mesh, sharding, axis=1), name='fixture read bands').carrier
            shape = (nk, width, ns, ng)

            def values(index):
                axes = [np.arange(*sl.indices(n), dtype=np.int64)
                        for sl, n in zip(index, shape)]
                a, b, s, g = np.ix_(*axes)
                return np.where(b+lo < hi, coeff(a, b+lo, s, g), 0)
            return jax.make_array_from_callback(shape, NamedSharding(mesh, sharding), values)

    def local(a):
        dev = jax.local_devices()[0]
        return a.addressable_data(0), a.sharding.addressable_devices_indices_map(a.shape)[dev]

    def verify_store(a, interval, host=False):
        lo, hi = interval
        if host:
            rank = int(jax.process_index())
            shape = (nk, a.shape[1], ns, a.shape[-1]*int(mesh.size))
            index = (slice(None), slice(None), slice(None), slice(rank*a.shape[-1], (rank+1)*a.shape[-1]))
            actual = a
        else:
            part, index = local(a)
            shape, actual = a.shape, np.asarray(part)
        axes = [np.arange(*sl.indices(n), dtype=np.int64) for sl, n in zip(index, shape)]
        ka, ba, sa, ga = axes
        for start in range(0, len(ba), 128):
            b = ba[start:start+128]
            k_, b_, s_, g_ = np.ix_(ka, b+lo, sa, ga)
            expected = np.where(b_ < hi, coeff(k_, b_, s_, g_), 0)
            np.testing.assert_array_equal(actual[:, start:start+len(b)], expected)

    def verify_faces(faces, reference=None):
        result = []
        for position, a in enumerate(faces):
            part, index = local(a)
            actual = np.array(part)
            if reference is not None:
                np.testing.assert_array_equal(actual, reference[position])
            else:
                # Independent all-G Fourier sum, in bounded band slabs.
                target_shape = a.shape
                axes = [np.arange(*sl.indices(n), dtype=np.int64)
                        for sl, n in zip(index, target_shape)]
                ka, ba, sa, ma = axes if position == 0 else (axes[0], axes[2], axes[3], axes[1])
                for start in range(0, len(ba), 128):
                    bs = ba[start:start+128]
                    expected = np.zeros((len(ka), len(bs), len(sa), len(ma)), complex)
                    for ik, k in enumerate(ka):
                        live_mu = ma < m
                        if not live_mu.any():
                            continue
                        g = np.arange(ng)
                        g3 = np.stack(np.unravel_index(g, grid), axis=-1)
                        phase = np.exp(2j*np.pi*((g3+np.array([k/4., 0., 0.]))
                                                @ (coords[ma[live_mu]]/np.asarray(grid)).T))/np.sqrt(nrtot)
                        raw = coeff(k, (bs+read[0])[:, None, None], sa[None, :, None], g[None, None, :])
                        values = np.einsum('bsg,gm->bsm', raw, phase)
                        values[(bs+read[0]) >= user_stop] = 0
                        expected[ik, :, :, live_mu] = np.moveaxis(values, -1, 0)
                    wanted = expected if position == 0 else np.conj(expected.transpose(0, 3, 1, 2))
                    got = actual[:, start:start+len(bs)] if position == 0 else actual[:, :, start:start+len(bs), :]
                    np.testing.assert_allclose(got, wanted, rtol=2e-12, atol=2e-12)
            result.append(actual)
        return result

    records, reference = [], None
    cases = [('default', 'device', None), ('retained', 'device', keep)]
    if not args.large:
        cases.append(('host_retained', 'host', keep))
    for name, placement, retained in cases:
        loader = Loader()
        started = time.monotonic()
        product = load_parent_psi_G(wfn=loader, mesh_xy=mesh, meta=meta,
            band_range=read, band_chunk=128 if args.large else 4,
            centroid_indices=coords, placement=placement,
            mu_tile_bytes=16*((ng+3)//4)*2, store_band_range=retained)
        interval = read if retained is None else retained
        assert product.band_range == interval
        store = product.psi_G if placement == 'device' else product.host_tile
        verify_store(store, interval, host=placement == 'host')
        # The large case compares every face element with the full-store
        # result; the small case also uses the independent Fourier oracle.
        faces = verify_faces(product.faces, reference if args.large else None)
        if reference is None:
            reference = faces
        row = dict(name=name, seconds=time.monotonic()-started,
                   read_calls=loader.reads, retained_range=interval,
                   store_shape=list(store.shape), face_shapes=[list(a.shape) for a in product.faces])
        held = list(product.faces) + [product.sphere_index]
        if placement == 'device':
            held.append(store)
        distinct = {}
        for a in held:
            part = a.addressable_data(0)
            distinct[int(part.unsafe_buffer_pointer())] = int(part.nbytes)
        row['unique_held_bytes_per_rank'] = sum(distinct.values())
        row['unique_held_buffers'] = len(distinct)
        rss = {}
        for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
            if line.startswith(('Rss:', 'Pss:')):
                key, value = line.split(':', 1)
                rss[key+'_bytes'] = int(value.split()[0])*1024
        row['resident_after_load'] = rss
        if placement == 'device':
            ex = jax.jit(jnp.conj, out_shardings=store.sharding, donate_argnums=(0,)).lower(store).compile()
            mm = ex.memory_analysis()
            row['conjugation'] = dict(arguments=mm.argument_size_in_bytes,
                outputs=mm.output_size_in_bytes, aliases=mm.alias_size_in_bytes,
                temporary=mm.temp_size_in_bytes, new=compiled_new_bytes(ex))
            row['store_bytes_per_rank'] = int(store.addressable_data(0).nbytes)
            store.delete()
        for a in product.faces:
            a.delete()
        del product, store
        gc.collect()
        row['rss_hwm_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        records.append(row)
    assert all(r['read_calls'] == records[0]['read_calls'] for r in records)
    rejected = 0
    for invalid in ((read[0]-1, read[1]), (read[0], read[1]+1), (read[0], read[0])):
        loader = Loader()
        try:
            load_parent_psi_G(wfn=loader, mesh_xy=mesh, meta=meta,
                band_range=read, band_chunk=4, store_band_range=invalid)
        except ValueError:
            assert not loader.reads
            rejected += 1
    assert rejected == 3
    result = dict(rank=jax.process_index(), host=socket.gethostname(), physical_cores=len(pairs),
                  cpu_devices=len(jax.local_devices()), omp=os.environ['OMP_NUM_THREADS'],
                  scope='Dense synthetic raw-loader fixture; production retention, transport and face kernels',
                  large=args.large, cases=records, invalid_ranges_rejected=rejected)
    (args.out/f'rank{jax.process_index()}.json').write_text(json.dumps(result, indent=2)+'\n')
    assert np.all(np.asarray(all_gather_processes(np.array(1, np.int32))) == 1)
    if jax.process_index() == 0:
        print('PARENT_PSI_G_RETENTION PASS', flush=True)
    return 0


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
