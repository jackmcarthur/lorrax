"""P4 route-G corrected sample seam against explicit band-pair sums.

Run as a script under four ranks, one GPU per rank. The deliberately changed
sample endpoint must affect the smooth RHS while the reciprocal wavefunction
endpoint stays fixed. This is a kernel test, not a reconstruction certificate.
"""
from pathlib import Path
import json
import os
import time

def check_samples(runtime):
    import numpy as np
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.wfn_transforms import psi_cylinder_tables
    from isdf.zeta_mubatch import make_route_g_kernel, zeta_plane_tables
    from runtime.padding import padded_axis

    mesh = runtime.mesh
    nranks = int(mesh.size)
    assert nranks == 4, f"this distributed kernel check requires P4, got {nranks}"
    grid, kgrid = (4, 4, 4), (2, 1, 1)
    nk, nb, ns, b = 2, 3, 4, 8
    N = int(np.prod(grid))
    xyz = np.indices(grid).reshape(3, -1).T
    x = xyz / np.asarray(grid)
    k = np.array([[0., 0., 0.], [.5, 0., 0.]])
    xmu = x[[0, 1, 5, 9, 17, 29, 42, 63]]
    rng = np.random.default_rng(9371)
    psi = (rng.normal(size=(nk, nb, ns, N))
           + 1j * rng.normal(size=(nk, nb, ns, N))) / np.sqrt(N * ns)
    # Independent full reciprocal sum, including physical Bloch phases.
    E = np.exp(2j * np.pi * np.einsum('kgd,rd->kgr', k[:, None] + xyz, x))
    psi_r = np.einsum('knsg,kgr->knsr', psi, E) / np.sqrt(N)
    Emu = np.exp(2j * np.pi * np.einsum('kgd,md->kgm', k[:, None] + xyz, xmu))
    samples = np.einsum('knsg,kgm->knsm', psi, Emu) / np.sqrt(N)
    changed = samples.copy()
    changed[0, 0, 3, 2] += 0.47 - 0.23j
    changed[1, 2, 1, 5] -= 0.31 + 0.17j
    live = np.ones(b)
    live[-1] = 0.0
    changed[..., -1] = 1e5  # a padded sample must remain inert
    wl, wr = np.array([1., 1., 0.]), np.array([0., 1., 1.])
    qneg = np.array([0, 1], np.int32)

    def explicit(sample):
        sample = sample * live
        dl = np.einsum('n,knam,knbr->kabmr', wl, sample, psi_r.conj())
        dr = np.einsum('n,knam,knbr->kabmr', wr, sample, psi_r.conj())
        z = np.stack([sum(np.einsum('abmr,abmr->mr', dl[i], dr[(i+q)%nk].conj())
                          for i in range(nk)) for q in range(nk)])
        z = z + z[qneg].conj()
        demod = np.exp(-2j * np.pi * (k @ x.T))
        return np.fft.fftn((z * demod[:, None]).reshape(nk, b, *grid),
                          axes=(-3, -2, -1)).reshape(nk, b, N)

    rep = NamedSharding(mesh, P())
    put = lambda a, spec=P(): device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    box = np.broadcast_to(np.arange(N, dtype=np.int32), (nk, N))
    ci, ca, pfc = psi_cylinder_tables(put(box), grid, 0, ngkmax=N)
    gvec = np.broadcast_to(xyz.T, (nk, 3, N))
    zt = zeta_plane_tables(gvec, np.full(nk, N), grid, 0,
                           padded_axis(N, 16, name="test zeta G tiles"))
    qa = padded_axis(nk, nranks, name="test q rows")
    factory = dict(mesh=mesh, kgrid=kgrid, fft_grid=grid, ns=ns, b=b,
                   q_sel=np.arange(nk), q_axis=qa, q_neg=qneg, qvec_frac=k,
                   n_col=int(ci.shape[1]), n_s=int(ci.shape[2]),
                   plane_from_col=np.asarray(jax.device_get(pfc)), n_pg=2,
                   axis=0, n_src=nk, n_pc=1, c_out=1, n_blk=2)
    unf = tuple(put(a) for a in (
        np.arange(nk, dtype=np.int32), np.zeros(nk, np.int32), np.zeros(nk, bool),
        np.broadcast_to(np.eye(ns), (nk, ns, ns)).astype(np.complex128),
        box, np.ones((nk, N), np.complex128), k))
    c = b // nranks
    lp = np.broadcast_to(np.arange(c, dtype=np.int32), (nranks, 1, c))
    ll = np.zeros((nranks, 1, c, 3), np.int32)
    args = (put(psi.conj(), P(None, None, None, ('x', 'y'))), put(wl), put(wr),
            put(k), put(np.broadcast_to(xyz, (nk, N, 3)), P(None, ('x', 'y'), None)),
            put(xmu), put(live), (put(np.asarray(ci)), put(np.asarray(ca))),
            tuple(put(a) for a in zt), unf,
            (put(lp, P(('x', 'y'))), put(ll, P(('x', 'y')))))
    smooth_kernel = make_route_g_kernel(**factory)
    augmented_kernel = make_route_g_kernel(**factory, use_augmented_samples=True)
    sample_put = lambda a: put(a, P(None, None, None, ('x', 'y')))
    t0 = time.monotonic()
    smooth = np.asarray(gather_to_host(smooth_kernel(*args)[0]))
    zero = np.asarray(gather_to_host(augmented_kernel(*args, sample_put(samples))[0]))
    altered = np.asarray(gather_to_host(augmented_kernel(*args, sample_put(changed))[0]))
    # A larger source batch retains the same streamed one-row plane stage.
    # This is the fallback planner's independent axis: all logical rows must
    # agree, while additional source slots remain exact zero padding.
    expanded_b = 2*b
    expanded_args = list(args)
    expanded_args[5] = put(np.pad(xmu,((0,b),(0,0))))
    expanded_args[6] = put(np.pad(live,(0,b)))
    expanded_c = expanded_b//nranks
    expanded_lp = np.broadcast_to(np.arange(expanded_c,dtype=np.int32),(nranks,1,expanded_c))
    expanded_ll = np.zeros((nranks,1,expanded_c,3),np.int32)
    expanded_args[-1] = (put(expanded_lp,P(('x','y'))),put(expanded_ll,P(('x','y'))))
    expanded_sample = np.pad(changed,((0,0),(0,0),(0,0),(0,b)))
    expanded_kernel = make_route_g_kernel(**dict(factory,b=expanded_b),use_augmented_samples=True)
    expanded = np.asarray(gather_to_host(expanded_kernel(*expanded_args,sample_put(expanded_sample))[0]))
    reference, expected_changed = explicit(samples), explicit(changed)
    rel = lambda a, e: float(np.linalg.norm(a-e) / np.linalg.norm(e))
    results = dict(P=nranks, ns=ns, parents=nk, parent_chunk=1, batch=b,
                   rows_in_flight=1, plane_blocks=2, asymmetric_windows=True,
                   zero_augmentation_relative_error=rel(zero, smooth),
                   smooth_direct_relative_error=rel(smooth, reference),
                   altered_direct_relative_error=rel(altered, expected_changed),
                   expanded_source_stream_relative_error=rel(expanded[:,:b],altered),
                   expanded_source_padding_max=float(np.max(abs(expanded[:,b:]))),
                   planted_signal_relative_norm=rel(altered, smooth),
                   pad_max_abs=float(np.max(np.abs(altered[:, -1]))),
                   wall_s=time.monotonic()-t0)
    assert results['zero_augmentation_relative_error'] < 2e-13, results
    assert results['smooth_direct_relative_error'] < 2e-13, results
    assert results['altered_direct_relative_error'] < 2e-13, results
    assert results['expanded_source_stream_relative_error'] < 2e-13, results
    assert results['expanded_source_padding_max'] == 0.,results
    assert results['planted_signal_relative_norm'] > 0.1, results
    assert results['pad_max_abs'] == 0.0, results
    executable = augmented_kernel.lower(*args, sample_put(changed)).compile()
    hlo = executable.as_text()
    results['collectives'] = {name: hlo.count(name+'(') + hlo.count(name+'-start(')
                              for name in ('all-gather', 'all-reduce', 'all-to-all')}
    mem = executable.memory_analysis()
    results['compiled_bytes_per_rank'] = dict(argument=mem.argument_size_in_bytes,
                                              output=mem.output_size_in_bytes,
                                              temporary=mem.temp_size_in_bytes)
    out = os.environ.get('AUGMENTED_SAMPLE_REPORT')
    if jax.process_index() == 0:
        print(json.dumps(results, sort_keys=True), flush=True)
        if out:
            Path(out).write_text(json.dumps(results, indent=2)+'\n')
            Path(out).with_suffix('.hlo.txt').write_text(hlo)
    return results


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack()

    def main():
        check_samples(runtime)
        return 0

    run_main_and_finalize(main)
