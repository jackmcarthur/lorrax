"""Completed route-G Hartree response versus literal physical band pairs.

Run as a canonical P4 script. Rectangular axes, padded planes, owner row
chunks, changed augmented samples and nonunit fitting loss are intentional.
The physical potential is independent of those fitting weights.
"""
from pathlib import Path
import json
import os


def check_response(runtime):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.wfn_transforms import psi_cylinder_tables
    from isdf.zeta_mubatch import (make_route_g_kernel, zeta_plane_tables,
                                  q0_response_from_batches,
                                  finalize_charge_q0_response, ZetaG)
    from runtime.padding import padded_axis
    from types import SimpleNamespace
    import hashlib

    mesh = runtime.mesh
    assert int(mesh.size) == 4
    grid, kgrid, axis = (4, 3, 4), (2, 1, 1), 1
    nk, nb, ns, b = 2, 3, 4, 8
    N = int(np.prod(grid))
    xyz = np.indices(grid).reshape(3, -1).T
    x = xyz/np.asarray(grid)
    k = np.array([[0., 0., 0.], [.5, 0., 0.]])
    xmu = x[[0, 1, 5, 9, 17, 29, 42, 47]]
    rng = np.random.default_rng(1064)
    psi = (rng.normal(size=(nk, nb, ns, N))
           + 1j*rng.normal(size=(nk, nb, ns, N)))/np.sqrt(N*ns)
    E = np.exp(2j*np.pi*np.einsum('kgd,rd->kgr', k[:, None]+xyz, x))
    psi_r = np.einsum('knsg,kgr->knsr', psi, E)/np.sqrt(N)
    Emu = np.exp(2j*np.pi*np.einsum('kgd,md->kgm', k[:, None]+xyz, xmu))
    samples = np.einsum('knsg,kgm->knsm', psi, Emu)/np.sqrt(N)
    samples[0, 0, 3, 2] += .47-.23j
    samples[1, 2, 1, 5] -= .31+.17j
    live = np.ones(b)
    live[-1] = 0
    samples[..., -1] = 1e5
    wl, wr = np.array([4., 1., 0.]), np.array([4., 1., 1.])
    potentials = np.stack((.3+np.sin(2*np.pi*x[:, 0])+.7*x[:, 1],
                           -.2+np.cos(2*np.pi*x[:, 2])+.4*x[:, 0]))
    potential_grid = potentials.reshape(2, *grid).astype(np.float64)

    # Literal alpha^0 band-pair features and physical smooth densities,
    # independently of the pair-projector/convolution implementation.
    sm = samples*live
    eta = np.einsum('kmsu,knsu->kmnu', sm.conj(), sm)
    density = np.einsum('kmsr,knsr->kmnr', psi_r.conj(), psi_r)
    z0 = np.einsum('m,n,kmnu,kmnr->ur', wl, wr, eta.conj(), density)
    z0 = z0+z0.conj()  # LR+RL at Gamma
    expected = np.einsum('ur,vr->uv', z0, potentials)

    put = lambda a, spec=P(): device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    host = lambda a: np.asarray(gather_to_host(a))
    box = np.broadcast_to(np.arange(N, dtype=np.int32), (nk, N))
    ci, ca, pfc = psi_cylinder_tables(put(box), grid, axis, ngkmax=N)
    gvec = np.broadcast_to(xyz.T, (nk, 3, N))
    zt = zeta_plane_tables(gvec, np.full(nk, N), grid, axis,
                           padded_axis(N, 16, name='response test G'))
    qa = padded_axis(nk, 4, name='response test q')
    factory = dict(mesh=mesh, kgrid=kgrid, fft_grid=grid, ns=ns, b=b,
        q_sel=np.arange(nk), q_axis=qa, q_neg=np.array([0, 1]), qvec_frac=k,
        n_col=int(ci.shape[1]), n_s=int(ci.shape[2]),
        plane_from_col=np.asarray(jax.device_get(pfc)), n_pg=2,
        axis=axis, n_src=nk, n_pc=1, c_out=1, n_blk=2,
        use_augmented_samples=True)
    unf = tuple(put(a) for a in (
        np.arange(nk, dtype=np.int32), np.zeros(nk, np.int32), np.zeros(nk, bool),
        np.broadcast_to(np.eye(ns), (nk, ns, ns)).astype(np.complex128),
        box, np.ones((nk, N), np.complex128), k))
    c = b//4
    lp = np.broadcast_to(np.arange(c, dtype=np.int32), (4, 1, c))
    ll = np.zeros((4, 1, c, 3), np.int32)
    args = (put(psi.conj(), P(None, None, None, ('x', 'y'))), put(wl), put(wr),
        put(k), put(np.broadcast_to(xyz, (nk, N, 3)), P(None, ('x', 'y'), None)),
        put(xmu), put(live), (put(np.asarray(ci)), put(np.asarray(ca))),
        tuple(put(a) for a in zt), unf,
        (put(lp, P(('x', 'y'))), put(ll, P(('x', 'y')))),
        put(samples, P(None, None, None, ('x', 'y'))))
    default = make_route_g_kernel(**factory)(*args)[0]
    kernel = make_route_g_kernel(**factory, q0_response=True)
    rows, raw = kernel(*args, put(potential_grid))
    np.testing.assert_array_equal(host(rows[0]), host(default))
    observed = host(raw)
    relative = float(np.linalg.norm(observed-expected)/np.linalg.norm(expected))
    assert relative < 3e-13, relative
    assert np.max(abs(observed[-1])) == 0
    wrong_norm = float(np.linalg.norm(observed/N-expected)/np.linalg.norm(expected))
    assert wrong_norm > .9

    # A non-prefix mapping across ranks/batches and inert ghost slot must use
    # exactly the same packed map as the reciprocal Z store.
    second = raw*(.4+.2j)
    batched = jax.jit(lambda a, b: jnp.stack((a, b)),
        out_shardings=NamedSharding(mesh, P(None, ('x', 'y'), None)))(raw, second)
    slots = np.array([10, 1, 12, -1, 0, 6, 9, 3], np.int32)
    packed = q0_response_from_batches(mesh, batched, q_axis=qa,
                                      q0_slot=0, packed_from_slot=slots)
    flat = np.concatenate((observed, observed*(.4+.2j)))
    expected_packed = np.where(slots[:, None] >= 0, flat[np.maximum(slots, 0)], 0)
    ph = host(packed)
    np.testing.assert_allclose(ph[0], expected_packed, rtol=2e-15, atol=1e-17)
    assert np.max(abs(ph[1:])) == 0

    # Tiny physical SPD normal equation: add local response BEFORE one solve.
    # This uses the existing pseudoinverse factor owner, no Hartree factor.
    matrix = rng.normal(size=(b, b))+1j*rng.normal(size=(b, b))
    matrix = matrix.conj().T@matrix+np.eye(b)
    from isdf.core import factor_c_q
    gram = put(np.broadcast_to(matrix, (qa.carrier, b, b)).copy(), P(None, 'x', 'y'))
    factor = factor_c_q(gram, mesh, n_rmu_logical=b,
                        solver_kind='replicated_rank_truncate', zeta_rcond=1e-8)
    local = put(ph*(.2-.1j), P(('x', 'y'), None, None))
    store = SimpleNamespace(Q=nk, Q_pad=qa.carrier, mu_pad=b)
    zeta = ZetaG(store, mesh=mesh, L_q=factor, lu_piv=None,
        solver_kind='replicated_rank_truncate', batched_route='batch_reshard', n_rmu_solve=b,
        n_rmu=b, mu_basis=None, ngk_per_q=np.full(nk, N),
        gvec_components=gvec, path='unused')
    zeta.fit_vertex_mu_L = 0
    zeta.q0_response_rhs = packed
    zeta.fit_q_full_indices = np.arange(nk)
    zeta.q0_response_source_identity = 'independent_unit_normal_equation_source'
    zeta.fit_centroid_geometry = dict(coordinate_kind='fractional',canonical_shape=list(xmu.shape),
        canonical_coordinates_sha256=hashlib.sha256(xmu.astype(np.float64).tobytes()).hexdigest())
    zeta.q0_response_metadata = dict(schema='lorrax.charge_q0_response.v1', complete=True,
        source_identity=zeta.q0_response_source_identity,
        centroid_geometry=zeta.fit_centroid_geometry,
        q0_slot=0, q0_full_index=0, q_full_indices=list(range(nk)), source_count=2,
        units='physical_Ry_potential_times_grid_density_sum',
        packed_to_canonical_sha256=hashlib.sha256(np.arange(b, dtype=np.int64).tobytes()).hexdigest(),
        active_mu_sha256=hashlib.sha256(np.ones(b, bool).tobytes()).hexdigest())
    solved = host(finalize_charge_q0_response(zeta, local))
    reference = np.linalg.solve(matrix, expected_packed*(1.2-.1j))
    solve_relative = float(np.linalg.norm(solved-reference)/np.linalg.norm(reference))
    assert solve_relative < 3e-14, solve_relative

    zeta.q0_response_metadata['q0_slot'] = 1
    try:
        finalize_charge_q0_response(zeta, local)
    except ValueError:
        pass
    else:
        raise AssertionError('changed physical q0 placement must refuse before solve')

    refused = 0
    for changes in (dict(vertices=(1,)), dict(stop_at='kconv'),
                    dict(qvec_frac=np.array([[.1, 0., 0.], [.5, 0., 0.]]))):
        try:
            make_route_g_kernel(**dict(factory, **changes), q0_response=True)
        except ValueError:
            refused += 1
    assert refused == 3
    result = dict(P=4, grid=grid, axis=axis, sources=2, physical_gamma=0,
        padded_q=qa.carrier, LR_RL=True, occupied_endpoint_weight=4,
        source_potential_independent_of_fit_loss=True,
        raw_response_relative_error=relative, wrong_1_over_N_relative_error=wrong_norm,
        same_factor_solve_relative_error=solve_relative, default_rows_bitwise=True,
        point_and_q_ghosts_zero=True, invalid_route_refusals=refused)
    if jax.process_index() == 0:
        print(json.dumps(result, sort_keys=True), flush=True)
        output = os.environ.get('CHARGE_Q0_RESPONSE_REPORT')
        if output:
            Path(output).write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack()
    run_main_and_finalize(lambda: (check_response(runtime), 0)[1])
