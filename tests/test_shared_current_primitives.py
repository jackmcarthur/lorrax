"""P4 current channel reuse against literal band-pair normal sources.

Run on the distributed runtime. This does not certify a material replay's
capacity or timing; the fixture checks completion, selected q, projection,
primitive reuse and distinct canonical coefficient solves.
"""
import json


def check_shared_current_primitives(runtime):
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.gamma_matrices import gamma0, gamma1, gamma2, gamma3, current_fit_terms
    from common.staged_reshard import face_to_batch_reshard
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.isdf_augmentation import _rhs_storage_kernels
    from gw.augmentation_current import _shared_current_packet_plan, _shared_current_rhs_packets
    from isdf import core
    from isdf.local_rhs import local_density_rhs
    from isdf.zeta_mubatch import _local_coefficient_solve
    from symmetry_maps import q_negation_index

    mesh = runtime.mesh
    assert int(mesh.size) == 4
    nb, ns, kgrid, grid = 4, 4, (3, 1, 1), (4, 4, 4)
    qneg = np.asarray(q_negation_index(kgrid))
    eye = np.eye(4, dtype=np.complex128)
    sym = SimpleNamespace(sym_matrices=np.eye(3, dtype=np.int32)[None],
        translations=np.zeros((1, 3)), irr_idx_k=np.arange(3, dtype=np.int32),
        sym_idx_k=np.zeros(3, dtype=np.int32), kirr_fullids=np.arange(3),
        spinor_action=lambda rows, nspinor: np.repeat(eye[None], len(rows), axis=0))
    kpar = np.zeros((3, 3)); kpar[:, 0] = np.arange(3) / 3.
    # Neither point axis divides P: typed zero ghosts participate in transport.
    mu = np.array([[0,0,0], [1,0,0], [2,0,0], [0,1,0], [0,0,1]], np.int32)
    r = np.array([[.071,0,0], [0,.117,0], [0,0,.159], [.103,.081,0],
                  [0,.093,.133], [.097,0,.183], [.031,.057,.109]])
    lp = build_centroid_k_unfold_plan(sym, mu, grid, mesh, nspinor=4, parent_k_frac=kpar)
    rp = build_centroid_k_unfold_plan(sym, r, grid, mesh, nspinor=4,
                                    parent_k_frac=kpar, coordinate_kind='fractional')
    rng = np.random.default_rng(9173)
    def faces(plan):
        x = (rng.normal(size=(3,nb,ns,plan.layout.axis.n_logical))
             + 1j*rng.normal(size=(3,nb,ns,plan.layout.axis.n_logical))) / 4.
        x[:, -1] = 1e30  # The endpoint weights must keep this inert.
        x = plan.layout.axis.pack_host(x, axis=3, fill_value=0.)
        return x, (device_put_process_local(x, NamedSharding(mesh,P(None,'x',None,'y'))),
                   device_put_process_local(x.transpose(0,2,3,1),
                                            NamedSharding(mesh,P(None,None,'x','y'))))
    left, lf = faces(lp); ae, af = faces(rp); ps, pf = faces(rp)
    wl, wr = np.array([1.,1.,0.,0.]), np.array([0.,1.,1.,0.])
    operands = dict(centroid_faces=lf, atom_ae_faces=af, atom_ps_faces=pf,
        left_plan=lp, right_plan=rp, weight_l=wl, weight_r=wr,
        kgrid=kgrid, mesh_xy=mesh, q_neg_idx=qneg)
    circular = np.array([[1,1j,0],[1,-1j,0],[0,0,2**.5]],complex) / 2**.5
    terms = tuple(current_fit_terms(i, circular) for i in (1,2,3))
    gammas = tuple(np.asarray(gather_to_host(x)) for x in (gamma0,gamma1,gamma2,gamma3))
    left_phase = tuple(np.vdot(g,g.conj()) / np.vdot(g,g) for g in gammas)
    def literal(gl, gr):
        value = np.zeros((3,left.shape[-1],ae.shape[-1]),complex)
        for q in range(3):
            for k in range(3):
                kp = (k+q) % 3
                for m in range(nb):
                    for n in range(nb):
                        if not wl[m]*wr[n]:
                            continue
                        a = np.einsum('su,st,tu->u',left[k,m].conj(),gammas[gl],left[kp,n])
                        da = np.einsum('sr,st,tr->r',ae[k,m].conj(),gammas[gr],ae[kp,n])
                        dp = np.einsum('sr,st,tr->r',ps[k,m].conj(),gammas[gr],ps[kp,n])
                        value[q] += wl[m]*wr[n]*a.conj()[:,None]*(da-dp)[None,:]
        return left_phase[gl]*value
    raw = {(i,j): literal(i,j) for _,i,j in terms[0]}
    expected = tuple(sum(w*(raw[i,j]+raw[i,j][qneg].conj()) for w,i,j in t)
                     for t in terms[:2])
    # A summed complex channel cannot be completed as if its weights were real.
    wrong = sum(w*raw[i,j] for w,i,j in terms[0])
    wrong = wrong + wrong[qneg].conj()
    completion_red = np.linalg.norm(wrong-expected[0]) / np.linalg.norm(expected[0])
    assert completion_red > 1e-3

    qsh = NamedSharding(mesh,P(('x','y'),None,None))
    packet = jax.ShapeDtypeStruct((4,left.shape[-1],ae.shape[-1]),jnp.complex128,sharding=qsh)
    shared = _shared_current_packet_plan(mesh, terms, packet)
    assert shared is not None and shared['receipt']['channels'] == [1,2]
    assert len(shared['pairs']) == 4
    assert _shared_current_packet_plan(mesh,tuple(current_fit_terms(i,None) for i in (1,2,3)),packet) is None
    assert _shared_current_packet_plan(mesh,(terms[0],tuple(reversed(terms[1])),terms[2]),packet) is None
    to_q = face_to_batch_reshard(mesh)
    # A complex linear angular-like map makes weighting/projection commutation
    # observable. It preserves typed ghosts and is independent of the RHS.
    phase_host = np.exp(.31j*np.arange(ae.shape[-1]))
    projection_phase = device_put_process_local(phase_host,NamedSharding(mesh,P('y')))
    from jax import shard_map
    angular = jax.jit(shard_map(lambda z,p:z*p[None,None,:],mesh=mesh,
        in_specs=(P(None,'x','y'),P('y')),out_specs=P(None,'x','y'),check_vma=False))
    cases = []
    for rows in (np.array([0,1,2],np.int32), np.array([2,0],np.int32), np.array([1],np.int32)):
        select,_ = _rhs_storage_kernels(mesh,np.arange(len(rows)),4,1,1,
                                       ae.shape[-1],ae.shape[-1],ae.shape[-1])
        project = lambda value: to_q(angular(select(value),projection_phase))
        calls = []
        original = core._c_q_dirac_quarters
        def count(*args, **kwargs):
            calls.append((kwargs['gamma_L'],kwargs['gamma_R']))
            return original(*args, **kwargs)
        core._c_q_dirac_quarters = count
        try:
            got = _shared_current_rhs_packets(shared, project, **operands, q_indices=rows)
        finally:
            core._c_q_dirac_quarters = original
        assert calls == [(i,j) for i,j in shared['pairs'] for _ in range(2)]
        scalar = tuple(project(local_density_rhs(**operands,q_indices=rows,vertex_terms=t))
                       for t in terms[:2])
        errors = []
        for channel,(value,old,oracle) in enumerate(zip(got,scalar,expected),1):
            host, oldhost = np.asarray(gather_to_host(value)), np.asarray(gather_to_host(old))
            target = np.pad(oracle[rows]*phase_host,((0,4-len(rows)),(0,0),(0,0)))
            error = np.linalg.norm(host-target) / np.linalg.norm(target)
            assert error < 3e-13, (rows,channel,error)
            np.testing.assert_allclose(host,oldhost,rtol=3e-13,atol=3e-15)
            assert value.sharding.spec == qsh.spec
            assert np.max(abs(host[len(rows):])) == 0.
            assert np.max(abs(host[...,~rp.layout.axis.active_mask])) == 0.
            # Deliberately distinct positive factors, carried unchanged to the
            # actual q-owned solve owner. Reuse must never merge the OWN solves.
            diagonal = np.arange(1,left.shape[-1]+1,dtype=float) + channel
            f = np.repeat(np.diag(1./np.sqrt(diagonal))[None],4,axis=0).astype(complex)
            factor = device_put_process_local(f,qsh)
            solve = _local_coefficient_solve(mesh,'positive_rank_truncate',left.shape[-1],len(rows))
            newcoef, oldcoef = solve(factor,value), solve(factor,old)
            np.testing.assert_allclose(np.asarray(gather_to_host(newcoef)),
                                       np.asarray(gather_to_host(oldcoef)),rtol=4e-13,atol=4e-15)
            errors.append(float(error))
        cases.append(dict(q_rows=rows.tolist(),relative_errors=errors,endpoint_calls=len(calls)))
    result = dict(scope='P4 typed rectangular RHS, selected q and distinct OWN solve fixture',
        cases=cases,completion_negative_control=float(completion_red),
        actual_AOT=shared['receipt']['actual_AOT'],material_capacity_claim=False)
    if jax.process_index() == 0:
        print(json.dumps(result),flush=True)
    return result


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack()
    def main():
        check_shared_current_primitives(runtime)
        return 0
    run_main_and_finalize(main)
