"""P4 rectangular four-component local RHS versus explicit band-pair sums."""
from pathlib import Path
import json
import os


def check_local_rhs(runtime):
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.gamma_matrices import gamma0, gamma1, gamma2, gamma3
    from distrib_la import gemm_plan
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from isdf.core import c_q_from_psi_sm
    from isdf.local_rhs import local_density_rhs
    from symmetry_maps import q_negation_index

    mesh = runtime.mesh
    assert int(mesh.size) == 4
    grid, kgrid, nb, ns = (4, 4, 4), (3, 1, 1), 4, 4
    eye = np.eye(4, dtype=np.complex128)
    inversion = np.diag([1., 1., -1., -1.]).astype(np.complex128)
    trs = np.kron(np.eye(2), np.array([[0., 1.], [-1., 0.]])).astype(np.complex128)
    gamma = tuple(np.asarray(gather_to_host(g)) for g in (gamma0, gamma1, gamma2, gamma3))
    mu_idx = np.array([[0,0,0],[1,0,0],[3,0,0],[2,0,0],
                       [0,1,0],[0,3,0],[0,0,1],[0,0,3]], np.int32)
    half_local = np.array([[.071,0,0],[0,.117,0],[0,0,.159],
                           [.103,.081,0],[0,.093,.133],[.097,0,.183]])
    local_frac = np.concatenate((half_local, (-half_local) % 1.))
    wl, wr = np.array([1.,1.,0.,0.]), np.array([0.,1.,1.,0.])
    qneg = q_negation_index(kgrid)
    rng = np.random.default_rng(5401)
    receipts = []

    def dense_unfold(parent, plan):
        rows = []
        for p, s, U in zip(plan.irr_idx, plan.sym_idx, plan.spin_action_full):
            perm = plan.sym_perm[s]
            phase = np.exp(2j*np.pi*(plan.L_table[s] @ plan.k_parent_frac[p]))
            row = parent[p][..., perm] * phase
            if s >= plan.n_sym_spatial:
                row = row.conj()
            rows.append(np.einsum('st,ntm->nsm', U, row))
        return np.stack(rows)

    def direct(left, right, gl, gr):
        out = np.zeros((3, left.shape[-1], right.shape[-1]), np.complex128)
        for q in range(3):
            for k in range(3):
                for m in range(nb):
                    for n in range(nb):
                        if not wl[m]*wr[n]:
                            continue
                        a = np.einsum('sm,st,tm->m', left[k,m].conj(), gamma[gl], left[(k+q)%3,n])
                        r = np.einsum('sr,st,tr->r', right[k,m].conj(), gamma[gr], right[(k+q)%3,n])
                        out[q] += wl[m]*wr[n]*a.conj()[:,None]*r[None,:]
        # The native Cartesian-alpha2 convention puts the phase on both
        # endpoints. Its left vertex is minus the conjugate LS vertex.
        return -out if gl == 2 else out

    for kind in ('identity', 'inversion', 'time_reversal'):
        npar = 3 if kind == 'identity' else 2
        irr = np.arange(3, dtype=np.int32) if npar == 3 else np.array([0,1,1], np.int32)
        sidx = np.array([0,0,1], np.int32) if npar == 2 else np.zeros(3, np.int32)
        spatial = np.stack((np.eye(3, dtype=np.int32), -np.eye(3, dtype=np.int32))) \
            if kind == 'inversion' else np.eye(3, dtype=np.int32)[None]
        actions = (eye, inversion) if kind == 'inversion' else (eye, trs)
        sym = SimpleNamespace(sym_matrices=spatial,
            translations=np.zeros((len(spatial),3)), irr_idx_k=irr, sym_idx_k=sidx,
            kirr_fullids=np.arange(npar), spinor_action=lambda rows,nspinor: np.stack([actions[i] for i in rows]))
        kpar = np.zeros((npar,3))
        kpar[:,0] = np.arange(npar)/3.
        left_plan = build_centroid_k_unfold_plan(sym, mu_idx, grid, mesh,
                                                nspinor=4, parent_k_frac=kpar)
        right_plan = build_centroid_k_unfold_plan(sym, local_frac, grid, mesh,
                    nspinor=4, parent_k_frac=kpar, coordinate_kind='fractional')

        def faces(plan):
            p = (rng.normal(size=(npar,nb,ns,plan.layout.axis.n_logical))
                 + 1j*rng.normal(size=(npar,nb,ns,plan.layout.axis.n_logical))) / 4.
            p[:,-1] = 0.0
            p = plan.layout.axis.pack_host(p, axis=3, fill_value=0.)
            put = lambda a,spec: device_put_process_local(a, NamedSharding(mesh,spec))
            return p, (put(p,P(None,'x',None,'y')),
                       put(p.transpose(0,2,3,1),P(None,None,'x','y')))

        left, lf = faces(left_plan)
        ae, aef = faces(right_plan)
        ps = .83*ae
        psf = (device_put_process_local(ps, NamedSharding(mesh,P(None,'x',None,'y'))),
               device_put_process_local(ps.transpose(0,2,3,1), NamedSharding(mesh,P(None,None,'x','y'))))
        options = dict(centroid_faces=lf, atom_ae_faces=aef, atom_ps_faces=psf,
                       left_plan=left_plan,right_plan=right_plan,weight_l=wl,weight_r=wr,
                       kgrid=kgrid,mesh_xy=mesh,q_neg_idx=qneg)
        L, AE, PS = dense_unfold(left,left_plan), dense_unfold(ae,right_plan), dense_unfold(ps,right_plan)
        tests = (((1.,0,0),), ((1.,2,2),),
                 ((.5,1,1),(.5,2,2),(.5j,1,2),(-.5j,2,1)))
        errors = []
        for terms in tests:
            got = local_density_rhs(**options,vertex_terms=terms)
            result = np.asarray(gather_to_host(got))
            expected = sum(w*(lambda z:z+z[qneg].conj())(
                direct(L,AE,i,j)-direct(L,PS,i,j)) for w,i,j in terms)
            err = float(np.linalg.norm(result-expected)/np.linalg.norm(expected))
            assert err < 2e-13, (kind,terms,err)
            assert got.sharding.spec == P(None,'x','y')
            errors.append(err)
            for selected in (np.asarray([2,0],np.int32),np.asarray([1],np.int32),
                             np.asarray([0],np.int32)):
                indexed = local_density_rhs(**options,vertex_terms=terms,q_indices=selected)
                np.testing.assert_array_equal(np.asarray(gather_to_host(indexed)),result[selected])
        zero = local_density_rhs(**dict(options,atom_ps_faces=aef))
        assert float(jax.device_get(jax.numpy.max(jax.numpy.abs(zero)))) == 0.0
        delta, smooth = local_density_rhs(**options, return_smooth=True)
        delta_default = np.asarray(gather_to_host(local_density_rhs(**options)))
        np.testing.assert_array_equal(np.asarray(gather_to_host(delta)), delta_default)
        expected_smooth = direct(L,PS,0,0)
        expected_smooth += expected_smooth[qneg].conj()
        smooth_error = np.linalg.norm(np.asarray(gather_to_host(smooth))-expected_smooth)/np.linalg.norm(expected_smooth)
        assert smooth_error < 2e-13, (kind,smooth_error)
        uncompleted_delta, uncompleted_smooth = local_density_rhs(
            **dict(options,q_neg_idx=None), return_smooth=True)
        np.testing.assert_allclose(np.asarray(gather_to_host(uncompleted_smooth)),
                                   direct(L,PS,0,0),rtol=2e-13,atol=2e-15)
        np.testing.assert_array_equal(np.asarray(gather_to_host(uncompleted_delta)),
            np.asarray(gather_to_host(local_density_rhs(**dict(options,q_neg_idx=None)))))
        selected = np.asarray([1],np.int32)
        indexed_delta,indexed_smooth = local_density_rhs(**options,return_smooth=True,q_indices=selected)
        np.testing.assert_array_equal(np.asarray(gather_to_host(indexed_delta)),delta_default[selected])
        np.testing.assert_array_equal(np.asarray(gather_to_host(indexed_smooth)),
                                      np.asarray(gather_to_host(smooth))[selected])
        indexed_no_completion = local_density_rhs(**dict(options,q_neg_idx=None),q_indices=selected)
        np.testing.assert_array_equal(np.asarray(gather_to_host(indexed_no_completion)),
                                      np.asarray(gather_to_host(uncompleted_delta))[selected])
        for bad_rows in ([-1],[3],[1,1],[.5],[]):
            try:
                local_density_rhs(**options,q_indices=np.asarray(bad_rows))
            except ValueError as exc:
                assert 'unique full-q rows' in str(exc)
            else:
                raise AssertionError('invalid selected q admitted')
        try:
            local_density_rhs(**dict(options,q_neg_idx=np.arange(3)),q_indices=selected)
        except ValueError as exc:
            assert 'canonical q negation' in str(exc)
        else:
            raise AssertionError('selected completion admitted a noncanonical involution')
        try:
            local_density_rhs(**options,vertex_terms=((1.,1,1),),return_smooth=True)
        except ValueError as exc:
            assert 'scalar charge vertex' in str(exc)
        else:
            raise AssertionError('scalar smooth retention admitted a current vertex')
        # The rectangular generalization's default square route stays equal
        # to the incumbent public C_q entry point with identical operands.
        from isdf.core import _c_q_dirac_quarters
        m = int(left_plan.n_centroid_packed)
        gemm = gemm_plan(mesh,m=ns*m,n=ns*m,k=nb,nq=npar,dtype=jax.numpy.complex128,
                         layout='face',warmup=False)
        square = c_q_from_psi_sm(lf[1],lf[0],wl,wr,k_unfold_plan=left_plan,
                                kgrid=kgrid,mesh_xy=mesh,gemm=gemm)
        explicit_square = _c_q_dirac_quarters(lf[1],lf[0],wl,wr,plan=left_plan,
                         right_plan=left_plan,kgrid=kgrid,mesh_xy=mesh,gemm=gemm,gamma_L=0,gamma_R=0)
        square_error = float(jax.device_get(jax.numpy.max(jax.numpy.abs(square-explicit_square))))
        assert square_error == 0.0
        bad = replace(right_plan, k_parent_frac=np.array(right_plan.k_parent_frac)+.01)
        try:
            local_density_rhs(**dict(options,right_plan=bad))
        except ValueError as exc:
            assert 'k_parent_frac' in str(exc)
        else:
            raise AssertionError('mismatched parent gauge accepted')
        receipts.append(dict(kind=kind,parents=npar,mu=m,
                             local_points=int(right_plan.n_centroid_packed),
                             relative_errors=errors,zero_exact=True,square_exact=True,
                             mismatched_parent_refused=True,smooth_relative_error=float(smooth_error),
                             retained_delta_bitwise=True,smooth_current_refused=True,
                             indexed_quarters_and_completion_bitwise=True,
                             indexed_q_validation_refused=True))
    out = os.environ.get('LOCAL_RHS_REPORT')
    if jax.process_index() == 0:
        print(json.dumps(receipts),flush=True)
        if out:
            Path(out).write_text(json.dumps(receipts,indent=2)+'\n')


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack()

    def main():
        check_local_rhs(runtime)
        return 0

    run_main_and_finalize(main)
