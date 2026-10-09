"""Canonical P4 charge-functional RHS, two atoms, images, TRS and pads."""
from pathlib import Path
import json
import os


def check_atomic_moments(runtime):
    from types import SimpleNamespace
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.staged_reshard import face_to_batch_reshard
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from isdf.local_rhs import local_density_rhs
    from isdf import cplus
    from isdf.atomic_moments import (auxiliary_charge_geometry, evaluate_auxiliary_charge,
        exact_pair_moments, make_auxiliary_monopole_compressor)
    from psp.augmentation_spinors import spinor_spherical_harmonic
    from symmetry_maps import q_negation_index
    from tests.test_atomic_moments import _synthetic_cache

    mesh = runtime.mesh
    assert mesh.size == 4
    cache = _synthetic_cache()
    geometry = auxiliary_charge_geometry(cache, auxiliary_radius=.18)
    nb, kgrid, fftgrid = 8, (3, 1, 1), (4, 4, 4)
    centers = np.asarray([[0., 0., 0.], [.5, 0., 0.]])
    point_frac = (centers[:, None]+geometry['relative_points'][None]).reshape(-1, 3)
    canonical_weights = np.zeros((2, len(point_frac)))
    npoint = len(geometry['relative_points'])
    for atom in range(2):
        canonical_weights[atom, atom*npoint:(atom+1)*npoint] = geometry['integration_weights']
    mu_idx = np.asarray([[0,0,0],[1,0,0],[3,0,0],[2,0,0],
                         [0,1,0],[0,3,0],[0,0,1],[0,0,3]], np.int32)
    wl = np.asarray([1., .8, .7, 0., 0., 0., 0., 0.])
    wr = np.asarray([.9, 1., 1., .8, .6, 0., 0., 0.])
    qneg = q_negation_index(kgrid)
    grid_scale = np.sqrt(95./64)
    eye = np.eye(4, dtype=complex)
    inversion = np.diag([1.,1.,-1.,-1.]).astype(complex)
    jspin = np.asarray([[0.,1.],[-1.,0.]], complex)
    trs = np.kron(np.eye(2), jspin)
    rng = np.random.default_rng(610121)
    random = lambda shape: (rng.normal(size=shape)+1j*rng.normal(size=shape))/17
    receipts = []

    def dense_unfold(parent, plan):
        rows = []
        for p, s, U in zip(plan.irr_idx, plan.sym_idx, plan.spin_action_full):
            values = parent[p][..., plan.sym_perm[s]]
            values = values*np.exp(2j*np.pi*(plan.L_table[s]@plan.k_parent_frac[p]))
            if s >= plan.n_sym_spatial:
                values = values.conj()
            rows.append(np.einsum('st,ntm->nsm', U, values))
        return np.stack(rows)

    for kind in ('identity', 'inversion', 'time_reversal'):
        npar = 3 if kind == 'identity' else 2
        irr = np.arange(3) if npar == 3 else np.asarray([0,1,1])
        sidx = np.zeros(3, int) if npar == 3 else np.asarray([0,0,1])
        spatial = (np.stack((np.eye(3, dtype=int), -np.eye(3, dtype=int)))
                   if kind == 'inversion' else np.eye(3, dtype=int)[None])
        actions = (eye, inversion) if kind == 'inversion' else (eye, trs)
        sym = SimpleNamespace(sym_matrices=spatial, translations=np.zeros((len(spatial),3)),
            irr_idx_k=irr, sym_idx_k=sidx, kirr_fullids=np.arange(npar),
            spinor_action=lambda rows,nspinor: np.stack([actions[i] for i in rows]))
        kpar = np.zeros((npar,3)); kpar[:,0] = np.arange(npar)/3
        left_plan = build_centroid_k_unfold_plan(sym, mu_idx, fftgrid, mesh, nspinor=4, parent_k_frac=kpar)
        right_plan = build_centroid_k_unfold_plan(sym, point_frac, fftgrid, mesh,
            nspinor=4, parent_k_frac=kpar, coordinate_kind='fractional')
        compressor = make_auxiliary_monopole_compressor(mesh, right_plan, canonical_weights)
        C, D = [random((2,npar,nb,len(cache['labels']))) for _ in range(2)]
        C[:,:, -1] = 1e5*(1+1j); D[:,:, -1] = 1e5*(1-1j)
        plus, minus = zip(*(evaluate_auxiliary_charge(C[a], D[a], geometry,
                               grid_sample_scale=grid_scale) for a in range(2)))
        plus, minus = np.concatenate(plus, axis=-1), np.concatenate(minus, axis=-1)
        plus = right_plan.layout.axis.pack_host(plus, axis=3, fill_value=0.)
        minus = right_plan.layout.axis.pack_host(minus, axis=3, fill_value=0.)
        left = random((npar,nb,4,len(mu_idx)))
        left[:,-1] = 1e5*(1+1j)
        left = left_plan.layout.axis.pack_host(left, axis=3, fill_value=0.)
        put = lambda a,spec: device_put_process_local(a, NamedSharding(mesh,spec))
        faces = lambda a: (put(a,P(None,'x',None,'y')), put(a.transpose(0,2,3,1),P(None,None,'x','y')))
        rhs = local_density_rhs(centroid_faces=faces(left), atom_ae_faces=faces(plus), atom_ps_faces=faces(minus),
            left_plan=left_plan, right_plan=right_plan, weight_l=wl, weight_r=wr,
            kgrid=kgrid, mesh_xy=mesh, q_neg_idx=qneg)
        compressed = compressor['compress'](rhs)
        # Pad q after canonical LR+RL, then use the incumbent face-to-q owner.
        compressed = jax.jit(lambda a: jnp.pad(a,((0,1),(0,0),(0,0))),
            out_shardings=NamedSharding(mesh,P(None,'x','y')))(compressed)
        owned = face_to_batch_reshard(mesh)(compressed)
        value = np.asarray(gather_to_host(owned))
        LF = dense_unfold(left, left_plan)
        # Independent κ coefficient transport, including the Rhalf atom image.
        CF, DF = [], []
        labels = cache['labels']; opf,mj = labels.T
        for atom in range(2):
            outC, outD = [], []
            for p,s in zip(irr,sidx):
                cc,dd = C[atom,p].copy(),D[atom,p].copy()
                if s == 1:
                    for k in np.unique(cache['upper']['kappa']):
                        radial_ids = np.flatnonzero(cache['upper']['kappa']==k)
                        ell = int(cache['upper']['ell'][radial_ids[0]])
                        mvals = np.arange(-2*abs(k)+1,2*abs(k),2)
                        ids = np.asarray([[np.flatnonzero((opf==i)&(mj==m))[0] for m in mvals] for i in radial_ids])
                        if kind == 'inversion':
                            factor = (-1)**ell*np.exp(-2j*np.pi*(kpar[p]@(2*centers[atom])))
                            cc[:,ids] *= factor; dd[:,ids] *= factor
                        else:
                            # Same physical i sigma_y K used by the point service.
                            from scipy.integrate import lebedev_rule
                            directions,wa = lebedev_rule(15); directions=directions.T
                            omega=np.stack([spinor_spherical_harmonic(k,m,directions) for m in mvals])
                            trmatrix=np.einsum('mas,nas,a->mn',omega.conj(),np.einsum('st,mat->mas',jspin,omega.conj()),wa)
                            cc[:,ids]=np.einsum('nim,pm->nip',C[atom,p][:,ids].conj(),trmatrix)
                            dd[:,ids]=np.einsum('nim,pm->nip',D[atom,p][:,ids].conj(),trmatrix)
                outC.append(cc);outD.append(dd)
            CF.append(np.stack(outC));DF.append(np.stack(outD))
        reference = np.zeros_like(value)
        for q in range(3):
            for k in range(3):
                right=(k+q)%3
                pair=np.einsum('nsm,psm->npm',LF[k].conj(),LF[right])
                for atom in range(2):
                    moment=grid_scale**2*exact_pair_moments(CF[atom][k],DF[atom][k],CF[atom][right],DF[atom][right],cache['B'])
                    reference[q,:,atom]+=np.einsum('npm,np,n,p->m',pair.conj(),moment,wl,wr)/np.sqrt(4*np.pi)
        reference[:3] += reference[qneg].conj()
        error=float(np.max(abs(value-reference)))
        assert error<2e-11,(kind,error)
        assert np.count_nonzero(value[3])==0
        normals=[]
        for q in range(4):
            x=random((len(mu_idx),len(mu_idx)))
            normals.append(np.eye(len(mu_idx))*(1+q)+x.conj().T@x)
        factor=cplus.factor(put(np.asarray(normals),P(('x','y'),None,None)),rcond=1e-8,rank_log=False,n_log=len(mu_idx))
        solved=np.asarray(gather_to_host(cplus.apply(factor,owned)))
        expected=np.asarray([np.linalg.solve(normal,ref) for normal,ref in zip(normals,reference)])
        solve_error=float(np.max(abs(solved-expected)))
        assert solve_error<2e-11,(kind,solve_error)
        receipts.append(dict(kind=kind,atoms=2,raw_parent_rows=npar,true_points=len(point_frac),
            signed_radial_monopole_RHS_error=error,same_cplus_solution_error=solve_error,
            zero_weight_huge_ghost_band_inert=True,q_padding_zero=True,carrier=geometry['carrier']))
    if jax.process_index()==0:
        output=Path(os.environ['ATOMIC_MOMENT_P4_REPORT'])
        output.write_text(json.dumps(dict(cases=receipts,scope='Canonical P4 two-atom charge functional, finite q, Rhalf and TRS; actual-source frame oracle is separate.'),indent=2)+'\n')
    return receipts


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime=initialize_communicator_stack()
    run_main_and_finalize(lambda: (check_atomic_moments(runtime), 0)[1])
