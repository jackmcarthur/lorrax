"""P4 explicit factor consumption before V unfold preserves shells and reuse defaults."""
import runtime
import numpy as np


def main():
    mesh=runtime.initialize_communicator_stack().mesh
    import gc
    import weakref
    from types import SimpleNamespace
    from unittest.mock import patch
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local,gather_to_host
    from isdf.zeta_mubatch import ZetaG
    from gw.compute_vcoul import compute_all_V_q
    from symmetry_maps import unfold_isdf_operator
    assert mesh.size==4
    mu=8;parents=2
    base=(np.arange(parents*mu*mu).reshape(parents,mu,mu)+.3j)
    parent=device_put_process_local(base,NamedSharding(mesh,P(None,'x','y')))
    shell=device_put_process_local(np.ones((parents,mu,1),complex),NamedSharding(mesh,P(None,('x','y'),None)))
    g0=device_put_process_local(np.ones((3,mu),complex),NamedSharding(mesh,P(None,'x')))
    perm=np.tile(np.arange(mu),(2,1));wraps=np.zeros((2,mu,3));rows=np.array([0,0,1],np.int32)
    metadata=(None,np.zeros((parents,3)),np.array([0,1,1],np.int32),rows,perm,wraps,True)
    cases=0
    for solver in ('inverse','lu'):
        for consume in (False,True):
            closed=[]
            factor=device_put_process_local(np.ones((parents,mu,mu),complex),NamedSharding(mesh,P(None,'x','y')))
            piv=(device_put_process_local(np.ones((parents,mu),np.int32),NamedSharding(mesh,P(None,'x'))) if solver=='lu' else None)
            factor_ref=weakref.ref(factor);piv_ref=None if piv is None else weakref.ref(piv)
            zeta=ZetaG(SimpleNamespace(close=lambda:closed.append(True)),mesh=mesh,L_q=factor,lu_piv=piv,
                solver_kind=solver,batched_route='auto',n_rmu_solve=mu,n_rmu=mu,mu_basis=None,
                ngk_per_q=np.ones(parents,np.int32),gvec_components=np.zeros((parents,3,1),np.int32),path='unused')
            zeta.shell=shell;zeta.shell_slots=np.zeros((parents,1),np.int32)
            del factor,piv
            def contracted(*a,**kw):
                assert zeta.factor is not None
                return parent,g0
            def unfolded(value,**kw):
                gc.collect()
                assert (factor_ref() is None)==consume
                if piv_ref is not None:assert (piv_ref() is None)==consume
                assert zeta.shell is shell and not closed
                np.testing.assert_array_equal(zeta.head_columns(np.zeros((parents,1),np.int32)),np.zeros((parents,1),np.int32))
                return unfold_isdf_operator(value,**kw)
            with patch('gw.v_q_g_flat._compute_V_q_g_flat_one_tile',contracted),patch('gw.v_q_g_flat._resolve_ibz_q_list',lambda **kw:metadata),patch('gw.qgrid_symmetry.qgrid_trs_policy_for',lambda **kw:SimpleNamespace(unfold_sym_idx=rows,n_sym_spatial=1)),patch('symmetry_maps.unfold_isdf_operator',unfolded):
                for _ in range(1 if consume else 2):
                    result,got_g0=compute_all_V_q(zeta,kgrid=(3,1,1),fft_grid=(2,2,2),bvec=np.eye(3),cell_volume=1.,mesh_xy=mesh,sys_dim=3,mc_average_vcoul_body=False,consume_fit_factor=consume)
                    np.testing.assert_array_equal(np.asarray(gather_to_host(result)),np.stack((base[0],base[1],base[1].conj())))
                    assert got_g0 is g0
            if consume:
                try:zeta.factor
                except ValueError as exc:assert 'retired' in str(exc)
                else:raise AssertionError('consumed factor remained available')
            zeta.close();assert closed==[True] and zeta.shell is None
            cases+=1
    print('ZETA_FACTOR_LIFETIME PASS',cases,'LU/face/default-reuse/explicit-consume cases',flush=True)


if __name__=='__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
