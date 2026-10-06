"""P4 distributed explicit charge equilibration with canonical factor/solve."""
from pathlib import Path
import json
import os


def check_charge_equilibration(runtime):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local,gather_to_host
    from gw.isdf_fitting import (equilibrate_charge_gram,scale_charge_factor_rows,
                                add_pad_diagonal_sharded)
    from isdf.core import factor_c_q,zeta_factor_resident
    from isdf.zeta_mubatch import _local_coefficient_solve

    mesh = runtime.mesh
    assert mesh.size == 4
    rng = np.random.default_rng(20817)
    active = np.asarray([True,True,False,True,True,False,True,True])
    slots = np.flatnonzero(active)
    features = rng.normal(size=(3,19,6))+1j*rng.normal(size=(3,19,6))
    features *= np.geomspace(.05,20,6)
    physical = np.einsum('qpm,qpn->qmn',features.conj(),features)
    C = np.zeros((3,8,8),np.complex128)
    C[:,slots[:,None],slots] = physical
    put = lambda a,s:device_put_process_local(a,NamedSharding(mesh,s))
    get = lambda a:np.asarray(gather_to_host(a))
    Ceq,D = equilibrate_charge_gram(put(C,P(None,'x','y')),active,mesh_xy=mesh)
    expected_D = np.ones((3,8))
    expected_D[:,slots] = 1/np.sqrt(np.diagonal(physical,axis1=-2,axis2=-1).real)
    np.testing.assert_allclose(get(D),expected_D,rtol=3e-15,atol=1e-15)
    np.testing.assert_allclose(get(Ceq),C*expected_D[:,:,None]*expected_D[:,None,:],rtol=5e-15,atol=1e-15)
    np.testing.assert_array_equal(get(D)[:,~active],1.)
    Ceq = add_pad_diagonal_sharded(Ceq,active,len(slots),mesh_xy=mesh)
    B = factor_c_q(Ceq,mesh,n_rmu_logical=8,solver_kind='replicated_rank_truncate',zeta_rcond=1e-8)
    Bphysical = scale_charge_factor_rows(B,D,mesh_xy=mesh)
    rhs = rng.normal(size=(3,8,7))+1j*rng.normal(size=(3,8,7))
    rhs[:,~active] = 0
    dense_inverse = np.linalg.inv(physical)
    expected = np.zeros_like(rhs)
    expected[:,slots] = np.einsum('qmn,qnr->qmr',dense_inverse,rhs[:,slots])
    inverse = get(jax.jit(lambda b:b@jnp.swapaxes(b.conj(),-1,-2))(Bphysical))
    obtained = np.einsum('qmn,qnr->qmr',inverse,rhs)
    inverse_error = np.linalg.norm(obtained-expected)/np.linalg.norm(expected)
    assert inverse_error < 3e-12,inverse_error
    assert np.max(np.abs(obtained[:,~active])) < 1e-12
    # The exact same physical factor serves independently stored PW/local RHSs
    # on q owners, including a completely inert q pad row.
    Bowned,_ = zeta_factor_resident(Bphysical,None,mesh,solver_kind='replicated_rank_truncate')
    rhs_pad = np.pad(rhs,((0,1),(0,0),(0,0)))
    solve = _local_coefficient_solve(mesh,'replicated_rank_truncate',8,3)
    answer = get(solve(Bowned,put(rhs_pad,P(('x','y'),None,None))))
    pw = get(solve(Bowned,put(rhs_pad[:,:,:4],P(('x','y'),None,None))))
    local = get(solve(Bowned,put(rhs_pad[:,:,4:],P(('x','y'),None,None))))
    np.testing.assert_allclose(answer[:3],expected,rtol=3e-12,atol=3e-12)
    np.testing.assert_array_equal(answer[3],0.)
    np.testing.assert_allclose(pw,answer[:,:,:4],rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(local,answer[:,:,4:],rtol=3e-13,atol=3e-13)
    # A star action changes only the centroid/q permutation. Both D and
    # the physical inverse must commute with this authenticated transport.
    permutation = np.asarray([6,3,2,7,0,5,4,1])
    qperm = np.asarray([2,0,1])
    Cs = C[qperm][:,permutation][:,:,permutation]
    Ces,Ds = equilibrate_charge_gram(put(Cs,P(None,'x','y')),active,mesh_xy=mesh)
    np.testing.assert_allclose(get(Ds),expected_D[qperm][:,permutation],rtol=3e-15)
    Ces = add_pad_diagonal_sharded(Ces,active,len(slots),mesh_xy=mesh)
    Bs = scale_charge_factor_rows(factor_c_q(Ces,mesh,n_rmu_logical=8,
        solver_kind='replicated_rank_truncate',zeta_rcond=1e-8),Ds,mesh_xy=mesh)
    star_inverse = get(jax.jit(lambda b:b@jnp.swapaxes(b.conj(),-1,-2))(Bs))
    np.testing.assert_allclose(star_inverse,inverse[qperm][:,permutation][:,:,permutation],rtol=4e-12,atol=3e-12)
    # Unit diagonal makes the optional branch an exact no-op, including
    # canonical eigensolver bits and the factor applied to either RHS.
    unit = np.eye(8,dtype=np.complex128)[None].repeat(3,axis=0)
    unit[:,0,1] = .2+.1j;unit[:,1,0] = .2-.1j
    original = put(unit,P(None,'x','y'))
    no_op,D1 = equilibrate_charge_gram(original,np.ones(8,bool),mesh_xy=mesh)
    np.testing.assert_array_equal(get(no_op),unit)
    old_B = factor_c_q(original,mesh,n_rmu_logical=8,solver_kind='replicated_rank_truncate',zeta_rcond=1e-8)
    new_B = scale_charge_factor_rows(factor_c_q(no_op,mesh,n_rmu_logical=8,
        solver_kind='replicated_rank_truncate',zeta_rcond=1e-8),D1,mesh_xy=mesh)
    np.testing.assert_array_equal(get(new_B),get(old_B))
    # Explicitly demonstrate changed truncated metric without weakening
    # rcond or certified amplification. A small physical diagonal direction
    # can be represented while its equilibrated condition is exactly one.
    tiny = np.eye(8,dtype=np.complex128)[None].repeat(3,axis=0)
    tiny[:,1,1] = 1e-10
    old = factor_c_q(put(tiny,P(None,'x','y')),mesh,n_rmu_logical=8,
        solver_kind='replicated_rank_truncate',zeta_rcond=1e-8)
    small,Ds = equilibrate_charge_gram(put(tiny,P(None,'x','y')),np.ones(8,bool),mesh_xy=mesh)
    new = scale_charge_factor_rows(factor_c_q(small,mesh,n_rmu_logical=8,
        solver_kind='replicated_rank_truncate',zeta_rcond=1e-8),Ds,mesh_xy=mesh)
    assert np.max(np.abs(get(old)[:,1])) == 0
    assert np.min(np.sum(np.abs(get(new)[:,1])**2,axis=-1)) > 9e9
    # Physical invalid diagonals refuse. Transport pad zeros above did not.
    negative_controls = 0
    for value in (0.,-1.,np.nan,1.+1j):
        invalid = unit.copy();invalid[:,0,0] = value
        try:
            equilibrate_charge_gram(put(invalid,P(None,'x','y')),np.ones(8,bool),mesh_xy=mesh)
        except ValueError:
            negative_controls += 1
        else:
            raise AssertionError('invalid diagonal admitted')
    receipt = dict(relative_complex_full_rank_solve_error=float(inverse_error),
        common_pw_local_factor=True,q_owner_and_q_pad=True,interleaved_transport_pad_scale_one=True,
        centroid_and_q_star_covariance=True,unit_diagonal_legacy_factor_bitwise_equal=True,
        changed_truncated_metric_at_same_rcond=True,invalid_diagonal_refusals=negative_controls,
        rank_policy='unchanged default refuse',rcond=1e-8)
    if jax.process_index() == 0:
        print(json.dumps(receipt),flush=True)
        out=os.environ.get('CHARGE_EQUILIBRATION_REPORT')
        if out:Path(out).write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def main():
        check_charge_equilibration(runtime)
        return 0
    run_main_and_finalize(main)
