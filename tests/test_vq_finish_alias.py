"""P4 Coulomb finishing preserves resident buffers and explicit sharding."""
import runtime
import numpy as np


def main():
    mesh=runtime.initialize_communicator_stack().mesh
    import jax
    from types import SimpleNamespace
    from unittest.mock import patch
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local,gather_to_host
    from gw.v_q_g_flat import _finish_vq_tile
    from gw.restart_q_storage import capture_scope,peek_pre_unfold
    assert mesh.size==4
    nq,mu=3,12
    physical=np.arange(nq*mu*mu).reshape(nq,mu,mu)*(1+.2j)
    target=NamedSharding(mesh,P(None,'x','y'))
    vec=np.arange(nq*mu).reshape(nq,mu)*(2-.3j)
    def pointer(a):return a.addressable_data(0).unsafe_buffer_pointer()
    def finish(v,g0,*,write=False,one=False,ibz=False,charge=False):
        s=dict(V=v,g0=g0,is_charge_cc=charge,one_leg=one,write_g0=write,L=SimpleNamespace(n_rmu=mu))
        if one:s.update(parts=[g0],one_leg_action='scalar',source_component=0)
        policy=SimpleNamespace(measure_covariance=lambda *a,**kw:None,
                               project_fixed_q=lambda value,*a:(value,0.))
        return _finish_vq_tile(s,V_sh=target,mesh_xy=mesh,stack_cols=None,
            unfold_isdf_one_leg=lambda value,**kw:value,one_leg_gvec=None,
            sym=SimpleNamespace(q_irr_full_idx=np.arange(nq),sym_mats_k=np.eye(3)[None]),
            unfold_sym=np.zeros(nq,np.int32),sym_perm=np.tile(np.arange(mu),(2,1)),
            L_table=np.zeros((2,mu,3)),q_irr_frac=np.zeros((nq,3)),
            kgrid=(3,1,1),use_ibz=ibz,policy=policy,full_to_irr_idx=np.arange(nq))
    cases=0
    for matching in (True,False):
        vspec=P(None,'x','y') if matching else P(None,'y','x')
        value=device_put_process_local(physical,NamedSharding(mesh,vspec))
        for dims,one in ((0,False),(2,False),(3,False),(2,True)):
            host=vec if dims!=3 else np.stack((vec,2*vec),axis=1)
            spec=(P(None,'x') if matching else P(None,'y')) if dims!=3 else (P(None,None,'x') if matching else P(None,None,'y'))
            g0=device_put_process_local(host,NamedSharding(mesh,spec))
            out,gout=finish(value,g0,write=bool(dims),one=one)
            np.testing.assert_array_equal(np.asarray(gather_to_host(out)),physical)
            assert out.sharding.spec==target.spec
            if matching:assert pointer(out)==pointer(value)
            if dims:
                np.testing.assert_array_equal(np.asarray(gather_to_host(gout)),host)
                if matching:assert pointer(gout)==pointer(g0)
            else:assert gout is None
            cases+=1
        # The traced path still inserts the target constraint, including a
        # genuine mismatch; this does not promise physical aliasing under jit.
        traced=jax.jit(lambda v:finish(v,None)[0],out_shardings=target)(value)
        np.testing.assert_array_equal(np.asarray(gather_to_host(traced)),physical)
    # On an IBZ charge path the authenticated capture and returned value
    # must be the same physical canonical operator after projection.
    value=device_put_process_local(physical,target)
    with capture_scope(),patch('common.sanity.report_parent_covariance',lambda *a,**kw:None):
        out,_=finish(value,None,ibz=True,charge=True)
        capture=peek_pre_unfold('V_qmunu')
        assert pointer(out)==pointer(capture.X_ibz)
    print('VQ_FINISH_ALIAS PASS',cases,'branches plus traced reshards and captured-parent alias',flush=True)


if __name__=='__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
