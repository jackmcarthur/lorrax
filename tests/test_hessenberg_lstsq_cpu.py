"""Active-prefix least squares keeps full-capacity cutoff and minimum norm."""
import runtime  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)


def test_bucketed_prefix_full_svd_parity_breakdown_and_boundaries(monkeypatch):
    from distrib_la import local_svd as owner
    monkeypatch.setattr(owner, '_process_leg_key', lambda: 'ROCM')
    run=jax.jit(owner.lstsq_hessenberg_local)
    rng=np.random.default_rng(9831);cap=300
    for active in (0,1,7,8,9,16,17,30,300):
        h=np.zeros((cap+1,cap),complex);g=np.zeros(cap+1,complex);g[0]=1
        if active:
            block=np.triu(rng.normal(size=(active+1,active))+1j*rng.normal(size=(active+1,active)),-1)
            block *= .03  # Keep this capacity/parity control well-conditioned.
            block[:active]+=3*np.eye(active)
            h[:active+1,:active]=block
        actual=np.asarray(run(h,g,jnp.int32(active)))
        expected=np.linalg.lstsq(h,g,rcond=None)[0]
        np.testing.assert_allclose(actual,expected,rtol=3e-10,atol=3e-10)
        np.testing.assert_array_equal(actual[active:],0)
    h=np.zeros((301,300),complex);g=np.zeros(301,complex);g[0]=1
    h[:6,:5]=rng.normal(size=(6,5))+1j*rng.normal(size=(6,5));h[:,4]=h[:,1]
    np.testing.assert_allclose(run(h,g,jnp.int32(5)),np.linalg.lstsq(h,g,rcond=None)[0],rtol=2e-12,atol=2e-12)
    np.testing.assert_array_equal(run(np.zeros_like(h),g,jnp.int32(5)),0)


def test_bucket_cutoff_is_full_301_epsilon_with_inclusive_equality(monkeypatch):
    from distrib_la import local_svd as owner
    monkeypatch.setattr(owner,'_process_leg_key',lambda:'ROCM')
    h=np.zeros((301,300),complex);g=np.zeros(301,complex);eps=np.finfo(float).eps
    # A bucket-local default would wrongly keep20eps. The301eps tie must stay.
    diagonal=[1.,20*eps,600*eps,301*eps]
    h[np.arange(4),np.arange(4)]=diagonal;g[:4]=diagonal
    actual=owner.lstsq_hessenberg_local(h,g,jnp.int32(4))
    expected=np.zeros(300,complex);expected[:4]=[1,0,1,1]
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual,owner.lstsq_local(h,g)[0])


def test_invalid_carriers_are_not_hidden_and_other_legs_remain_bitwise(monkeypatch):
    from distrib_la import local_svd as owner
    h=np.zeros((17,16),complex);h[:3,:2]=[[2,0],[.1,3],[0,.2]]
    g=np.zeros(17,complex);g[0]=1
    for leg in ('CPU','CUDA'):
        monkeypatch.setattr(owner,'_process_leg_key',lambda:leg)
        np.testing.assert_array_equal(owner.lstsq_hessenberg_local(h,g,jnp.int32(2)),jnp.linalg.lstsq(h,g)[0])
    monkeypatch.setattr(owner,'_process_leg_key',lambda:'ROCM')
    bad=h.copy();bad[10,10]=1
    assert np.isnan(owner.lstsq_hessenberg_local(bad,g,jnp.int32(2))).all()
    badg=g.copy();badg[-1]=np.nan
    assert np.isnan(owner.lstsq_hessenberg_local(np.zeros_like(h),badg,jnp.int32(0))).all()
    for active in (-1,17):
        assert np.isnan(owner.lstsq_hessenberg_local(h,g,jnp.int32(active))).all()


def test_invalid_carriers_skip_the_svd_execution_arm(monkeypatch):
    from distrib_la import local_svd as owner
    monkeypatch.setattr(owner,'_process_leg_key',lambda:'ROCM')
    original=owner.svd_local;calls=[]

    def observed(a,*args,**kwargs):
        jax.debug.callback(lambda width:calls.append(int(width)),jnp.int32(a.shape[1]))
        return original(a,*args,**kwargs)

    monkeypatch.setattr(owner,'svd_local',observed)
    run=jax.jit(lambda h,g,k:owner.lstsq_hessenberg_local(h,g,k))
    h=np.zeros((17,16),complex);h[0,0]=2;h[1,1]=3
    g=np.zeros(17,complex);g[0]=1
    np.asarray(run(h,g,jnp.int32(2)));jax.effects_barrier()
    assert calls==[8]
    calls.clear()
    for bad_h,bad_g,active in ((h.copy(),g,2),(h.copy(),g,-1),
                               (h.copy(),g,17),(np.zeros_like(h),g.copy(),0)):
        if active==2:bad_h[10,10]=np.nan
        if active==0:bad_g[-1]=np.nan
        assert np.isnan(run(bad_h,bad_g,jnp.int32(active))).all()
        jax.effects_barrier()
        assert not calls
    np.testing.assert_array_equal(run(np.zeros_like(h),g,jnp.int32(0)),0)
    jax.effects_barrier()
    assert not calls
