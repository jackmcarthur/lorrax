"""Capacity eigensolves keep the shared provider window and active semantics."""
import runtime  # noqa: F401
import numpy as np


def _setup(monkeypatch, capacity, roots):
    import jax
    jax.config.update('jax_enable_x64', True)
    from distrib_la import local_eigh as owner
    from distrib_la.active_subspace import LocalSubspacePlan
    monkeypatch.setattr(owner, '_process_leg_key', lambda: 'ROCM')
    return jax, owner, LocalSubspacePlan(capacity, roots, 0,
                                       handlers='devsize', leg_key='ROCM')


def _check(a, active, values, vectors):
    m, roots = int(active), len(values)
    kept = min(m, roots)
    if kept:
        physical = np.tril(a[:m,:m])+np.tril(a[:m,:m],-1).conj().T
        np.testing.assert_allclose(values[:kept],np.linalg.eigvalsh(physical)[:kept],atol=3e-13)
        v = vectors[:m,:kept]
        np.testing.assert_allclose(physical@v,v*values[None,:kept],atol=3e-13)
        np.testing.assert_allclose(v.conj().T@v,np.eye(kept),atol=3e-13)
    np.testing.assert_array_equal(vectors[m:],0)
    np.testing.assert_array_equal(vectors[:,kept:],0)
    assert np.isposinf(values[kept:]).all()


def test_capacity_below_native_window_dynamic_active_and_padding(monkeypatch):
    jax, owner, plan = _setup(monkeypatch,72,4)
    import jax.numpy as jnp
    def forbidden(*_args):
        raise AssertionError('capacity72 must not bypass the shared native lower bound128')
    monkeypatch.setattr(owner,'_native_batched_eigh',forbidden)
    solve = jax.jit(plan.eigh)
    rng = np.random.default_rng(21772)
    for active in (0,1,2,7,72):
        # Inactive poison must never reach the full eigensolver. A positive
        # active spectrum also exposes accidental inactive-zero selection.
        raw = rng.normal(size=(active,active))+1j*rng.normal(size=(active,active))
        block = .02*(raw+raw.conj().T)+3*np.eye(active)
        a = np.full((72,72),complex(np.nan,np.nan))
        a[:active,:active] = block
        e,v = map(np.asarray,solve(jnp.asarray(a),jnp.int32(active)))
        _check(a,active,e,v)
    # The material failure was a well-conditioned later Gram near I.
    raw = rng.normal(size=(72,72))+1j*rng.normal(size=(72,72))
    a = np.eye(72)+1e-13*(raw+raw.conj().T)
    e,v = map(np.asarray,solve(a,jnp.int32(72)))
    _check(a,72,e,v)


def test_capacity_inside_window_keeps_native_and_failure_mask(monkeypatch):
    jax, owner, plan = _setup(monkeypatch,128,4)
    import jax.numpy as jnp
    received=[]
    def native(_row,b):
        received.append(b.shape)
        return jnp.linalg.eigh(b)
    monkeypatch.setattr(owner,'_native_batched_eigh',native)
    a = np.zeros((128,128),complex)
    a[:7,:7] = np.diag(np.arange(7)+2.)
    a[1,0],a[0,1] = .1j,-.1j
    e,v = map(np.asarray,jax.jit(plan.eigh)(a,jnp.int32(7)))
    _check(a,7,e,v)
    assert received==[(1,128,128)]
    def failed(_row,b):
        return jnp.full(b.shape[:-1],jnp.nan),jnp.full_like(b,jnp.nan)
    monkeypatch.setattr(owner,'_native_batched_eigh',failed)
    e,v = plan.eigh(jnp.asarray(a),jnp.int32(7))
    assert np.isnan(e).all() and np.isnan(v[:7]).any()


def test_capacity_policy_workspace_includes_jaxlib_outside_window(monkeypatch):
    jax, owner, _plan = _setup(monkeypatch,72,4)
    from distrib_la import plan, workspace
    from jax.sharding import Mesh
    monkeypatch.setattr(workspace,'mesh_platform',lambda _mesh:'ROCM')
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),('x','y'))
    workspace.local_eigh_workspace.cache_clear()
    row = owner.BATCHED_EIGH_ROUTE['ROCM']
    win = row.windows['complex128']
    for n in (72,128):
        # The real CPU plan requires no ROCm library; only its query surface
        # is priced for ROCm, without allocating any matrix or workspace.
        policy = plan('eigh',mesh,n=n,backend='off',batched_route='auto')
        expected = (win.workspace_tiles*n*n*16+n*8 if n<win.min_n else
                    owner.batched_workspace_bytes((1,n,n),np.complex128,'ROCM'))
        assert expected>0
        assert workspace.local_eigh_workspace(mesh,1,n,np.complex128)==expected
        assert workspace.workspace_bytes_per_rank(
            policy,'eigh',((n,n),),np.complex128)==expected
        for share in (1,3):
            assert workspace.workspace_bytes_per_rank(
                policy,'eigh',((share,n,n),),np.complex128)==workspace.local_eigh_workspace(
                    mesh,share,n,np.complex128)
    assert owner.batched_eigh_row((72,72),np.complex128,'ROCM') is None
    assert owner.batched_eigh_row((128,128),np.complex128,'ROCM') is not None
