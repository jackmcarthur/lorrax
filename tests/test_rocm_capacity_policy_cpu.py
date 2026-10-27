"""Capacity eigensolves keep the shared provider window and active semantics."""
import runtime  # noqa: F401
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
import numpy as np


def _setup(monkeypatch, capacity, roots):
    import jax
    jax.config.update('jax_enable_x64', True)
    from distrib_la import local_eigh as owner
    from distrib_la.active_subspace import LocalSubspacePlan
    monkeypatch.setattr(owner, '_process_leg_key', lambda: 'ROCM')
    import jax.numpy as jnp
    monkeypatch.setattr(owner, 'eigh_local_jacobi', jnp.linalg.eigh)
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
    monkeypatch.setattr(owner,'eigh_local_jacobi',lambda b: (
        jnp.full(b.shape[:-1],jnp.nan),jnp.full_like(b,jnp.nan)))
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
        expected = win.workspace_tiles*n*n*16+n*8  # legacy/scalar Jacobi scratch
        assert expected>0
        assert workspace.local_eigh_workspace(mesh,1,n,np.complex128)==expected
        assert workspace.workspace_bytes_per_rank(
            policy,'eigh',((n,n),),np.complex128)==expected
        for share in (1,3):
            assert workspace.workspace_bytes_per_rank(
                policy,'eigh',((share,n,n),),np.complex128)==workspace.local_eigh_workspace(
                    mesh,share,n,np.complex128,batched=True)
    assert owner.batched_eigh_row((72,72),np.complex128,'ROCM') is None
    assert owner.batched_eigh_row((128,128),np.complex128,'ROCM') is not None


def test_distributed_elpa_prices_its_arena_once_and_local_routes_keep_scratch(monkeypatch):
    """Metadata-only: real Plan route selection, no ROCm library or field.

    The native gate checks the actual ELPA executable's u8 arena. Here an
    already-resolved ELPA plan is represented on the available CPU mesh to
    challenge the extra-workspace query, including its misleading rank-3 case.
    """
    from dataclasses import replace
    import jax
    from jax.sharding import Mesh
    from distrib_la import plan, workspace
    from distrib_la.plan import ROUTE_BATCH_RESHARD, ROUTE_SCAN
    mesh=Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),('x','y'))
    monkeypatch.setattr(workspace,'mesh_platform',lambda _mesh:'ROCM')
    workspace.local_eigh_workspace.cache_clear()
    try:
        for dtype in (np.float64,np.complex128):
            n=18436
            native=plan('eigh',mesh,n=n,backend='off',batched_route='auto')
            provider=replace(native,requested='distributed',backend='elpa')
            staged=replace(provider,requested_batched_route='batch_reshard')
            whole_local=replace(provider,backend='one_device')
            for batch in (1,3):
                shape=(batch,n,n)
                assert provider.known_route(shape,dtype)==ROUTE_SCAN
                assert staged.known_route(shape,dtype)==ROUTE_BATCH_RESHARD
                assert workspace.workspace_bytes_per_rank(provider,'eigh',(shape,),dtype)==0
                expected=workspace.local_eigh_workspace(mesh,batch,n,dtype)
                assert expected>0
                for policy in (native,staged,whole_local):
                    assert workspace.workspace_bytes_per_rank(policy,'eigh',(shape,),dtype)==expected
            # A rank-2 ELPA solve remains distributed even if its stack
            # preference is staged. The preference applies only to stacks.
            for policy in (provider,staged):
                assert workspace.workspace_bytes_per_rank(policy,'eigh',((n,n),),dtype,checked=True)==0
            for policy in (native,whole_local):
                assert workspace.workspace_bytes_per_rank(policy,'eigh',((n,n),),dtype)>0
    finally:
        workspace.local_eigh_workspace.cache_clear()


def test_checked_workspace_prices_only_admitted_gathered_retry(monkeypatch):
    """Real gathered-program price and room boundary; provider queries are metadata-only.

    The CPU compiler supplies the actual gathered executable's buffer price.
    An already-resolved ELPA/Mp Plan and their scratch-query surfaces stand in
    for the GPU libraries, without executing a solver or allocating its fields.
    Neither gathered eligibility nor its compiled price is mocked.
    """
    from dataclasses import replace
    import jax
    from jax.sharding import Mesh
    from distrib_la import plan, workspace, _cusolvermp
    from distrib_la.plan import (_gathered_admitted, _gathered_program, compiled_bytes,
                                 GATHERED_EIGH_BYTES)

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1,1), ('x','y'))
    n = 2048
    # Distinguish the distributed and local CUDA query, including a primary
    # workspace larger than its retry: the two solves run sequentially.
    monkeypatch.setattr(_cusolvermp, 'get_or_init_context', lambda _mesh: 17)
    primary = [512]
    monkeypatch.setattr(workspace, '_vendor_query',
                        lambda ctx, *_args: (primary[0] if ctx else 4096, 0))
    try:
        for platform, provider in (('ROCM','elpa'), ('CUDA','cusolvermp')):
            monkeypatch.setattr(workspace, 'mesh_platform', lambda _mesh: platform)
            workspace.local_eigh_workspace.cache_clear()
            for dtype in (np.float64, np.complex128):
                native = plan('eigh', mesh, n=n, backend='off', batched_route='auto')
                extra = workspace.local_eigh_workspace(mesh, 1, n, dtype)
                need = compiled_bytes(_gathered_program(mesh), mesh, (n,n), dtype) + extra
                assert need > GATHERED_EIGH_BYTES  # otherwise room cannot exclude this retry
                for room in (need-1, need):
                    policy = replace(native, requested='distributed', backend=provider,
                                     budget_bytes=room)
                    included = room == need
                    assert _gathered_admitted(mesh, (n,n), dtype, room) == included
                    for primary[0] in (512, 65536):
                        first = 0 if platform == 'ROCM' else primary[0]
                        assert workspace.workspace_bytes_per_rank(
                            policy, 'eigh', ((n,n),), dtype) == first
                        assert workspace.workspace_bytes_per_rank(
                            policy, 'eigh', ((n,n),), dtype, checked=True) == (
                                max(first, extra) if included else first)
    finally:
        workspace.local_eigh_workspace.cache_clear()


def test_measured_stack_placement_keeps_batches_and_unavailable_provider_local(monkeypatch):
    """Actual Plan decisions with quote/provider metadata, no matrix execution.

    Positive local/provider quote stand-ins isolate the timing decision; they
    are not a hardware capacity measurement. Gathered eligibility keeps its
    real CPU compiler owner. Native matched probes qualify the numerical routes.
    """
    from dataclasses import replace
    import importlib
    import jax
    import pytest
    from jax.sharding import Mesh
    from lxkit import memory
    pm = importlib.import_module('distrib_la.plan')
    if len(jax.devices()) < 4:
        pytest.skip('requires four virtual CPU devices for the P4 policy')
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2,2), ('x','y'))
    monkeypatch.setattr(pm, '_STACK_CLASSES', {})
    monkeypatch.setattr(pm, '_STACK_ROUTES', {})
    monkeypatch.setattr(memory, '_RUN', {**memory._RUN, 'nmu': 0})
    provider_calls = []
    provider_price_override = [None]

    def local_quote(_op, _mesh, nb, n, dtype, rounds, *_args):
        share = -(-nb // (4*rounds))
        return 8*share*n*n*np.dtype(dtype).itemsize + 1024

    def provider_quote(policy, op, nb, n, dtype, phase):
        provider_calls.append((policy.backend, nb, n))
        if provider_price_override[0] is not None:
            return provider_price_override[0]
        return nb*n*n*np.dtype(dtype).itemsize//4 + (1 << 20)

    monkeypatch.setattr(pm, '_stack_bytes', local_quote)
    monkeypatch.setattr(pm.Plan, 'mesh_bytes', provider_quote)
    room = 64 << 30
    for platform, provider in (('ROCM','elpa'), ('CUDA','cusolvermp')):
        pm._STACK_CLASSES.clear()
        pm._STACK_ROUTES.clear()
        monkeypatch.setattr(pm, 'mesh_platform', lambda _mesh: platform)
        monkeypatch.setattr(pm, 'resolve_eigh_capacity', lambda *_a, **_k: (provider, None))
        for dtype in (np.float64, np.complex128):
            for n in (8192, 16384):
                policy = pm.plan('eigh', mesh, backend='auto', n=n,
                                 budget_bytes=room, batched_route='auto')
                assert policy.backend == provider
                assert policy.stack_route((1,n,n), dtype).route == pm.ROUTE_SCAN
                for nb in (4, 8):
                    result = policy.stack_route((nb,n,n), dtype)
                    assert result.route == pm.ROUTE_BATCH_RESHARD
                    assert result.program_bytes <= room
            if platform == 'ROCM':
                for n in (256, 2048, 3120, 4096):
                    policy = pm.plan('eigh', mesh, backend='auto', n=n,
                                     budget_bytes=room, batched_route='auto')
                    assert policy.stack_route((1,n,n), dtype).route == pm.ROUTE_BATCH_RESHARD
                for n in (5120, 6144):
                    policy = pm.plan('eigh', mesh, backend='auto', n=n,
                                     budget_bytes=room, batched_route='auto')
                    expected = pm.ROUTE_BATCH_RESHARD if dtype == np.float64 else pm.ROUTE_SCAN
                    assert policy.stack_route((1,n,n), dtype).route == expected

        # Time alone must not select a provider whose own quote is over the
        # room while the local quote fits. This is not an allocator bound.
        pm._STACK_CLASSES.clear()
        pm._STACK_ROUTES.clear()
        provider_price_override[0] = room + 1
        priced = pm.plan('eigh', mesh, backend='auto', n=8192,
                         budget_bytes=room, batched_route='auto')
        assert priced.stack_route((1,8192,8192), np.complex128).route == pm.ROUTE_BATCH_RESHARD
        provider_price_override[0] = None
        monkeypatch.setattr(pm, 'resolve_eigh_capacity', lambda *_a, **_k: (
            'native', 'test capability metadata: provider unavailable'))
        absent = pm.plan('eigh', mesh, backend='auto', n=8192,
                         budget_bytes=room, batched_route='auto')
        before = len(provider_calls)
        assert absent.stack_route((1,8192,8192), np.complex128).route == pm.ROUTE_BATCH_RESHARD
        assert len(provider_calls) == before
        # An explicit one-device backend is still a local backend even when
        # the platform's whole-mesh timing row would be faster for B1.
        explicit = replace(policy, requested='one_device', backend='one_device', n=8192)
        assert explicit.stack_route((1,8192,8192), np.complex128).route == pm.ROUTE_BATCH_RESHARD

    monkeypatch.setattr(pm, 'mesh_platform', lambda _mesh: 'ROCM')
    monkeypatch.setattr(pm, 'resolve_eigh_capacity', lambda *_a, **_k: ('elpa', None))
    for dtype in (np.float32, np.complex64):
        for kind in ('one', 'mesh'):
            assert pm._eigh_seconds(kind, 8192, 4, 'ROCM', dtype) is None
            assert pm._eigh_seconds(kind, 8192, 4, 'CUDA', dtype) == pm._eigh_seconds(
                kind, 8192, 4, 'CUDA', np.complex128)
        policy = pm.plan('eigh', mesh, backend='auto', n=8192,
                         budget_bytes=room, batched_route='auto')
        before = len(provider_calls)
        assert policy.stack_route((1,8192,8192), dtype).route == pm.ROUTE_BATCH_RESHARD
        assert len(provider_calls) == before
    # The unchanged interpolation can round an anchor by one floating-point ulp.
    for kind, n, seconds in (('one',8192,2.05), ('mesh',8192,6.53/4),
                             ('one',16384,13.7), ('mesh',16384,29/4)):
        assert pm._eigh_seconds(kind,n,4,'CUDA') == pytest.approx(seconds,rel=2e-15)
