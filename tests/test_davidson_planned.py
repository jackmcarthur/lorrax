"""Shared solver migration: distributed windows, restart and host API."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local, gather_to_host
from distrib_la import plan_subspace
from solvers import plan_davidson
from solvers.davidson import davidson, LAST_RUN
from solvers.davidson_fixed import CONVERGED, STALLED


def mesh_sharding():
    if jax.device_count() != 4:
        pytest.skip('requires real P4')
    return NamedSharding(Mesh(np.asarray(jax.devices()).reshape(2, 2), ('x', 'y')), P(None, 'x', 'y'))


def test_distributed_window_and_incremental_projection():
    sh = mesh_sharding()
    rng = np.random.default_rng(329)
    cap, b = 12, 3
    v = np.linalg.qr(rng.normal(size=(128, cap))+1j*rng.normal(size=(128, cap)))[0].T.reshape(cap, 8, 16)
    hv = v*np.arange(128).reshape(8, 16)
    p = (rng.normal(size=(b, 8, 16))+1j*rng.normal(size=(b, 8, 16)))
    plan = plan_subspace(capacity=cap, n_eig=b, vector_sharding=sh)
    vv, hh, pp = [device_put_process_local(a, sh) for a in (v, hv, p)]
    def project(v, hv):
        h = jnp.zeros((cap, cap), jnp.complex128)
        h = plan.project(v, hv, 4, h, 0, 4)
        return plan.project(v, hv, 7, h, 4, 3)
    h = jax.jit(project)(vv, hh)
    ref = v[:7].reshape(7, -1).conj()@hv[:7].reshape(7, -1).T
    np.testing.assert_allclose(gather_to_host(h)[:7, :7], ref, atol=1e-12)
    # Poison both prefix and tail outside an interior window.
    poison = v.copy(); poison[:2] = np.nan; poison[7:] = np.nan
    poisoned = device_put_process_local(poison, sh)
    def ortho(v, p, count):
        return plan.orthogonalize(v, p, count, start=2)
    q = jax.jit(ortho)(poisoned, pp, jnp.int32(5))
    expected = p.reshape(b, -1).copy()
    window = v[2:7].reshape(5, -1)
    for _ in range(2):
        expected -= (window.conj()@expected.T).T@window
    np.testing.assert_allclose(gather_to_host(q).reshape(b, -1), expected, atol=3e-12)
    np.testing.assert_allclose(gather_to_host(jax.jit(ortho)(poisoned, pp, jnp.int32(0))), p, atol=0)
    assert q.sharding.is_equivalent_to(sh, q.ndim)


def test_distributed_davidson_restart_and_changed_data(tmp_path):
    sh = mesh_sharding()
    rng = np.random.default_rng(338)
    # Separable diagonal operator needs no vector gather or dense H.
    d = np.linspace(.5, 20, 128).reshape(8, 16)
    x = rng.normal(size=(3, 8, 16))+1j*rng.normal(size=(3, 8, 16))
    x = device_put_process_local(x, sh)
    ds = NamedSharding(sh.mesh, P('x', 'y'))
    d = device_put_process_local(d, ds)
    plan = plan_davidson(lambda d, x: d*x, lambda d, r, e, x: r,
        n_eig=3, capacity=24, vector_shape=(8, 16), vector_sharding=sh)
    compiled = plan.solve.lower(d, x, 1e-8, 300).compile()
    hlo = compiled.as_text()
    (tmp_path/'davidson.hlo').write_text(hlo)
    # Include async op names: an exact-opcode analyzer can miss these.
    assert ' all-gather' not in hlo and ' all-to-all' not in hlo
    (tmp_path/'memory.txt').write_text(str(compiled.memory_analysis()))
    for shift in (0., .2):
        e, v, info = compiled(d+shift, x, 1e-8, 300)
        assert int(gather_to_host(info.status)) == CONVERGED
        assert int(gather_to_host(info.restarts)) > 0
        np.testing.assert_allclose(gather_to_host(e), np.linspace(.5, 20, 128)[:3]+shift, atol=1e-10)
        assert v.sharding.is_equivalent_to(sh, v.ndim)
        residual = jnp.sqrt(jnp.sum(jnp.abs((d+shift)*v-e[:,None,None]*v)**2, axis=(1,2)))
        assert np.max(gather_to_host(residual)) < 1.01e-8


def test_host_api_uses_planned_solver():
    d = jnp.arange(32, dtype=jnp.float64)+.5
    x = jnp.eye(32, dtype=jnp.complex128)[:3]
    e, v = davidson(lambda v: d*v, n_eig=3, X0=x, verbose=False)
    np.testing.assert_allclose(e, [.5, 1.5, 2.5], atol=1e-12)
    assert LAST_RUN['status'] == CONVERGED
    assert LAST_RUN['history_kind'] == 'final_snapshot'


def test_stall_guard_is_explicit_status():
    rng = np.random.default_rng(6)
    h = jnp.asarray(rng.normal(size=(20, 20))+1j*rng.normal(size=(20, 20)))
    x = jnp.eye(20, dtype=jnp.complex128)[:2]
    plan = plan_davidson(lambda h, x: x@h.T, lambda h,r,e,x:r,
        n_eig=2, capacity=12, vector_shape=(20,))
    _, _, info = plan.solve(h, x, 1e-14, 200, 2)
    assert int(info.status) == STALLED


def test_deficient_seed_never_calls_operator_on_zero_tail():
    def apply(d, x):
        return jnp.where(jnp.all(jnp.sum(jnp.abs(x)**2, axis=1) > 0), d*x, jnp.nan)
    from solvers.davidson_fixed import BAD_INITIAL
    plan = plan_davidson(apply, lambda d,r,e,x:r,
                         n_eig=3, capacity=12, vector_shape=(32,))
    x = jnp.zeros((3, 32), jnp.complex128).at[0, 0].set(1)
    _, _, info = plan.solve(jnp.arange(32), x, 1e-8, 10)
    assert int(info.status) == BAD_INITIAL
    assert int(info.matvecs) == 0


@pytest.mark.parametrize('spec', [P(), P(None, 'x', None)])
def test_distributed_plan_refuses_partial_mesh_replication(spec):
    sh = mesh_sharding()
    with pytest.raises(ValueError, match='every nontrivial mesh axis'):
        plan_subspace(capacity=12, n_eig=3,
                      vector_sharding=NamedSharding(sh.mesh, spec))


@pytest.mark.parametrize('shape,dependent', [((8,16),False), ((8,16),True), ((2,4),False)])
def test_distributed_tsqr_preserves_local_vectors(shape, dependent):
    sh = mesh_sharding()
    rng = np.random.default_rng(820)
    x = rng.normal(size=(3,)+shape)+1j*rng.normal(size=(3,)+shape)
    if dependent:
        x[2] = x[0]+1e-12*x[2]
    rows = device_put_process_local(x, sh)
    plan = plan_subspace(capacity=12, n_eig=2, max_block_size=3, vector_sharding=sh)
    compiled = jax.jit(plan.qr).lower(rows).compile()
    q, r = compiled(rows)
    qh, rh = gather_to_host(q).reshape(3, -1), gather_to_host(r)
    np.testing.assert_allclose(qh.conj()@qh.T, np.eye(3), atol=3e-14)
    np.testing.assert_allclose(rh.T@qh, x.reshape(3,-1), atol=3e-14)
    assert q.sharding.is_equivalent_to(sh, q.ndim)
    # QR gathers only reduced R panels, including local entry count < width.
    hlo = compiled.as_text()
    for line in hlo.splitlines():
        if ' all-gather-start(' in line or ' all-gather(' in line:
            assert 'c128[3,3]' in line or 'c128[2,3]' in line, line
