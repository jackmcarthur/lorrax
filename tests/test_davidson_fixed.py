"""Local fixed-memory Davidson: spectra, restart, rank loss and explicit data."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from solvers import plan_local_davidson
from solvers.davidson_fixed import (
    CONVERGED, ITERATION_LIMIT, NO_DIRECTIONS, BAD_INITIAL, NONFINITE, BAD_TOLERANCE,
)


@pytest.fixture(scope='module')
def problem():
    if jax.default_backend() != 'gpu':
        pytest.skip('local active Davidson requires CUDA')
    rng = np.random.default_rng(7)
    b, n = 4, 96
    z = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n))
    u = np.linalg.qr(z)[0]
    values = np.linspace(.5, 30, n)
    values[:b] = [.5, .5, .6, .6]
    h = jnp.asarray((u*values)@u.conj().T)
    x = jnp.asarray((u[:, :b]+.04*(rng.normal(size=(n, b))+1j*rng.normal(size=(n, b)))).T)
    def apply(h, x):
        return jnp.einsum('ij,bj->bi', h, x)
    plan = plan_local_davidson(apply, lambda h, r, e, x: r,
                               n_eig=b, capacity=12, vector_shape=(n,))
    return plan, h, x, values[:b], apply


def test_degenerate_spectrum_restart_and_changed_operator(problem):
    plan, h, x, reference, apply = problem
    executable = plan.solve.lower(
        jax.ShapeDtypeStruct(h.shape, h.dtype),
        jax.ShapeDtypeStruct(x.shape, x.dtype), 1e-8, 300).compile()
    for shift in (0., 2.):
        operator = h+shift*jnp.eye(h.shape[0])
        e, v, info = executable(operator, x, 1e-8, 300)
        assert int(info.status) == CONVERGED
        assert int(info.restarts) > 0
        np.testing.assert_allclose(e, reference+shift, atol=1e-9, rtol=0)
        norms = np.linalg.norm(np.asarray(apply(operator, v)-e[:, None]*v), axis=1)
        assert np.all(norms < 1e-8*np.maximum(1., np.abs(e)))
        np.testing.assert_allclose(np.asarray(v).conj()@np.asarray(v).T, np.eye(4), atol=1e-12)
    assert executable.memory_analysis().temp_size_in_bytes > 0
    assert plan.workspace_specs['basis'].shape == (12, 96)


@pytest.mark.parametrize('case,status', [
    ('budget', ITERATION_LIMIT), ('empty_budget', ITERATION_LIMIT),
    ('bad_seed', BAD_INITIAL), ('nonfinite', NONFINITE), ('bad_tol', BAD_TOLERANCE),
])
def test_termination_status(problem, case, status):
    plan, h, x, _, _ = problem
    tolerance, budget = 1e-8, 10
    if case == 'budget':
        tolerance, budget = 1e-14, 1
    elif case == 'empty_budget':
        budget = 0
    elif case == 'bad_seed':
        x = jnp.zeros_like(x)
    elif case == 'nonfinite':
        x = x.at[0, 0].set(jnp.nan)
    elif case == 'bad_tol':
        tolerance = 0.
    _, _, info = plan.solve(h, x, tolerance, budget)
    assert int(info.status) == status


def test_zero_preconditioner_is_not_convergence(problem):
    _, h, x, _, apply = problem
    plan = plan_local_davidson(apply, lambda h, r, e, x: jnp.zeros_like(r),
                               n_eig=4, capacity=12, vector_shape=(96,))
    _, _, info = plan.solve(h, x, 1e-8, 10)
    assert int(info.status) == NO_DIRECTIONS
    assert float(jnp.max(info.residuals)) > 1e-8


def test_rank_one_tail_never_applies_h_to_zero_rows():
    d = jnp.arange(16, dtype=jnp.float64)+.5
    x = jnp.eye(16, dtype=jnp.complex128)[:4]
    x = x.at[3, 3].set(1/jnp.sqrt(2.)).at[3, 4].set(1/jnp.sqrt(2.))
    def apply(d, x):
        # Deliberate negative control: padded H calls poison the solve.
        nonzero = jnp.sum(jnp.abs(x)**2, axis=1) > 0
        return jnp.where(nonzero[:, None], x*d, jnp.nan)
    plan = plan_local_davidson(apply, lambda d, r, e, x: r,
                               n_eig=4, capacity=12, vector_shape=(16,))
    e, _, info = plan.solve(d, x, 1e-10, 10)
    assert int(info.status) == CONVERGED
    assert int(info.matvecs) == 5
    np.testing.assert_allclose(e, np.arange(4)+.5, atol=1e-12, rtol=0)


def test_nonfinite_preconditioner_reports_failure(problem):
    _, h, x, _, apply = problem
    plan = plan_local_davidson(apply, lambda h, r, e, x: jnp.full_like(r, jnp.nan),
                               n_eig=4, capacity=12, vector_shape=(96,))
    _, _, info = plan.solve(h, x, 1e-8, 10)
    assert int(info.status) == NONFINITE


def test_distributed_vectors_refuse_before_native_execution():
    if jax.process_count() != 4:
        pytest.skip('four-rank sharding refusal control')
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ('x', 'y'))
    sh = NamedSharding(mesh, P(None, 'x', 'y'))
    x = jax.make_array_from_callback(
        (4, 8, 8), sh,
        lambda index: np.eye(64, dtype=np.complex128)[:4].reshape(4, 8, 8)[index])
    d = jnp.arange(64, dtype=jnp.float64).reshape(8, 8)+.5
    plan = plan_local_davidson(lambda d, x: d*x, lambda d, r, e, x: r,
                               n_eig=4, capacity=12, vector_shape=(8, 8))
    with pytest.raises(ValueError, match='distributed'):
        plan.solve(d, x, 1e-8, 10)
    outer = jax.jit(lambda x: plan.solve(d, x, 1e-8, 10), in_shardings=sh)
    with pytest.raises(ValueError, match='distributed'):
        outer(x)
