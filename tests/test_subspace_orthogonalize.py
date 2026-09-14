"""Numerical and communication contract for planned CGS2 orthogonalization."""
from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local, gather_to_host
from distrib_la import plan_orthogonalization, plan_subspace


def _orthonormal_rows(count, size, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(size, count)) + 1j * rng.normal(
        size=(size, count))
    return np.linalg.qr(matrix)[0].T.astype(np.complex128)


def _cgs2_reference(v, p, active, *, start=0):
    """Apply the specified interval twice without reading any other row."""
    result = np.array(p, copy=True).reshape(p.shape[0], -1)
    if active == 0:
        return result.reshape(p.shape)
    basis = np.asarray(v[start:start + active]).reshape(active, result.shape[1])
    for _ in range(2):
        coefficients = basis.conj() @ result.T
        result -= coefficients.T @ basis
    return result.reshape(p.shape)


def _regular_inputs(*, capacity=12, block=3, vector_shape=(32,), seed=419):
    size = int(np.prod(vector_shape))
    if capacity > size:
        raise ValueError("the test basis needs at least capacity vector entries")
    rng = np.random.default_rng(seed)
    v = _orthonormal_rows(capacity, size, seed).reshape(
        (capacity,) + tuple(vector_shape))
    p = rng.normal(size=(block,) + tuple(vector_shape))
    p = (p + 1j * rng.normal(size=p.shape)).astype(np.complex128)
    return v, p


@pytest.mark.parametrize("start,active", [
    (0, 0), (5, 0), (0, 1), (3, 5), (0, 12),
])
def test_orthogonalize_matches_two_pass_definition_and_excludes_poison(
        start, active):
    capacity, block = 12, 3
    v, p = _regular_inputs(capacity=capacity, block=block)
    stop = start + active
    # Values outside the selected interval must not enter either projection.
    poisoned = v.copy()
    poisoned[:start] = np.nan
    poisoned[stop:] = np.nan
    expected = _cgs2_reference(v, p, active, start=start)
    original_v, original_p = poisoned.copy(), p.copy()

    plan = plan_subspace(capacity=capacity, n_eig=block)
    operands = jnp.asarray(poisoned), jnp.asarray(p)
    actual = jax.jit(plan.orthogonalize)(
        *operands, jnp.int32(active), start=jnp.int32(start))

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-13,
                               atol=3e-12)
    # Compiled aliases may reuse donated buffers, but ordinary calls must keep
    # their inputs unchanged.
    np.testing.assert_array_equal(np.asarray(operands[0]), original_v)
    np.testing.assert_array_equal(np.asarray(operands[1]), original_p)


def test_one_compilation_accepts_distinct_runtime_intervals():
    """Changing interval bounds does not change shapes or compile a new loop."""
    capacity, block = 12, 3
    v, p = _regular_inputs(capacity=capacity, block=block, seed=420)
    poisoned = v.copy()
    poisoned[:2] = np.nan
    poisoned[8:] = np.nan
    plan = plan_subspace(capacity=capacity, n_eig=block)
    compiled = jax.jit(plan.orthogonalize).lower(
        jnp.asarray(poisoned), jnp.asarray(p), jnp.int32(6),
        start=jnp.int32(2)).compile()

    for start, active in ((2, 6), (3, 2), (7, 1), (4, 0)):
        got = compiled(jnp.asarray(poisoned), jnp.asarray(p),
                       jnp.int32(active), start=jnp.int32(start))
        want = _cgs2_reference(v, p, active, start=start)
        np.testing.assert_allclose(np.asarray(got), want, rtol=2e-13,
                                   atol=3e-12)


def test_standalone_plan_does_not_query_an_eigensolver(monkeypatch):
    """Planning CGS2 alone must not allocate or query unrelated eigenscratch."""
    import distrib_la.active_subspace as active_subspace

    def refuse(*args, **kwargs):
        raise AssertionError("standalone orthogonalization queried eigenscratch")

    monkeypatch.setattr(active_subspace, "active_eigh_workspace", refuse)
    plan = plan_orthogonalization(capacity=12, max_block_size=3)
    assert not ({"eigh", "eigenvalues", "info", "qr_r"} &
                set(plan.workspace_specs))
    assert set(plan.workspace_specs) <= {
        "gram", "blas", "orthogonalization_ranges"}
    v, p = _regular_inputs()
    got = jax.jit(plan)(jnp.asarray(v), jnp.asarray(p), jnp.int32(5),
                        start=jnp.int32(3))
    np.testing.assert_allclose(np.asarray(got),
                               _cgs2_reference(v, p, 5, start=3),
                               rtol=2e-13, atol=3e-12)


@pytest.mark.parametrize("kwargs", [
    {"capacity": 0, "max_block_size": 3},
    {"capacity": 12.0, "max_block_size": 3},
    {"capacity": 12, "max_block_size": 0},
    {"capacity": 12, "max_block_size": 3.0},
])
def test_standalone_plan_refuses_invalid_static_geometry(kwargs):
    with pytest.raises((TypeError, ValueError)):
        plan_orthogonalization(**kwargs)


def test_standalone_plan_checks_operand_shape_dtype_and_block_width():
    plan = plan_orthogonalization(capacity=12, max_block_size=2)
    v, p = _regular_inputs(block=2)
    with pytest.raises(ValueError, match="planned geometry"):
        plan(jnp.asarray(v[:-1]), jnp.asarray(p), jnp.int32(5))
    with pytest.raises(ValueError, match="planned geometry"):
        plan(jnp.asarray(v), jnp.asarray(np.concatenate([p, p[:1]])),
             jnp.int32(5))
    with pytest.raises(TypeError, match="complex128"):
        plan(jnp.asarray(v.real), jnp.asarray(p.real), jnp.int32(5))


def _nearly_dependent_inputs(*, capacity=12, block=2, start=2, active=7,
                             size=48, seed=421):
    rng = np.random.default_rng(seed)
    v = _orthonormal_rows(capacity, size, seed)
    basis = v[start:start + active]
    coefficients = rng.normal(size=(block, active))
    coefficients = coefficients + 1j * rng.normal(size=coefficients.shape)
    remainder = rng.normal(size=(block, size))
    remainder = remainder + 1j * rng.normal(size=remainder.shape)
    remainder -= (basis.conj() @ remainder.T).T @ basis
    remainder /= np.linalg.norm(remainder, axis=1)[:, None]
    p = coefficients @ basis + 1e-10 * remainder
    return v.astype(np.complex128), p.astype(np.complex128), remainder


def test_second_pass_recovers_a_nearly_dependent_remainder():
    """The second projection removes roundoff amplified by near dependence."""
    capacity, block, start, active = 12, 2, 2, 7
    v, p, remainder = _nearly_dependent_inputs(
        capacity=capacity, block=block, start=start, active=active)
    basis = v[start:start + active]

    coefficients = basis.conj() @ p.T
    one_pass = p - coefficients.T @ basis
    one_pass_relative_overlap = (
        np.linalg.norm(basis.conj() @ one_pass.T) /
        np.linalg.norm(one_pass))
    assert one_pass_relative_overlap > 1e-9

    expected = _cgs2_reference(v, p, active, start=start)
    plan = plan_subspace(capacity=capacity, n_eig=block)
    actual = jax.jit(plan.orthogonalize)(
        jnp.asarray(v), jnp.asarray(p), jnp.int32(active),
        start=jnp.int32(start))
    actual = np.asarray(actual)
    relative_overlap = (np.linalg.norm(basis.conj() @ actual.T) /
                        np.linalg.norm(actual))
    assert relative_overlap < 2e-12
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-15)
    # The small component itself survives; the test is not satisfied by zero.
    assert np.all(np.linalg.norm(actual, axis=1) > 0.8e-10)
    assert np.all(np.linalg.norm(actual, axis=1) < 1.2e-10)
    np.testing.assert_allclose(
        np.abs(np.sum(actual.conj() * remainder, axis=1)), 1e-10,
        rtol=2e-5, atol=2e-15)


@pytest.mark.parametrize("start,active", [
    (-1, 1), (0, -1), (8, 5), (13, 0),
    (2**32, 0), (0, 2**32),
])
def test_orthogonalize_refuses_invalid_runtime_intervals(start, active):
    v, p = _regular_inputs()
    plan = plan_subspace(capacity=12, n_eig=3)
    run = jax.jit(plan.orthogonalize)
    with pytest.raises(Exception, match=re.compile(r"ortho", re.IGNORECASE)):
        jax.block_until_ready(run(
            jnp.asarray(v), jnp.asarray(p), jnp.asarray(active, jnp.int64),
            start=jnp.asarray(start, jnp.int64)))


def _p4_sharding():
    if jax.process_count() != 4 or jax.device_count() != 4:
        pytest.skip("requires four real GPU ranks")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    return NamedSharding(mesh, P(None, "x", "y"))


@pytest.mark.parametrize("start,active", [(4, 0), (2, 7)])
def test_p4_default_orthogonalize_matches_global_cgs2_without_vector_collectives(
        tmp_path, start, active):
    sharding = _p4_sharding()
    capacity, block, vector_shape = 12, 2, (8, 6)
    v, p, _ = _nearly_dependent_inputs(
        capacity=capacity, block=block, start=2, active=7,
        size=np.prod(vector_shape), seed=422)
    v = v.reshape((capacity,) + vector_shape)
    p = p.reshape((block,) + vector_shape)
    stop = start + active
    poisoned = v.copy()
    poisoned[:start] = np.nan
    poisoned[stop:] = np.nan
    expected = _cgs2_reference(v, p, active, start=start)

    standalone = plan_orthogonalization(
        capacity=capacity, max_block_size=block, vector_sharding=sharding)
    assert standalone.backend == "cuda_jax_collectives"
    plan = plan_subspace(capacity=capacity, n_eig=block,
                         vector_sharding=sharding)
    vv = device_put_process_local(poisoned, sharding)
    pp = device_put_process_local(p, sharding)
    compiled = jax.jit(plan.orthogonalize).lower(
        vv, pp, jnp.int32(active), start=jnp.int32(start)).compile()
    actual = compiled(vv, pp, jnp.int32(active),
                      start=jnp.int32(start))
    np.testing.assert_allclose(gather_to_host(actual), expected,
                               rtol=3e-12, atol=4e-12)
    assert actual.sharding.is_equivalent_to(sharding, actual.ndim)

    hlo = compiled.as_text()
    (tmp_path / f"orthogonalize_{start}_{active}.hlo").write_text(hlo)
    (tmp_path / f"memory_{start}_{active}.txt").write_text(
        str(compiled.memory_analysis()))
    forbidden = ("all-gather", "reduce-scatter", "all-to-all",
                 "collective-permute")
    assert not any(token in hlo for token in forbidden)
    # The default keeps the two small coefficient reductions visible to XLA.
    # Three local calls update the correction in place; no call or collective
    # communicates a complete distributed vector.
    target_lines = {
        target: [line for line in hlo.splitlines()
                 if f'custom_call_target="{target}"' in line]
        for target in (
            "lorrax_active_subspace_gram",
            "lorrax_active_subspace_subtract_gram",
            "lorrax_active_subspace_subtract",
        )
    }
    assert {target: len(lines) for target, lines in target_lines.items()} == {
        "lorrax_active_subspace_gram": 1,
        "lorrax_active_subspace_subtract_gram": 1,
        "lorrax_active_subspace_subtract": 1,
    }
    assert "lorrax_active_subspace_distributed_ortho" not in hlo
    assert all("output_to_operand_aliasing={{0}: (1, {})}" in line
               for line in target_lines["lorrax_active_subspace_subtract"])
    assert all("output_to_operand_aliasing={{0}: (1, {}), {1}: (2, {})}"
               in line
               for line in target_lines["lorrax_active_subspace_subtract_gram"])
    all_reduce_lines = [line for line in hlo.splitlines()
                        if re.search(r"\ball-reduce(?:-start)?\(", line)]
    assert len(all_reduce_lines) == 2
    # Compiler text is per-rank: the local basis is (12,4,3), sometimes
    # flattened to (12,12). Checking the global shape would be vacuous.
    assert not any(("c128[12,4,3]" in line or "c128[12,12]" in line)
                   and "copy(" in line for line in hlo.splitlines())


def test_p4_explicit_native_collectives_keep_single_distributed_call(tmp_path):
    """The higher-memory NCCL route remains available only by explicit opt-in."""
    sharding = _p4_sharding()
    capacity, block = 12, 2
    v, p = _regular_inputs(block=block, vector_shape=(8, 6), seed=425)
    expected = _cgs2_reference(v, p, 7, start=2)
    plan = plan_orthogonalization(
        capacity=capacity, max_block_size=block, vector_sharding=sharding,
        native_collectives=True)
    assert plan.backend == "cuda_nccl"
    vv = device_put_process_local(v.reshape(12, 8, 6), sharding)
    pp = device_put_process_local(p.reshape(2, 8, 6), sharding)
    compiled = jax.jit(plan).lower(
        vv, pp, jnp.int32(7), start=jnp.int32(2)).compile()
    actual = compiled(vv, pp, jnp.int32(7), start=jnp.int32(2))
    np.testing.assert_allclose(gather_to_host(actual), expected,
                               rtol=3e-12, atol=4e-12)
    hlo = compiled.as_text()
    (tmp_path / "orthogonalize_native_collectives.hlo").write_text(hlo)
    lines = [line for line in hlo.splitlines()
             if 'custom_call_target="lorrax_active_subspace_distributed_ortho"'
             in line]
    assert len(lines) == 1
    assert "output_to_operand_aliasing={{0}: (1, {})}" in lines[0]
    assert not re.search(r"\ball-reduce(?:-start)?\(", hlo)
    assert plan.workspace_specs["orthogonalization_ranges"].shape == (8,)


@pytest.mark.parametrize("start,active", [(8, 5), (2**32, 0)])
def test_p4_distributed_provider_refuses_invalid_common_interval(start, active):
    sharding = _p4_sharding()
    v, p = _regular_inputs(block=2, vector_shape=(8, 6), seed=423)
    plan = plan_subspace(capacity=12, n_eig=2,
                         vector_sharding=sharding,
                         native_collectives=True)
    vv = device_put_process_local(v.reshape(12, 8, 6), sharding)
    pp = device_put_process_local(p.reshape(2, 8, 6), sharding)
    run = jax.jit(plan.orthogonalize)
    with pytest.raises(Exception, match=re.compile(r"ortho", re.IGNORECASE)):
        jax.block_until_ready(run(
            vv, pp, jnp.asarray(active, jnp.int64),
            start=jnp.asarray(start, jnp.int64)))


def test_p4_distributed_provider_refuses_rank_disagreement():
    """Every rank reports a mismatched interval instead of entering NCCL alone."""
    sharding = _p4_sharding()
    v, p = _regular_inputs(block=2, vector_shape=(8, 6), seed=424)
    plan = plan_subspace(capacity=12, n_eig=2,
                         vector_sharding=sharding,
                         native_collectives=True)
    vv = device_put_process_local(v.reshape(12, 8, 6), sharding)
    pp = device_put_process_local(p.reshape(2, 8, 6), sharding)
    active = jnp.asarray(2 + jax.process_index(), jnp.int64)
    with pytest.raises(Exception, match=re.compile(
            r"(?:ortho|rank|range)", re.IGNORECASE)):
        jax.block_until_ready(jax.jit(plan.orthogonalize)(
            vv, pp, active, start=jnp.asarray(1, jnp.int64)))
