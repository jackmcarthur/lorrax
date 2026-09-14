"""Fused active-subspace insertion and projected-Hamiltonian update."""
from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local, gather_to_host
from distrib_la import plan_subspace


def _inputs(*, cap=12, block=4, vector_shape=(8,), seed=937):
    rng = np.random.default_rng(seed)
    shape = (cap,) + tuple(vector_shape)
    v = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    hv = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    p = rng.normal(size=(block,) + tuple(vector_shape))
    p = p + 1j * rng.normal(size=p.shape)
    hp = rng.normal(size=p.shape) + 1j * rng.normal(size=p.shape)
    h = rng.normal(size=(cap, cap)) + 1j * rng.normal(size=(cap, cap))
    return tuple(np.asarray(x, np.complex128) for x in (v, hv, p, hp, h))


def _numpy_reference(v, hv, p, hp, h, start, count):
    """The specified store followed by project, with no work for count zero."""
    vv, hh, projected = v.copy(), hv.copy(), h.copy()
    if count:
        stop = start + count
        vv[start:stop] = p[:count]
        hh[start:stop] = hp[:count]
        flat_v = vv[:stop].reshape(stop, -1)
        flat_new_hv = hh[start:stop].reshape(count, -1)
        panel = flat_v.conj() @ flat_new_hv.T
        projected[:stop, start:stop] = panel
        projected[start:stop, :stop] = panel.conj().T
    return vv, hh, projected


@pytest.mark.parametrize("start,count", [(0, 0), (0, 1), (4, 3), (8, 4)])
def test_store_project_matches_sequential_definition_and_preserves_inputs(
        start, count):
    """Runtime ranges select exact rows; inactive poison is neither read nor lost."""
    cap, block = 12, 4
    v, hv, p, hp, h = _inputs(cap=cap, block=block)
    # The previous prefix is meaningful. Rows beyond the new active end are
    # poisoned so any capacity-wide projection makes the result nonfinite.
    active = start + count
    if active < cap:
        v[active:] = np.nan
        hv[active:] = np.nan
    originals = tuple(x.copy() for x in (v, hv, p, hp, h))
    plan = plan_subspace(capacity=cap, n_eig=block)
    fused = jax.jit(plan.store_project)
    operands = tuple(jnp.asarray(x) for x in (v, hv, p, hp, h))
    actual = fused(*operands, jnp.int32(start), jnp.int32(count))
    actual = tuple(np.asarray(x) for x in jax.block_until_ready(actual))
    expected = _numpy_reference(v, hv, p, hp, h, start, count)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=0, atol=2e-12,
                                   equal_nan=True)
    # FFI aliases describe internal compiled buffers. Without donation the
    # caller's arrays retain their original values.
    for got, want in zip(jax.device_get(operands), originals):
        np.testing.assert_array_equal(got, want)


def test_store_project_equals_public_store_then_project():
    """The fused primitive has exactly the two-call service semantics."""
    if jax.default_backend() != "gpu":
        pytest.skip("the separate CUDA store/project reference needs the provider")
    cap, block, start, count = 12, 4, 5, 3
    v, hv, p, hp, h = _inputs(cap=cap, block=block, seed=938)
    v[start + count:] = np.nan
    hv[start + count:] = np.nan
    plan = plan_subspace(capacity=cap, n_eig=block)

    def separate(v, hv, p, hp, h, start, count):
        v, hv = plan.store(v, hv, p, hp, start, count)
        h = plan.project(v, hv, start + count, h, start, count)
        return v, hv, h

    operands = tuple(jnp.asarray(x) for x in (v, hv, p, hp, h))
    expected = jax.jit(separate)(*operands, jnp.int32(start), jnp.int32(count))
    actual = jax.jit(plan.store_project)(
        *operands, jnp.int32(start), jnp.int32(count))
    for got, want in zip(jax.device_get(actual), jax.device_get(expected)):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("start,count", [
    (-1, 1), (0, -1), (0, 5), (9, 4), (13, 0),
])
def test_store_project_refuses_invalid_runtime_ranges(start, count):
    if jax.default_backend() != "gpu":
        pytest.skip("runtime refusal is a CUDA provider contract")
    plan = plan_subspace(capacity=12, n_eig=4)
    args = tuple(jnp.asarray(x) for x in _inputs())
    run = jax.jit(plan.store_project)
    with pytest.raises(Exception, match=re.compile(
            r"store/project range outside capacity", re.IGNORECASE)):
        jax.block_until_ready(run(*args, jnp.int32(start), jnp.int32(count)))


def test_store_project_declares_donated_capacity_aliases():
    if jax.default_backend() != "gpu":
        pytest.skip("compiled CUDA alias accounting")
    plan = plan_subspace(capacity=12, n_eig=4)
    args = tuple(jnp.asarray(x) for x in _inputs())
    compiled = jax.jit(
        plan.store_project, donate_argnums=(0, 1, 4)).lower(
            *args, jnp.int32(4), jnp.int32(3)).compile()
    memory = compiled.memory_analysis()
    expected_alias = args[0].nbytes + args[1].nbytes + args[4].nbytes
    assert memory.alias_size_in_bytes >= expected_alias
    hlo = compiled.as_text()
    assert "lorrax_active_subspace_store_project" in hlo
    # One fused custom call replaces the previous store and project calls.
    assert hlo.count("lorrax_active_subspace_store_project") >= 1
    assert not re.search(r'custom_call_target="lorrax_active_subspace_(?:store|project)"', hlo)
    fused_lines = [line for line in hlo.splitlines()
                   if 'custom_call_target="lorrax_active_subspace_store_project"' in line]
    assert fused_lines
    assert all("output_to_operand_aliasing={{0}: (0, {}), {1}: (1, {}), "
               "{2}: (4, {})}" in line for line in fused_lines)
    assert not any(token in hlo for token in (
        " all-gather(", " all-gather-start(", " all-reduce(",
        " reduce-scatter(", " all-to-all(", " collective-permute("))
    assert not any("c128[12,8]" in line and "copy(" in line
                   for line in hlo.splitlines())


def _p4_sharding():
    if jax.process_count() != 4 or jax.device_count() != 4:
        pytest.skip("requires four real GPU ranks")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    return NamedSharding(mesh, P(None, "x", "y"))


@pytest.mark.parametrize("start,count", [(0, 0), (4, 3)])
def test_distributed_store_project_reduces_only_the_new_global_panel(start, count):
    sharding = _p4_sharding()
    cap, block = 12, 3
    arrays = list(_inputs(cap=cap, block=block, vector_shape=(8, 16), seed=939))
    v, hv, p, hp, h = arrays
    v[start + count:] = np.nan
    hv[start + count:] = np.nan
    plan = plan_subspace(capacity=cap, n_eig=block,
                         vector_sharding=sharding)
    vectors = [device_put_process_local(x, sharding) for x in (v, hv, p, hp)]
    h_device = jax.device_put(h, NamedSharding(sharding.mesh, P()))
    run = jax.jit(plan.store_project)
    actual = run(*vectors, h_device, jnp.int32(start), jnp.int32(count))
    expected = _numpy_reference(v, hv, p, hp, h, start, count)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(gather_to_host(got), want, rtol=0,
                                   atol=3e-12, equal_nan=True)
    assert actual[0].sharding.is_equivalent_to(sharding, actual[0].ndim)
    assert actual[1].sharding.is_equivalent_to(sharding, actual[1].ndim)
    assert actual[2].sharding.is_fully_replicated
