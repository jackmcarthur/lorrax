"""Exact active-band ranges for the platform-independent local GEMM plan.

These cells run both on a four-device emulated CPU mesh and on a real
four-process CUDA mesh.  A spin-pair stream's band-complete copies use
this axis-layout contraction on GPU, while the same pure-JAX kernel is the planned
active-range implementation available on CPU.  The distributed face plan
remains a CUDA/cuBLASMp service and is covered separately.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from lxkit.testing import require_devices

import distrib_la as D
from common.collectives import device_put_process_local
from distrib_la.matmul_plan import _prepare_host_bounds


def _mesh() -> Mesh:
    platform = jax.default_backend()
    require_devices(4, platform)
    return Mesh(np.asarray(jax.devices(platform)[:4]).reshape(2, 2), ("x", "y"))


def _values(rng, shape, dtype):
    value = rng.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        value = value + 1j * rng.standard_normal(shape)
    return value.astype(dtype)


def _put(value, sharding):
    return device_put_process_local(value, sharding)


def _poison_outside(a, b, lo, hi):
    poisoned_a, poisoned_b = a.copy(), b.copy()
    for iq, (begin, end) in enumerate(zip(lo, hi)):
        poisoned_a[iq, :, :begin] = np.nan
        poisoned_a[iq, :, end:] = np.nan
        poisoned_b[iq, :begin, :] = np.nan
        poisoned_b[iq, end:, :] = np.nan
    return poisoned_a, poisoned_b


def _assert_allclose(actual, expected, *, exact=False):
    """Compare every addressable output tile without gathering multi-host data."""
    for shard in actual.addressable_shards:
        want = np.asarray(expected)[shard.index]
        if exact:
            np.testing.assert_array_equal(np.asarray(shard.data), want)
        else:
            np.testing.assert_allclose(
                np.asarray(shard.data), want, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    "dtype,alpha,beta",
    [
        (np.float64, 0.75, 0.0),
        (np.complex128, 0.75 - 0.2j, -0.3 + 0.1j),
    ],
)
def test_axis_plan_active_range_per_q_excludes_nan_bands(dtype, alpha, beta):
    """Mixed parent ranges use only live values and retain all-P output."""
    mesh = _mesh()
    nq, m, k, n = 4, 8, 16, 12
    rng = np.random.default_rng(20260913)
    a = _values(rng, (nq, m, k), dtype)
    b = _values(rng, (nq, k, n), dtype)
    c = _values(rng, (nq, m, n), dtype)
    lo = np.asarray([0, 3, 7, 11], np.int32)
    hi = np.asarray([16, 12, 7, 16], np.int32)
    aa, bb = _poison_outside(a, b, lo, hi)

    # The public resolver path is the production wiring used by the axis
    # wavefunction carrier, rather than a test-only call to a private helper.
    plan = D.gemm_plan(
        mesh, m=m, k=k, n=n, nq=nq, dtype=dtype, layout="axis",
        alpha=alpha, beta=beta, enable_active_range=True,
    )
    args = (_put(aa, plan.in_sharding_a), _put(bb, plan.in_sharding_b))

    @jax.jit
    def apply(a_arg, b_arg, lo_arg, hi_arg, c_arg):
        if beta == 0:
            return plan.active_range(a_arg, b_arg, lo_arg, hi_arg)
        return plan.active_range(a_arg, b_arg, lo_arg, hi_arg, C=c_arg)

    actual = apply(
        *args,
        _put(lo, NamedSharding(mesh, P())),
        _put(hi, NamedSharding(mesh, P())),
        _put(c, plan.out_sharding),
    )
    expected = np.empty_like(c)
    for iq, (begin, end) in enumerate(zip(lo, hi)):
        expected[iq] = (
            alpha * (a[iq, :, begin:end] @ b[iq, begin:end, :])
            + beta * c[iq]
        )
    _assert_allclose(actual, expected)
    assert actual.sharding == plan.out_sharding
    assert tuple(actual.sharding.spec) == (None, "x", "y")


def test_local_active_range_full_and_empty_semantics():
    """The full fast path is unchanged; empty ranges never inspect A/B."""
    mesh = _mesh()
    nq, m, k, n = 2, 8, 15, 12
    rng = np.random.default_rng(20260914)
    a = _values(rng, (nq, m, k), np.complex128)
    b = _values(rng, (nq, k, n), np.complex128)
    plan = D.local_gemm_plan(
        mesh, m=m, k=k, n=n, nq=nq, dtype=np.complex128,
        enable_active_range=True,
    )
    a_j = _put(a, plan.in_sharding_a)
    b_j = _put(b, plan.in_sharding_b)

    dense = plan(a_j, b_j)
    full = plan.active_range(a_j, b_j, 0, k)
    for full_shard, dense_shard in zip(full.addressable_shards,
                                       dense.addressable_shards):
        np.testing.assert_array_equal(
            np.asarray(full_shard.data), np.asarray(dense_shard.data))

    nan_a = _put(np.full_like(a, np.nan), plan.in_sharding_a)
    nan_b = _put(np.full_like(b, np.nan), plan.in_sharding_b)
    empty = jax.jit(lambda x, y, lo, hi: plan.active_range(x, y, lo, hi))(
        nan_a, nan_b,
        _put(np.int32(9), NamedSharding(mesh, P())),
        _put(np.int32(9), NamedSharding(mesh, P())))
    _assert_allclose(empty, np.zeros((nq, m, n)), exact=True)

    beta = -0.625
    c = _values(rng, (nq, m, n), np.complex128)
    accumulate = D.local_gemm_plan(
        mesh, m=m, k=k, n=n, nq=nq, dtype=np.complex128, beta=beta,
        enable_active_range=True,
    )
    got = accumulate.active_range(
        nan_a, nan_b, 4, 4, C=_put(c, accumulate.out_sharding))
    _assert_allclose(got, beta * c, exact=True)


def test_local_active_range_composes_inside_scan_without_retracing_bounds():
    """One executable accepts scalar and per-parent runtime range changes."""
    mesh = _mesh()
    nstep, nq, m, k, n = 3, 3, 8, 17, 12
    rng = np.random.default_rng(20260915)
    a = _values(rng, (nstep, nq, m, k), np.float64)
    b = _values(rng, (nstep, nq, k, n), np.float64)
    lo = np.asarray([[1, 1, 1], [0, 4, 8], [6, 2, 13]], np.int32)
    hi = np.asarray([[16, 16, 16], [17, 11, 8], [6, 13, 17]], np.int32)
    expected = np.empty((nstep, nq, m, n), np.float64)
    for step in range(nstep):
        for iq in range(nq):
            begin, end = lo[step, iq], hi[step, iq]
            expected[step, iq] = (
                a[step, iq, :, begin:end] @ b[step, iq, begin:end, :])
        a[step], b[step] = _poison_outside(a[step], b[step], lo[step], hi[step])

    plan = D.local_gemm_plan(
        mesh, m=m, k=k, n=n, nq=nq, dtype=np.float64,
        enable_active_range=True,
    )
    a_j = _put(a, NamedSharding(mesh, P(None, None, "x", None)))
    b_j = _put(b, NamedSharding(mesh, P(None, None, None, "y")))

    @jax.jit
    def scanned(a_stack, b_stack, los, his):
        def body(carry, values):
            a_step, b_step, lo_step, hi_step = values
            return carry, plan.active_range(a_step, b_step, lo_step, hi_step)
        return jax.lax.scan(body, None, (a_stack, b_stack, los, his))[1]

    got = scanned(
        a_j, b_j,
        _put(lo, NamedSharding(mesh, P())),
        _put(hi, NamedSharding(mesh, P())),
    )
    _assert_allclose(got, expected)


def test_local_active_range_applies_complex_weights_only_inside_live_slices():
    """Phase/selector weights do not recreate or touch inactive band tails."""
    mesh = _mesh()
    nq, m, k, n = 3, 8, 19, 12
    rng = np.random.default_rng(20260917)
    a = _values(rng, (nq, m, k), np.complex128)
    b = _values(rng, (nq, k, n), np.complex128)
    weights = _values(rng, (nq, k), np.complex128)
    lo = np.asarray([2, 5, 11], np.int32)
    hi = np.asarray([17, 5, 19], np.int32)
    # Exercise an exact selector hole inside the first interval as well as
    # arbitrary complex phase/weight values on the other live columns.
    weights[0, 9] = 0
    aa, bb = _poison_outside(a, b, lo, hi)
    poisoned_weights = weights.copy()
    for iq, (begin, end) in enumerate(zip(lo, hi)):
        poisoned_weights[iq, :begin] = np.nan
        poisoned_weights[iq, end:] = np.nan

    alpha = 0.4 + 0.1j
    plan = D.local_gemm_plan(
        mesh, m=m, k=k, n=n, nq=nq, dtype=np.complex128, alpha=alpha,
        enable_active_range=True,
    )

    @jax.jit
    def apply(a_arg, b_arg, lo_arg, hi_arg, weight_arg):
        return plan.active_range(
            a_arg, b_arg, lo_arg, hi_arg, weights=weight_arg)

    got = apply(
        _put(aa, plan.in_sharding_a),
        _put(bb, plan.in_sharding_b),
        _put(lo, NamedSharding(mesh, P())),
        _put(hi, NamedSharding(mesh, P())),
        _put(poisoned_weights, NamedSharding(mesh, P())),
    )
    expected = np.empty((nq, m, n), np.complex128)
    for iq, (begin, end) in enumerate(zip(lo, hi)):
        expected[iq] = alpha * (
            (a[iq, :, begin:end] * weights[iq, None, begin:end])
            @ b[iq, begin:end, :]
        )
    _assert_allclose(got, expected)


def test_local_active_range_invalid_bounds_are_loud_or_poisoned():
    """Static mistakes refuse; traced mistakes cannot silently clamp."""
    mesh = _mesh()
    nq, m, k, n = 2, 8, 16, 12
    plan = D.local_gemm_plan(
        mesh, m=m, k=k, n=n, nq=nq, dtype=np.float64,
        enable_active_range=True,
    )
    a = _put(np.ones((nq, m, k)), plan.in_sharding_a)
    b = _put(np.ones((nq, k, n)), plan.in_sharding_b)

    for lo, hi in ((-1, 3), (7, 6), (0, k + 1)):
        with pytest.raises(ValueError, match="0 <= lo <= hi <= K"):
            plan.active_range(a, b, lo, hi)
    with pytest.raises(ValueError, match="weights must have shape"):
        plan.active_range(a, b, 0, k, weights=jnp.ones((nq, k - 1)))

    @jax.jit
    def dynamic(a_arg, b_arg, lo, hi):
        # A/B are explicit arguments: closing over a multi-host array would
        # ask JAX to materialize its non-addressable shards as a constant.
        return plan.active_range(a_arg, b_arg, lo, hi)

    for lo, hi in ((jnp.int32(-1), jnp.int32(3)),
                   (jnp.int32(7), jnp.int32(6)),
                   (jnp.int32(0), jnp.int32(k + 1))):
        got = dynamic(
            a,
            b,
            _put(np.asarray(lo), NamedSharding(mesh, P())),
            _put(np.asarray(hi), NamedSharding(mesh, P())),
        )
        for shard in got.addressable_shards:
            assert np.isnan(np.asarray(shard.data)).all()


def test_local_active_range_refuses_complex_weights_for_real_plan():
    """A real plan cannot silently discard the phase of complex weights."""
    mesh = _mesh()
    nq, m, k, n = 2, 8, 16, 12
    plan = D.local_gemm_plan(
        mesh, m=m, k=k, n=n, nq=nq, dtype=np.float64,
        enable_active_range=True,
    )
    a = _put(np.ones((nq, m, k)), plan.in_sharding_a)
    b = _put(np.ones((nq, k, n)), plan.in_sharding_b)
    weights = _put(
        np.ones((nq, k), dtype=np.complex128), NamedSharding(mesh, P()))

    with pytest.raises(TypeError, match="complex weights require a complex plan"):
        plan.active_range(a, b, 0, k, weights=weights)


def test_prepared_local_active_range_captures_bounds_outside_jit():
    """Prepared CPU calls have no runtime bounds and preserve weighted slices."""
    mesh = _mesh()
    nq, m, k, n = 3, 8, 19, 12
    rng = np.random.default_rng(20260919)
    a = _values(rng, (nq, m, k), np.complex128)
    b = _values(rng, (nq, k, n), np.complex128)
    weights = _values(rng, (nq, k), np.complex128)
    lo = np.asarray([1, 7, 14], np.int64)
    hi = np.asarray([18, 7, 19], np.int64)
    aa, bb = _poison_outside(a, b, lo, hi)
    poisoned_weights = weights.copy()
    for iq, (begin, end) in enumerate(zip(lo, hi)):
        poisoned_weights[iq, :begin] = np.nan
        poisoned_weights[iq, end:] = np.nan

    alpha = 0.6 - 0.15j
    plan = D.local_gemm_plan(
        mesh, m=m, k=k, n=n, nq=nq, dtype=np.complex128, alpha=alpha,
        enable_active_range=True,
    )
    prepared = plan.prepare_active_range(lo, hi)

    @jax.jit
    def apply(a_arg, b_arg, weight_arg):
        return prepared(a_arg, b_arg, weights=weight_arg)

    got = apply(
        _put(aa, plan.in_sharding_a),
        _put(bb, plan.in_sharding_b),
        _put(poisoned_weights, NamedSharding(mesh, P())),
    )
    expected = np.empty((nq, m, n), np.complex128)
    for iq, (begin, end) in enumerate(zip(lo, hi)):
        expected[iq] = alpha * (
            (a[iq, :, begin:end] * weights[iq, None, begin:end])
            @ b[iq, begin:end, :]
        )
    _assert_allclose(got, expected)
    assert got.sharding == plan.out_sharding


def test_prepare_active_range_validates_eager_bounds_and_accumulate_contract():
    """The factory refuses invalid metadata and retains beta*C semantics."""
    mesh = _mesh()
    nq, m, k, n = 2, 8, 16, 12
    rng = np.random.default_rng(20260920)
    beta = -0.375
    a = _values(rng, (nq, m, k), np.float64)
    b = _values(rng, (nq, k, n), np.float64)
    c = _values(rng, (nq, m, n), np.float64)
    plan = D.local_gemm_plan(
        mesh, m=m, k=k, n=n, nq=nq, dtype=np.float64, beta=beta,
        enable_active_range=True,
    )

    for lo, hi in ((-1, 3), (8, 7), (0, k + 1)):
        with pytest.raises(ValueError, match="0 <= lo <= hi <= K"):
            plan.prepare_active_range(lo, hi)
    with pytest.raises(TypeError, match=r"integer scalars or shape\(nq,\)"):
        plan.prepare_active_range(np.zeros(nq + 1, np.int32), k)

    prepared = plan.prepare_active_range(3, 13)
    with pytest.raises(ValueError, match="C is required"):
        prepared(
            _put(a, plan.in_sharding_a),
            _put(b, plan.in_sharding_b),
        )
    got = prepared(
        _put(a, plan.in_sharding_a),
        _put(b, plan.in_sharding_b),
        C=_put(c, plan.out_sharding),
    )
    expected = a[:, :, 3:13] @ b[:, 3:13, :] + beta * c
    _assert_allclose(got, expected)


def test_prepare_active_range_refuses_bounds_outside_native_int32_metadata():
    """Host validation cannot wrap an otherwise valid large bound to int32."""
    int32_max = np.iinfo(np.int32).max
    synthetic_plan = SimpleNamespace(nq=1, k=int32_max + 2)
    with pytest.raises(ValueError, match="fit signed int32"):
        _prepare_host_bounds(synthetic_plan, 0, int32_max + 1)


def test_local_active_range_refuses_unimplemented_collective_variants():
    """Active ranges do not silently alter reduction or output layouts."""
    mesh = _mesh()
    with pytest.raises(NotImplementedError, match="active_range"):
        D.local_gemm_plan(
            mesh, m=8, k=16, n=12, nq=2, dtype=np.float64,
            reduction_axis="y", enable_active_range=True,
        )
    with pytest.raises(NotImplementedError, match="active_range"):
        D.local_gemm_plan(
            mesh, m=8, k=16, n=12, nq=2, dtype=np.float64,
            out_spec=P(None, "x", None),
            enable_active_range=True,
        )
