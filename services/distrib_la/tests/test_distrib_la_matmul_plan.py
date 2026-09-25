"""Planned N,N GEMM (``distrib_la.gemm_plan``/``GemmPlan``) — logic tier.

Real cuBLASMp execution needs a CUDA mesh with one process per device, so
none of it is reachable here: every cell in this file either exercises a
pure-Python helper directly, or drives ``gemm_plan()`` on an emulated CPU
mesh far enough to observe its EAGER refusal ladder (backend='off', an
explicit non-cuBLASMp provider, mesh topology, shape divisibility, dtype)
BEFORE it would reach ``get_or_init_context``/the FFI call.  The CPU face
plan (K gathered, XLA dot) is executed here against a dense reference.  Numerics
inside nested ``jit``/``lax.scan``, the donated-``out=`` path, and the
internal-zero-``C`` kernel are the real four-rank CUDA gate,
``check_gemm_plan_cublasmp`` in ``test_distrib_la_multiproc.py``
(``gemm_plan_cublasmp`` CLI cell) — this file names that split rather than
padding a CPU-only report to look like coverage it does not have.
"""
from __future__ import annotations

import numpy as np
import pytest
from lxkit.testing import require_devices

import distrib_la as D
from distrib_la.matmul_plan import _as_extent, _as_scalar, _validate_dtype


def _mesh(px=2, py=2):
    import jax
    from jax.sharding import Mesh
    require_devices(px * py, "cpu")
    return Mesh(np.asarray(jax.devices("cpu")[:px * py]).reshape(px, py),
               ("x", "y"))


# ---------------------------------------------------------------------------
# Pure helpers — no mesh, no jax device work.
# ---------------------------------------------------------------------------

def test_as_extent_accepts_positive_ints_only():
    assert _as_extent("m", 8) == 8
    assert _as_extent("m", np.int64(8)) == 8
    for bad in (0, -1, 1.5, "8", True):
        with pytest.raises(ValueError):
            _as_extent("m", bad)


def test_as_scalar_rejects_non_numeric():
    assert _as_scalar("alpha", 1) == 1 + 0j
    assert _as_scalar("alpha", 1 - 2j) == 1 - 2j
    with pytest.raises(ValueError):
        _as_scalar("alpha", "one")


def test_validate_dtype_only_f64_c128():
    import jax.numpy as jnp
    _validate_dtype(jnp.dtype(jnp.float64))
    _validate_dtype(jnp.dtype(jnp.complex128))
    with pytest.raises(TypeError):
        _validate_dtype(jnp.dtype(jnp.complex64))
    with pytest.raises(TypeError):
        _validate_dtype(jnp.dtype(jnp.float32))


# ---------------------------------------------------------------------------
# Eager construction-time refusals, reachable on an emulated CPU mesh.
# ---------------------------------------------------------------------------

def test_backend_off_refuses_by_name():
    mesh = _mesh()
    with pytest.raises(ValueError, match="never selects batch_reshard"):
        D.gemm_plan(mesh, m=4, k=4, n=4, nq=2, dtype="complex128",
                   backend="off")


def _dense(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


@pytest.mark.parametrize("grid", [(1, 1), (2, 2)])
@pytest.mark.parametrize("beta", [0.0, 0.3 - 0.1j])
def test_cpu_face_plan_matches_dense(grid, beta):
    """On a CPU mesh the face plan gathers K and matches the dense GEMM:
    __call__, active_range (per-q bounds, weights), prepare_active_range and
    local_call inside a manual shard_map.  The output stays P(None,'x','y')."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from distrib_la._shard_map import shard_map
    mesh = _mesh(*grid)
    nq, m, k, n = 3, 8, 12, 6
    alpha = 0.7 + 0.2j
    rng = np.random.default_rng(20260925)
    a, b, c = _dense(rng, (nq, m, k)), _dense(rng, (nq, k, n)), _dense(rng, (nq, m, n))
    w = _dense(rng, (nq, k))
    plan = D.gemm_plan(mesh, m=m, k=k, n=n, nq=nq, dtype="complex128",
                       alpha=alpha, beta=beta, enable_active_range=True)
    face = NamedSharding(mesh, P(None, "x", "y"))
    assert plan.face_gather and "K gathered" in plan.describe()
    assert plan.in_sharding_a == face and plan.in_sharding_b == face
    assert plan.out_sharding == face
    aa, bb, cc = (jax.device_put(v, face) for v in (a, b, c))

    def want(prod):
        return alpha * prod + (beta * c if beta else 0)

    got = plan(aa, bb, C=cc) if beta else plan(aa, bb)
    assert got.sharding == face
    np.testing.assert_allclose(np.asarray(got), want(a @ b), atol=1e-12, rtol=0)

    lo = np.asarray([0, 3, 5], np.int32)
    hi = np.asarray([12, 9, 5], np.int32)
    mask = (np.arange(k)[None, :] >= lo[:, None]) & (np.arange(k)[None, :] < hi[:, None])
    active_want = want((a * (w * mask)[:, None, :]) @ b)
    rep = NamedSharding(mesh, P())
    lo_d, hi_d = jax.device_put(lo, rep), jax.device_put(hi, rep)
    kw = dict(C=jax.device_put(c, face)) if beta else {}
    got = jax.jit(lambda x, y, l, h: plan.active_range(x, y, l, h, weights=w, **kw))(
        aa, bb, lo_d, hi_d)
    np.testing.assert_allclose(np.asarray(got), active_want, atol=1e-12, rtol=0)

    prepared = plan.prepare_active_range(3, 9)
    band = np.zeros(k, bool)
    band[3:9] = True
    got = prepared(aa, bb, weights=w, **kw)
    np.testing.assert_allclose(np.asarray(got), want((a * (w * band)[:, None, :]) @ b),
                               atol=1e-12, rtol=0)

    def body(x, y, z):
        return plan.local_call(x, y, C=z) if beta else plan.local_call(x, y)
    manual = jax.jit(shard_map(body, mesh=mesh, in_specs=(P(None, "x", "y"),) * 3,
                               out_specs=P(None, "x", "y"), check_vma=False))
    got = manual(aa, bb, jax.device_put(c, face))
    np.testing.assert_allclose(np.asarray(got), want(a @ b), atol=1e-12, rtol=0)


def test_cpu_face_plan_keeps_the_face_contract():
    """The CPU face plan goes through gemm_plan's one face-contract check:
    k must tile both mesh axes (it is A's column face and B's row face)."""
    mesh = _mesh()
    with pytest.raises(ValueError, match="k=5 does not tile the 2x2 mesh"):
        D.gemm_plan(mesh, m=4, k=5, n=4, nq=1, dtype="complex128")


def test_cpu_explicit_scalapack_refuses_through_the_probe():
    """No host batched-GEMM handler exists; the probe refuses by name
    (``ProbeResult`` is a tuple, so this needs ``probe.ok``).  A 1x1 mesh
    passes the one-process-per-cell guard, so the probe is what refuses."""
    mesh = _mesh(1, 1)
    with pytest.raises(RuntimeError, match="'scalapack' is unavailable"):
        D.gemm_plan(mesh, m=4, k=4, n=4, nq=2, dtype="complex128",
                   backend="scalapack")


def test_explicit_cublasmp_refuses_off_cuda():
    # A 1x1 mesh keeps px*py == jax.process_count() == 1, so the platform
    # guard is the one that fires (a 2x2 emulated single-process mesh
    # trips the EARLIER one-process-per-cell guard instead, which is its
    # own refusal and not what this test targets).
    mesh = _mesh(px=1, py=1)
    with pytest.raises(RuntimeError, match="CUDA-only"):
        D.gemm_plan(mesh, m=4, k=4, n=4, nq=2, dtype="complex128",
                   backend="cublasmp")


def test_mesh_axes_must_be_exact_and_y_minor():
    import jax
    from jax.sharding import Mesh
    require_devices(4, "cpu")
    mesh = Mesh(np.asarray(jax.devices("cpu")[:4]).reshape(2, 2),
               ("y", "x"))
    with pytest.raises(ValueError, match="exactly 2-D with y-minor axes"):
        D.gemm_plan(mesh, m=4, k=4, n=4, nq=2, dtype="complex128",
                   backend="cublasmp")


def test_real_dtype_rejects_complex_alpha_beta():
    mesh = _mesh()
    with pytest.raises(ValueError, match="must be real"):
        D.gemm_plan(mesh, m=4, k=4, n=4, nq=2, dtype="float64",
                   backend="cublasmp", alpha=1 + 1j)


def test_bad_extent_refuses_before_backend_resolution():
    """A malformed shape is refused before any provider/mesh work, on any
    platform -- so this is legitimately testable without CUDA."""
    mesh = _mesh()
    with pytest.raises(ValueError, match="must be positive"):
        D.gemm_plan(mesh, m=0, k=4, n=4, nq=2, dtype="complex128")
    with pytest.raises(ValueError, match="must be positive"):
        D.gemm_plan(mesh, m=4, k=4, n=4, nq=-1, dtype="complex128")


def test_unsupported_dtype_refuses():
    mesh = _mesh()
    with pytest.raises(TypeError):
        D.gemm_plan(mesh, m=4, k=4, n=4, nq=2, dtype="float32")


# ---------------------------------------------------------------------------
# ``GemmPlan.__call__``'s own Python-level guards -- shape/dtype/sharding
# and the out=/beta interaction -- are pure checks on static metadata, run
# BEFORE either compiled kernel is invoked (module docstring, "safe on a
# tracer").  Constructing a ``GemmPlan`` directly (bypassing ``gemm_plan()``,
# which needs real cuBLASMp) with stand-in ``_fn_with_c``/``_fn_no_c``
# lambdas exercises exactly that guard logic without any backend at all --
# a real warmed plan's numerics are the four-rank CUDA gate's job
# (``check_gemm_plan_cublasmp``), not this file's.
# ---------------------------------------------------------------------------

def _fake_plan(mesh, *, beta):
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from distrib_la.matmul_plan import GemmPlan

    sharding = NamedSharding(mesh, P(None, "x", "y"))
    return GemmPlan(
        mesh=mesh, backend="cublasmp", m=2, k=2, n=2, nq=1,
        dtype=jnp.dtype("complex128"), alpha=1 + 0j, beta=complex(beta),
        in_sharding_a=sharding, in_sharding_b=sharding, out_sharding=sharding,
        ctx_handle=0,   # stand-in: no real FFI call reachable via __call__ here
        _fn_with_c=lambda A, B, C: C,   # stand-in: no real GEMM needed here
        _fn_no_c=(lambda A, B: A) if beta == 0 else None,
    )


def test_gemm_plans_expose_operand_shardings():
    """Both production layout plans carry the Green operand constraints."""
    mesh = _mesh()
    plans = (_fake_plan(mesh, beta=0),
             D.local_gemm_plan(mesh, m=2, k=2, n=2, nq=1,
                               dtype="complex128"))
    for plan in plans:
        assert plan.in_sharding_a is not None and plan.in_sharding_b is not None


def _sharded(mesh, value, shape):
    """(nq,m,k)-shaped P(None,'x','y') operand -- ``_check_operand``
    refuses a plain ``jnp.zeros`` (single-device sharded), so every fake
    operand in this section needs a real placement, exactly like
    ``distrib_la.matmul._zeros``."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    return jax.jit(lambda: jnp.full(shape, value, dtype=jnp.complex128),
                   out_shardings=sharding)()


def test_out_refuses_on_a_beta_nonzero_plan():
    """A caller who read only the general ``out=`` docstring framing
    ("content is ignored") and reaches for it on an accumulate (beta!=0)
    plan must get a named refusal, not a silent wrong answer built from
    whatever the buffer happened to hold."""
    mesh = _mesh()
    plan = _fake_plan(mesh, beta=1.0)
    A = _sharded(mesh, 0, (1, 2, 2))
    B = _sharded(mesh, 0, (1, 2, 2))
    scratch = _sharded(mesh, 0, (1, 2, 2))
    with pytest.raises(ValueError, match="content-ignored donation"):
        plan(A, B, out=scratch)


def test_out_is_accepted_on_a_beta_zero_plan():
    """The same call shape is legal -- and reaches the compiled kernel --
    on a beta==0 plan, where out='s content-ignored contract actually
    holds."""
    mesh = _mesh()
    plan = _fake_plan(mesh, beta=0.0)
    A = _sharded(mesh, 0, (1, 2, 2))
    B = _sharded(mesh, 0, (1, 2, 2))
    scratch = _sharded(mesh, 0, (1, 2, 2))
    out = plan(A, B, out=scratch)
    assert out is scratch   # this plan's stand-in _fn_with_c returns C
                             # unchanged, so out= must reach it as C


def test_c_still_works_unaffected_on_a_beta_nonzero_plan():
    """The new out=/beta guard must not touch the existing C= path."""
    mesh = _mesh()
    plan = _fake_plan(mesh, beta=1.0)
    A = _sharded(mesh, 0, (1, 2, 2))
    B = _sharded(mesh, 0, (1, 2, 2))
    C = _sharded(mesh, 1, (1, 2, 2))
    out = plan(A, B, C=C)
    assert out is C


@pytest.mark.parametrize("reduction_axis", [None, "x", "y"])
def test_local_gemm_plan_contracts_random_complex_operands(reduction_axis):
    """Local band GEMMs and centroid reductions reproduce the dense contraction."""
    import jax
    mesh = _mesh()
    rng = np.random.default_rng(20260906)
    a = rng.normal(size=(2, 8, 6)) + 1j * rng.normal(size=(2, 8, 6))
    b = rng.normal(size=(2, 6, 10)) + 1j * rng.normal(size=(2, 6, 10))
    plan = D.local_gemm_plan(mesh, m=8, k=6, n=10, nq=2,
                            dtype="complex128", reduction_axis=reduction_axis)
    assert isinstance(plan, D.GemmPlan)
    assert plan.in_sharding_a is not None and plan.in_sharding_b is not None
    result = plan(jax.device_put(a, plan.in_sharding_a),
                  jax.device_put(b, plan.in_sharding_b))
    np.testing.assert_allclose(np.asarray(result), a @ b, atol=1e-12, rtol=0)
    assert result.sharding == plan.out_sharding


def test_local_beta_zero_keeps_out_live_without_donation_warning():
    """The ignored beta-zero addend remains live and emits no donation warning."""
    import warnings
    import jax
    mesh = _mesh()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = D.local_gemm_plan(mesh, m=4, k=4, n=4, nq=1,
                                 dtype="complex128", beta=0)
        a = jax.device_put(np.ones((1, 4, 4), np.complex128), plan.in_sharding_a)
        b = jax.device_put(np.ones((1, 4, 4), np.complex128), plan.in_sharding_b)
        out = jax.device_put(np.full((1, 4, 4), 17+0j), plan.out_sharding)
        result = plan(a, b, out=out)
        result.block_until_ready()
    assert not out.is_deleted()
    np.testing.assert_array_equal(np.asarray(out), 17)
    np.testing.assert_array_equal(np.asarray(result), 4)
    assert not [w for w in caught if "donat" in str(w.message).lower()]
