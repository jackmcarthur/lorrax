"""Planned N,N GEMM (``distrib_la.gemm_plan``/``GemmPlan``) — logic tier.

Real cuBLASMp execution needs a CUDA mesh with one process per device, so
none of it is reachable here: every cell in this file either exercises a
pure-Python helper directly, or drives ``gemm_plan()`` on an emulated CPU
mesh far enough to observe its EAGER refusal ladder (backend='off', a
resolved non-cuBLASMp provider, mesh topology, shape divisibility, dtype)
BEFORE it would reach ``get_or_init_context``/the FFI call.  Numerics
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


def test_cpu_mesh_resolves_scalapack_and_gemm_plan_refuses_by_name():
    """cuBLASMp is CUDA-only, so 'auto' on a CPU mesh resolves to
    scalapack -- for which this module has no warmed kernel (the GEMM FFI
    symbol itself does not exist in the tree either, per
    KNOWN_LORRAX_ISSUES.md).  Either way the caller gets a NAMED refusal,
    never a silent construction of something unusable."""
    mesh = _mesh()
    with pytest.raises((RuntimeError, NotImplementedError)):
        D.gemm_plan(mesh, m=4, k=4, n=4, nq=2, dtype="complex128",
                   backend="auto")


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
