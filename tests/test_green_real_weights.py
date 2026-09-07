"""Real-weight tracing and complex-weight transpose partners share the Green equation."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from gw.greens_function_kernel import build_G, build_G_tau


def _operands():
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    calls = []
    def gemm(a, b):
        calls.append(1)
        return a @ b
    gemm.in_sharding_a = gemm.in_sharding_b = NamedSharding(mesh, P())
    plan = SimpleNamespace(sym_idx=np.array([1]), n_sym_spatial=1, mesh_xy=mesh,
        unfold_operator=lambda g, *, operator_transpose, right_plan: operator_transpose)
    left = jnp.full((1, 1, 1, 1), 2+3j)
    right = jnp.full((1, 1, 1, 1), 4+1j)
    return left, right, gemm, plan, calls


def test_static_real_weights_trace_only_one_gemm():
    """A real-weight contract excludes the transpose-partner GEMM at trace time."""
    left, right, gemm, plan, calls = _operands()
    jax.make_jaxpr(lambda w: build_G(left, right, phases=w, gemm=gemm,
        k_unfold_plan=plan, real_weights=True))(jnp.ones((1, 1), dtype=jnp.complex128))
    assert len(calls) == 1


def test_complex_band_weights_use_the_correct_transpose_partner():
    """A real time node does not make a complex band weight real."""
    left, right, gemm, plan, _ = _operands()
    weight = jnp.array([[1+2j]])
    got = build_G_tau(left, right, jnp.zeros((1, 1)), 0.0,
        band_weight=weight, gemm=gemm, k_unfold_plan=plan)
    np.testing.assert_allclose(np.asarray(got), (2-3j)*(1+2j)*(4+1j))


def test_occupation_projector_refuses_complex_weights():
    """Occupation construction enforces the real dtype assumed by static Sigma."""
    from gw.cohsex_sigma import build_Gij
    with pytest.raises(TypeError, match="real occupation weights"):
        build_Gij(SimpleNamespace(), None,
                  SimpleNamespace(f_kn=np.ones((1, 1), dtype=np.complex128)))
