"""Pivoted face LU: exact general solves, padded support, and bounded programs."""
import importlib
import re

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import distrib_la
from distrib_la import _native2d

jax.config.update("jax_enable_x64", True)


def _mesh(side):
    if len(jax.devices()) < side * side:
        pytest.skip(f"requires {side * side} virtual CPU devices")
    return Mesh(np.asarray(jax.devices()[:side * side]).reshape(side, side), ("x", "y"))


def _fixture(n, nrhs, dtype, *, logical=None):
    """General indefinite A with dominant entries on cross-owner rows."""
    logical = n if logical is None else logical
    rng = np.random.default_rng(218)
    a = .025 * rng.standard_normal((2, logical, logical))
    exact = rng.standard_normal((2, logical, nrhs))
    if np.issubdtype(dtype, np.complexfloating):
        a = a + .025j * rng.standard_normal(a.shape)
        exact = exact + 1j * rng.standard_normal(exact.shape)
    diag = np.arange(logical)
    a[:, diag, diag] += np.where(diag % 2, -2., 3.)
    # Partial pivoting must move rows between process rows, not just pivot
    # the diagonal owner. Neither Hermiticity nor positive definiteness holds.
    a = a[:, np.roll(np.arange(logical), logical // 2 + 1), :].astype(dtype)
    exact = exact.astype(dtype)
    b = a @ exact
    full_a = np.broadcast_to(np.eye(n, dtype=dtype), (2, n, n)).copy()
    full_a[:, :logical, :logical] = a
    full_x = np.zeros((2, n, nrhs), dtype=dtype)
    full_x[:, :logical] = exact
    full_b = np.zeros_like(full_x)
    full_b[:, :logical] = b
    return full_a, full_b, full_x


def _put(mesh, a):
    return jax.device_put(a, NamedSharding(mesh, P(None, "x", "y")))


@pytest.mark.parametrize("side", [2, 3, 4])
def test_partial_pivot_general_and_padding(side):
    mesh = _mesh(side)
    n, nrhs = side * 7, side * 3
    for dtype in (np.float32, np.complex64, np.float64, np.complex128):
        a, b, exact = _fixture(n, nrhs, dtype, logical=n - 1)
        b[:, :, -1], exact[:, :, -1] = 0, 0     # independent RHS carrier padding
        # Width five crosses owners and leaves a short final panel.
        x = _native2d.batched_distributed_solve_lu(
            _put(mesh, a), _put(mesh, b), mesh=mesh, block_size=5)
        assert x.sharding.is_equivalent_to(NamedSharding(mesh, P(None, "x", "y")), 3)
        result = np.asarray(x)
        eps = np.finfo(np.empty((), dtype=dtype).real.dtype).eps
        assert np.isfinite(result).all()
        assert np.max(np.abs(result - exact)) < 40 * n * eps
        assert np.array_equal(result[:, n - 1], np.zeros_like(result[:, n - 1]))
        assert np.array_equal(result[:, :, -1], np.zeros_like(result[:, :, -1]))
        error = np.linalg.norm(a @ result - b) / (
            np.linalg.norm(a) * np.linalg.norm(result) + np.linalg.norm(b))
        assert np.isfinite(error) and error < 20 * eps


def test_public_capacity_route_and_singular_refusal(monkeypatch):
    mesh = _mesh(2)
    resolve = importlib.import_module("distrib_la.resolve")
    monkeypatch.setattr(resolve, "mesh_platform", lambda _: "ROCM")
    # A room too small for one whole matrix resolves to the existing 2-D
    # backend; no room also has a complete distributed implementation.
    for room in (None, 1):
        plan = distrib_la.plan("solve_lu", mesh, backend="distributed", n=14,
                               budget_bytes=room)
        assert plan.backend == "native2d"
        assert plan.route_for((2, 14, 14), np.complex128) == "backend_batched"
        a, b, exact = _fixture(14, 10, np.complex128, logical=13)
        result = plan.batched(_put(mesh, a), _put(mesh, b), block_size=5)
        np.testing.assert_allclose(np.asarray(result), exact, atol=1e-12, rtol=0)
    a[:, 5] = a[:, 4]
    with pytest.raises(ValueError, match="distrib_la_result_check"):
        plan.batched(_put(mesh, a), _put(mesh, b), block_size=5)
    # The old efficient local selection remains when its complete program fits.
    from distrib_la._one_device import one_device_bytes
    plan = distrib_la.plan("solve_lu", mesh, backend="distributed", n=14,
                           budget_bytes=one_device_bytes("solve_lu", 14))
    assert plan.backend == "one_device"


@pytest.mark.parametrize("side", [2, 3, 4, 12])
def test_actual_partitioned_program_has_only_faces_and_bounded_panels(side):
    mesh = _mesh(side)
    # The large carrier ensures that a bounded panel cannot be mistaken for
    # a whole matrix. Abstract compilation allocates no numerical field.
    n = side * 171
    nrhs = side * 97
    a = jax.ShapeDtypeStruct((1, n, n), np.complex128,
                            sharding=NamedSharding(mesh, P(None, "x", "y")))
    b = jax.ShapeDtypeStruct((1, n, nrhs), np.complex128,
                            sharding=NamedSharding(mesh, P(None, "x", "y")))
    executable = _native2d._lu_kernel(mesh, n, nrhs, 128).lower(a, b).compile()
    assert "all-gather" not in executable.as_text(), "LU exchanges preselected rows/panels only"
    plan = distrib_la.plan("solve_lu", mesh, backend="native2d", n=n)
    checked = distrib_la.checked_program(lambda a, b: plan.batched(a, b), mesh,
                                        NamedSharding(mesh, P(None, "x", "y")),
                                        in_shardings=(a.sharding, b.sharding))
    # Include the production plan's independent pre-donation result check:
    # a bounded backend alone does not certify its enclosing caller.
    checked_executable = checked.lower(a, b).compile()
    largest = max(171 * 171, 171 * 128, 128 * 171, 128 * 128, 171 * 97)
    for program in (executable, checked_executable):
        for dims in re.findall(r"c128\[([0-9,]+)\]", program.as_text()):
            count = int(np.prod([int(v) for v in dims.split(",")]))
            assert count <= largest, (dims, side, largest)
        assert program.memory_analysis().temp_size_in_bytes < 64 * largest * 16


def test_pivot_ties_are_ordered_and_shape_refusals():
    mesh = _mesh(2)
    # Equal-magnitude alternatives on distinct row owners; lowest global
    # row is selected. The independent answer does not depend on tie choice.
    a = np.array([[0, 1, 0, 0], [2, 0, 1, 0], [-2, 0, 0, 1], [0, 1, 1, 2]], float)
    b = np.arange(8, dtype=float).reshape(1, 4, 2)
    result = _native2d.batched_distributed_solve_lu(
        _put(mesh, a[None]), _put(mesh, b), mesh=mesh, block_size=3)
    np.testing.assert_allclose(np.asarray(result), np.linalg.solve(a, b), atol=1e-12)
    with pytest.raises(ValueError, match="mesh-padded"):
        _native2d.batched_distributed_solve_lu(
            jax.numpy.zeros((1, 3, 3)), jax.numpy.zeros((1, 3, 2)), mesh=mesh)
