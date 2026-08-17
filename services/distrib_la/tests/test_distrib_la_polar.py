"""Hermitian-dilation polar/SVD contract on a four-device CPU mesh."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import distrib_la as D


RTOL = 2e-11


def _mesh(px=2, py=2):
    import jax
    from jax.sharding import Mesh
    from lxkit.testing import require_devices

    require_devices(px * py, "cpu")
    devices = np.asarray(jax.devices("cpu")[:px * py]).reshape(px, py)
    return Mesh(devices, ("x", "y"))


def _shardings(mesh):
    from jax.sharding import NamedSharding, PartitionSpec as P
    return NamedSharding(mesh, P("x", "y")), NamedSharding(mesh, P())


def _put(A, mesh, *, replicated=False):
    import jax
    tile, rep = _shardings(mesh)
    return jax.device_put(np.asarray(A), rep if replicated else tile)


def _random_complex(rng, n):
    return (rng.standard_normal((n, n))
            + 1j * rng.standard_normal((n, n))).astype("complex128")


def _unitary(rng, n):
    Q, R = np.linalg.qr(_random_complex(rng, n))
    phase = np.diag(R)
    phase = np.where(np.abs(phase) == 0, 1, phase / np.abs(phase))
    return Q * np.conj(phase)[None, :]


def _relative(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want))) / max(
        float(np.max(np.abs(want))), 1e-300)


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_polar_factor_matches_numpy_svd(dtype):
    """Synthetic full-rank parity, including the returned SVD ordering."""
    mesh = _mesh()
    rng = np.random.default_rng(701)
    A = rng.standard_normal((8, 8))
    if np.dtype(dtype).kind == "c":
        A = A + 1j * rng.standard_normal((8, 8))
    A = A.astype(dtype)

    L, s = D.polar_factor(_put(A, mesh), mesh, backend="off")
    U, s_ref, Vh = np.linalg.svd(A, full_matrices=False)

    assert _relative(s, s_ref) < RTOL
    assert _relative(L, U @ Vh) < RTOL
    assert _relative(np.asarray(L).conj().T @ np.asarray(L), np.eye(8)) < RTOL
    tile, rep = _shardings(mesh)
    assert L.sharding.spec == tile.spec and L.sharding.mesh == mesh
    assert s.sharding.spec == rep.spec and s.sharding.mesh == mesh


def test_unitary_and_degenerate_singular_spaces_are_gauge_invariant():
    """An eight-fold degenerate SVD must still return the matrix itself."""
    mesh = _mesh()
    A = _unitary(np.random.default_rng(702), 8)
    L, s = D.polar_factor(_put(A, mesh), mesh, backend="off")
    assert _relative(L, A) < RTOL
    assert _relative(s, np.ones(8)) < RTOL


def test_repeated_nontrivial_singular_values_match_numpy():
    """Degenerate singular-vector gauges cannot affect U @ Vh."""
    mesh = _mesh()
    rng = np.random.default_rng(703)
    U = _unitary(rng, 8)
    V = _unitary(rng, 8)
    s_ref = np.asarray([5, 5, 2, 2, 0.75, 0.75, 0.2, 0.2])
    A = (U * s_ref[None, :]) @ V.conj().T
    L, s = D.polar_factor(_put(A, mesh), mesh, backend="off")
    assert _relative(s, s_ref) < RTOL
    assert _relative(L, U @ V.conj().T) < RTOL


def test_ill_conditioned_singular_values_are_not_squared():
    """The dilation resolves a value whose square is below float64 epsilon."""
    mesh = _mesh()
    rng = np.random.default_rng(704)
    U = _unitary(rng, 8)
    V = _unitary(rng, 8)
    s_ref = np.asarray([1, 0.1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-9, 1e-10])
    A = (U * s_ref[None, :]) @ V.conj().T
    _, s = D.polar_factor(_put(A, mesh), mesh, backend="off", rcond=0)
    assert abs(np.asarray(s)[-1] - s_ref[-1]) / s_ref[-1] < 2e-5


def test_rank_deficient_input_returns_the_canonical_partial_isometry():
    """Null directions are omitted, not paired by an arbitrary zero gauge."""
    mesh = _mesh()
    rng = np.random.default_rng(705)
    U = _unitary(rng, 8)
    V = _unitary(rng, 8)
    spectrum = np.asarray([4, 3, 2, 1, 0, 0, 0, 0], dtype="float64")
    A = (U * spectrum[None, :]) @ V.conj().T

    L, s = D.polar_factor(
        _put(A, mesh), mesh, backend="off", rcond=1e-12)
    Un, s_ref, Vhn = np.linalg.svd(A, full_matrices=False)
    rank = int(np.count_nonzero(s_ref > 1e-12 * s_ref[0]))
    partial_ref = Un[:, :rank] @ Vhn[:rank]

    assert rank == 4
    assert _relative(s, s_ref) < RTOL
    assert _relative(L, partial_ref) < RTOL
    assert _relative(np.asarray(L) @ np.asarray(L).conj().T,
                     partial_ref @ partial_ref.conj().T) < RTOL


def test_zero_matrix_has_zero_spectrum_and_zero_partial_isometry():
    mesh = _mesh()
    A = np.zeros((8, 8), dtype="complex128")
    L, s = D.polar_factor(
        _put(A, mesh), mesh, backend="off", rcond=1e-12)
    assert np.array_equal(np.asarray(s), np.zeros(8))
    assert np.array_equal(np.asarray(L), np.zeros((8, 8)))


def test_nondivisible_logical_extent_survives_zero_padding():
    """Seven logical bands are padded to eight for a 2x2 physical tile."""
    mesh = _mesh()
    n_log, n_pad = 7, 8
    assert n_log % 2 and n_pad % 2 == 0
    with pytest.raises(ValueError, match="Zero-pad A to at least n=8"):
        D.plan_polar_factor(mesh, n=n_log, backend="off")

    A = _random_complex(np.random.default_rng(706), n_log)
    A_pad = np.zeros((n_pad, n_pad), dtype=A.dtype)
    A_pad[:n_log, :n_log] = A
    L_pad, s_pad = D.polar_factor(
        _put(A_pad, mesh), mesh, backend="off", rcond=1e-12)
    U, s_ref, Vh = np.linalg.svd(A, full_matrices=False)

    assert _relative(np.asarray(L_pad)[:n_log, :n_log], U @ Vh) < RTOL
    assert _relative(np.asarray(s_pad)[:n_log], s_ref) < RTOL
    assert abs(np.asarray(s_pad)[-1]) < 1e-12
    assert np.max(np.abs(np.asarray(L_pad)[n_log:, :])) < 1e-12
    assert np.max(np.abs(np.asarray(L_pad)[:, n_log:])) < 1e-12


def test_planned_operation_is_trace_safe_and_reuses_its_answer():
    import jax

    mesh = _mesh()
    tile, rep = _shardings(mesh)
    A = _put(_random_complex(np.random.default_rng(707), 8), mesh)
    operation = D.plan_polar_factor(mesh, n=8, backend="off")
    eager = operation(A)
    traced = jax.jit(operation, in_shardings=tile,
                     out_shardings=(tile, rep))(A)
    assert np.array_equal(np.asarray(eager[0]), np.asarray(traced[0]))
    assert np.array_equal(np.asarray(eager[1]), np.asarray(traced[1]))
    assert "Hermitian dilation n=16" in operation.describe()


def test_the_convenience_call_refuses_to_plan_inside_a_trace():
    import jax

    mesh = _mesh()
    tile, _ = _shardings(mesh)
    A = _put(np.eye(8, dtype="complex128"), mesh)
    with pytest.raises(RuntimeError, match="plan_polar_factor"):
        jax.jit(lambda x: D.polar_factor(x, mesh, backend="off"),
                in_shardings=tile)(A)


@pytest.mark.parametrize("shape,match", [
    ((2, 8, 8), "rank 2"),
    ((8, 6), "square"),
])
def test_rank_and_shape_refusals_happen_before_planning(shape, match):
    mesh = _mesh()
    A = _put(np.zeros(shape, dtype="complex128"), mesh, replicated=True)
    with pytest.raises(ValueError, match=match):
        D.polar_factor(A, mesh, backend="off")


@pytest.mark.parametrize("dtype", ["int64", "float32", "complex64"])
def test_unsupported_dtypes_refuse_by_name(dtype):
    mesh = _mesh()
    A = _put(np.eye(8, dtype=dtype), mesh)
    with pytest.raises(TypeError, match="float64.*complex128"):
        D.polar_factor(A, mesh, backend="off")


@pytest.mark.parametrize("bad", [-1, np.inf, np.nan, True, "not-a-number"])
def test_invalid_rcond_refuses_before_work(bad):
    mesh = _mesh()
    with pytest.raises(ValueError, match="finite non-negative real"):
        D.plan_polar_factor(mesh, n=8, backend="off", rcond=bad)


def test_wrong_layout_refuses_an_implicit_full_matrix_reshard():
    mesh = _mesh()
    A = _put(np.eye(8, dtype="complex128"), mesh, replicated=True)
    operation = D.plan_polar_factor(mesh, n=8, backend="off")
    with pytest.raises(ValueError, match="implicit n.2 reshard"):
        operation(A)


def test_public_door_and_distributed_default_are_pinned():
    assert D.polar_factor.__module__ == "distrib_la.polar"
    assert D.plan_polar_factor.__module__ == "distrib_la.polar"
    assert inspect.signature(D.polar_factor).parameters["backend"].default \
        == "distributed"
    assert "polar_factor" in D.__all__


def test_polar_planning_delegates_once_to_eigh_at_twice_the_extent(
        monkeypatch):
    """The composite has no second resolver or hidden backend selection."""
    import sys
    module = sys.modules["distrib_la.polar"]
    mesh = _mesh()
    calls = []

    class _EighPlan:
        backend = "scalapack"
        is_native = False

    def fake_plan(op, got_mesh, *, backend, n):
        calls.append((op, got_mesh, backend, n))
        return _EighPlan()

    monkeypatch.setattr(module, "plan", fake_plan)
    operation = D.plan_polar_factor(mesh, n=8)
    assert calls == [("eigh", mesh, "distributed", 16)]
    assert operation.backend == "scalapack"


def test_a_gram_eigh_or_caller_side_svd_is_not_the_implementation():
    """Structural red control for the two numerically forbidden routes."""
    import ast
    import sys
    module = sys.modules["distrib_la.polar"]
    tree = ast.parse(inspect.getsource(module))

    def dotted(node):
        names = []
        while isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            names.append(node.id)
        return ".".join(reversed(names))

    calls = {dotted(node.func) for node in ast.walk(tree)
             if isinstance(node, ast.Call)}
    assert "jnp.linalg.svd" not in calls
    assert "jnp.linalg.eigvalsh" not in calls
    assert "eigh" in calls and "jnp.pad" in calls

    gram_products = [node for node in ast.walk(tree)
                     if isinstance(node, ast.BinOp)
                     and isinstance(node.op, ast.MatMult)
                     and dotted(node.left) in ("A.H", "A.T")]
    assert not gram_products
