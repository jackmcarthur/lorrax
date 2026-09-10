"""Public distributed matmul: provider dispatch and batch-reshard route."""
from __future__ import annotations

import numpy as np
import pytest
from lxkit.testing import require_devices

import distrib_la as D


def _mesh():
    import jax
    from jax.sharding import Mesh
    require_devices(4, "cpu")
    return Mesh(np.asarray(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))


def _mesh_shape(px, py):
    import jax
    from jax.sharding import Mesh
    require_devices(px * py, "cpu")
    return Mesh(
        np.asarray(jax.devices("cpu")[:px * py]).reshape(px, py), ("x", "y"))


def _put(a, mesh):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    spec = P("x", "y") if np.ndim(a) == 2 else P(None, "x", "y")
    return jax.device_put(np.asarray(a), NamedSharding(mesh, spec))


def _op(a, op):
    if op == "N":
        return a
    a = np.swapaxes(a, -1, -2)
    return np.conj(a) if op == "C" else a


@pytest.mark.parametrize("transa,transb", [("N", "N"), ("C", "N"),
                                             ("N", "T")])
def test_batch_reshard_matmul_ragged_and_transposed(transa, transb):
    from jax.sharding import PartitionSpec as P
    mesh = _mesh()
    rng = np.random.default_rng(9100 + ord(transa) + ord(transb))
    nb, m, k, n = 5, 8, 6, 10
    ashape = (nb, m, k) if transa == "N" else (nb, k, m)
    bshape = (nb, k, n) if transb == "N" else (nb, n, k)
    A = rng.standard_normal(ashape) + 1j*rng.standard_normal(ashape)
    B = rng.standard_normal(bshape) + 1j*rng.standard_normal(bshape)
    C = rng.standard_normal((nb,m,n)) + 1j*rng.standard_normal((nb,m,n))
    alpha, beta = 1.3-0.2j, -0.4+0.1j
    want = alpha*(_op(A,transa) @ _op(B,transb)) + beta*C
    got = D.matmul(_put(A,mesh), _put(B,mesh), _put(C,mesh), mesh=mesh,
                   alpha=alpha, beta=beta, transa=transa, transb=transb,
                   backend="off", batched_route="batch_reshard")
    assert got.sharding.spec == P(None,"x","y")
    np.testing.assert_allclose(np.asarray(got), want, rtol=2e-13, atol=2e-13)


def test_rank2_surface_and_zero_c_default():
    from jax.sharding import PartitionSpec as P
    mesh = _mesh()
    rng = np.random.default_rng(9201)
    A, B = rng.standard_normal((8,6)), rng.standard_normal((6,10))
    got = D.matmul(_put(A,mesh), _put(B,mesh), mesh=mesh, backend="off",
                   batched_route="batch_reshard")
    assert got.shape == (8,10) and got.sharding.spec == P("x","y")
    np.testing.assert_allclose(np.asarray(got), A@B, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("px,py", [(4, 1), (1, 4)])
def test_batch_reshard_rectangular_axis_order(px, py):
    mesh = _mesh_shape(px, py)
    rng = np.random.default_rng(9250 + 10*px + py)
    A = rng.standard_normal((5, 8, 8))
    B = rng.standard_normal((5, 8, 8))
    got = D.matmul(_put(A, mesh), _put(B, mesh), mesh=mesh, backend="off",
                   batched_route="batch_reshard")
    np.testing.assert_allclose(np.asarray(got), A @ B,
                               rtol=2e-13, atol=2e-13)


def test_provider_vocabulary_and_off_refusal():
    mesh = _mesh()
    assert "cusolvermp" in D.MATMUL_BACKEND_CHOICES
    assert "scalapack" in D.MATMUL_BACKEND_CHOICES
    assert "slate" in D.MATMUL_BACKEND_CHOICES
    assert D.resolve_matmul_backend(
        "off", mesh, batched_route="batch_reshard") == "off"
    assert D.resolve_matmul_backend("off", mesh) == "off"
    with pytest.raises(RuntimeError, match="no distributed provider"):
        D.resolve_matmul_backend("off", mesh, batched_route="auto")


def test_mesh_axes_must_be_exact_and_y_minor():
    import jax
    from jax.sharding import Mesh
    require_devices(4, "cpu")
    mesh = Mesh(np.asarray(jax.devices("cpu")[:4]).reshape(2, 2),
                ("y", "x"))
    A = _put(np.zeros((4, 4)), mesh)
    with pytest.raises(ValueError, match="exactly 2-D with y-minor axes"):
        D.matmul(A, A, mesh=mesh, backend="off",
                 batched_route="batch_reshard")


def test_provider_refuses_non_y_minor_process_ownership(monkeypatch):
    import importlib
    public_matmul = importlib.import_module("distrib_la.matmul")

    class Device:
        def __init__(self, process_index):
            self.process_index = process_index

    class FakeMesh:
        axis_names = ("x", "y")
        shape = {"x": 2, "y": 2}
        devices = np.asarray([
            [Device(0), Device(2)],
            [Device(1), Device(3)],
        ])

    monkeypatch.setattr(public_matmul.jax, "process_count", lambda: 4)
    with pytest.raises(RuntimeError, match="y-minor process grid"):
        D.resolve_matmul_backend("scalapack", FakeMesh())


def test_slate_provider_refuses_rectangular_before_ffi(monkeypatch):
    import importlib
    import inspect
    public_matmul = importlib.import_module("distrib_la.matmul")
    slate = importlib.import_module("distrib_la._slate")

    class Device:
        process_index = 0

    class FakeMesh:
        axis_names = ("x", "y")
        shape = {"x": 1, "y": 2}
        devices = np.asarray([[Device(), Device()]])

    with pytest.raises(ValueError, match="needs a square mesh"):
        D.resolve_matmul_backend("slate", FakeMesh())
    assert "validate_mesh(mesh, require_square=True)" in inspect.getsource(
        slate.batched_distributed_matmul)


def test_transposed_output_face_refuses_before_zero_c_placement():
    mesh = _mesh_shape(4, 1)
    # Raw A(8,6) and B(8,10) both tile 4x1, but op(A)=A.T makes the
    # output rows 6, which cannot be P('x','y') on this mesh.
    A = _put(np.zeros((3, 8, 6)), mesh)
    B = _put(np.zeros((3, 8, 10)), mesh)
    with pytest.raises(ValueError, match="output face must tile"):
        D.matmul(A, B, mesh=mesh, transa="T", backend="off",
                 batched_route="batch_reshard")


def test_batch_reshard_source_is_two_forward_then_two_inverse_exchanges():
    import importlib
    import inspect
    import re
    public_matmul = importlib.import_module("distrib_la.matmul")
    source = inspect.getsource(public_matmul)
    assert "_face_to_batch" in source and "_batch_to_face" in source
    assert "jnp.matmul" in source
    assert "all_gather" not in source
    assert re.search(r"\bring\b", source.lower()) is None


def test_provider_wrappers_call_library_gemm_not_jax_panels():
    import inspect
    from distrib_la import _scalapack, _slate
    for wrapper, target in (
            (_scalapack.batched_distributed_matmul,
             "lorrax_scalapack_batched_gemm"),
            (_slate.batched_distributed_matmul,
             "lorrax_slate_batched_gemm")):
        source = inspect.getsource(wrapper)
        assert target in source or "_GEMM_TARGET" in source
        assert "ffi_call" in source
        assert "all_gather" not in source
