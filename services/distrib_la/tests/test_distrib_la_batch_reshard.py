"""Route (c): staged batch-axis reshard around device-local dense LA."""
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


def _put(a, mesh):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(
        np.asarray(a), NamedSharding(mesh, P(None, "x", "y")))


def _herm(rng, nb, n):
    z = rng.standard_normal((nb, n, n)) + 1j * rng.standard_normal((nb, n, n))
    return (0.5 * (z + np.conj(np.swapaxes(z, -1, -2)))).astype("complex128")


def _hpd(rng, nb, n):
    A = _herm(rng, nb, n)
    return A + (n + 4) * np.eye(n, dtype=A.dtype)[None]


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-300)


def test_public_route_grammar_and_plan_provenance():
    from jax.sharding import PartitionSpec as P

    mesh = _mesh()
    assert D.BATCHED_ROUTE_CHOICES == ("auto", D.ROUTE_BATCH_RESHARD)

    default = D.plan("eigh", mesh, backend="off")
    assert default.requested_batched_route == D.BATCHED_ROUTE_DEFAULT
    assert default.batched_route == D.ROUTE_BATCH_RESHARD
    assert default.in_sharding.spec == P("x", "y")
    assert default.batch_in_sharding.spec == P(None, "x", "y")

    auto = D.plan("eigh", mesh, backend="off", batched_route="auto")
    assert auto.batched_route == D.ROUTE_BACKEND_BATCHED
    assert auto.in_sharding is None and auto.batch_in_sharding is None

    staged = D.plan(
        "eigh", mesh, backend="off", batched_route=D.ROUTE_BATCH_RESHARD)
    assert staged.is_native
    assert staged.batched_route == D.ROUTE_BATCH_RESHARD
    assert staged.in_sharding.spec == P("x", "y")
    assert staged.batch_in_sharding.spec == P(None, "x", "y")
    line = staged.describe()
    assert "'batch_reshard' -> batch_reshard" in line

    with pytest.raises(ValueError, match="auto\\|batch_reshard"):
        D.plan("eigh", mesh, backend="off", batched_route="reshard-ish")


@pytest.mark.parametrize("nb", [1, 3, 5, 8])
def test_eigh_route_handles_ragged_batches_and_restores_layout(nb):
    from jax.sharding import PartitionSpec as P

    mesh = _mesh()
    A = _herm(np.random.default_rng(8100 + nb), nb, 8)
    p = D.plan(
        "eigh", mesh, backend="off", n=8,
        batched_route=D.ROUTE_BATCH_RESHARD)
    W, Z = p.batched(_put(A, mesh))

    assert W.shape == (nb, 8) and Z.shape == (nb, 8, 8)
    assert W.sharding.is_fully_replicated
    assert Z.sharding.spec == P(None, "x", "y")
    assert _rel(W, np.linalg.eigvalsh(A)) < 1e-12
    resid = _rel(A @ np.asarray(Z), np.asarray(Z) * np.asarray(W)[:, None, :])
    assert resid < 1e-12


def test_cholesky_route_uses_safe_ragged_padding_and_ignores_block_size():
    from jax.sharding import PartitionSpec as P

    mesh = _mesh()
    A = _hpd(np.random.default_rng(8201), 5, 8)
    want = np.linalg.cholesky(A)
    p = D.plan(
        "cholesky", mesh, backend="off", n=8,
        batched_route=D.ROUTE_BATCH_RESHARD)
    # ``block_size`` belongs to the distributed descriptor.  If it leaks to
    # jnp.linalg.cholesky this raises TypeError instead of producing L.
    L = p.batched(_put(A, mesh), block_size=3)
    assert L.shape == A.shape and L.sharding.spec == P(None, "x", "y")
    assert _rel(L, want) < 1e-12
    assert _rel(np.asarray(L) @ np.conj(np.swapaxes(np.asarray(L), -1, -2)), A) < 1e-12


def test_solve_lu_route_skips_ragged_padding_and_restores_layout():
    from jax.sharding import PartitionSpec as P

    mesh = _mesh()
    rng = np.random.default_rng(8301)
    A = _hpd(rng, 5, 8)
    B = (rng.standard_normal((5, 8, 6))
         + 1j * rng.standard_normal((5, 8, 6))).astype("complex128")
    want = np.linalg.solve(A, B)
    p = D.plan(
        "solve_lu", mesh, backend="off", n=8,
        batched_route=D.ROUTE_BATCH_RESHARD)
    X = p.batched(_put(A, mesh), _put(B, mesh))
    assert X.shape == B.shape and X.sharding.spec == P(None, "x", "y")
    assert _rel(X, want) < 1e-12
    assert _rel(A @ np.asarray(X), B) < 1e-12


@pytest.mark.parametrize("op", ["eigh", "cholesky", "solve_lu"])
def test_ragged_route_never_calls_dense_kernel_for_synthetic_rows(
    op, monkeypatch,
):
    """Q=5 on P=4 executes five dense rows, not Qp=8."""
    import jax
    import jax.numpy as jnp
    import distrib_la._batch_reshard as br

    mesh = _mesh()
    rng = np.random.default_rng(8350)
    A = (_herm(rng, 5, 8) if op == "eigh" else _hpd(rng, 5, 8))
    B = (rng.standard_normal((5, 8, 6))
         + 1j * rng.standard_normal((5, 8, 6))).astype("complex128")
    calls = []

    original = getattr(jnp.linalg, "solve" if op == "solve_lu" else op)

    def _counted(*args, **kwargs):
        # This callback is staged inside the scalar lax.cond's real branch;
        # it therefore counts runtime dense executions rather than traces.
        jax.debug.callback(lambda _unused: calls.append(1), jnp.int32(0))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        jnp.linalg, "solve" if op == "solve_lu" else op, _counted)
    br._JIT_CACHE.clear()
    p = D.plan(
        op, mesh, backend="off", n=8,
        batched_route=D.ROUTE_BATCH_RESHARD)
    operands = [_put(A, mesh)]
    if op == "solve_lu":
        operands.append(_put(B, mesh))
    result = p.batched(*operands)
    leaves = jax.tree.leaves(result)
    for leaf in leaves:
        leaf.block_until_ready()

    assert len(calls) == 5, (
        f"{op} executed {len(calls)} dense rows for Q=5; Qp=8 means "
        "synthetic slots reached the expensive branch")


def test_dispatch_public_route_does_not_take_the_native_early_return(monkeypatch):
    """The legacy door passes the public route into Plan construction."""
    import jax.numpy as jnp

    mesh = _mesh()
    A = _herm(np.random.default_rng(8401), 3, 8)
    seen = []
    import sys
    dm = sys.modules["distrib_la.dispatch"]
    orig = dm._plan

    def recording(*args, **kwargs):
        seen.append(kwargs.get("batched_route"))
        return orig(*args, **kwargs)

    monkeypatch.setattr(dm, "_plan", recording)
    W, Z = D.dispatch_batched_eigh(
        jnp.asarray(A), mesh, backend="off",
        batched_route=D.ROUTE_BATCH_RESHARD)
    assert seen == [D.ROUTE_BATCH_RESHARD]
    assert _rel(W, np.linalg.eigvalsh(A)) < 1e-12
    assert Z.shape == A.shape


def test_selected_route_remains_trace_safe_inside_a_callers_jit():
    import jax

    mesh = _mesh()
    A = _herm(np.random.default_rng(8451), 3, 8)
    p = D.plan(
        "eigh", mesh, backend="off", n=8,
        batched_route=D.ROUTE_BATCH_RESHARD)
    W, Z = jax.jit(p.batched)(_put(A, mesh))
    assert _rel(W, np.linalg.eigvalsh(A)) < 1e-12
    resid = _rel(
        A @ np.asarray(Z), np.asarray(Z) * np.asarray(W)[:, None, :])
    assert resid < 1e-12


def test_staged_forward_then_inverse_is_bit_exact():
    """Positive movement-only round trip, independent of a dense kernel."""
    import jax
    from jax.sharding import PartitionSpec as P
    from distrib_la._batch_reshard import _batch_to_face, _face_to_batch
    from distrib_la._shard_map import shard_map

    mesh = _mesh()
    a = np.arange(5 * 8 * 8, dtype=np.float64).reshape(5, 8, 8)

    def body(local):
        # The production route pads a ragged B=5 to 8 before the exchanges.
        local = jax.numpy.pad(local, ((0, 3), (0, 0), (0, 0)))
        local = _face_to_batch(local, px=2, py=2)
        local = _batch_to_face(local, px=2, py=2)
        return local[:5]

    roundtrip = jax.jit(shard_map(
        body, mesh=mesh, in_specs=(P(None, "x", "y"),),
        out_specs=P(None, "x", "y"), check_vma=False))
    got = roundtrip(_put(a, mesh))
    assert np.array_equal(np.asarray(got), a)


def test_red_twin_wrong_inverse_order_scrambles_the_batch():
    """The round-trip gate fails if inverse exchanges run x then y."""
    import jax
    from jax.sharding import PartitionSpec as P
    from distrib_la._batch_reshard import _face_to_batch
    from distrib_la._shard_map import shard_map

    mesh = _mesh()
    a = np.arange(8 * 8 * 8, dtype=np.float64).reshape(8, 8, 8)

    def wrong(local):
        local = _face_to_batch(local, px=2, py=2)
        # Shape-correct but order-wrong inverse: the true inverse is y then x.
        local = jax.lax.all_to_all(
            local, "x", split_axis=1, concat_axis=0, tiled=True)
        local = jax.lax.all_to_all(
            local, "y", split_axis=2, concat_axis=0, tiled=True)
        return local

    bad = jax.jit(shard_map(
        wrong, mesh=mesh, in_specs=(P(None, "x", "y"),),
        out_specs=P(None, "x", "y"), check_vma=False))(_put(a, mesh))
    assert not np.array_equal(np.asarray(bad), a), (
        "the wrong inverse schedule returned the input, so the positive "
        "round-trip cannot distinguish exchange order and is vacuous")


def test_extent_refusals_fire_before_the_kernel(monkeypatch):
    import jax.numpy as jnp

    mesh = _mesh()
    calls = []
    monkeypatch.setattr(jnp.linalg, "eigh", lambda A: calls.append(A))
    p = D.plan(
        "eigh", mesh, backend="off",
        batched_route=D.ROUTE_BATCH_RESHARD)
    with pytest.raises(ValueError, match="matrix face must tile"):
        p.batched(jnp.zeros((3, 7, 7), dtype=jnp.complex128))
    assert calls == []


def test_the_service_route_has_no_upward_or_host_gather_dependency():
    """Standalone distrib_la cannot reach LORRAX common or host gathers."""
    import inspect
    import distrib_la._batch_reshard as br

    source = inspect.getsource(br)
    assert "from common" not in source and "import common" not in source
    assert "process_allgather" not in source and "np.asarray" not in source
    assert "lax.all_to_all" in source and "lax.all_gather" in source
