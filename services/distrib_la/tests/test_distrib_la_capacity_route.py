"""Batch-layout operands and capacity routing (plan revision 1, A1b and A2).

CPU (pytest, 2x2 host mesh):
* a spectral selection on a batch-layout stack P(('x','y'),None,...) equals the face route bit for bit
  (spectra and Q, rank 3 and rank 4), with no movement;
* ``real_rows`` never solves a synthetic slot (runtime kernel count) and retains nothing for it;
* ``fits_local`` is exactly live bytes + the LAPACK ?heevd workspace formula at the boundary;
* ``'auto'`` with a budget routes by capacity: a provider plan whose stack fits takes route (c), one that
  does not keeps its provider route; matmul likewise.
GPU (``python test_distrib_la_capacity_route.py OUT.json`` at P4): ``fits_local`` against
``workspace_bytes_per_rank`` of the local plan, and a ``distributed`` eigh plan with a fitting stack runs
route (c) bit for bit, a non-fitting one runs cuSOLVERMp (1e-12).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import distrib_la as D


def _mesh():
    import jax
    from jax.sharding import Mesh
    from lxkit.testing import require_devices
    require_devices(4, "cpu")
    return Mesh(np.asarray(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))


def _put(a, mesh, spec):
    import jax
    from jax.sharding import NamedSharding
    return jax.make_array_from_callback(a.shape, NamedSharding(mesh, spec), lambda idx: a[idx])


def _stacks(rng, leading, n):
    def unitary():
        return np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    w = np.empty(leading + (n, n), complex)
    h = np.empty(leading + (n, n), complex)
    for idx in np.ndindex(*leading):
        cut = 1 + (sum(idx) + int(np.prod(idx) if idx else 0)) % (n - 1)
        spectrum = np.where(np.arange(n) < cut, 10.0 - np.arange(n), 1e-4 * (1 + np.arange(n)))
        u, v = unitary(), unitary()
        w[idx] = (u * spectrum) @ v.conj().T
        h[idx] = (v * spectrum) @ v.conj().T
    return w, h


@pytest.mark.parametrize("leading", [(4,), (4, 3)])
def test_batch_layout_selection_equals_the_face_route_bitwise(leading):
    from jax.sharding import PartitionSpec as P
    mesh = _mesh()
    n = 8
    w, h = _stacks(np.random.default_rng(4040 + len(leading)), leading, n)
    extent = lambda r: 2 * ((r + 1) // 2)
    none = (None,) * (len(leading) - 1)
    face = P(*((None,) * len(leading)), "x", "y")
    batch = P(("x", "y"), *none, None, None)
    svd = D.plan("eigh", mesh, backend="off", n=2 * n, batched_route="batch_reshard")
    eig = D.plan("eigh", mesh, backend="off", n=n, batched_route="batch_reshard")
    for name, matrix, select in (
            ("svd", w, lambda a: D.right_singular_vectors(a, 1e-3, eigh_plan=svd, column_extent=extent)),
            ("eigh", h, lambda a: D.leading_eigenvectors(a, 3, eigh_plan=eig, column_extent=extent))):
        qf, vf = select(_put(matrix, mesh, face))
        qb, vb = select(_put(matrix, mesh, batch))
        assert qb.sharding.spec == batch and qf.sharding.spec == face
        assert np.array_equal(np.asarray(qb), np.asarray(qf)), name
        def flat(v):
            return [np.asarray(x) for row in v for x in (row if isinstance(row, tuple) else (row,))]
        assert len(flat(vb)) == int(np.prod(leading))
        assert all(np.array_equal(a, b) for a, b in zip(flat(vb), flat(vf))), name


def test_real_rows_never_solve_synthetic_slots(monkeypatch):
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    import distrib_la._batch_reshard as br
    mesh = _mesh()
    n = 8
    w, _ = _stacks(np.random.default_rng(4050), (4, 2), n)
    calls = []
    original = jnp.linalg.eigh

    def counted(a, *args, **kwargs):
        jax.debug.callback(lambda _unused: calls.append(1), jnp.int32(0))
        return original(a, *args, **kwargs)

    monkeypatch.setattr(jnp.linalg, "eigh", counted)
    br._JIT_CACHE.clear()
    svd = D.plan("eigh", mesh, backend="off", n=2 * n, batched_route="batch_reshard")
    q, values = D.right_singular_vectors(_put(w, mesh, P(("x", "y"), None, None, None)), 1e-3,
                                         eigh_plan=svd, column_extent=lambda r: 2 * ((r + 1) // 2),
                                         real_rows=3)
    q.block_until_ready()
    assert len(calls) == 3 * 2, calls                     # 3 real parents x 2 samples; slot 3 never solved
    assert [len(row) for row in values] == [2, 2, 2, 2]
    assert all(np.asarray(v).size == 0 for v in values[3])
    assert all(np.asarray(v).size > 0 for row in values[:3] for v in row)
    assert float(jnp.max(jnp.abs(q[3]))) == 0.0
    with pytest.raises(ValueError, match="batch-layout"):
        D.right_singular_vectors(_put(w, mesh, P(None, None, "x", "y")), 1e-3, eigh_plan=svd,
                                 column_extent=lambda r: 2 * ((r + 1) // 2), real_rows=3)


def test_fits_local_is_live_bytes_plus_the_lapack_workspace_at_the_boundary():
    mesh = _mesh()
    plan = D.plan("eigh", mesh, backend="off", n=24, batched_route="batch_reshard")
    n = 24
    live = ((2, n, n), (2, n, n))
    workspace = 16 * (2 * n + n * n) + 8 * (1 + 5 * n + 2 * n * n) + 4 * (3 + 5 * n)
    need = 2 * 2 * n * n * 16 + workspace
    assert D.fits_local(plan, "eigh", live, np.complex128, need)
    assert not D.fits_local(plan, "eigh", live, np.complex128, need - 1)


def test_auto_with_a_budget_routes_eigh_and_matmul_by_capacity():
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    mesh = _mesh()
    n = 8
    face = NamedSharding(mesh, P(None, "x", "y"))
    # A provider plan built without resolution: the routing question needs no library.
    provider = D.Plan(op="eigh", requested="scalapack", backend="scalapack", mesh=mesh, n=n,
                      in_sharding=NamedSharding(mesh, P("x", "y")), batch_in_sharding=face,
                      requested_batched_route="auto", budget_bytes=10 ** 9)
    assert provider.batched_route == D.ROUTE_BACKEND_BATCHED          # the static answer is unchanged
    assert provider.route_for((5, n, n), np.complex128) == D.ROUTE_BATCH_RESHARD
    tight = D.Plan(**{**provider.__dict__, "budget_bytes": 64})
    assert tight.route_for((5, n, n), np.complex128) == D.ROUTE_BACKEND_BATCHED
    unbudgeted = D.Plan(**{**provider.__dict__, "budget_bytes": None})
    assert unbudgeted.route_for((5, n, n), np.complex128) == D.ROUTE_BACKEND_BATCHED
    rng = np.random.default_rng(4060)
    a = rng.normal(size=(5, 8, 6)) + 1j * rng.normal(size=(5, 8, 6))
    b = rng.normal(size=(5, 6, 10)) + 1j * rng.normal(size=(5, 6, 10))
    staged = D.matmul(jax.device_put(a, face), jax.device_put(b, face), mesh=mesh, backend="off",
                      batched_route="batch_reshard")
    routed = D.matmul(jax.device_put(a, face), jax.device_put(b, face), mesh=mesh, backend="off",
                      batched_route="auto", budget_bytes=10 ** 9)
    assert np.array_equal(np.asarray(routed), np.asarray(staged))
    with pytest.raises(RuntimeError, match="no distributed provider"):
        D.matmul(jax.device_put(a, face), jax.device_put(b, face), mesh=mesh, backend="off",
                 batched_route="auto", budget_bytes=64)


def check_gpu(mesh):
    """P4 CUDA: fits_local against the vendor query, and a distributed plan routed by capacity."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    n, nb = 96, 8
    rows = {}
    local = D.plan("eigh", mesh, backend="off", n=n, batched_route="batch_reshard")
    ws = D.workspace_bytes_per_rank(local, "eigh", ((1, n, n),), np.complex128)
    need = 2 * n * n * 16 + ws
    assert D.fits_local(local, "eigh", ((1, n, n),) * 2, np.complex128, need)
    assert not D.fits_local(local, "eigh", ((1, n, n),) * 2, np.complex128, need - 1)
    rows["fits_local_boundary_bytes"] = int(need)
    rng = np.random.default_rng(4070)
    z = rng.normal(size=(nb, n, n)) + 1j * rng.normal(size=(nb, n, n))
    a = 0.5 * (z + np.conj(np.swapaxes(z, -1, -2)))
    face = NamedSharding(mesh, P(None, "x", "y"))
    stack = jax.make_array_from_callback(a.shape, face, lambda idx: a[idx])
    from jax.experimental import multihost_utils
    host = lambda x: np.asarray(multihost_utils.process_allgather(x, tiled=True))
    ws_c, vs_c = local.batched(stack)
    fitting = D.plan("eigh", mesh, backend="distributed", n=n, batched_route="auto", budget_bytes=2 ** 34)
    assert fitting.route_for(stack.shape, stack.dtype) == D.ROUTE_BATCH_RESHARD
    wf, vf = fitting.batched(stack)
    assert np.array_equal(host(wf), host(ws_c)) and np.array_equal(host(vf), host(vs_c))
    tight = D.plan("eigh", mesh, backend="distributed", n=n, batched_route="auto", budget_bytes=1024)
    assert tight.route_for(stack.shape, stack.dtype) != D.ROUTE_BATCH_RESHARD
    wt, vt = tight.batched(stack)
    wt, vt, wc = host(wt), host(vt), host(ws_c)
    rows["provider_eigenvalue_rel"] = float(np.max(np.abs(wt - wc)) / np.max(np.abs(wc)))
    rows["provider_residual_rel"] = float(np.max(np.linalg.norm(a @ vt - vt * wt[:, None, :], axis=(-2, -1)))
                                          / np.max(np.abs(wc)))
    assert rows["provider_eigenvalue_rel"] < 1e-12 and rows["provider_residual_rel"] < 1e-11, rows
    return dict(status="PASS", **rows)


if __name__ == "__main__":
    from runtime import initialize_communicator_stack, run_main_and_finalize
    stack = initialize_communicator_stack(platform="gpu")

    def main():
        import sys
        import jax
        result = check_gpu(stack.mesh)
        result["job_step"] = os.environ.get("SLURM_JOB_ID", "") + "." + os.environ.get("SLURM_STEP_ID", "")
        if jax.process_index() == 0:
            Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        return 0
    run_main_and_finalize(main)
