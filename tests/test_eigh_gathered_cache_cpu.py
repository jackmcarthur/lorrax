"""An eigh chain's gathered attempt depends on the plan's room (budget_bytes, ignored by plan
equality): a program built in one room must not be reused in another (census run30k: the
(1, 30000^2) stack priced 40.68 GB/rank at room 20.4 GB with the gathered attempt of a larger
room's program; the (2, 30000^2) stack 18.77 GB without it)."""
import dataclasses
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

pytestmark = pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices")


def test_room_decides_gathered_attempt_per_program(monkeypatch):
    import distrib_la
    import distrib_la._cusolvermp as mp
    plan = importlib.import_module("distrib_la.plan")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    face = NamedSharding(mesh, P("x", "y"))

    def stub(A, *, mesh, compute_evecs=True, block=None, side=None):
        work = jax.lax.with_sharding_constraint(A * (1 + 0j), face)
        return (jnp.zeros(A.shape[-1], jnp.float64), jax.lax.with_sharding_constraint(work * 1, face))
    monkeypatch.setattr(mp, "distributed_eigh", stub)
    monkeypatch.setattr(distrib_la, "workspace_bytes_per_rank", lambda *a, **k: 0)
    n = 2048
    need = plan.compiled_bytes(plan._gathered_program(mesh), mesh, (n, n), np.complex128)
    assert need > plan.GATHERED_EIGH_BYTES
    got = {}
    for room in (4 * need, need // 2):          # admitted first, then refused, in one process
        p = distrib_la.plan("eigh", mesh, n=n, backend="off", batched_route="auto",
                            budget_bytes=int(room), price_only=True)
        p = dataclasses.replace(p, backend="cusolvermp", in_sharding=face,
                                batch_in_sharding=NamedSharding(mesh, P(None, "x", "y")))
        got[room > need] = p.mesh_bytes("eigh", 1, n, np.complex128, "all")
    assert got[True] > got[False], (got, need)      # the refused room compiles its own chain


def test_face_programs_are_keyed_by_the_plans_room():
    """Face plans that differ only by room compare equal; their programs must not be shared."""
    from gw import shared_pole_execution as spe
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    small, large = (spe.face_eigh(mesh, 256, room=r, backend="auto") for r in (1 << 20, 1 << 33))
    assert small == large
    programs = {spe.face_round_check_program(mesh, True, p, spe.plan_room(p)) for p in (small, large)}
    assert len(programs) == 2


def test_cuda_workspace_query_reads_the_stacks_own_n(monkeypatch):
    """place() prices any stack with the plan it holds (run30k 2026-10-20: h_vv beside a
    pencil-sized plan raised 'eigh shape differs from the planned n' on CUDA only). The CUDA
    branch runs here with the platform and the native query stubbed; the plan/shape guard
    runs on every platform."""
    import distrib_la
    import distrib_la._cusolvermp as mp
    from distrib_la import workspace
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    face = NamedSharding(mesh, P("x", "y"))

    def stub(A, *, mesh, compute_evecs=True, block=None, side=None):
        return (jnp.zeros(A.shape[-1], jnp.float64), jax.lax.with_sharding_constraint(A * (1 + 0j), face))
    asked = []
    monkeypatch.setattr(mp, "distributed_eigh", stub)
    monkeypatch.setattr(mp, "get_or_init_context", lambda mesh, **k: 1)
    monkeypatch.setattr(workspace, "_vendor_query", lambda ctx, op, sizes, dtype: asked.append(sizes) or (7, 0))
    p = distrib_la.plan("eigh", mesh, n=64, backend="off", batched_route="auto", price_only=True)
    cuda = dataclasses.replace(p, backend="cusolvermp", in_sharding=face,
                               batch_in_sharding=NamedSharding(mesh, P(None, "x", "y")))
    with pytest.raises(ValueError, match="planned n"):       # CPU platform: the guard still runs
        distrib_la.workspace_bytes_per_rank(p, "eigh", ((1, 32, 32),), np.complex128)
    monkeypatch.setattr(workspace, "mesh_platform", lambda mesh: "CUDA")
    got = cuda.mesh_bytes("eigh", 1, 32, np.complex128)
    assert asked and asked[-1][0] == mp.solve_layout(32, 2)[0] and got > 7, (asked, got)


def test_mesh_price_compiles_once_per_room_class(monkeypatch):
    """A stack's mesh price is compiled once per class of rooms deciding it alike (production
    hsuite: one compile per constructor round, each round's room new): two rooms that both
    admit the gathered retry share one compile; a room that refuses it compiles its own."""
    import distrib_la
    import distrib_la._cusolvermp as mp
    plan = importlib.import_module("distrib_la.plan")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    face = NamedSharding(mesh, P("x", "y"))

    def stub(A, *, mesh, compute_evecs=True, block=None, side=None):
        return (jnp.zeros(A.shape[-1], jnp.float64), jax.lax.with_sharding_constraint(A * (1 + 0j), face))
    monkeypatch.setattr(mp, "distributed_eigh", stub)
    monkeypatch.setattr(distrib_la, "workspace_bytes_per_rank", lambda *a, **k: 0)
    n = 2048
    need = plan.compiled_bytes(plan._gathered_program(mesh), mesh, (n, n), np.complex128)
    monkeypatch.setattr(plan, "_MESH_PROGRAMS", {})
    compiles, real = [], plan._program_bytes
    monkeypatch.setattr(plan, "_program_bytes", lambda *a, **k: compiles.append(1) or real(*a, **k))
    got = []
    for room in (4 * need, 8 * need, need // 2, 4 * need + 1):
        p = distrib_la.plan("eigh", mesh, n=n, backend="off", batched_route="auto",
                            budget_bytes=int(room), price_only=True)
        p = dataclasses.replace(p, backend="cusolvermp", in_sharding=face,
                                batch_in_sharding=NamedSharding(mesh, P(None, "x", "y")))
        before = len(compiles)
        got.append((p.mesh_bytes("eigh", 1, n, np.complex128, "all"), len(compiles) - before))
    assert got[1] == (got[0][0], 0) and got[3] == (got[0][0], 0), got     # same decisions: no compile
    assert got[2][1] > 0 and got[2][0] < got[0][0], got                   # gathered refused: its own



def test_a_memo_hit_replays_its_room_tests(monkeypatch):
    """A stack decided before a room-class build (the report memo) still records its room tests
    inside the build, so the class is not served at a room that decides the stack otherwise
    (REVIEW_C8 A1)."""
    import distrib_la
    import distrib_la._cusolvermp as mp
    plan = importlib.import_module("distrib_la.plan")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    face = NamedSharding(mesh, P("x", "y"))

    def stub(A, *, mesh, compute_evecs=True, block=None, side=None):
        return (jnp.zeros(A.shape[-1], jnp.float64), jax.lax.with_sharding_constraint(A * (1 + 0j), face))
    monkeypatch.setattr(mp, "distributed_eigh", stub)
    monkeypatch.setattr(distrib_la, "workspace_bytes_per_rank", lambda *a, **k: 0)
    n, big = 64, 1 << 30
    base = distrib_la.plan("eigh", mesh, n=n, backend="off", batched_route="auto", budget_bytes=big,
                           price_only=True)
    at = lambda room: dataclasses.replace(base, backend="cusolvermp", budget_bytes=int(room), price_only=False,
                                          in_sharding=face, batch_in_sharding=NamedSharding(mesh, P(None, "x", "y")))
    shape = (4, n, n)
    first = at(big).stack_route(shape, np.complex128, traced=False)      # the memo, outside any build
    cache, builds = {}, []
    decide = lambda room: distrib_la.room_class(cache, "stack", room, lambda: builds.append(room) or
                                                at(room).stack_route(shape, np.complex128, traced=False).route)
    assert decide(big) == first.route
    small = 1
    got = decide(small)
    assert builds == [big, small] and got != first.route, (builds, got, first.route)
