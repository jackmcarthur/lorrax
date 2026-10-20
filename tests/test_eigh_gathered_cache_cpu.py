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
