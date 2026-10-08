"""Program identity on CPU (compile audit 2026-10-05, rows 1 and 4).

Row 1: the persistent-cache namespace names what JAX's key cannot see (jax,
jaxlib, the FFI bundle, this file's key schema) and never the LORRAX source.
Row 4: a route-(c) decision, eager or traced, prices its candidates from the
shapes: nothing is compiled to be measured and no size is exchanged.
"""
import importlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)


def test_namespace_names_no_source(monkeypatch):
    from common import jax_compile_cache as jcc
    assert not hasattr(jcc, "_source_identity")
    name = jcc.cache_namespace()
    assert re.fullmatch(r"jax[^_]+-jaxlib[^_]+_.+_k\d+", name), name
    assert "git-" not in name and "source-" not in name
    monkeypatch.setattr(jcc, "_ffi_identity", lambda: "bundle-other")
    assert jcc.cache_namespace() != name
    monkeypatch.setattr(jcc, "_ffi_identity", lambda: name.split("_")[1])
    monkeypatch.setattr(jcc, "_KEY_SCHEMA", "k999")
    assert jcc.cache_namespace() != name


def _mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _plan(mesh):
    from distrib_la.plan import Plan
    face = NamedSharding(mesh, P(None, "x", "y"))
    return Plan(op="eigh", requested="distributed", backend="scalapack", mesh=mesh, n=8,
                in_sharding=NamedSharding(mesh, P("x", "y")), batch_in_sharding=face,
                requested_batched_route="auto", budget_bytes=1 << 30)


def test_stack_route_is_priced_from_the_shapes():
    """Eager and traced decisions take the same shape price: no sizing compile, no exchange."""
    plan_mod = importlib.import_module("distrib_la.plan")
    plan_mod._STACK_ROUTES.clear()
    p = _plan(_mesh())
    traced = p.stack_route((4, 8, 8), np.complex128, traced=True)
    eager = p.stack_route((4, 8, 8), np.complex128, traced=False)
    price = plan_mod._stack_price("eigh", _mesh(), 4, 8, "complex128", 1)
    assert traced.route == eager.route == plan_mod.ROUTE_BATCH_RESHARD
    assert traced.program_bytes == eager.program_bytes == price
    assert price >= 4 * plan_mod.BATCH_EIGH_TILES * 8 * 8 * 16 and eager.per_rank == 4 and eager.rounds == 1
    assert "priced from the shapes" in plan_mod._describe_stack(eager)


def test_face_plans_differ_by_room_share_programs(monkeypatch):
    from gw import shared_pole_execution as spe
    from gw.shared_pole_capacity import constructor_eigenplan
    from gw.shared_pole_execution import face_eigh
    mesh = _mesh()
    monkeypatch.setattr(spe, "_face_eigh", lambda mesh, n: _plan(mesh))   # CPU has no distributed eigh
    a, b = face_eigh(mesh, 8, 1 << 30), face_eigh(mesh, 8, 1 << 31)
    assert a == b and hash(a) == hash(b) and a.budget_bytes != b.budget_bytes
    assert constructor_eigenplan(mesh, 8, "face", 1 << 30) == constructor_eigenplan(mesh, 8, "face", None)
    assert constructor_eigenplan(mesh, 8, "local") is constructor_eigenplan(mesh, 8, "local", 1 << 30)


def test_program_keys_hold_no_site_or_budget():
    import inspect
    plan_mod = importlib.import_module("distrib_la.plan")
    for fn in (plan_mod._stack_price, plan_mod._reshard_stack_program):
        names = inspect.signature(getattr(fn, "__wrapped__", fn)).parameters
        assert not {"site", "budget_bytes", "room"} & set(names), (fn.__name__, list(names))
