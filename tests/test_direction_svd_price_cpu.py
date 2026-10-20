"""A face selection's direction solve is priced by the normal-matrix executable it then runs
(distrib_la.direction_svd_bytes, polar._normal_program through plan._executable)."""
import importlib

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

pytestmark = pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices")


def test_the_direction_price_is_the_normal_program_the_solve_runs(monkeypatch):
    polar = importlib.import_module("distrib_la.polar")
    pl = importlib.import_module("distrib_la.plan")
    ws = importlib.import_module("distrib_la.workspace")
    from gw.shared_pole_execution import face_eigh_price
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    shape = (2, 64, 64)
    plan = face_eigh_price(mesh, 64, backend="auto")
    # This host has no whole-mesh eigh: pin the mesh route and its stack price.
    monkeypatch.setattr(type(plan), "known_route", lambda self, s, d: "mesh")
    monkeypatch.setattr(ws, "eigh_stack_bytes", lambda *a, **k: 1000)
    got = polar.direction_svd_bytes(plan, shape, np.complex128)
    compiled = len(pl._EXECUTABLES)
    W = jax.device_put(np.ones(shape, np.complex128), NamedSharding(mesh, P(None, "x", "y")))
    ex = pl._executable(polar._normal_program(mesh, 3, polar._gemm_backend(plan),
                                              plan.backend == "cusolvermp"), (W,))
    assert len(pl._EXECUTABLES) == compiled                  # the run's lookup hits the price's compile
    assert got == max(pl._analysis_bytes(ex), ex.memory_analysis().output_size_in_bytes + 1000)
