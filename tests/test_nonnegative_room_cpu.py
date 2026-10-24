"""A room-routed LU serves nonnegative quadratics, or refuses before donation."""
import os
from pathlib import Path
import subprocess
import sys


def _controls():
    import runtime  # Install thread policy and source closure before JAX.
    from runtime.source_closure import ensure_source_closure
    ensure_source_closure()
    import importlib
    import json
    import jax
    import numpy as np
    import pytest
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from lxkit import memory

    jax.config.update("jax_enable_x64", True)
    owner = importlib.import_module("distrib_la.nonnegative")
    resolver = importlib.import_module("distrib_la.resolve")
    local = importlib.import_module("distrib_la._one_device")
    mesh = Mesh(np.array(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))
    face, rows = NamedSharding(mesh, P("x", "y")), NamedSharding(mesh, P("x"))
    monkeypatch = pytest.MonkeyPatch()
    # Only the leg's route table is substituted; the real Plan, redistribution,
    # principal-pivot programs and LU math run on the virtual CPU mesh.
    monkeypatch.setattr(resolver, "mesh_platform", lambda _mesh: "ROCM")
    monkeypatch.setattr(local, "mesh_platform", lambda _mesh: "ROCM")
    original = owner.plan
    decisions = []

    def route(*args, **kwargs):
        assert kwargs["budget_bytes"] is not None and kwargs["budget_bytes"] >= 0
        result = original(*args, **kwargs)
        assert result.backend == "one_device"
        decisions.append(int(kwargs["budget_bytes"]))
        return result

    monkeypatch.setattr(owner, "plan", route)
    errors = []
    for budget in (128 << 20, 64 << 20):
        memory.set_budget(budget, linalg="distributed")
        for n in (2, 7):
            # Low-symmetry nodes may have one free variable per centroid.
            # The shared room-only LU policy still admits a fitting matrix;
            # a generic backup tag must not veto it after route admission.
            memory.set_nmu(n)
            rng = np.random.default_rng(31+n)
            matrix = rng.normal(size=(2*n, n))
            a = matrix.T @ matrix + np.eye(n)*.2
            truth = rng.random(n)
            truth[::2] = 0.
            dual = np.where(truth == 0., .5+rng.random(n), 0.)
            b = a @ truth - dual
            padded = n + n % 2
            expected = np.pad(truth, (0, padded-n))
            for scale in (1e-8, 1., 1e8):
                aa = jax.device_put(np.pad(a*scale, ((0, padded-n), (0, padded-n))), face)
                bb = jax.device_put(np.pad(b*scale, (0, padded-n)), rows)
                answer = owner.solve_nonnegative_quadratic(aa, bb, mesh=mesh, n_logical=n)
                np.testing.assert_allclose(answer, expected, rtol=2e-11, atol=2e-12)
                assert answer.sharding.is_equivalent_to(rows, 1)
                assert np.all(np.asarray(answer)[expected == 0] == 0)
                errors.append(float(np.max(np.abs(np.asarray(answer)-expected))))
    assert min(decisions[:6]) > max(decisions[6:])  # Current room reaches each new call.
    # The geometry is warm. Its earlier admitted plan cannot admit a new
    # solve against zero room, nor consume either input before refusing.
    memory.set_budget(0, linalg="distributed")
    aa, bb = jax.device_put(np.eye(2), face), jax.device_put(np.ones(2), rows)
    with pytest.raises(ValueError, match="room"):
        owner.solve_nonnegative_quadratic(aa, bb, mesh=mesh, n_logical=2)
    assert not aa.is_deleted() and not bb.is_deleted()
    # A changed live set after route selection also has a strict executable
    # check: fixed whole-matrix scratch cannot be priced as 1/P face storage.
    linear = original("solve_lu", mesh, backend="distributed", n=2,
                      batched_route="auto", budget_bytes=64 << 20)
    monkeypatch.setattr(memory, "room", lambda: 0)
    product = jax.jit(lambda a: a @ a, out_shardings=face)
    with pytest.raises(RuntimeError, match="GATE nonnegative_quadratic_capacity"):
        owner._principal(mesh, 2, 2, linear, (aa,),
                         (None, product, None, None, None))
    print(json.dumps(dict(cases=len(errors), max_absolute_error=max(errors),
                          budgets=decisions, output_layout="P(x)",
                          centroid_scale_principal_route_admitted=True,
                          insufficient_room_refuses_before_donation=True,
                          compiled_whole_matrix_over_room_refuses=True)))


def test_nonnegative_room_route_and_compiled_capacity_on_p4_cpu():
    root = Path(__file__).parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=4", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--controls"],
                            cwd=root, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"insufficient_room_refuses_before_donation": true' in result.stdout


if __name__ == "__main__" and "--controls" in sys.argv:
    _controls()
