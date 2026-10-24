"""Actual face-Dyson algebra and fallback admission on the room-only LU leg."""
import os
from pathlib import Path
import subprocess
import sys


def _controls():
    import runtime
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
    owner = importlib.import_module("gw.response_bank")
    w_owner = importlib.import_module("gw.w_isdf")
    resolver = importlib.import_module("distrib_la.resolve")
    local = importlib.import_module("distrib_la._one_device")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(resolver, "mesh_platform", lambda _mesh: "ROCM")
    monkeypatch.setattr(local, "mesh_platform", lambda _mesh: "ROCM")
    mesh = Mesh(np.array(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))
    face = NamedSharding(mesh, P(None, "x", "y"))
    memory.set_budget(128 << 20, linalg="distributed")

    class Ledger:
        live_stages = ()
        cap = 0

        def at_phase(self):
            pass

        def room_bytes_per_rank(self, concurrent_with=()):
            assert concurrent_with == ()
            return self.cap

    ledger = Ledger()
    n, q, pref = 4, 3, .2
    built = owner._response_programs(mesh, n, "distributed", "auto", pref, False, None)
    # Binding a charge-moment program performs no LU, even on the no-library
    # leg with zero room. Actual solve-bearing calls admit their operands later.
    dyson, moments, lu = built.bind(ledger)
    assert lu is None and not isinstance(moments, owner._Decided)
    h = np.broadcast_to(np.diag(1+.1*np.arange(n)), (q, n, n)).astype(complex).copy()
    chi = np.stack([np.diag(-.1-.01*np.arange(n)-.02*k) for k in range(q)]).astype(complex)
    put = lambda a: jax.device_put(a, face)
    H, C = put(h), put(chi)
    with pytest.raises(ValueError, match="room"):
        dyson.value.resolve(H, C)
    ledger.cap = 64 << 20
    program = dyson.value.resolve(H, C)
    assert program.lu.backend == "one_device"
    executable = program.lower(H, C).compile()
    got = program.finish((H, C), executable(H, C))
    x = pref * (h @ chi @ h)
    wanted = h @ np.linalg.solve(np.eye(n)[None]-x, x) @ h
    np.testing.assert_allclose(got, wanted, rtol=2e-12, atol=2e-13)
    assert got.sharding.is_equivalent_to(face, 3)
    C = put(chi)  # The successful program consumed its original fresh response rows.
    ledger.cap //= 2
    assert dyson.value.resolve(H, C) is program  # Same actual decision, no room cache key.
    ledger.cap = 0
    with pytest.raises(ValueError, match="room"):
        dyson.value.resolve(H, C)  # A warm variant never bypasses new admission.
    assert not H.is_deleted() and not C.is_deleted()
    base = memory.room(bucketed=False)
    abstract = jax.ShapeDtypeStruct((q, n, n), np.complex128, sharding=face)
    assert owner._dyson_room(None, mesh, (H, abstract)) == base-q*n*n*16//mesh.size
    # An uninitialized ledger uses the same shared run room, not an unbounded route.
    class Unknown(Ledger):
        @property
        def live_stages(self):
            raise ValueError("no phase selected")
    assert owner._dyson_room(Unknown(), mesh, (H, abstract)) == base-q*n*n*16//mesh.size

    # A bounded policy fixture: enough room for the face-backed fallback plus
    # one whole local LU, while whole-parent Dyson storage exceeds the budget.
    memory.set_budget(12000, linalg="distributed")
    monkeypatch.setattr(memory, "room", lambda **_kwargs: 4320)
    assert w_owner._dyson_route("local", n, 40, mesh) == ("distributed", 4320)
    assert w_owner._dyson_route("local", n, 1, mesh) == ("local", None)
    memory.set_budget(8000, linalg="distributed")
    monkeypatch.setattr(memory, "room", lambda **_kwargs: 320)
    with pytest.raises(RuntimeError, match="GATE w_dyson_capacity"):
        w_owner._dyson_route("local", n, 40, mesh)
    print(json.dumps(dict(max_absolute_error=float(np.max(np.abs(np.asarray(got)-wanted))),
                          actual_q_batch=q, output_layout="P(None,x,y)",
                          charge_moments_bind_without_lu=True,
                          warm_variant_rechecks_room=True,
                          uninitialized_ledger_prices_future_faces=True,
                          whole_parent_fallback_admitted=True,
                          unavailable_oversized_fallback_refuses=True)))


def test_rocm_dyson_room_route_on_p4_cpu():
    root = Path(__file__).parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=4", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--controls"],
                            cwd=root, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"unavailable_oversized_fallback_refuses": true' in result.stdout


if __name__ == "__main__" and "--controls" in sys.argv:
    _controls()
