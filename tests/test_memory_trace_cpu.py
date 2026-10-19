"""The memory trace (LORRAX_MEMORY_TRACE, lxkit.memory) and its residual table on 4 CPU devices."""
import json
import os
import subprocess
import sys
from pathlib import Path

RUN = r"""
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from common import timing
from lxkit import memory
mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
face = NamedSharding(mesh, P("x", "y"))
memory.set_budget(1e9)
with timing.section("demo.outer"):
    a = jax.device_put(jnp.ones((1024, 1024), jnp.complex64), face)
    with timing.section("demo.inner"):
        memory.book(memory.Price(vendor=memory.Term(1e6, source="a library's size query")))
        b = jax.jit(lambda x: (x @ x.T) * 2, out_shardings=face)(a)
    del b
"""


def test_trace_records_live_arrays_compiles_and_residuals(tmp_path):
    root = Path(__file__).resolve().parents[1]
    trace = tmp_path / "trace.json"
    env = dict(os.environ, JAX_PLATFORMS="cpu", LORRAX_MEMORY_TRACE=str(trace),
               XLA_FLAGS="--xla_force_host_platform_device_count=4")
    run = subprocess.run([sys.executable, "-c", RUN], env=env, capture_output=True, text=True,
                         timeout=300, cwd=root)
    assert run.returncode == 0, run.stderr[-2000:]
    sections = {s["name"]: s for s in json.loads(trace.read_text())["sections"]}
    inner, outer = sections["demo.inner"], sections["demo.outer"]
    face = 1024 * 1024 * 8 // 4                        # complex64 on a 2 x 2 mesh, per device
    assert inner["open"]["live"] == face               # a, created in demo.outer, inherited
    assert inner["peak"]["live"] == 2 * face           # a and b
    assert outer["close"]["live"] == face              # b deleted
    born = {(tuple(x["shape"]), x["born"]) for x in inner["peak"]["arrays"]}
    assert {((1024, 1024), "demo.outer"), ((1024, 1024), "demo.inner")} <= born
    matmul = [c for c in inner["compiles"] if c["output"] == face]
    assert matmul and all(k in matmul[0] for k in ("temp", "argument", "alias", "code"))
    assert inner["priced"]["vendor"] == 1e6
    from lxkit import memory_residuals
    rows = {r["section"]: r for r in memory_residuals.rows(json.loads(trace.read_text()))}
    r = rows["demo.inner"]
    assert r["live"] == face and r["vendor"] == 1e6
    assert r["residual"] == r["measured"] - (r["live"] + r["compiled"] + r["vendor"])
    assert "demo.inner" in memory_residuals.table(json.loads(trace.read_text()), True)
