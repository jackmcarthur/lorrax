import json
import os
import subprocess
import sys
from pathlib import Path


def test_reshard_mu_to_r_uses_all_to_all_and_profiles(tmp_path: Path) -> None:
    """Check A_q[mu_XY, r] -> A_q[mu, r_XY] resharding on 4 forced CPU devices."""
    trace_dir = tmp_path / "reshard_trace"
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_PLATFORM_NAME"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    script = f"""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

trace_dir = Path({str(trace_dir)!r})
trace_dir.mkdir(parents=True, exist_ok=True)

devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 2), ('x', 'y'))
src = NamedSharding(mesh, P(None, ('x', 'y'), None))   # A_q[mu_XY, r]
dst = NamedSharding(mesh, P(None, None, ('x', 'y')))   # A_q[mu, r_XY]

@jax.jit(in_shardings=src, out_shardings=dst)
def reshard(a):
    return jax.lax.with_sharding_constraint(a, dst)

x = jax.device_put(jnp.arange(4 * 64 * 64, dtype=jnp.float64).reshape(4, 64, 64), src)
compiled = reshard.lower(x).compile()
y = compiled(x)
y.block_until_ready()

with jax.profiler.trace(str(trace_dir)):
    for _ in range(3):
        y = compiled(x)
        y.block_until_ready()

text = compiled.as_text().lower()
files = sorted(str(p.relative_to(trace_dir)) for p in trace_dir.rglob('*') if p.is_file())
print(json.dumps({{
    "device_count": len(devices),
    "platforms": sorted({{d.platform for d in devices}}),
    "has_all_to_all": "all-to-all" in text,
    "has_all_gather": "all-gather" in text,
    "has_collective_permute": "collective-permute" in text,
    "files": files,
}}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (
        "Reshard subprocess failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    payload = None
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            payload = json.loads(line)
            break
    assert payload is not None, f"No JSON payload found in stdout:\n{result.stdout}"

    assert payload["device_count"] == 4, payload
    assert payload["platforms"] == ["cpu"], payload
    assert payload["has_all_to_all"], payload
    assert any(name.endswith(".xplane.pb") for name in payload["files"]), payload
    assert any(name.endswith(".trace.json.gz") for name in payload["files"]), payload
