"""Transverse distributed-LU divisibility contract (resolve time).

The indefinite transverse ζ solve must run at the LOGICAL μ extent
(ROOT_CAUSE.md 2026-07-08), and the block-cyclic descriptors of the two
distributed LU backends need ``n_log % px == n_log % py == 0``.  Before
2026-07-27 a non-dividing extent demoted to the per-q replicated LU via
a ``warnings.warn`` deep inside ``solve_zeta`` — the ledgered "silent
replicated-LU fallback".  The contract now (mirrors the charge two-plan
W treatment, quality pattern #6/#8):

  * EXPLICIT request (``distributed_lu=cusolvermp|on|scalapack``) with a
    non-dividing ``n_rmu_logical`` → ValueError at RESOLVE time.
  * ``auto`` resolution → announced demotion to the per-q ``'lu'``.

The 2×2 mesh comes from 4 forced host CPU devices in a subprocess (env
must precede ``import jax``).  On a CPU mesh the ladder itself never
yields 'cusolvermp_lu' (auto is CPU-safe), so the subprocess fakes the
2D-GPU decision only where the ladder is override-driven ('cusolvermp'
→ kind_cusolvermp on any true-2D mesh, no FFI probe involved).
"""
import os
import subprocess
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")

_WORKER = r"""
import numpy as np
import jax
from jax.sharding import Mesh

from isdf.core import _resolve_solver_kind_transverse

devices = jax.devices()
assert len(devices) >= 4, f"need 4 forced host devices, got {len(devices)}"
mesh22 = Mesh(np.array(devices[:4]).reshape(2, 2), ("x", "y"))

# 1. Explicit distributed request + non-dividing n_log → resolve-time
#    ValueError naming the fix (never a silent demotion).
try:
    _resolve_solver_kind_transverse(mesh22, "cusolvermp", n_rmu_logical=135)
    raise SystemExit("FAIL: explicit cusolvermp with n=135 did not raise")
except ValueError as e:
    msg = str(e)
    assert "135" in msg and "distributed_lu" in msg, msg

# 2. Explicit request + dividing n_log → honored.
kind = _resolve_solver_kind_transverse(mesh22, "cusolvermp", n_rmu_logical=136)
assert kind == "cusolvermp_lu", kind

# 3. auto on a CPU mesh → per-q 'lu' (CPU-safe ladder), any extent.
assert _resolve_solver_kind_transverse(mesh22, "auto",
                                       n_rmu_logical=135) == "lu"

# 4. No n_rmu_logical (callers that cannot know it) → pure ladder,
#    unchanged behavior.
assert _resolve_solver_kind_transverse(mesh22, "cusolvermp") == "cusolvermp_lu"
print("TRANSVERSE_LU_RESOLVE_OK")
"""


def test_transverse_lu_divisibility_contract(tmp_path):
    env = os.environ.copy()
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + " --xla_force_host_platform_device_count=4").strip()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", _WORKER], env=env,
        capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
    assert "TRANSVERSE_LU_RESOLVE_OK" in out.stdout
