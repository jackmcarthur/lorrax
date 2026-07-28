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
  * ``auto`` resolution → demotion to the per-q ``'lu'`` that is
    ANNOUNCED (rank-0 print) — both halves are pinned here, including
    the announcement text (audit fix/zq 2026-07-28: the demote branch
    used to be unreachable on the CPU test mesh, so deleting the
    announcement or the demotion would not have failed any test).

The 2×2 mesh comes from 4 forced host CPU devices in a subprocess (env
must precede ``import jax``).  The worker STUBS the ffi.linalg facade
(``isdf.core._resolve_linalg_backend``): since the fix/zq audit every
EXPLICIT FFI backend ('cusolvermp'/'on'/'scalapack') runs the facade's
platform/capability/geometry guard ladder, which on this CPU test mesh
refuses (cusolvermp is CUDA-only; scalapack needs the compiled host FFI)
BEFORE the divisibility contract under test is reachable.  The facade's
own guards are pinned by tests/test_ffi_linalg_contract.py.  NOTE:
consequently nothing here asserts that 'cusolvermp' is a promise a CPU
handler can run — it is not, and on a real CPU mesh the facade refuses
it at resolve time.
"""
import os
import subprocess
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")

_WORKER = r"""
import contextlib
import io

import numpy as np
import jax
from jax.sharding import Mesh

import isdf.core as core
from isdf.core import _resolve_solver_kind_transverse

devices = jax.devices()
assert len(devices) >= 4, f"need 4 forced host devices, got {len(devices)}"
mesh22 = Mesh(np.array(devices[:4]).reshape(2, 2), ("x", "y"))

# Stub the facade guard ladder (see module docstring): this worker pins
# the RESOLVER's divisibility/announce contract, not the facade's
# platform/capability guards.
core._resolve_linalg_backend = lambda *a, **k: None

# 1. Explicit distributed request + non-dividing n_log -> resolve-time
#    ValueError naming the fix (never a silent demotion) — BOTH explicit
#    backends, since both take the same divisibility contract.
for _ov in ("cusolvermp", "scalapack"):
    try:
        _resolve_solver_kind_transverse(mesh22, _ov, n_rmu_logical=135)
        raise SystemExit(f"FAIL: explicit {_ov} with n=135 did not raise")
    except ValueError as e:
        msg = str(e)
        assert "135" in msg and "distributed_lu" in msg, msg

# 2. Explicit request + dividing n_log -> honored.
kind = _resolve_solver_kind_transverse(mesh22, "cusolvermp", n_rmu_logical=136)
assert kind == "cusolvermp_lu", kind
kind = _resolve_solver_kind_transverse(mesh22, "scalapack", n_rmu_logical=136)
assert kind == "scalapack_lu", kind

# 3. auto on a CPU mesh -> per-q 'lu' (CPU-safe ladder), any extent, and
#    NO demotion announcement (nothing was demoted).
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    kind = _resolve_solver_kind_transverse(mesh22, "auto", n_rmu_logical=135)
assert kind == "lu", kind
assert "demoting" not in buf.getvalue(), buf.getvalue()

# 4. auto DEMOTE-AND-ANNOUNCE (the doctrine-3 "auto may demote but must
#    ANNOUNCE" half): fake a 2D GPU mesh so auto resolves to
#    'cusolvermp_lu'; a non-dividing extent must then return the per-q
#    'lu' AND say so on stdout (rank 0).
core._mesh_is_cpu = lambda m: False
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    kind = _resolve_solver_kind_transverse(mesh22, "auto", n_rmu_logical=135)
out = buf.getvalue()
assert kind == "lu", kind
assert "demoting" in out and "135" in out and "cusolvermp_lu" in out, out

# 5. Dividing extent on the same faked 2D GPU mesh: auto keeps the
#    distributed kind, no announcement.
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    kind = _resolve_solver_kind_transverse(mesh22, "auto", n_rmu_logical=136)
assert kind == "cusolvermp_lu", kind
assert "demoting" not in buf.getvalue(), buf.getvalue()

# 6. No n_rmu_logical (callers that cannot know it) -> pure ladder,
#    unchanged behavior (facade still stubbed; see module docstring).
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
