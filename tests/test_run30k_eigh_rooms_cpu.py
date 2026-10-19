"""run30k's bank line selection and constructor eigh stacks, decided in the rooms the run hands them.

Probe 59995725 (2026-10-19, 9c103b668 + ffb6b6f9a) refused the line selection's direction eigh:
'normal_eigh stack (1, 30000, 30000) complex128 needs 8.10 GB/rank ... over its room 0.00 GB/rank'.
The selection's face plans were built with no room (face_eigh: room 0), and a planner replay at a
nominal room (16.11 GB) did not see it. Here every room comes from the runtime helpers on the run's
ledger: CapacityLedger at the run's budget (63414637611 B/rank, 4x4, n_mu 29998, nspinor 2) with
the bank's Coulomb roots live; line_selection_execution / line_selection_room; the constructor's
face_eigh_room on its selection row (46.49 GB admitted) and check_eigh_room. Only the vendor is
faked (a CPU 4x4 mesh standing in for A100s, cuSOLVERMp's whole-mesh eigh, the measured local
syevd workspace), in a subprocess with 16 host devices.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPLAY = r'''
import dataclasses, json, sys
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from functools import lru_cache
from types import SimpleNamespace
from jax.sharding import Mesh
import lxkit, distrib_la
from distrib_la import loader, workspace
import importlib
_pl, _rs = (importlib.import_module("distrib_la." + m) for m in ("plan", "resolve"))
_pl.mesh_platform = _rs.mesh_platform = lambda mesh: "CUDA"
def query(ctx, n, cplx, device, host):      # cusolverDnXsyevd_bufferSize: 2x LAPACK (A100, measured)
    device._obj.value, host._obj.value = 2 * (16 * (2 * n + n * n) + 8 * (1 + 5 * n + 2 * n * n)), 0
    return 0
loader.get_lib = lambda platform: type("Lib", (), {"lrx_eigh_workspace_bytes": staticmethod(query)})()
lxkit.device_vendor = lambda device: "cuda"; workspace.device_vendor = lambda device: "cuda"
from common import gpu_utils
gpu_utils._RUN_DEVICE_BUDGET_GB = 63.41
mesh = Mesh(np.array(jax.devices()[:16]).reshape(4, 4), ("x", "y"))
N, BUDGET, NATIVE_EIGH = 30000, 63414637611, 7382397440

import gw.shared_pole_execution as spe
@lru_cache(maxsize=None)
def mesh_plan(mesh, n, backend):            # the leg's whole-mesh provider: cuSOLVERMp on A100
    p = distrib_la.plan("eigh", mesh, n=int(n), backend="off", batched_route="auto", budget_bytes=0)
    return dataclasses.replace(p, backend="cusolvermp")
spe._face_eigh = mesh_plan
# The leg's eigh workspace queries: the run's whole-mesh reservation (native_eigh) and the
# measured local cusolverDnXsyevd workspace (2x LAPACK, 57.6 GB at n 30000), per stack.
def wbpr(plan, op, shapes, dtype):
    if plan.backend == "cusolvermp":
        return NATIVE_EIGH
    nb, n = int(shapes[0][0]), int(shapes[0][-1])
    return -(-nb // mesh.size) * 2 * (16 * (2 * n + n * n) + 8 * (1 + 5 * n + 2 * n * n))
distrib_la.workspace_bytes_per_rank = wbpr

from gw.shared_pole_recipe import CapacityLedger
from gw.shared_pole_execution import line_selection_execution, line_selection_room
from gw.shared_pole_capacity import face_eigh_room, check_eigh_room, constructor_eigenplan
from gw.shared_pole_directions import charge_line_selection

out = {}
def decide(plan, nb, op):
    try:
        d = plan.stack_route((nb, N, N), np.complex128, op)
        return dict(route=d.route, price=int(d.program_bytes or 0), room=None if d.room is None else int(d.room))
    except (RuntimeError, ValueError) as e:
        return dict(refused=str(e)[:240])

# --- bank line selection (response_bank.line_route) ---
ledger = CapacityLedger(SimpleNamespace(nk_tot=1, nspinor=2, n_rmu=29998), mesh_xy=mesh,
                        device_budget_bytes=BUDGET)
face_bytes = 16 * N * N // mesh.size
ledger.reserve("coulomb_roots:2", resident_bytes_per_rank=face_bytes, workspace_bytes_per_rank=0)
ledger.live_stages = ("coulomb_roots:2",)
carry = 2 * 1 * face_bytes
execution, resident, ws = line_selection_execution([N], mesh=mesh, ledger=ledger, nq=1, carry=carry,
                                                   eigh_backend="auto")
room = line_selection_room(ledger, resident, carry=carry)
out["line"] = dict(execution=execution, face_aggregate=resident + ws + carry + face_bytes, room=room)
meta = SimpleNamespace(mu_basis=SimpleNamespace(active_mask=np.arange(N) < 29998, n_packed=N),
                       n_rmu=29998, shared_pole_recipe={})
sel = charge_line_selection(meta, mesh_xy=mesh, ordered=False, execution=execution, nq=1,
                            eigh_backend="auto", room=room)
out["line_normal_eigh"] = decide(sel.plans["charge"], 1, "normal_eigh")
out["line_no_room_plan"] = decide(constructor_eigenplan(mesh, N, "face", None, backend="auto"), 1, "normal_eigh")
try:
    charge_line_selection(meta, mesh_xy=mesh, ordered=False, execution="face", nq=1, eigh_backend="auto")
    out["line_no_room_selector"] = "built"
except ValueError as e:
    out["line_no_room_selector"] = str(e)[:120]

# --- constructor (shared_pole_constructor face branch; passivity checks) ---
selection_row = dict(available_device_bytes_per_rank=BUDGET, aggregate_bytes_per_rank=46_490_000_000)
face_room = face_eigh_room(selection_row, 0)
cons = constructor_eigenplan(mesh, N, "face", face_room, backend="auto")
out["constructor_room"] = face_room
out["constructor_eigh"] = [decide(cons, nb, "eigh") for nb in (1, 2)]
out["constructor_normal_eigh"] = decide(cons, 1, "normal_eigh")
pre = int(8.5 * (1 << 30))                  # passivity's pre-floor room, 8-9 GiB (run30k)
model_row = dict(available_device_bytes_per_rank=BUDGET, native_workspace={"eigh": NATIVE_EIGH},
                 aggregate_bytes_per_rank=BUDGET - pre - 1_000_000_000 + NATIVE_EIGH)
check_room = check_eigh_room(model_row, 1_000_000_000)
out["passivity_room"] = check_room
out["passivity_eigh"] = [decide(constructor_eigenplan(mesh, N, "face", check_room, backend="auto"), nb, "eigh")
                         for nb in (1, 2)]
print("RESULT " + json.dumps(out))
'''


def _replay():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=16",
               PYTHONPATH=os.pathsep.join([*map(str, sorted((ROOT / "services").glob("*/src"))),
                                           str(ROOT / "src")]))
    run = subprocess.run([sys.executable, "-c", REPLAY], env=env, capture_output=True, text=True,
                         timeout=600, cwd=ROOT)
    lines = [line for line in run.stdout.splitlines() if line.startswith("RESULT ")]
    assert run.returncode == 0 and lines, run.stdout[-2000:] + run.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT "):])


def test_run30k_line_selection_and_constructor_eighs_are_decided_in_their_runtime_rooms():
    out = _replay()
    print(json.dumps(out, indent=1))
    line = out["line"]
    # The route line the probe logged: face 13.68 GB of room 63.41 GB.
    assert line["execution"] == "face" and abs(line["face_aggregate"] / 1e9 - 13.68) < 0.01
    assert line["room"] > 8.10e9
    decided = [out["line_normal_eigh"], *out["constructor_eigh"], out["constructor_normal_eigh"],
               *out["passivity_eigh"]]
    assert all("refused" not in d and d["room"] and d["price"] <= d["room"] for d in decided), decided
    assert abs(out["constructor_room"] / 1e9 - 16.11) < 0.01        # the run's logged room
    assert 8 << 30 == out["passivity_room"]                         # floored to 8 GiB
    # The probe's failure, reproduced with no room, and the guard that now names it.
    assert "over its room 0.00 GB/rank" in out["line_no_room_plan"]["refused"]
    assert "GATE shared_pole_line_room" in out["line_no_room_selector"]
