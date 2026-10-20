"""run30k's eigh plans, made as the run makes them, in the rooms the runtime gives them.

Two probes refused an eigh stack in room 0 (2026-10-19): the bank's line selection (built with no
room, 59995725) and the constructor's gram_reduction (its round admitted over budget, 89.26 of
63.41 GB/rank: the round's room function returned 0). ``replays/run30k_plan_census.py`` replays,
on a 16-device CPU mesh with the A100 leg's vendor faked below the decision, every room-decided
eigh plan of the run (the 4 ``plan('eigh', budget_bytes=...)`` sites: isdf charge, bank Coulomb
root, the face plans of face_eigh, the pinned staged plan) in the rooms the runtime helpers give
them on the run's ledger (CapacityLedger at 63414637611 B/rank, n_mu 29998, nspinor 2, 4x4):
line_selection_execution/room, face_batch_width + ConstructorCapacity.plan, face_eigh_room and
the round's room function, check_eigh_room, shared_device_room. The constructor's round program
(face_parent_program) is traced with jax.eval_shape on abstract run30k operands (side 48640, n
30000, carrier for pole budget 14999, infinity 3750), first and whole-chain phases, so every stack
its pencil decides is decided. Not traced: the eager direction selection and the coulomb stage
(host decisions on device values); their stacks are decided on the plans those stages build.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAY = Path(__file__).resolve().parent / "replays" / "run30k_plan_census.py"


def _census():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=16",
               PYTHONPATH=os.pathsep.join([*map(str, sorted((ROOT / "services").glob("*/src"))),
                                           str(ROOT / "src")]))
    run = subprocess.run([sys.executable, str(REPLAY)], env=env, capture_output=True, text=True,
                         timeout=1200, cwd=ROOT)
    lines = [line for line in run.stdout.splitlines() if line.startswith("RESULT ")]
    assert run.returncode == 0 and lines, run.stdout[-3000:] + run.stderr[-3000:]
    stages = [line for line in run.stdout.splitlines() if line.startswith("STAGE ")]
    warnings = [line for line in run.stderr.splitlines() if "memory over budget at distrib_la" in line]
    return json.loads(lines[-1][len("RESULT "):]), stages, warnings


def test_run30k_eigh_plans_decide_every_stack_in_their_runtime_rooms():
    out, stages, warnings = _census()
    assert out["refusals"] == [] and all(": ok" in s for s in stages), (out["refusals"], stages)
    info, census = out["info"], out["census"]
    # The rooms and prices the run logged.
    assert info["line"]["face_aggregate"] == 13682397440                       # 'face 13.68 GB'
    # The width's program compiled (XLA:CPU), below the byte formula's proposal (137.9 GB).
    assert 0 < info["constructor"]["program_gb"] < 137.9
    assert info["reduction"]["side"] == 48640
    assert info["reduction"]["room"].startswith("RoomExhausted(stage='shared-pole constructor reduction round'")
    # The stack the second probe refused, now on the whole mesh under the exhausted room.
    hvv = [c for c in census if c["shape"] == [1, 24320, 24320]]
    assert hvv and all(c["exhausted"] == "shared-pole constructor reduction round" for c in hvv)
    assert any(c["price"] for c in hvv)                                        # compiled
    assert any("(1, 24320, 24320)" in w and "over budget" in w for w in warnings)
    # The routes are the run's (every stack on the whole mesh); a stack over its room is
    # admitted by the one policy (a warning), never refused.
    assert {c["route"] for c in census} == {"scan"}, census
    assert len(census) >= 14
    # A missing room is refused where the plan is built, naming it.
    missing = info["missing_rooms"]
    assert "GATE distrib_la_room" in missing["room=0"] and "caller:" in missing["room=0"]
    assert "GATE shared_pole_eigh_room" in missing["room=None"]
    assert missing["no room"].startswith("TypeError")
