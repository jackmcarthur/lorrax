"""The shared-pole ledger admits beside what was live outside it at bind (the driver's arrays).

Leg 59998942 (run30k, 45fe62feb): every no-planner row sat on a 3.13-3.19 GB/rank floor the
ledger never counted, so ledger rows read under-priced against the pool (head sample terms
8.77 measured vs 6.60 priced: 5.64 beside the floor) and passivity reached 66.02 of 63.41 GB."""
from types import SimpleNamespace

import numpy as np

from gw.shared_pole_recipe import CapacityLedger

MESH = SimpleNamespace(shape={"x": 4, "y": 4})
META = SimpleNamespace(nk_tot=1, nspinor=2, n_rmu=29998)
BUDGET, BASELINE = 63_414_637_611, 3_130_000_000


def test_the_baseline_is_taken_from_every_admission_and_room(monkeypatch):
    prices = []
    from common import gpu_utils
    monkeypatch.setattr(gpu_utils, "record_stage_price", lambda stage, b: prices.append((stage, b)))
    bare = CapacityLedger(META, mesh_xy=MESH, device_budget_bytes=BUDGET)
    held = CapacityLedger(META, mesh_xy=MESH, device_budget_bytes=BUDGET, baseline_bytes=BASELINE)
    a = bare.preview(resident_bytes_per_rank=10**9, workspace_bytes_per_rank=0)
    b = held.preview(resident_bytes_per_rank=10**9, workspace_bytes_per_rank=0)
    assert a["available_device_bytes_per_rank"] - b["available_device_bytes_per_rank"] == BASELINE
    assert b["aggregate_bytes_per_rank"] == a["aggregate_bytes_per_rank"] == 10**9   # 3U scaling: ledger objects only
    assert b["baseline_bytes_per_rank"] == BASELINE
    assert bare.room_bytes_per_rank() - held.room_bytes_per_rank() == BASELINE
    # Over budget only beside the baseline: warned and admitted, status FAIL.
    over = held.preview(resident_bytes_per_rank=BUDGET - BASELINE + 1, workspace_bytes_per_rank=0)
    assert over["device_budget_status"] == "FAIL"
    # Prices are recorded as the pool measures them, the label naming the baseline.
    held.reserve("head:0", resident_bytes_per_rank=6_600_000_000, workspace_bytes_per_rank=0)
    bare.reserve("head:0", resident_bytes_per_rank=6_600_000_000, workspace_bytes_per_rank=0)
    assert prices == [("shared-pole ledger + 3.13 GB baseline", 6_600_000_000 + BASELINE),
                      ("shared-pole ledger", 6_600_000_000)]
