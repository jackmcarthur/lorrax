"""The distributed W Dyson's price against its measured pool high-water (CPU, no device; ms).

``gw.w_isdf.dyson_route_prices`` (the admission's price, and ``gw.stage_inventory``'s W) for
the solve ``_get_w_solve_fn_distributed`` runs: V, χ, A, B and the LU jit's transposed A_T, B_T
beside cuSOLVERMp's workspace, or the A-build's q block where that is larger.  Measured on one
2x2 A100-80GB node (planner-60k-calibration, 2026-11-16, job 61436272) at the 60k MoS₂/hBN P = 36
tile: n 20004, 2 q, 10002 a rank (M 1.6 GB, as at 60k), the runtime's cuda_async pool,
cuSOLVERMp's stacked entry (the route a 60k system takes; a whole 20004 system would take the
one-device size route).  The 60k run itself (job 60432704, W 29.34 GB) ran the one-tile layout
main no longer takes there.
"""
import pytest

GB = 1e9
N, STACK, P = 20004, 2, 4
#: Pool high-water per rank: before the solve (V and χ) and its two phases.
MEASURED = {
    "relabeled": dict(room=None, base=6.40, build=18.50, lu=19.66),
    "one tile": dict(room=40e9, base=6.40, build=18.50, lu=49.99),
}


@pytest.mark.parametrize("layout", MEASURED)
def test_distributed_w_dyson_prices_its_measured_peak(layout):
    from lxkit import memory
    from gw.w_isdf import dyson_lu_workspace, dyson_route_prices
    m = MEASURED[layout]
    ws = dyson_lu_workspace(N, 2, 2, room=m["room"])
    _, dist = dyson_route_prices(N, STACK, P, memory.Term(), ws)
    priced = dist.bytes() / GB
    measured = max(m["build"], m["lu"])
    assert 0.9 <= measured / priced <= 1.0, (layout, measured, priced)
    # The phases' residents: 6 face stacks at the LU (V, chi, A, B, A_T, B_T), vendor beside.
    tile = 16.0 * N * N / P
    assert dist.operands.at(P) == pytest.approx(6 * STACK * tile)
    assert (3 * STACK + 6) * tile / GB >= m["build"]


def test_a_build_binds_a_one_q_stack():
    """One q: V, chi and A (3 tiles) plus the q block's 6 exceed the LU's 6, so the xla term
    carries the difference beyond the vendor workspace."""
    from lxkit import memory
    from gw.w_isdf import dyson_lu_workspace, dyson_route_prices
    ws = dyson_lu_workspace(N, 2, 2)
    _, dist = dyson_route_prices(N, 1, P, memory.Term(), ws)
    tile = 16.0 * N * N / P
    assert dist.bytes() == pytest.approx(max(9 * tile, 6 * tile + ws.at(P)))


def test_room_keeps_what_the_solve_holds():
    """The one-tile decision's room leaves out V, chi, A, B and every q's transposed pair but the
    one ``lu_one_tile_bytes`` prices: the 2x2 one-tile point fits 63.4 GB with its contexts."""
    from gw.w_isdf import dyson_room
    from distrib_la.workspace import lu_one_tile_bytes
    contexts = 5.58e9
    room = dyson_room(N, STACK, P, 63.4e9, contexts)
    assert lu_one_tile_bytes(N, 2, 2) <= room
    assert MEASURED["one tile"]["lu"] * GB + contexts <= 63.4e9
    assert room == int(63.4e9 - 16.0 * N * N * (6 * STACK - 2) / P - contexts)
