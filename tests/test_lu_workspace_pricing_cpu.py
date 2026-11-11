"""cuSOLVERMp LU on CUDA meshes, priced off the device (CPU): the layout its room admits, the
planner's LU workspace and context terms, and the default route's whole-matrix cap.

The measured figures are runs/lu_workspace_pricing_20261111 (2x2 A100, complex128, nrhs = n;
each solve's mempool high-water equals the native query to 0.1 MB) and the distrib_la shape
audit (2026-11-11: tile 8191 on 3x3, 27.2 GB/rank).
"""
from types import SimpleNamespace

import pytest

GB = 1e9


def test_lu_layout_follows_the_room():
    from distrib_la._cusolvermp import lu_layout
    from distrib_la.workspace import lu_workspace_model
    big = int(40 * GB)
    # Either side of the former fixed 8192 bound on 2x2: one tile where it fits the room,
    # relabeled where it does not (no room: the 2 GiB getrs room), on both sides alike.
    for n in (16382, 16396):
        assert lu_layout(n, 2, 2, big)[1] == n // 2
        assert lu_layout(n, 2, 2)[1] <= 256
    assert lu_workspace_model(16382, 2, 2, room=big).fixed / GB == pytest.approx(20.7, rel=0.05)
    assert lu_workspace_model(16382, 2, 2).fixed / GB == pytest.approx(2.147, rel=0.01)
    # 3x3 at tile 8191: one tile priced over the 27.2 GB measured, by under 25 %.
    assert 1.0 <= lu_workspace_model(3 * 8191, 3, 3, room=big).fixed / (27.2 * GB) <= 1.25
    assert lu_layout(4 * 1000, 4, 4, 1 << 62)[1] <= 256       # 4x4 and up always relabel
    # A non-square mesh has one tile only: run where it fits, refused by name where not.
    assert lu_layout(4096, 2, 4) == (4096, 2048, 1024)
    assert lu_layout(16400, 2, 4, 1 << 62) == (16400, 8200, 4100)
    with pytest.raises(ValueError, match=r"mesh 2x4 is not square.*2x2 \(4 ranks\) or 3x3 \(9 ranks\)"):
        lu_layout(16400, 2, 4)


def test_w_dyson_distributed_route_prices_lu_workspace_and_both_contexts():
    from distrib_la.workspace import GETRS_DEFAULT_ROOM
    from gw.stage_inventory import CONTEXT, SECOND_CONTEXT, Deck, stage_prices
    # 60k slab (N_mu 60000 -> 60012 on 6x6, relabeled): the planner's distributed W carries the
    # getrs-room floor and both cuSOLVERMp grids' contexts; on 2x4 only the local route is left.
    deck = Deck(nk=4, n_parent=2, nq=2, ns=2, nb=9010, nb_qp=832, ng=229376, fft=(128, 128, 250),
                nmu=60000, kgrid=(2, 2, 1), sys_dim=2)
    dist = dict(stage_prices(deck, 36)["W"])["distributed"]
    assert dist.vendor.fixed == GETRS_DEFAULT_ROOM
    assert dist.outside.fixed == pytest.approx(CONTEXT + SECOND_CONTEXT)
    assert [r for r, _ in stage_prices(deck, 8)["W"]] == ["local"]


def test_room_leg_default_route_holds_no_matrix_past_the_budget(monkeypatch):
    import importlib
    from lxkit import memory
    dla_plan = importlib.import_module("distrib_la.plan")       # the package exports plan()
    resolve = importlib.import_module("distrib_la.resolve")
    monkeypatch.setattr(resolve, "mesh_platform", lambda mesh: "CUDA")
    monkeypatch.setattr(memory, "run", lambda: {"budget": int(60 * GB)})
    mesh = SimpleNamespace(shape={"x": 2, "y": 2})
    bound = resolve.one_device_max_n("solve_lu", "CUDA", int(60 * GB))
    # Route (c) holds each matrix whole on one device: kept up to what the budget holds, the
    # library's own route past it.
    assert not dla_plan._size_routed_library("solve_lu", "distributed", mesh, bound, "cusolvermp")
    assert dla_plan._size_routed_library("solve_lu", "distributed", mesh, bound + 1, "cusolvermp")
