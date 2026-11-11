"""cuSOLVERMp LU: the non-square refusal and the planner's LU workspace term (CPU, no device).

The measured figures are runs/lu_workspace_pricing_20261111 (2x2 A100, complex128, nrhs = n;
each solve's mempool high-water equals the native query to 0.1 MB).
"""
import pytest

GB = 1e9


def test_non_square_lu_past_the_relabel_bound_is_refused():
    from distrib_la._cusolvermp import lu_layout
    assert lu_layout(4096, 2, 4) == (4096, 2048, 1024)          # a small tile stays one per rank
    with pytest.raises(ValueError, match=r"mesh 2x4 is not square.*2x2 \(4 ranks\) or 3x3 \(9 ranks\)"):
        lu_layout(16400, 2, 4)                                   # tile 8200 > 8192


def test_w_dyson_distributed_route_prices_the_lu_workspace():
    from distrib_la.workspace import lu_workspace_model
    from gw.stage_inventory import Deck, stage_prices
    # The model against the measured query: one tile per rank (n 16384 on 2x2, 20.71 GB) and
    # relabeled at the 60k Dyson tile (n 20008 on 2x2: tile 10004, block 244, 0.44 GB).
    assert lu_workspace_model(16384, 2, 2, 16).fixed / GB == pytest.approx(20.71, rel=0.05)
    assert 0.44 <= lu_workspace_model(20008, 2, 2, 16).fixed / GB <= 2.2
    # 60k slab (N_mu 60000 -> 60012 on 6x6): the planner's distributed W carries that term;
    # on 2x4 (no square relabel, tile past 8192) only the local route is left.
    deck = Deck(nk=4, n_parent=2, nq=2, ns=2, nb=9010, nb_qp=832, ng=229376, fft=(128, 128, 250),
                nmu=60000, kgrid=(2, 2, 1), sys_dim=2)
    dist = dict(stage_prices(deck, 36)["W"])["distributed"]
    assert dist.vendor.fixed == lu_workspace_model(60012, 6, 6, 16).fixed > 0
    assert [r for r, _ in stage_prices(deck, 8)["W"]] == ["local"]
