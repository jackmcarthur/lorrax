"""The run30k Dyson LU (n 30000, complex128, 16 A100s on a 4x4 mesh) at the layout that ran it in
22 s per dispatch (try 8, e57a2071a, 2026-10-18): cuSOLVERMp past the one_device bound, block-cyclic
at block 250 (relabeled), the right-hand sides (nrhs = n) in 250-wide blocks, 30 per rank, which
getrs chunks under its 2 GiB workspace cap. One tile per rank (7500) ran getrf/getrs in four
serialized panel steps across nodes (try 5). Pure functions: no GPU and no context."""
import types

import numpy as np
import pytest

from distrib_la import _cusolvermp as cm
from distrib_la import resolve
from lxkit.memory import bucket as room_bucket

N, P = 30000, 4


@pytest.mark.parametrize("room_gb", [16.11, 23.42, 63.41])
def test_run30k_dyson_lu_is_cusolvermp(room_gb):
    bound = resolve.one_device_max_n("solve_lu", "CUDA", int(room_gb * 1e9))
    assert bound is not None and bound < N


def test_run30k_dyson_lu_layout_is_block_250_with_chunked_rhs():
    mb, nb = cm.lu_blocks(N, P, P)
    assert (mb, nb) == (250, 250)
    nb_b = cm.rhs_block(N, N, P, P, mb)
    assert nb_b == 250
    assert (N // P) // nb_b == 30          # local RHS blocks: getrs can chunk them


def test_one_tile_below_the_relabel_bound_on_2x2():
    assert cm.lu_blocks(312, 2, 2) == (156, 156)


def test_the_lu_layout_is_announced_once_per_program(capsys, monkeypatch):
    mesh = types.SimpleNamespace(shape={"x": P, "y": P}, devices=np.empty((P, P)))
    monkeypatch.setattr(cm, "_validate_mesh", lambda m: (P, P))
    cm._announce_lu("solve_lu", mesh, 1, N, N, 250, 250, 250, 250)
    out = capsys.readouterr().out
    assert "A mb 250 nb 250 (relabeled block-cyclic)" in out and "30 RHS block(s) per rank" in out


@pytest.mark.parametrize("n, measured", [(30000, 680619408), (24000, 540338384), (16384, 20708678656),
                                         (4096, 1402455040)])
def test_the_lu_workspace_price_is_within_the_measured_band_on_2x2(n, measured):
    # lrx_lu_workspace_bytes on 2x2 A100 (2026-10-19): getrf + getrs (nrhs = n) + pivots.
    import distrib_la
    mesh = types.SimpleNamespace(shape={"x": 2, "y": 2})
    price = distrib_la.lu_workspace_bytes(mesh, n, np.complex128)
    assert measured <= price <= 1.25 * measured


def test_the_run30k_dyson_lu_workspace_is_priced_from_shapes_alone(monkeypatch):
    import distrib_la
    from distrib_la import workspace
    monkeypatch.setattr(workspace, "_vendor_query", lambda *a: pytest.fail("vendor query at bind"))
    mesh = types.SimpleNamespace(shape={"x": P, "y": P})
    assert distrib_la.lu_workspace_bytes(mesh, N, np.complex128) == int(11.2 * (7500 + 250) * 250 * 16)


def test_room_bucket_floors_to_gib_and_keeps_a_sub_gib_room():
    gib = 1 << 30
    assert room_bucket(16.11e9) == 15 * gib
    assert room_bucket(gib - 1) == gib - 1
    assert room_bucket(-5) == 0
