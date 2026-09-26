"""The chunk planners that size from the run budget's room (``common.gpu_utils``).

Each planner is checked at a large room (its ceiling), a small room (its
floor) and in between, with ``device_room_bytes`` pinned, so the arithmetic
is tested without a device.  Red twins: a planner that ignores the room would
return the same size at both ends.
"""
from types import SimpleNamespace

import pytest

import common.gpu_utils as gpu_utils


class _Mesh(SimpleNamespace):
    def __init__(self, px, py):
        super().__init__(shape={'x': px, 'y': py}, devices=SimpleNamespace(size=px * py),
                         axis_names=('x', 'y'))


@pytest.fixture
def room(monkeypatch):
    state = {'room': 0, 'budget': 40e9}
    monkeypatch.setattr(gpu_utils, 'device_room_bytes', lambda **_: int(state['room']))
    monkeypatch.setattr(gpu_utils, 'device_budget_bytes', lambda: float(state['budget']))
    monkeypatch.setattr(gpu_utils, 'record_stage_price', lambda *a, **k: None)
    return state


def test_head_wing_mu_block_follows_the_room(room):
    from gw.qsgw_head import head_wing_mu_block, _HEAD_WING_MU_MIN
    kw = dict(mu_local=1544, nk=36, ns=4, nb_full=496, n_ends=2)
    per_mu = 16 * 36 * 4 * 496 * 2
    room['room'] = 1e12
    assert head_wing_mu_block(**kw) == 1544
    room['room'] = 0
    assert head_wing_mu_block(**kw) == _HEAD_WING_MU_MIN
    room['room'] = 2 * 100 * per_mu            # half the room holds 100 centroids
    assert head_wing_mu_block(**kw) == 100
    assert head_wing_mu_block(**{**kw, 'mu_local': 8}) == 8


def test_green_panel_bytes_is_one_tile_bounded_by_the_room():
    from gw.greens_function_kernel import green_panel_bytes
    mesh = _Mesh(2, 2)
    tile = 16 * 7 * (400 // 2) * (400 // 2)
    column = 16 * 7 * (400 // 2 + 400 // 2)
    assert green_panel_bytes(n_rows=7, m=400, n=400, mesh=mesh) == tile
    assert green_panel_bytes(n_rows=7, m=400, n=400, mesh=mesh, room=10 * tile) == tile
    assert green_panel_bytes(n_rows=7, m=400, n=400, mesh=mesh, room=tile // 3) == tile // 3
    assert green_panel_bytes(n_rows=7, m=400, n=400, mesh=mesh, room=-5) == column


def test_moment_q_width_takes_half_the_smaller_room(room):
    from gw.response_bank import moment_q_width
    face = 16 * 1000 ** 2 // 4
    ledger = SimpleNamespace(live_stages=(), room_bytes_per_rank=lambda live: 10 ** 15)
    room['room'] = 2 * (16 + 8 * 5) * face     # half holds 5 q at 8 faces + 16
    assert moment_q_width(ledger, n_q=59, face_bytes=face, per_q=8) == 5
    room['room'] = 10 ** 15
    assert moment_q_width(ledger, n_q=59, face_bytes=face, per_q=8) == 59
    ledger.room_bytes_per_rank = lambda live: 0
    assert moment_q_width(ledger, n_q=59, face_bytes=face, per_q=8) == 1


def test_build_cq_q_chunk_prices_the_accumulator_and_psi(room):
    from bse.vq_interp import build_cq_q_chunk
    mesh = _Mesh(2, 2)
    kw = dict(nq=64, nb=40, ns=2, n_mu=600, n_r=64, mesh_xy=mesh)
    face = 16 * 2 ** 2 * 600 ** 2 / 4
    per_q = 2 * 16 * 40 * 2 * 600 + face
    room['room'] = 2 * 64 * face + 10 * per_q
    assert build_cq_q_chunk(**kw) == 10
    room['room'] = 1e15
    assert build_cq_q_chunk(**kw) == 64
    room['room'] = 0
    assert build_cq_q_chunk(**kw) == 1


def test_loader_band_chunk_spans_the_floor_to_all_bands(room):
    from gw.gflat_memory_model import loader_band_chunk
    kw = dict(nb=496, nk=36, ns=4, ngkmax=20000, n_rmu=3088, mesh_xy=_Mesh(2, 2),
              p_band=2, floor=16)
    room['room'] = 1e15
    assert loader_band_chunk(**kw) == 496
    room['room'] = 0
    assert loader_band_chunk(**kw) == 16
    room['room'] = 2e9
    b = loader_band_chunk(**kw)
    assert 16 < b < 496 and b % 2 == 0
    assert -(-496 // b) * b - 496 < b      # the least band padding for its tile count
