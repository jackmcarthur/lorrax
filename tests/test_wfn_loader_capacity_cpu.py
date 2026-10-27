"""Centroid scan proposals respect current room without double-charging borrowed psi."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('stream', [False, True])
def test_centroid_scan_respects_current_and_certified_room(monkeypatch, stream):
    from common import gpu_utils, wfn_transforms as owner
    from lxkit import memory
    from runtime.tiles import TILE_BYTES
    monkeypatch.setattr(gpu_utils, 'device_budget_bytes', lambda: None)
    room = [512 << 20]
    monkeypatch.setattr(memory, 'room', lambda **_k: room[0])
    memory.set_budget(1 << 30)
    ns, nr, copies, persistent = 4, 32768, 3, 32 << 20
    row = ns * nr * 16 * copies
    def choose(cert):
        return owner._centroid_fft_scan_chunk(128, 'ibz', cert, 4, nr, 32,
                                              ns, copies, persistent, False, stream)
    broad = choose(1 << 30)
    assert broad * row <= min(TILE_BYTES, room[0] - persistent)
    room[0] = persistent + 3 * row
    tighter = choose(1 << 30)
    assert tighter == 3 < broad
    # An explicitly smaller certificate remains an upper bound even with more live room.
    room[0] = 1 << 30
    assert choose(persistent + 2 * row) == 2
    room[0] = 0
    with pytest.warns(RuntimeWarning, match='over budget'):
        assert choose(1 << 30) == 1


def test_centroid_borrowed_gflat_is_already_in_room():
    from common.wfn_transforms import _centroid_resident_bytes
    mesh = SimpleNamespace(shape={'x': 2, 'y': 3}, axis_names=('x', 'y'))
    kwargs = dict(band_tile=24, bispinor=True, k_tile=2,
        loader=SimpleNamespace(ngkmax=17, nspinor=2, nkpts=3), max_parent_star=4,
        mesh_xy=mesh, n_rmu=11, n_rmu_padded=12, n_rtot=4096,
        nb_accum=24, nb_total=23, nk_accum=3, nk_tot=8, nspinor=4,
        p_band=6, parent_stream_active=True, return_parents=True, stream_tiles=True)
    own = _centroid_resident_bytes(psi_G_flat=None, **kwargs)
    borrowed = _centroid_resident_bytes(psi_G_flat=object(), **kwargs)
    assert own[:2] == borrowed[:2]
    assert own[2] > borrowed[2] > 0
    # New X/Y output and tile faces survive in both figures; only the new G-flat,
    # raw-parent/star and request phase storage is omitted for an existing operand.
    without_parent = _centroid_resident_bytes(
        psi_G_flat=object(), **dict(kwargs, return_parents=False))
    assert without_parent[2] < borrowed[2]
