"""``ladder_probe_chunk``: the facade threads w_ladder's memory knob.

Register row closed here (sandbox KNOWN_LORRAX_ISSUES 2026-08-20): the GW
ladder facade (``gw.screening_bse._ladder_wedge``) called
``bse.w_ladder.compute_wc_qwedge`` without threading its public
``probe_chunk`` argument, so the bounded-memory probe control was
unreachable from any deck and the fully relativistic LiF 666-centroid
solve attempted one 77.83-GiB block allocation (JID 57288835) while the
same deck at probe_chunk=64 passed every production-W gate
(JID 57280453).

The deck key is ``ladder_probe_chunk`` (ScreeningConfig, default 0 =
whole padded basis, bit-identical).  These cells drive the REAL facade
with a recording stand-in for ``compute_wc_qwedge`` — the seam under
test is the facade's call, not the ladder solve.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _import_facade():
    pytest.importorskip("jax")
    try:
        import gw.screening_bse as sb
    except RuntimeError as exc:  # FFI host library not built here
        if "FFI" in str(exc) or "liblorrax" in str(exc):
            pytest.skip(f"FFI host library unavailable: {exc}")
        raise
    return sb


def test_config_key_default_and_refusal():
    pytest.importorskip("jax")
    from gw.gw_config import ScreeningConfig, _DEFAULTS

    assert _DEFAULTS["ladder_probe_chunk"] == 0
    kw = dict(method="minimax", occ_broadening_ev=0.0,
              minimax_target_error=1e-6, minimax_max_nodes=64,
              regenerate_minimax_tables=False,
              minimax_energy_reference="midgap")
    assert ScreeningConfig(**kw).ladder_probe_chunk == 0
    with pytest.raises(ValueError, match="ladder_probe_chunk"):
        ScreeningConfig(ladder_probe_chunk=-1, **kw)


def _drive_facade(monkeypatch, *, deck_chunk, p_y):
    sb = _import_facade()
    import bse.w_ladder as w_ladder

    seen = {}

    def _fake_wedge(restart_path, z, mesh_xy, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            gmres_resid=np.full((1, 1, 4), 1e-9),
            gmres_iters=np.full((1, 1, 4), 3, dtype=np.int64),
            wc=None)

    monkeypatch.setattr(w_ladder, "compute_wc_qwedge", _fake_wedge)
    config = SimpleNamespace(
        screening=SimpleNamespace(ladder_probe_chunk=deck_chunk),
        head=SimpleNamespace(correction=None),
        input_file="deck.in")
    mesh = SimpleNamespace(
        devices=np.zeros((1, p_y)), axis_names=("x", "y"),
        shape={"x": 1, "y": p_y})
    lines = []
    sb._ladder_wedge("restart.h5", [0.0 + 0.0j], mesh,
                     input_file="deck.in", config=config,
                     print_fn=lines.append)
    return seen, lines


def test_facade_threads_and_rounds_the_deck_chunk(monkeypatch):
    """probe_chunk=10 on a 'y'=4 mesh arrives at the kernel as 12 (the
    reduce-scatter tiles the probe axis over 'y'), and the rounding is
    announced.  Pre-fix red twin: the facade passed no probe_chunk at
    all, so ``seen`` would miss the key entirely."""
    seen, lines = _drive_facade(monkeypatch, deck_chunk=10, p_y=4)
    assert "probe_chunk" in seen, "facade still does not thread probe_chunk"
    assert seen["probe_chunk"] == 12
    assert any("ladder_probe_chunk=10" in ln and "12" in ln
               for ln in lines), lines


def test_default_zero_keeps_the_whole_basis_block(monkeypatch):
    """ladder_probe_chunk=0 (and a config with no screening section at
    all) must reach the kernel as probe_chunk=None — the historical
    whole-padded-basis block, bit-identical for every existing deck."""
    seen, _ = _drive_facade(monkeypatch, deck_chunk=0, p_y=4)
    assert seen["probe_chunk"] is None
