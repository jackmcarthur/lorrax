"""The internal screened head has independent placement and restart identity."""
from dataclasses import dataclass
from types import SimpleNamespace as NS

import numpy as np
import pytest


def test_static_head_identity_binds_resolved_scalar_geometry_and_policy(monkeypatch, tmp_path):
    from bse.w_ladder import ladder_static_head_identity, LADDER_INTERNAL_HEAD_POLICY
    from file_io import restart_bundle
    header = dict(grid=np.array([3, 3, 1]), head_vector=np.array([1., .8+0.1j]),
                  bare_head=31., screened_head=np.array([18.+0j]))
    monkeypatch.setattr(restart_bundle, "read_head_metadata", lambda _path: header)
    def identity(volume=20., deck=None):
        return ladder_static_head_identity("unused", deck, cell_volume=volume)
    original = identity()
    assert original["policy"] == LADDER_INTERNAL_HEAD_POLICY
    assert original["q0_head"] == "diagonal"
    assert original["q0_head_ry"] == 18./(20.*9)
    # Each of these changes the physical static operator independently of W0.
    for name, changed in (("screened_head", np.array([19.+0j])),
                          ("head_vector", np.array([1., .81+0.1j])),
                          ("grid", np.array([4, 3, 1]))):
        old = header[name]
        header[name] = changed
        assert identity() != original
        header[name] = old
    assert identity(volume=21.) != original
    deck = tmp_path / "cohsex.in"
    deck.write_text("whead_0freq = 23\n")
    overridden = identity(deck=str(deck))
    assert overridden["whead_ry_bohr3"] == [23., 0.]
    assert overridden["q0_head_ry"] == 23./(20.*9)
    header["screened_head"] = None
    with pytest.raises(ValueError, match="ladder_static_head_identity"):
        identity()


def test_static_rpa_rung_is_persisted_with_its_producer_diagram(monkeypatch):
    from gw import screening_bse as owner
    from gw import gw_output
    from gw.gw_config import HeadCorrection, ScreeningDiagrams
    @dataclass(frozen=True)
    class Screening:
        diagrams: object
    @dataclass(frozen=True)
    class Config:
        screening: object
        head: object
    config = Config(Screening(ScreeningDiagrams.W_BSE), NS(correction=HeadCorrection.OFF))
    static = np.ones((1, 2, 2))
    seen = []
    monkeypatch.setattr(owner, "_refuse_metallic_mean_field", lambda *a, **k: None)
    monkeypatch.setattr(owner, "_refuse_unusable_restart", lambda *a, **k: None)
    monkeypatch.setattr(owner, "_gate_w_or_refuse", lambda *a, **k: None)
    monkeypatch.setattr(owner, "_assert_restart_is_loadable", lambda *a, **k: None)
    monkeypatch.setattr(owner, "compute_static_w", lambda *a, **k: NS(unfold=lambda _m: static))
    monkeypatch.setattr(gw_output, "persist_w0_and_head", lambda *a, **k: seen.append(k["config"]))
    result = owner.prepare_ladder_restart(None, None, quad=None, e_ref=0.,
        sym=NS(), centroid_indices=[], config=config, meta=NS(kgrid=(1, 1, 1)),
        mesh_xy=None, tensors_filename="unused", head_resolver=None,
        print_fn=lambda *a: None)
    assert result is static
    assert config.screening.diagrams is ScreeningDiagrams.W_BSE
    assert len(seen) == 1 and seen[0].screening.diagrams is ScreeningDiagrams.W_RPA
    assert seen[0].head is config.head
