"""Contract tests for the immutable candidate-Gram profile wrapper."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _entry_module():
    path = Path(__file__).resolve().parents[1] / "tools" / \
        "profile_candidate_gram_entry.py"
    spec = importlib.util.spec_from_file_location("_gram_profile_entry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_wrapper_records_the_public_isdf_owner():
    module = _entry_module()
    widths = []
    extents = []
    sentinel = object()

    def tiled(destination, *, tile_width):
        del destination, tile_width
        return sentinel

    owner = SimpleNamespace(gram_q0_tiled_from_psi_sm=tiled)
    original = module._install_tiled_recorder(owner, widths, extents)
    destination = SimpleNamespace(shape=(588, 588))
    assert owner.gram_q0_tiled_from_psi_sm(
        destination, tile_width=256) is sentinel
    assert original is tiled
    assert widths == [256]
    assert extents == [588]
