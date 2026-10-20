"""Host-only actual-deck/provenance/head routing; no material/GPU admission."""
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def load(name):
    path = Path(__file__).with_name(name+".py")
    spec = importlib.util.spec_from_file_location(name+"_qgrid", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


probe, zero = load("native_frozen_map_probe"), load("sc750_td_callback")


def source(grid=(4, 4, 1), window="fejer"):
    frame = dict(frame="accepted_input", input_h_sha256="actual-H", deck_sha256="actual-deck",
                 qgrid=list(grid), reference_ready=True)
    policy = dict(accepted_input_frame=frame["frame"], input_h_sha256=frame["input_h_sha256"],
        accepted_deck_sha256=frame["deck_sha256"], qgrid=list(grid), response_window=window)
    return dict(frame=frame, meta=NS(kgrid=(8, 8, 1), qgrid=grid, cell_volume=2.),
        config=NS(qgrid_response_window=window), reference_policy=policy)


def test_explicit_only_grid_avoids_unrequested_fine_rebuild():
    assert probe.selected_qgrids(None) == ((8, 8, 1), (4, 4, 1))
    assert probe.selected_qgrids("4") == ((4, 4, 1),)
    assert probe.selected_qgrids("8") == ((8, 8, 1),)


@pytest.mark.parametrize("filename", ["corrected_protected160_750_q8.in",
    "corrected_protected160_750_q4_fejer_from_accepted_q8.in"])
def test_actual_native_producer_deck_binds_grid_window_and_hash(filename):
    path = Path(__file__).with_name("inputs")/filename
    grid = [8, 8, 1] if filename == "corrected_protected160_750_q8.in" else [4, 4, 1]
    frame = dict(frame="accepted_actual", input_h_sha256="H", kgrid=[8, 8, 1], qgrid=grid,
        deck_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    policy = probe.accepted_response_policy(frame, path)
    assert policy["qgrid"] == grid
    assert policy["response_window"] == ("sharp" if grid[0] == 8 else "fejer")
    frame["deck_sha256"] = "wrong"
    with pytest.raises(ValueError, match="producer deck SHA"):
        probe.accepted_response_policy(frame, path)


def test_coarse_callback_requires_own_accepted_frame_and_exact_policy():
    value = source()
    report = zero.source_response_policy(value)
    assert report["response_q_count"] == 16 and report["electronic_kgrid"] == [8, 8, 1]
    probe.validate_callback_reference(value["frame"], value["meta"].qgrid,
                                     value["reference_policy"], "fejer")
    for field in ("qgrid", "response_window", "accepted_input_frame", "input_h_sha256", "accepted_deck_sha256"):
        broken = copy.deepcopy(value)
        broken["reference_policy"][field] = "wrong"
        with pytest.raises(ValueError, match="GATE"):
            zero.source_response_policy(broken)
    broken = copy.deepcopy(value)
    broken["frame"]["qgrid"] = [8, 8, 1]
    with pytest.raises(ValueError, match="relabeled"):
        zero.source_response_policy(broken)
    with pytest.raises(ValueError, match="relabeled"):
        probe.validate_callback_reference(broken["frame"], (4, 4, 1), None, "fejer")


def test_fine_identity_window_does_not_require_invented_old_frame_fields():
    value = source((8, 8, 1), "sharp")
    del value["reference_policy"]
    report = zero.source_response_policy(value)
    assert report["response_q_count"] == 64 and report["effective_response_window"] == "identity"


def test_bare_HF_head_uses_direct_fine_cell_not_coarse_iteration_head():
    value, calls = source(), []
    def direct(z):
        calls.append(z)
        return NS(vc0=128.+0j)
    value.update(head_resolver=NS(direct_at=direct),
                 iteration_head=NS(at=lambda z: pytest.fail("must not use coarse correlation head")))
    assert zero.fine_bare_head_coefficient(value) == 1.
    assert calls == [0j]
    value["head_resolver"].direct_at = lambda z: NS(vc0=1.+1j)
    with pytest.raises(ValueError, match="bare_head"):
        zero.fine_bare_head_coefficient(value)
