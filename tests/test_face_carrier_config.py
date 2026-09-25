"""The one ψ carrier: the retired ``low_mem_bands`` key and the Gij refusal.

ψ is always stored band-distributed (``docs/architecture/memory-model.md``,
ψ carriers).  The former ``low_mem_bands`` deck key refuses by name, and an
explicit dense ``Gij`` operand refuses before any Σ kernel runs.
"""
import pathlib

import pytest

from gw.gw_config import LorraxConfig, refuse_explicit_gij

_REPO = pathlib.Path(__file__).resolve().parents[1]

_BASE = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""


def _config(tmp_path, extra=""):
    path = tmp_path / "face_carrier.in"
    path.write_text(_BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: None)


@pytest.mark.parametrize("value", ["true", "false"])
def test_the_retired_low_mem_bands_key_refuses_by_name(tmp_path, value):
    """Either value refuses and names the new behaviour; nothing is ignored."""
    with pytest.raises(ValueError) as exc:
        _config(tmp_path, f"low_mem_bands = {value}\n")
    message = str(exc.value)
    assert "'low_mem_bands' is retired" in message
    assert "band-distributed" in message and "Remove the key" in message


def test_a_deck_without_the_key_parses(tmp_path):
    """The positive twin: the same deck minus the key resolves."""
    assert _config(tmp_path, "head_correction = off\n").memory.per_device_gb == 4.0


def test_an_explicit_gij_refuses_with_all_message_parts():
    """RED TWIN: a live Gij operand refuses, naming the rule id."""
    with pytest.raises(ValueError) as exc:
        refuse_explicit_gij("not-actually-an-array")
    message = str(exc.value)
    assert "GATE explicit_gij_unported" in message
    for part in ("got:", "want:", "fix:", "why:", "doc:"):
        assert part in message, f"Gij refusal is missing '{part}'"


def test_gij_none_is_the_positive_twin():
    """``Gij = None`` is every production call; it must never raise."""
    refuse_explicit_gij(None)


def test_compute_sigma_xc_checks_the_gij_row_before_any_kernel():
    """The Gij row runs at the top of the dispatch, before any kernel import
    or Gij-dependent allocation."""
    import inspect
    from gw import sigma_dispatch

    entry = inspect.getsource(sigma_dispatch.compute_sigma_xc)
    src = inspect.getsource(sigma_dispatch._validate_sigma_stage)
    assert src.index("refuse_unimplemented_compute_mode(") < src.index(
        "refuse_explicit_gij(")
    assert entry.index("_validate_sigma_stage(") < entry.index(
        "_static_sigma_channels("), (
        "compute_sigma_xc must validate the Gij row before allocating channels")


def test_the_docs_name_the_live_gij_refusal():
    """The input reference names the one live carrier refusal."""
    page = (_REPO / "docs" / "input_reference.md").read_text()
    assert "explicit_gij_unported" in page
