"""The restart file's Coulomb-kernel provenance stamp.

WHY THIS EXISTS.  A ``restart = true`` run reuses ``V_qmunu`` verbatim and
never re-runs ``compute_V_q``, and until this stamp the file recorded no
Coulomb-kernel policy anywhere — not ``mc_average_vcoul_body``, not the
mini-BZ placement, not ``bare_coulomb_cutoff``, not the BGW vcoul overlay.
So any averaging-policy change was inherited silently by every existing
restart, with the band-window guard, the n_rmu guard, the centroid md5
guard and the W0_ready guard all passing.  Same defect class as the
band-window bug (job 7874375), one layer over.

The stamp WARNS rather than refuses, and these tests pin that asymmetry:
a mismatch must produce a message that names the differing keys and both
values, and an unstamped legacy file must produce a message that says
"not stamped" rather than "matches".  Conflating those two is the original
defect re-created one level up.

numpy + h5py only; no mesh, no SlabIO, no deck.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

h5py = pytest.importorskip("h5py")

from file_io.tagged_arrays import (  # noqa: E402
    COULOMB_POLICY_DATASET,
    COULOMB_POLICY_KEYS,
    compare_coulomb_policy,
    coulomb_policy_from_config,
    describe_coulomb_policy_match,
    describe_coulomb_policy_stamp,
    format_coulomb_policy,
    parse_coulomb_policy,
    read_coulomb_policy_from_h5,
)


def _cfg(**over):
    head = dict(
        mc_average_vcoul_body=True,
        mc_average_placement="off",
        mc_average_placement_vcoul=None,
        head_minibz_average=False,
        bare_coulomb_cutoff=None,
        use_bgw_vcoul=False,
        bgw_vcoul_file=None,
        bispinor_tt_head_correction=False,
    )
    head.update(over)
    return SimpleNamespace(head=SimpleNamespace(**head))


def _meta(sys_dim=3):
    return SimpleNamespace(sys_dim=sys_dim)


def _write_stamp(path, text):
    with h5py.File(path, "w") as f:
        f.create_dataset(COULOMB_POLICY_DATASET,
                         data=np.asarray(text.encode("utf-8"), dtype="S"))


# ---------------------------------------------------------------------------
# Format round trip
# ---------------------------------------------------------------------------

def test_every_policy_key_is_stamped():
    """The stamp covers every key that can change v(q+G) or its placement."""
    pol = coulomb_policy_from_config(_cfg(), _meta())
    assert set(pol) == set(COULOMB_POLICY_KEYS)
    for k in ("mc_average_vcoul_body", "mc_average_placement",
              "bare_coulomb_cutoff", "use_bgw_vcoul",
              "bispinor_tt_head_correction", "sys_dim"):
        assert k in pol


def test_tt_head_correction_is_part_of_canonical_policy_identity():
    off = format_coulomb_policy(coulomb_policy_from_config(
        _cfg(bispinor_tt_head_correction=False), _meta()))
    on = format_coulomb_policy(coulomb_policy_from_config(
        _cfg(bispinor_tt_head_correction=True), _meta()))
    assert off != on
    assert "bispinor_tt_head_correction=false" in off
    assert "bispinor_tt_head_correction=true" in on


def test_format_parse_round_trip():
    pol = coulomb_policy_from_config(
        _cfg(mc_average_placement="bgw", bare_coulomb_cutoff=12.5), _meta())
    text = format_coulomb_policy(pol)
    assert text.startswith("v1;")
    assert "mc_average_placement=bgw" in text
    back = parse_coulomb_policy(text)
    assert back == pol


def test_bools_and_none_are_stamped_unambiguously():
    """``False`` and ``None`` must not collapse onto the same string.

    ``mc_average_vcoul_body = false`` is a decision; ``bare_coulomb_cutoff``
    unset is the absence of one, and a stamp that renders both as the empty
    string cannot tell a reader which was which.
    """
    pol = coulomb_policy_from_config(
        _cfg(mc_average_vcoul_body=False, bare_coulomb_cutoff=None), _meta())
    assert pol["mc_average_vcoul_body"] == "false"
    assert pol["bare_coulomb_cutoff"] == ""


def test_parse_of_absent_or_empty_is_none():
    assert parse_coulomb_policy(None) is None
    assert parse_coulomb_policy("") is None
    assert parse_coulomb_policy(b"") is None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def test_match_reports_no_differences():
    pol = coulomb_policy_from_config(_cfg(), _meta())
    assert compare_coulomb_policy(pol, pol) == []


def test_mismatch_names_the_key_and_both_values():
    a = coulomb_policy_from_config(_cfg(mc_average_placement="off"), _meta())
    b = coulomb_policy_from_config(_cfg(mc_average_placement="bgw"), _meta())
    diffs = compare_coulomb_policy(a, b)
    assert len(diffs) == 1
    key, stamped, running = diffs[0]
    assert key == "mc_average_placement"
    assert (stamped, running) == ("off", "bgw")


def test_unstamped_is_not_a_mismatch():
    """Absence of evidence is not evidence of a difference.

    A legacy file must not be reported as a MISMATCH — the caller has to be
    able to say "not stamped" in its own words, which it cannot do if the
    comparison has already invented differences against every key.
    """
    running = coulomb_policy_from_config(_cfg(), _meta())
    assert compare_coulomb_policy(None, running) == []


def test_unknown_key_from_a_newer_writer_is_reported_not_crashed():
    stamped = parse_coulomb_policy("v1;mc_average_placement=bgw;future_knob=7")
    running = coulomb_policy_from_config(_cfg(), _meta())
    diffs = dict((k, (a, b)) for k, a, b in
                 compare_coulomb_policy(stamped, running))
    assert "future_knob" in diffs
    assert diffs["future_knob"][1] == "<absent>"


# ---------------------------------------------------------------------------
# File-level disclosure
# ---------------------------------------------------------------------------

def test_read_from_file_round_trips(tmp_path):
    pol = coulomb_policy_from_config(
        _cfg(mc_average_placement="bgw"), _meta())
    path = tmp_path / "restart.h5"
    _write_stamp(path, format_coulomb_policy(pol))
    assert read_coulomb_policy_from_h5(str(path)) == pol


def test_missing_dataset_and_missing_file_both_read_as_none(tmp_path):
    path = tmp_path / "nostamp.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("V_qmunu", data=np.zeros((1, 2, 2)))
    assert read_coulomb_policy_from_h5(str(path)) is None
    assert read_coulomb_policy_from_h5(str(tmp_path / "absent.h5")) is None


def test_describe_match_says_matches(tmp_path):
    cfg, meta = _cfg(), _meta()
    path = tmp_path / "r.h5"
    _write_stamp(path, format_coulomb_policy(
        coulomb_policy_from_config(cfg, meta)))
    line = describe_coulomb_policy_match(str(path), cfg, meta)
    assert "matches" in line
    assert "WARNING" not in line


def test_describe_mismatch_is_loud_and_specific(tmp_path):
    path = tmp_path / "r.h5"
    _write_stamp(path, format_coulomb_policy(coulomb_policy_from_config(
        _cfg(mc_average_placement="off", mc_average_vcoul_body=True),
        _meta())))
    line = describe_coulomb_policy_match(
        str(path), _cfg(mc_average_placement="bgw"), _meta())
    assert "WARNING" in line
    assert "MISMATCH" in line
    assert "mc_average_placement" in line
    assert "'off'" in line and "'bgw'" in line
    # It must also say WHY the other guards will not catch this.
    assert "verbatim" in line


def test_describe_legacy_says_not_stamped(tmp_path):
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("V_qmunu", data=np.zeros((1, 2, 2)))
    line = describe_coulomb_policy_match(str(path), _cfg(), _meta())
    assert "NOT STAMPED" in line
    assert "matches" not in line
    assert "MISMATCH" not in line


def test_consumer_side_stamp_line(tmp_path):
    """The BSE reads W and owes the same disclosure, without a verdict."""
    path = tmp_path / "r.h5"
    _write_stamp(path, format_coulomb_policy(coulomb_policy_from_config(
        _cfg(mc_average_placement="bgw"), _meta())))
    line = describe_coulomb_policy_stamp(str(path))
    assert "mc_average_placement=bgw" in line
    assert "MISMATCH" not in line

    bare = tmp_path / "legacy.h5"
    with h5py.File(bare, "w") as f:
        f.create_dataset("V_qmunu", data=np.zeros((1, 2, 2)))
    assert "NOT STAMPED" in describe_coulomb_policy_stamp(str(bare))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
