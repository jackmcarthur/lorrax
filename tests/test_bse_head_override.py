"""The q=0 head override deck keys, and the GW/BSE parsing contract between them.

``vhead`` / ``whead_0freq`` pin the q=0 Coulomb head to an externally
supplied scalar -- in practice BerkeleyGW's, so that a cross-code
comparison is not confounded by the one term both codes treat specially.
The keys are read TWICE from the same ``cohsex.in``: once by the GW side
(``gw_config.HeadConfig``, via ``configparser`` with
``inline_comment_prefixes=('#',)``) and once by the BSE side
(``bse_io._parse_head_overrides``, a hand parser).

Two readers of one key is a standing hazard, and it had already bitten:
until 2026-08-09 the BSE reader did not strip inline comments and
swallowed a malformed value with a bare ``continue``.  On a deck written
``vhead = 3303.748102  # BGW`` the GW side overrode and the BSE side
silently used the restart's own head instead -- the two halves of one run
screening with two different q=0 heads, with nothing in the log to say
so.  These tests pin the contract that closes that.
"""
import configparser
import textwrap

import pytest

from bse.bse_io import _parse_head_overrides


# The GW side's parsing contract, quoted rather than described: this is
# exactly how gw_config builds its reader.
def _gw_side_parse(text):
    parser = configparser.ConfigParser(inline_comment_prefixes=('#',))
    parser.read_string(text)
    sec = parser["cohsex"]
    return (sec.get("vhead", fallback=None),
            sec.get("whead_0freq", fallback=None))


def _deck(tmp_path, body):
    p = tmp_path / "cohsex.in"
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_absent_keys_are_none(tmp_path):
    """Default is OFF: no key, no override, and the caller falls back."""
    f = _deck(tmp_path, """\
        [cohsex]
        centroids_file = centroids_frac_480_orbitclosed.txt
        """)
    assert _parse_head_overrides(f) == (None, None)


def test_plain_values_parse(tmp_path):
    f = _deck(tmp_path, """\
        [cohsex]
        vhead = 3303.748102
        whead_0freq = 150.395600
        """)
    vhead, whead0 = _parse_head_overrides(f)
    assert vhead == complex(3303.748102)
    assert whead0 == complex(150.395600)


@pytest.mark.parametrize("line,expected", [
    ("vhead = 3303.748102  # BGW's write_vcoul value", 3303.748102),
    ("vhead = 3303.748102# no space before the hash", 3303.748102),
    ("vhead = 3303.748102\t# tab then hash", 3303.748102),
])
def test_inline_comment_is_stripped(tmp_path, line, expected):
    """The regression proper.  A commented pin must still override.

    Before the fix these three lines each parsed to ``None`` here and to
    ``3303.748102`` on the GW side.
    """
    f = _deck(tmp_path, f"[cohsex]\n{line}\n")
    vhead, _ = _parse_head_overrides(f)
    assert vhead == complex(expected)


@pytest.mark.parametrize("body", [
    "[cohsex]\nvhead = 3303.748102  # BGW\n",
    "[cohsex]\nvhead = 3303.748102\nwhead_0freq = 150.3956 # W head\n",
    "[cohsex]\nvhead = 3303.748102\n",
    "[cohsex]\nwhead_0freq = 150.395600\n",
    "[cohsex]\ncentroids_file = c.txt\n",
])
def test_agrees_with_the_gw_side_reader(tmp_path, body):
    """The contract itself: both readers see the same value, or neither does.

    This is the assertion that matters.  Whatever either parser does with
    whitespace, comments or absence, a deck key that pins the head on one
    side of the run must pin it on the other -- a q=0 head that differs
    between the screening and the BSE assembly of a single run is not a
    physics choice anyone could have intended.
    """
    f = _deck(tmp_path, body)
    bse_v, bse_w = _parse_head_overrides(f)
    gw_v, gw_w = _gw_side_parse(body)

    for bse_val, gw_val in ((bse_v, gw_v), (bse_w, gw_w)):
        if gw_val is None:
            assert bse_val is None
        else:
            assert bse_val == complex(float(gw_val))


@pytest.mark.parametrize("bad", ["banana", "1.0d3", "3303.748102 150.0"])
def test_malformed_value_refuses_rather_than_falling_back(tmp_path, bad):
    """A validation knob that silently no-ops is worse than no knob.

    ``1.0d3`` is the interesting one: Fortran exponent notation is what a
    hand carrying a number over from a BerkeleyGW input would write, and
    it is exactly the case the old bare ``continue`` turned into a silent
    fallback to the restart's own head.
    """
    f = _deck(tmp_path, f"[cohsex]\nvhead = {bad}\n")
    with pytest.raises(ValueError, match="head override"):
        _parse_head_overrides(f)


def test_refusal_names_the_file_and_line(tmp_path):
    f = _deck(tmp_path, """\
        [cohsex]
        centroids_file = c.txt
        whead_0freq = not-a-number
        """)
    with pytest.raises(ValueError) as exc:
        _parse_head_overrides(f)
    assert "cohsex.in:3" in str(exc.value)
    assert "whead_0freq" in str(exc.value)


def test_missing_file_is_not_an_error(tmp_path):
    """Head resolution is optional; a caller with no deck gets None, not a raise."""
    assert _parse_head_overrides(None) == (None, None)
    assert _parse_head_overrides(str(tmp_path / "nope.in")) == (None, None)
