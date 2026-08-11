"""``zeta_nband``: the ζ-fit window stops being the χ0/Σ band-sum top.

WHY THE KEY EXISTS.  ``nband`` served two unrelated jobs.  It is ``b4``, the
top of the χ0/Σ band sum, and it was also the top of the band window the ISDF ζ
fit ran on.  Those want opposite things.  The fit wants a NARROW window,
because the per-Q ζ refit behind a dense exciton band path reaches its target Q
through an htransform Galerkin representation whose rank bound is
``n_μ·n_s ≥ nk·nb`` — on the Si 4×4×4 / 2628-centroid parent ``build_fH_R``
reads 3.44e-07 at nb 52 against a 1.0e-06 cap and 3.47e-06 at nb 60, so 52 is
the capacity point.  The band sum wants a WIDE one.  With one key for both,
narrowing the fit dragged ``ncond`` down with it (``BandSlices`` requires
b3 ≤ b4) and truncated the sum by eight bands: 222 meV median over the 4v8c
window, 48 meV in the direct gap, none of it a ζ-basis effect
(``tests/known_failures/2026-08-11-narrowed-zeta-window-clears-fh-and-the-tile-\
null-still-refuses.md`` §3).

THE THREE THINGS THIS FILE HAS TO HOLD.

1. **DEFAULT IDENTITY.**  A deck that does not name the key gets the ranges it
   always got — ``(b0, b3)`` and ``(b1, b4)``, the PADDED ``b4`` included.  Any
   other outcome silently re-fits every ζ in the tree.  ``zeta_nband = nband``
   collapses to "unset" at parse time for the same reason: it must not un-pad
   ``b4`` through a redundant statement of the default.
2. **ILLEGAL EDGES REFUSE BY NAME.**  The new edge is checked STRICT while the
   ``nband``/``ncond`` edges stay ``snap`` — not an inconsistency but the point
   of the grandfather clause: existing decks sit on edges chosen before the
   check existed, and naming ``zeta_nband`` is a brand-new explicit request.
   On the Si SOC deck 52 is legal by 6.870 meV and 56 is not (0.259 meV, a
   4-fold irrep at k=0); every odd edge splits a Kramers pair outright.
3. **THE DECOUPLING REACHES THE FIT.**  A key that parsed and was read by
   nobody is the failure mode ``AGENT_PREAMBLE``'s A/B rule names.  ``fit_zeta``
   must take its two ranges from this resolver, and the fit + the provenance
   stamp must both see the narrowed window while ``band_slices`` — χ0, Σ,
   psi_full_y — keeps ``b4``.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pytest

SRC_INIT = os.path.join(os.path.dirname(__file__), "..", "src", "gw",
                        "gw_init.py")
RY_PER_MEV = 1.0 / 13605.693122994


def _slices(b0=0, b1=0, b2=8, b3=60, b4=60):
    from gw.wavefunction_bundle import BandSlices
    return BandSlices.from_band_edges(b0, b1, b2, b3, b4)


def _init():
    pytest.importorskip("jax")
    from gw import gw_init
    return gw_init


def _soc_spectrum(nb=62, nk=8, pair_gap_mev=100.0, tight=None):
    """A spin-orbit spectrum: Kramers pairs, so every ODD edge splits one.

    ``tight`` maps an EVEN edge to the gap (meV) that should sit just below
    it, which is how a 4- or 6-fold irrep shows up: two adjacent Kramers pairs
    nearly coincident.  Returned as ``(nspin, nk, nb)`` — the shape
    ``wfn.energies`` carries.
    """
    tight = dict(tight or {})
    e = np.zeros(nb, dtype=np.float64)
    for b in range(1, nb):
        if b % 2 == 1:
            step = 0.0                       # inside a Kramers pair
        else:
            step = tight.get(b, pair_gap_mev) * RY_PER_MEV
        e[b] = e[b - 1] + step
    return np.tile(e, (1, nk, 1))


# ---------------------------------------------------------------------------
# (1) DEFAULT IDENTITY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("edges", [
    (0, 0, 8, 60, 60),          # the Si 4x4x4 production deck
    (0, 4, 8, 40, 64),          # b1 != b0, b3 != b4, b4 PADDED past nband
])
def test_unset_reproduces_the_historical_ranges_exactly(edges):
    """The bit-identity claim, stated as the identity it is: with the key
    unset the resolver returns ``(b0, b3)`` and ``(b1, b4)`` and nothing has
    happened."""
    bs = _slices(*edges)
    left, right = _init().zeta_fit_band_ranges(bs, None, log=lambda *_: None)
    assert left == (bs.b0, bs.b3)
    assert right == (bs.b1, bs.b4)


def test_the_padded_b4_is_passed_through_untouched():
    """``b4`` is ``round_up(nband, world_size)``.  A resolver that reached for
    the user's ``nband`` instead would silently drop the pad bands out of the
    ζ fit on every run whose world size does not divide nband."""
    bs = _slices(b3=60, b4=64)          # nband=60 at world 4 -> b4 = 64
    _, right = _init().zeta_fit_band_ranges(bs, None, log=lambda *_: None)
    assert right[1] == 64


def test_zeta_nband_equal_to_nband_collapses_to_unset_at_parse_time(tmp_path):
    """A deck stating the default explicitly must take the DEFAULT path, pad
    and all — otherwise ``zeta_nband = nband`` is a different calculation from
    omitting the line, which is the opposite of what "default" means."""
    pytest.importorskip("jax")
    from gw.gw_config import LorraxConfig, read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text("[cohsex]\nnval = 8\nncond = 52\nnband = 60\n"
                    "zeta_nband = 60\n")
    params = read_lorrax_input(str(deck))
    assert params["zeta_nband"] == 60          # the deck said what it said
    cfg = LorraxConfig.from_input_file(str(deck), print_fn=lambda *_: None)
    assert cfg.zeta_nband is None, (
        "a redundant zeta_nband = nband must resolve to 'follow nband', or it "
        "un-pads b4 and re-fits every zeta in the tree")


def test_an_unset_key_is_none_and_a_set_one_is_an_int(tmp_path):
    pytest.importorskip("jax")
    from gw.gw_config import LorraxConfig

    deck = tmp_path / "cohsex.in"
    deck.write_text("[cohsex]\nnval = 8\nncond = 44\nnband = 60\n")
    assert LorraxConfig.from_input_file(
        str(deck), print_fn=lambda *_: None).zeta_nband is None
    deck.write_text("[cohsex]\nnval = 8\nncond = 44\nnband = 60\n"
                    "zeta_nband = 52\n")
    cfg = LorraxConfig.from_input_file(str(deck), print_fn=lambda *_: None)
    assert cfg.zeta_nband == 52 and isinstance(cfg.zeta_nband, int)


def test_a_non_integer_edge_is_refused_rather_than_truncated(tmp_path):
    """The default is ``None``, and the parser's ``default is None`` branch
    otherwise means "nullable float".  A band edge that arrived as 52.5 and
    silently became 52 is a band edge nobody can reason about."""
    pytest.importorskip("jax")
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text("[cohsex]\nnband = 60\nzeta_nband = 52.5\n")
    with pytest.raises(ValueError):
        read_lorrax_input(str(deck))


def test_wider_than_nband_is_refused_by_name(tmp_path):
    """It can only NARROW: the centroid ψ is loaded once over [b0, b4) and
    there are no bands above b4 to fit."""
    pytest.importorskip("jax")
    from gw.gw_config import LorraxConfig

    deck = tmp_path / "cohsex.in"
    deck.write_text("[cohsex]\nnband = 60\nzeta_nband = 68\n")
    with pytest.raises(ValueError, match="zeta_nband"):
        LorraxConfig.from_input_file(str(deck), print_fn=lambda *_: None)


def test_an_edge_at_or_below_b1_is_refused():
    """``right = (b1, zeta_nband)`` has to be a window, not an empty or
    inverted range."""
    bs = _slices(b0=0, b1=4, b2=8, b3=40, b4=64)
    with pytest.raises(ValueError, match="zeta_nband"):
        _init().zeta_fit_band_ranges(bs, 4, log=lambda *_: None)


# ---------------------------------------------------------------------------
# (2) ILLEGAL EDGES REFUSE BY NAME
# ---------------------------------------------------------------------------

def test_the_zeta_nband_edge_is_strict_and_52_is_legal_where_56_is_not():
    """The Si 4×4×4 SOC deck's own table, reproduced on a synthetic spectrum:
    56 sits 0.259 meV above its neighbour (two Kramers pairs joined into a
    4-fold irrep) and 52 clears by 6.870 meV.  Tolerance is
    ``DEGENERACY_TOL_RY`` = 1.000 meV and is NOT touched here."""
    from common.band_degeneracy import BandWindowDegeneracyError
    gi = _init()
    enk = _soc_spectrum(tight={56: 0.259})
    # 52 — legal, and the check is genuinely running (mode is strict)
    gi.check_zeta_fit_windows(enk, (0, 52), (0, 52), 52, log=lambda *_: None)
    # 56 — REFUSED, and the message names the band
    with pytest.raises(BandWindowDegeneracyError, match="band 56"):
        gi.check_zeta_fit_windows(enk, (0, 56), (0, 56), 56,
                                  log=lambda *_: None)


@pytest.mark.parametrize("odd", [51, 53, 55, 57])
def test_every_odd_edge_splits_a_kramers_pair_and_refuses(odd):
    """On a spin-orbit deck this is structural, not a property of one
    spectrum: an odd edge cuts between the two members of a pair."""
    from common.band_degeneracy import BandWindowDegeneracyError
    gi = _init()
    enk = _soc_spectrum()
    with pytest.raises(BandWindowDegeneracyError, match=f"band {odd}"):
        gi.check_zeta_fit_windows(enk, (0, odd), (0, odd), odd,
                                  log=lambda *_: None)


def test_the_nband_edges_stay_snap_when_the_key_is_unset():
    """THE GRANDFATHER CLAUSE, asserted rather than assumed.  Flipping the
    ``nband``/``ncond`` edges to strict would refuse every deck in the tree
    whose window happens to slice, and that census has not been run (owner
    row).  A branch that made them strict as a side effect of this key would
    be that census happening by accident."""
    gi = _init()
    said = []
    enk = _soc_spectrum()
    gi.check_zeta_fit_windows(enk, (0, 55), (0, 55), None, log=said.append)
    assert any("cuts a degenerate multiplet" in s for s in said), (
        "the snap arm went silent — it must still SAY so, loudly")


def test_only_the_zeta_nband_edge_goes_strict_not_the_other_one():
    """``left`` is capped at ``min(b3, zeta_nband)``.  When b3 < zeta_nband the
    left edge is b3 — an ``ncond`` edge — and it keeps the grandfather clause
    even though the same call is checking a strict one."""
    gi = _init()
    said = []
    enk = _soc_spectrum()
    # left edge 55 (an ncond edge, splits a pair) + right edge 52 (legal)
    gi.check_zeta_fit_windows(enk, (0, 55), (0, 52), 52, log=said.append)
    assert any("band 55" in s for s in said)


def test_a_loader_with_no_energies_says_absence_not_pass():
    gi = _init()
    said = []
    gi.check_zeta_fit_windows(None, (0, 52), (0, 52), 52, log=said.append)
    assert any("NOT CHECKED" in s and "not a pass" in s for s in said)


# ---------------------------------------------------------------------------
# (3) THE DECOUPLING REACHES THE FIT
# ---------------------------------------------------------------------------

def test_the_narrowed_window_is_what_the_fit_gets_and_b4_is_untouched():
    """The whole point, in one cell: the ζ ranges come down to 52 and the
    caller's ``b4`` — χ0/Σ's band-sum top, and psi_full_y's extent — does
    not."""
    bs = _slices(b0=0, b1=0, b2=8, b3=52, b4=60)     # ncond 44, nband 60
    left, right = _init().zeta_fit_band_ranges(bs, 52, log=lambda *_: None)
    assert left == (0, 52) and right == (0, 52)
    assert bs.b4 == 60 and bs.nb_full == 60


def test_the_left_window_is_capped_too_and_the_extrapolation_is_announced():
    """``left`` is ``(b0, b3)``.  If b3 outran the fit window the bra leg of
    ρ_mn would carry bands ζ was never fitted on — so it is capped, and the
    fact that Σ then evaluates QP bands above the fit window is said out
    loud rather than discovered later."""
    bs = _slices(b0=0, b1=0, b2=8, b3=60, b4=60)     # ncond 52, nband 60
    said = []
    left, right = _init().zeta_fit_band_ranges(bs, 52, log=said.append)
    assert left == (0, 52) and right == (0, 52)
    assert any("EXTRAPOLATED" in s for s in said), said


def test_the_decoupling_is_announced_with_both_numbers():
    bs = _slices(b3=52, b4=60)
    said = []
    _init().zeta_fit_band_ranges(bs, 52, log=said.append)
    blob = " ".join(said)
    assert "DECOUPLED" in blob and "zeta_nband=52" in blob and "b4=60" in blob


def test_fit_zeta_takes_its_ranges_from_the_resolver():
    """A/B INSTRUMENT CHECK, applied to a deck key: a key that parsed and was
    read by nobody is a green that measures nothing.  ``fit_zeta`` must not
    keep a second copy of the band-range arithmetic beside the resolver."""
    src = open(SRC_INIT, encoding="utf8").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "fit_zeta")
    body = ast.get_source_segment(src, fn) or ""
    assert "zeta_fit_band_ranges(" in body, (
        "fit_zeta no longer calls the resolver — the deck key would parse and "
        "steer nothing")
    assert "check_zeta_fit_windows(" in body
    assert "(band_slices.b1, band_slices.b4)" not in body, (
        "fit_zeta grew a second copy of the right band range; the resolver is "
        "the only place that arithmetic may live")


def test_the_fit_window_travels_into_the_provenance_stamp():
    """``_zeta_fit_provenance`` records ``band_range_left``/``_right``, which
    is how ζ reuse notices a changed window AND how ``bse.vq_interp``'s refit
    learns which bands the producer actually fitted.  Both are the same two
    tuples this resolver returns."""
    src = open(SRC_INIT, encoding="utf8").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "fit_zeta")
    body = ast.get_source_segment(src, fn) or ""
    assert "band_range_left=band_range_left" in body
    assert "band_range_right=band_range_right" in body
    # and the same two names are what fit_zeta_to_h5 is handed
    i_prov = body.index("_zeta_fit_provenance(")
    i_fit = body.index("fit_zeta_to_h5(")
    assert i_fit > 0 and i_prov > 0
