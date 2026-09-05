"""``number_bands_chi`` / ``number_bands_sigma``: the band count stops being one axis.

WHY THE SPLIT EXISTS.  ``number_bands`` sized two convergence behaviours that
are not the same behaviour.  Measured on the Si 4×4×4 SOC deck, 2026-08-15/16:

* The **Σ** band sum extrapolates.  ``sigma_band_extrapolation`` fits
  ``S_∞ + A/N`` from three partial sums and takes the truncation error from
  106.2 meV MAE raw to 18.3 meV at 248 bands — Σ can be run SHORT and
  corrected.
* The **χ** band count does not.  Holding Σ fixed and sweeping only the
  screening's count 40 → 248 moves band-edge Σ_CH by 50–222 meV, and the last
  rung 224 → 248 still moves the median state by 40.7 meV NON-MONOTONICALLY.
  There is no 1/N to fit.

With one key the two could not be configured apart at all, and the study above
had to vary BerkeleyGW's ``epsilon`` count to isolate the W side.

THE FIVE THINGS THIS FILE HAS TO HOLD.

1. **DEFAULT IDENTITY.**  A deck that names only the umbrella — every deck in
   the tree — resolves to ``chi == sigma == umbrella`` and produces slices
   that are element-for-element what they were before the split existed, the
   PADDED ``b4`` included.  Anything else silently re-runs the whole tree.
2. **PRECEDENCE, AND ITS INVERSION.**  ``nband`` is a transitional ALIAS of
   ``number_bands``.  The defect this feature would most plausibly develop is
   the alias losing to the canonical key's DEFAULT — which is exactly what
   the first draft of ``resolve_band_counts`` did, and which is invisible on
   any deck that writes ``number_bands``.  ``test_alias_beats_the_canonical_
   default`` is that inversion, stated as a number.
3. **INCONSISTENCY REFUSES BY NAME.**  Umbrella + specific, or the two
   umbrella spellings, set to different values.  Not coerced to the max, the
   min, or either one of them.
4. **THE ISDF FIT IS SIZED BY THE MAX, VISIBLY.**  ``max(chi, sigma)``, with
   the winner named in a log line, and the loser demonstrably not sizing it.
5. **THE EXTRAPOLATION BRACKETS Σ.**  0.80/0.90/1.00 of ``number_bands_sigma``.
   A deck at χ=248 / Σ=100 brackets at ~(80, 90, 100), never ~(198, 223, 248).

Layering: everything in §1–§4 is pure and jax-free (it imports ``gw_config``
through the same stub path ``tests/test_env_grammar.py`` uses for the parser).
§5 and §6 need ``BandSlices`` / ``Meta`` and are gated on jax.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
RY_PER_MEV = 1.0 / 13605.693122994


# ---------------------------------------------------------------------------
#  A jax-free handle on gw_config
# ---------------------------------------------------------------------------
# ``import gw.gw_config`` pulls ``common/__init__`` -> ``common.meta`` ->
# ``jax``.  The resolver under test is pure integer logic and its whole point
# is that it is testable without a deck, a WFN or a device, so the module is
# loaded by path with the one constant it imports stubbed.  If jax IS present
# this still exercises the same file — ``importlib`` reads the source either
# way — so there is no second code path being tested here.

def _gw_config():
    if "gw.gw_config" in sys.modules:
        return sys.modules["gw.gw_config"]
    try:
        import jax  # noqa: F401
    except Exception:                                   # noqa: BLE001
        pass
    else:
        sys.path.insert(0, _SRC) if _SRC not in sys.path else None
        from gw import gw_config
        return gw_config
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    common = types.ModuleType("common")
    common.__path__ = [os.path.join(_SRC, "common")]
    sys.modules.setdefault("common", common)
    units = types.ModuleType("common.units")
    units.RYD_TO_EV = 13.6056980659
    units.EV_TO_RYD = 1.0 / 13.6056980659
    sys.modules.setdefault("common.units", units)
    gw = types.ModuleType("gw")
    gw.__path__ = [os.path.join(_SRC, "gw")]
    sys.modules.setdefault("gw", gw)
    spec = importlib.util.spec_from_file_location(
        "gw.gw_config", os.path.join(_SRC, "gw", "gw_config.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gw.gw_config"] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve(named=(), **values):
    """Resolve a params dict built the way ``read_lorrax_input`` builds one.

    Every one of the four keys is PRESENT and carrying its ``_DEFAULTS``
    value unless ``values`` overrides it — that is what the parser produces,
    and it is the shape in which the alias-vs-default inversion is visible.
    ``named`` is the deck's own key set.
    """
    cfg = _gw_config()
    keys = ("number_bands", "number_bands_chi", "number_bands_sigma", "nband")
    params = {k: cfg._DEFAULTS[k] for k in keys}
    params.update(values)
    return cfg.resolve_band_counts(params, deck_named=named)


# ---------------------------------------------------------------------------
# (1) DEFAULT IDENTITY — the bit-identity claim, at the config layer
# ---------------------------------------------------------------------------

def test_a_deck_naming_nothing_gets_the_single_documented_default():
    cfg = _gw_config()
    counts = _resolve()
    assert counts.chi == counts.sigma == int(cfg._DEFAULTS["number_bands"])
    assert not counts.split


@pytest.mark.parametrize("key", ["number_bands", "nband"])
def test_umbrella_only_ties_the_two_counts(key):
    """The whole of the bit-identity claim in one line: an umbrella-only deck
    has chi == sigma, so every slice downstream is the pre-split slice."""
    counts = _resolve((key,), **{key: 248})
    assert (counts.chi, counts.sigma) == (248, 248)
    assert counts.isdf == 248 and counts.isdf_source == "tied"
    assert not counts.split


def test_the_numeric_default_lives_in_exactly_one_entry():
    """``_DEFAULTS`` is the sole default owner (the brief's first bullet).

    Two entries holding 100 is two entries to change, and the day one moves
    the deck means two different runs depending on which key you wrote.  The
    alias and both specifics are ``None`` — "the deck did not say" — and only
    ``number_bands`` carries a number.
    """
    cfg = _gw_config()
    assert isinstance(cfg._DEFAULTS["number_bands"], int)
    for k in ("nband", "number_bands_chi", "number_bands_sigma"):
        assert cfg._DEFAULTS[k] is None, (
            f"{k} must default to None (= unset); a second numeric default "
            f"is a second source of truth")
        assert k in cfg._NULLABLE_INT, (
            f"{k} defaults to None, so the parser needs it in _NULLABLE_INT "
            f"or it parses as a nullable FLOAT and a band edge becomes 248.0")


# ---------------------------------------------------------------------------
# (2) PRECEDENCE — and the inversion this feature would most plausibly develop
# ---------------------------------------------------------------------------

def test_alias_beats_the_canonical_default():
    """THE PRECEDENCE-INVERSION GUARD.  ``nband`` is transitional but LIVE.

    ``number_bands`` carries the numeric default (100).  ``nband`` carries
    ``None``.  A resolver that reads ``params["number_bands"]``
    unconditionally — rather than only when the deck NAMED it — lets that 100
    outrank an explicit ``nband = 248``, and every deck and fixture in the
    tree silently runs at 100 bands.  This is not hypothetical: it is the bug
    the first draft of ``resolve_band_counts`` shipped with, and it is
    invisible to any test whose deck writes ``number_bands``.
    """
    counts = _resolve(("nband",), nband=248)
    assert counts.chi == counts.sigma == 248, (
        "the transitional alias must beat the canonical key's DEFAULT; "
        f"got chi={counts.chi} sigma={counts.sigma}")


def test_canonical_beats_the_alias_default_too():
    """The mirror image, so the guard above cannot be satisfied by inverting
    the precedence the other way."""
    counts = _resolve(("number_bands",), number_bands=248)
    assert counts.chi == counts.sigma == 248


def test_a_specific_key_overrides_only_its_own_consumer():
    counts = _resolve(("number_bands_chi", "number_bands_sigma"),
                      number_bands_chi=248, number_bands_sigma=100)
    assert (counts.chi, counts.sigma) == (248, 100)


def test_a_lone_specific_key_leaves_the_other_on_the_umbrella_default():
    """Terse but well defined: naming one consumer does not invent a value
    for the other, it falls back to the umbrella exactly as documented.  The
    startup line prints both, which is why this is a log problem and not a
    refusal."""
    cfg = _gw_config()
    counts = _resolve(("number_bands_chi",), number_bands_chi=248)
    assert counts.chi == 248
    assert counts.sigma == int(cfg._DEFAULTS["number_bands"])


# ---------------------------------------------------------------------------
# (3) INCONSISTENCY REFUSES BY NAME
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("named,values,must_name", [
    (("nband", "number_bands"), dict(nband=100, number_bands=248),
     ("nband", "number_bands")),
    (("number_bands", "number_bands_chi"),
     dict(number_bands=248, number_bands_chi=100),
     ("number_bands", "number_bands_chi")),
    (("number_bands", "number_bands_sigma"),
     dict(number_bands=248, number_bands_sigma=100),
     ("number_bands", "number_bands_sigma")),
    (("nband", "number_bands_sigma"),
     dict(nband=248, number_bands_sigma=100),
     ("nband", "number_bands_sigma")),
])
def test_inconsistent_pairings_refuse_by_name(named, values, must_name):
    """REFUSE, DON'T COERCE.  Every silent resolution here is wrong for
    somebody: the umbrella wins and the specific request the deck took the
    trouble to write is discarded; the specific wins and the umbrella is a
    lie for the other consumer; max or min invents a run nobody asked for.

    The message must quote BOTH keys and BOTH values — a refusal that says
    only "inconsistent band counts" sends the reader back to the deck to
    work out which two lines it meant.
    """
    cfg = _gw_config()
    with pytest.raises(cfg.BandCountConflict) as exc:
        _resolve(named, **values)
    msg = str(exc.value)
    for key in must_name:
        assert f"`{key}" in msg, f"the refusal must name `{key}`: {msg}"
    for value in values.values():
        assert str(value) in msg, f"the refusal must quote {value}: {msg}"


@pytest.mark.parametrize("named,values", [
    (("nband", "number_bands"), dict(nband=248, number_bands=248)),
    (("number_bands", "number_bands_chi"),
     dict(number_bands=248, number_bands_chi=248)),
])
def test_redundant_but_consistent_is_accepted(named, values):
    """Redundant is not wrong.  A deck that states the same number twice is
    saying one thing twice, and refusing it would make a mechanical deck
    generator's output illegal for no gain."""
    counts = _resolve(named, **values)
    assert counts.chi == counts.sigma == 248


def test_a_count_below_one_is_not_a_band_count():
    with pytest.raises(ValueError):
        _resolve(("number_bands_sigma",), number_bands_sigma=0)


# ---------------------------------------------------------------------------
# (4) THE ISDF FIT IS SIZED BY THE MAX, AND SAYS SO
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chi,sigma,winner", [
    (248, 100, "chi"),
    (100, 248, "sigma"),
    (248, 248, "tied"),
])
def test_the_isdf_window_is_the_max_and_names_its_source(chi, sigma, winner):
    counts = _resolve(("number_bands_chi", "number_bands_sigma"),
                      number_bands_chi=chi, number_bands_sigma=sigma)
    assert counts.isdf == max(chi, sigma)
    assert counts.isdf_source == winner


def test_the_max_is_never_silent():
    """"Log which count won the ``max`` and what the fit was built for.  A
    silent ``max`` is the kind of thing that gets mis-debugged for a day."

    The line has to carry three things a reader cannot otherwise recover:
    both counts, the number the fit was sized for, and WHICH key set it.
    ``nband`` in the deck echo is already the max, so a split deck and an
    unsplit deck at the larger count print the same number without this.
    """
    counts = _resolve(("number_bands_chi", "number_bands_sigma"),
                      number_bands_chi=248, number_bands_sigma=100)
    line = counts.describe()
    assert "248" in line and "100" in line
    assert "number_bands_chi" in line, "the line must name the WINNER"
    assert "ISDF" in line or "zeta" in line
    # and the loser must be identifiable as not having sized the fit
    assert "NOT size" in line or "not size" in line


def test_the_tied_case_says_tied_rather_than_naming_a_winner():
    line = _resolve(("number_bands",), number_bands=248).describe()
    assert "TIED" in line or "tied" in line
    assert "number_bands_chi" not in line and "number_bands_sigma" not in line


# ---------------------------------------------------------------------------
# (5) BandSlices / Meta — where the split becomes two band windows
# ---------------------------------------------------------------------------

def _slices(b0=0, b1=0, b2=8, b3=60, b4=64, **kw):
    pytest.importorskip("jax")
    from gw.wavefunction_bundle import BandSlices
    return BandSlices.from_band_edges(b0, b1, b2, b3, b4, **kw)


@pytest.mark.parametrize("edges", [
    (0, 0, 8, 60, 60),          # the Si 4x4x4 production deck
    (0, 4, 8, 40, 64),          # b4 PADDED past nband
])
def test_unsplit_slices_are_exactly_the_pre_split_slices(edges):
    """DEFAULT IDENTITY at the slice layer.  With no split supplied, ``cond``
    runs to the PADDED ``b4`` and ``sigma_sum`` is ``full`` — which is what
    ``cond`` and ``full`` were before the split existed.  Any drift here
    re-runs every deck in the tree."""
    b0, b1, b2, b3, b4 = edges
    s = _slices(*edges)
    assert s.cond == slice(b2 - b0, b4 - b0)
    assert s.sigma_sum == s.full == slice(0, b4 - b0)
    assert s.cond_all == s.cond
    assert (s.b4_chi, s.b4_sigma) == (b4, b4)
    assert not s.is_split
    assert s.nb_chi == s.nb_sigma_sum == s.nb_full


def test_split_slices_are_two_different_windows():
    s = _slices(b2=8, b3=60, b4=248, b4_chi=248, b4_sigma=100)
    assert s.cond == slice(8, 248), "chi keeps every band"
    assert s.sigma_sum == slice(0, 100), "Sigma stops at its own count"
    assert s.full == slice(0, 248), "psi is LOADED over the max"
    assert s.is_split and s.nb_chi == 248 and s.nb_sigma_sum == 100


def test_cond_all_is_the_union_not_the_chi_leg():
    """The minimax τ-axis is built once and reused by BOTH stages, so its
    interval must cover the union.  Under a Σ-larger split ``cond`` alone
    would under-cover Σ's transitions and the quadrature error would be
    invisible at the seam that caused it."""
    s = _slices(b2=8, b3=60, b4=248, b4_chi=100, b4_sigma=248)
    assert s.cond == slice(8, 100)
    assert s.cond_all == slice(8, 248)


def test_logical_conduction_union_excludes_process_padding():
    """A spectral consumer never promotes carrier padding into real bands."""
    p16 = _slices(
        b2=46, b3=72, b4=192, b4_chi=192, b4_sigma=192,
        b4_logical=184)
    p36 = _slices(
        b2=46, b3=72, b4=216, b4_chi=216, b4_sigma=216,
        b4_logical=184)

    assert p16.cond_all != p36.cond_all
    assert p16.cond_all_logical == p36.cond_all_logical == slice(46, 184)
    assert p16.nb_full_logical == p36.nb_full_logical == 184


def test_the_larger_count_must_own_the_padded_edge():
    """``b4`` is the padded top of the LARGER consumer.  A BandSlices whose
    two tops are both below ``b4`` would mean bands were loaded that no
    consumer sums — a silent waste that is also a silent inconsistency with
    ``Meta``."""
    with pytest.raises(ValueError, match="max"):
        _slices(b2=8, b3=60, b4=248, b4_chi=100, b4_sigma=100)


def test_meta_gives_the_padded_edge_to_the_larger_count():
    """The pad interaction, which is the genuinely new thing.  ψ is zeroed on
    ``[b_id_4_user, b_id_4)``; bands in ``[smaller, b_id_4_user)`` are REAL
    and the smaller consumer must not sum them, so its slice is load-bearing
    rather than a formality."""
    pytest.importorskip("jax")
    import jax
    from common.meta import Meta
    world = int(jax.device_count())
    wfn = types.SimpleNamespace(
        nelec=8, fft_grid=(8, 8, 8), cell_volume=1.0, nspin=1, nspinor=1,
        kgrid=(2, 2, 2))
    sym = types.SimpleNamespace(nk_tot=8)
    nband = 100 + (world - 100 % world) % world + 1   # force a real pad
    meta = Meta.from_system(wfn, sym, 4, 4, nband, 16, False,
                            nband_chi=nband, nband_sigma=40)
    assert meta.b_id_4_user == nband
    assert meta.b_id_4_chi == meta.b_id_4, "the larger count owns the pad"
    assert meta.b_id_4_sigma == 40, "the smaller count is literal"
    assert meta.b_id_4 >= nband and meta.b_id_4 % world == 0


def test_meta_refuses_an_nband_that_is_not_the_max():
    pytest.importorskip("jax")
    from common.meta import Meta
    wfn = types.SimpleNamespace(
        nelec=8, fft_grid=(8, 8, 8), cell_volume=1.0, nspin=1, nspinor=1,
        kgrid=(2, 2, 2))
    sym = types.SimpleNamespace(nk_tot=8)
    with pytest.raises(ValueError, match="max"):
        Meta.from_system(wfn, sym, 4, 4, 100, 16, False,
                         nband_chi=40, nband_sigma=40)


def test_meta_loaded_face_band_carrier_uses_specs_not_mesh_product():
    """Na's 86-band P16 face carrier is 88, not product-padded 96."""
    pytest.importorskip("jax")
    from common.meta import Meta

    mesh = types.SimpleNamespace(
        axis_names=("x", "y"), shape={"x": 4, "y": 4})
    wfn = types.SimpleNamespace(
        nelec=5, fft_grid=(8, 8, 8), cell_volume=1.0, nspin=1,
        nspinor=1, kgrid=(2, 2, 2))
    sym = types.SimpleNamespace(nk_tot=8)

    meta = Meta.from_system(
        wfn, sym, 5, 81, 86, 16, False,
        nband_chi=86, nband_sigma=86, mesh_xy=mesh)

    assert meta.b_id_4_user == 86
    assert meta.b_id_4 == 88
    assert meta.b_id_4_chi == 88
    assert meta.b_id_4_sigma == 88


# ---------------------------------------------------------------------------
# (6) THE DEGENERACY GUARDS, FIRING INDEPENDENTLY
# ---------------------------------------------------------------------------

def _soc_spectrum(nb=64, nk=8, pair_gap_mev=100.0, tight=None):
    """A spin-orbit spectrum: Kramers pairs, so every ODD edge splits one.

    ``tight`` maps an EVEN edge to the gap (meV) just below it — how a 4- or
    6-fold irrep shows up.  Shape ``(nspin, nk, nb)``, what ``wfn.energies``
    carries.
    """
    tight = dict(tight or {})
    e = np.zeros(nb, dtype=np.float64)
    for b in range(1, nb):
        step = 0.0 if b % 2 == 1 else tight.get(b, pair_gap_mev) * RY_PER_MEV
        e[b] = e[b - 1] + step
    return np.tile(e, (1, nk, 1))


def _cfg_stub(chi, sigma, named, restart=False):
    cfg = _gw_config()
    return types.SimpleNamespace(
        bands=cfg.BandCounts(chi=chi, sigma=sigma, named=frozenset(named)),
        restart=restart)


def _check(chi, sigma, named, *, b4=64, wfn_nb=None, env=None, lines=None):
    pytest.importorskip("jax")
    from gw import gw_init
    s = _slices(b2=8, b3=40, b4=b4,
                b4_chi=b4 if chi >= sigma else chi,
                b4_sigma=b4 if sigma >= chi else sigma)
    wfn = types.SimpleNamespace(energies=_soc_spectrum(
        nb=b4 if wfn_nb is None else wfn_nb))
    log = (lines if lines is not None else []).append
    old = os.environ.get("LORRAX_BAND_DEGENERACY")
    if env is None:
        os.environ.pop("LORRAX_BAND_DEGENERACY", None)
    else:
        os.environ["LORRAX_BAND_DEGENERACY"] = env
    try:
        gw_init.check_band_sum_degeneracy(
            wfn, _cfg_stub(chi, sigma, named), s, log=log)
    finally:
        if old is None:
            os.environ.pop("LORRAX_BAND_DEGENERACY", None)
        else:
            os.environ["LORRAX_BAND_DEGENERACY"] = old


@pytest.mark.parametrize("key,chi,sigma", [
    ("number_bands_chi", 33, 64),
    ("number_bands_sigma", 64, 33),
])
def test_each_named_count_refuses_its_own_split_multiplet(key, chi, sigma):
    """BOTH counts go through the guard, and INDEPENDENTLY: a bad χ edge must
    not be excused by a clean Σ edge or vice versa.  On a SOC spectrum every
    odd edge splits a Kramers pair, so 33 is illegal by construction."""
    pytest.importorskip("jax")
    from common.band_degeneracy import BandWindowDegeneracyError
    with pytest.raises(BandWindowDegeneracyError) as exc:
        _check(chi, sigma, (key,))
    msg = str(exc.value)
    assert key in msg, "the refusal must name the key that owns the edge"
    assert "33" in msg
    assert "Legal edges" in msg, (
        "the brief: the error should name the legal edges for each count")


def test_a_clean_edge_on_the_other_count_is_not_a_pass_for_the_bad_one():
    """The independence claim from the other side: χ clean at 32, Σ split at
    33, and the run still refuses — naming Σ."""
    pytest.importorskip("jax")
    from common.band_degeneracy import BandWindowDegeneracyError
    with pytest.raises(BandWindowDegeneracyError) as exc:
        # Two spare WFN bands make chi=64 provably clean, so this cell tests
        # the intended independent Sigma refusal rather than file-edge absence.
        _check(64, 33, ("number_bands_chi", "number_bands_sigma"), wfn_nb=66)
    assert "number_bands_sigma" in str(exc.value)


def test_the_umbrella_edge_is_grandfathered_to_a_warning():
    """The bit-identity claim needs this.  Every deck in the tree sits on an
    umbrella edge chosen before this check existed; a refusal there is not a
    byte-identical ``eqp0.dat``, it is a crash.  Same grandfather clause
    ``check_zeta_fit_windows`` documents, same census reason."""
    lines = []
    _check(33, 33, ("number_bands",), lines=lines)
    text = "\n".join(lines)
    assert "SPLITS A DEGENERATE MULTIPLET" in text, "silence is not allowed"
    assert "grandfathered" in text


def test_the_env_override_downgrades_and_says_it_did():
    lines = []
    _check(33, 64, ("number_bands_chi",), env="snap", lines=lines)
    text = "\n".join(lines)
    assert "SPLITS A DEGENERATE MULTIPLET" in text
    assert "LORRAX_BAND_DEGENERACY=snap" in text


def test_an_unrecognised_override_value_is_not_ignored():
    with pytest.raises(ValueError, match="LORRAX_BAND_DEGENERACY"):
        _check(64, 64, ("number_bands_chi",), env="yes-please")


def test_a_clean_pair_of_edges_prints_the_numbers_anyway():
    """"No news" and "a good number" must not look alike (preamble
    measurement rule 10)."""
    lines = []
    _check(32, 24, ("number_bands_chi", "number_bands_sigma"), lines=lines)
    text = "\n".join(lines)
    assert "clean" in text and "min gap" in text
    assert "chi0/W band sum" in text and "Sigma band sum" in text


def test_a_named_edge_at_the_wfn_extent_refuses_uncheckable_closure():
    from common.band_degeneracy import BandWindowDegeneracyError

    with pytest.raises(BandWindowDegeneracyError) as exc:
        _check(64, 32, ("number_bands_chi", "number_bands_sigma"),
               b4=64, wfn_nb=64)
    text = str(exc.value)
    assert "NOT CHECKABLE" in text
    assert "one spare band" in text
    assert "number_bands_chi" in text


def test_an_umbrella_edge_at_the_wfn_extent_warns_without_claiming_clean():
    lines = []
    _check(64, 32, ("number_bands",), b4=64, wfn_nb=64, lines=lines)
    text = "\n".join(lines)
    assert "NOT CHECKABLE" in text
    assert "grandfathered" in text
    assert "exempt" not in text


def test_restart_refuses_a_changed_chi_count_and_allows_a_changed_sigma():
    """THE ASYMMETRY IS THE POINT, and it is what makes the owner's "use
    ``restart`` after the first run of a deck" rule usable with a split.

    Every tensor in a restart bundle is a function of the SCREENING side or
    of the loaded extent: ``V_qmunu`` / ``W0_qmunu`` ARE the screening,
    ``psi_full_y`` / ``enk_full`` / zeta span ``[b0, b4) = max(chi, sigma)``
    which the 5-tuple's b4 already pins.  NOTHING on disk is a function of
    ``number_bands_sigma`` — Sigma slices ``[0, b4_sigma)`` out of tensors
    that already exist.  So a Sigma-count sweep at fixed chi may reuse the
    file (which is exactly the configuration the split exists to make cheap),
    and a changed chi must refuse.

    A file with NO ``band_window_split`` attr was written by an unsplit run;
    it resolves to ``(b4, b4)``, so it matches an unsplit run exactly and
    refuses a chi-changed one.  Widening ``band_window`` itself from 5 to 7
    entries would instead have stranded every restart file on disk.
    """
    h5py = pytest.importorskip("h5py")
    import tempfile
    from file_io.tagged_arrays import assert_restart_window_matches

    def _file(tmp, window, split):
        path = os.path.join(tmp, "restart.h5")
        with h5py.File(path, "w") as f:
            f["band_window"] = np.asarray(window, dtype=np.int64)
            if split is not None:
                f["band_window_split"] = np.asarray(split, dtype=np.int64)
        return path

    with tempfile.TemporaryDirectory() as tmp:
        # written by a split run: chi 28, sigma 20, loaded extent 28
        path = _file(tmp, (0, 0, 8, 20, 28), (28, 20))
        same = _slices(b0=0, b1=0, b2=8, b3=20, b4=28,
                       b4_chi=28, b4_sigma=20)
        assert_restart_window_matches(path, band_slices=same)   # identical

        # Sigma moved, chi held: legitimate reuse, no refusal
        moved_sigma = _slices(b0=0, b1=0, b2=8, b3=16, b4=28,
                              b4_chi=28, b4_sigma=16)
        assert_restart_window_matches(path, band_slices=moved_sigma)

        # chi moved: refuse, and say which key
        moved_chi = _slices(b0=0, b1=0, b2=8, b3=20, b4=24,
                            b4_chi=24, b4_sigma=20)
        with pytest.raises(ValueError, match="number_bands_chi"):
            assert_restart_window_matches(path, band_slices=moved_chi)

    with tempfile.TemporaryDirectory() as tmp:
        # a PRE-SPLIT file: no band_window_split attr at all
        legacy = _file(tmp, (0, 0, 8, 20, 28), None)
        unsplit = _slices(b0=0, b1=0, b2=8, b3=20, b4=28)
        assert_restart_window_matches(legacy, band_slices=unsplit)
        with pytest.raises(ValueError, match="number_bands_chi"):
            assert_restart_window_matches(
                legacy, band_slices=_slices(b0=0, b1=0, b2=8, b3=20, b4=28,
                                            b4_chi=20, b4_sigma=28))


# ---------------------------------------------------------------------------
# (7) THE ISDF SIZING RULE — the eV-scale failure this guards
# ---------------------------------------------------------------------------

def test_the_isdf_window_invariant_refuses_a_window_sized_by_the_smaller():
    """``docs/dev/isdf_basis_adequacy_at_large_nband.md``: an ISDF window
    clamped to a small band range and used for a large one returned a QP gap
    of 0.36 eV where the answer is 3.1-3.7 eV, with a NEGATIVE eqp1, and
    passed every gate in the suite because all of them are upstream of or
    orthogonal to Sigma_c.  The ``max`` is the guard against that, one index
    over, and a ``max`` that is only implied by how b4 is computed is one
    refactor away from being a ``min``.  So it is asserted where it fails.
    """
    pytest.importorskip("jax")
    from gw import gw_init
    s = _slices(b2=8, b3=20, b4=28, b4_chi=28, b4_sigma=20)
    meta = types.SimpleNamespace(
        b_id_4_user=28, b_id_4_chi_user=28, b_id_4_sigma_user=20)
    # the right range the resolver actually returns: tops out at the max
    gw_init.assert_isdf_window_is_the_max(
        s, (0, 28), None, meta=meta, log=lambda *_: None)
    with pytest.raises(ValueError, match="0.36 eV"):
        # a window sized by the SMALLER consumer
        gw_init.assert_isdf_window_is_the_max(s, (0, 20), None, meta=meta,
                                              log=lambda *_: None)


def test_the_isdf_sizing_line_names_the_count_that_set_it():
    pytest.importorskip("jax")
    from gw import gw_init
    lines = []
    s = _slices(b2=8, b3=20, b4=28, b4_chi=28, b4_sigma=20)
    meta = types.SimpleNamespace(
        b_id_4_user=28, b_id_4_chi_user=28, b_id_4_sigma_user=20)
    gw_init.assert_isdf_window_is_the_max(
        s, (0, 28), None, meta=meta, log=lines.append)
    text = "\n".join(lines)
    assert "28" in text and "20" in text
    assert "number_bands_chi" in text, "the winner must be named"


def test_narrowing_the_fit_below_a_band_sum_is_reported_per_consumer():
    """``zeta_nband`` may narrow the fit below a band sum's top — the BSE's
    Galerkin capacity bound is a legitimate reason — but the consumer left
    above it is then running on an EXTRAPOLATED zeta basis, which is the
    documented mechanism, and it must never be silent."""
    pytest.importorskip("jax")
    from gw import gw_init
    lines = []
    s = _slices(b2=8, b3=20, b4=28, b4_chi=28, b4_sigma=20)
    meta = types.SimpleNamespace(
        b_id_4_user=28, b_id_4_chi_user=28, b_id_4_sigma_user=20)
    gw_init.assert_isdf_window_is_the_max(
        s, (0, 16), 16, meta=meta, log=lines.append)
    text = "\n".join(lines)
    assert text.count("EXTRAPOLATED") == 2, (
        "both consumers top out above 16, so both must be reported")
    assert "number_bands_chi" in text and "number_bands_sigma" in text


def test_logical_250_is_not_extrapolation_into_the_exact_zero_256_carrier():
    """CrI3 P16: physical sums stop at 250; b4=256 is only a carrier."""
    pytest.importorskip("jax")
    from gw import gw_init
    s = _slices(b2=130, b3=190, b4=256,
                b4_chi=256, b4_sigma=256)
    meta = types.SimpleNamespace(
        b_id_4_user=250, b_id_4_chi_user=250,
        b_id_4_sigma_user=250)

    lines = []
    gw_init.assert_isdf_window_is_the_max(
        s, (0, 250), 250, meta=meta, log=lines.append)
    text = "\n".join(lines)
    assert "physical window top 250" in text
    assert "exact-zero inert pad [250, 256)" in text
    assert "EXTRAPOLATED" not in text

    narrowed = []
    gw_init.assert_isdf_window_is_the_max(
        s, (0, 190), 190, meta=meta, log=narrowed.append)
    narrow_text = "\n".join(narrowed)
    assert narrow_text.count("EXTRAPOLATED") == 2
    assert "Bands [190, 250)" in narrow_text
    assert "Bands [190, 256)" not in narrow_text

    with pytest.raises(ValueError, match="exact-zero mesh carrier pad"):
        gw_init.assert_isdf_window_is_the_max(
            s, (0, 252), 252, meta=meta, log=lambda *_: None)


# ---------------------------------------------------------------------------
# (8) DOCUMENTATION — the key that ships with no row is how a no-op survives
# ---------------------------------------------------------------------------

def test_every_deck_key_has_a_row_in_the_input_reference():
    """``docs/input_reference.md`` is the deck's documented surface, and
    ``_DEFAULTS`` is its source of truth.  A key present in one and absent
    from the other is how ``self_energy_eval_type`` shipped as a no-op nobody
    noticed: nothing read it, nothing documented it, and nothing said so.

    Measured when this test was written: the gap was already ZERO, so this
    pins a property the tree HAS rather than announcing one it lacks — which
    is the only kind of coverage assertion worth adding.  It costs nothing and
    it fails on the next key added without a row, including the four this
    feature introduces.

    Deliberately NOT the converse (a row with no key): retired keys keep rows
    on purpose, and ``_LEGACY_DECK_KEYS`` exists for exactly that.
    """
    import re
    import pathlib
    cfg = _gw_config()
    doc = (pathlib.Path(_REPO) / "docs" / "input_reference.md").read_text()
    rows = set(re.findall(r"^\|\s*`([^`]+)`\s*\|", doc, re.M))
    missing = sorted(k for k in cfg._DEFAULTS if k not in rows)
    assert not missing, (
        f"{len(missing)} deck key(s) in _DEFAULTS with no row in "
        f"docs/input_reference.md: {missing}")


@pytest.mark.parametrize("key", [
    "number_bands", "number_bands_chi", "number_bands_sigma", "nband"])
def test_the_band_count_family_is_documented_as_a_family(key):
    """Each of the four, and the three facts a reader needs that no single
    row can be trusted to carry by accident: that the umbrella overrides
    both, that disagreement refuses, and that the fit takes the max."""
    import pathlib
    doc = (pathlib.Path(_REPO) / "docs" / "input_reference.md").read_text()
    assert f"| `{key}` |" in doc, f"`{key}` has no row"
    row = next(l for l in doc.splitlines() if l.startswith(f"| `{key}` |"))
    if key == "nband":
        assert "alias" in row.lower(), "the alias must be documented as one"
    if key in ("number_bands_chi", "number_bands_sigma"):
        assert "refusal" in row.lower() or "refuses" in row.lower()
    if key == "number_bands":
        assert "BOTH" in row or "both" in row
