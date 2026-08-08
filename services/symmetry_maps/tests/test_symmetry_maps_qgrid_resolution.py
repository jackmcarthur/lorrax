"""``resolve_qgrid_symmetry`` — the decision, taken once, said out loud.

WHAT THIS PINS.  The service used to answer "may I reduce the q axis?" only
by raising: a caller called ``compute_centroid_sym_perm`` and read the
answer off whether an exception came back.  Three call sites in ``gw/`` did
exactly that, each with its own ``except``, its own private flag, and its
own wording — two of them behind a ``verbose`` argument that production
passes as ``False``.  The result on today's 960-centroid deck is a run that
is ~8× wider in the q axis than the design intends and never says so.

:func:`symmetry_maps.resolve_qgrid_symmetry` replaces the exception with an
answer.  The cells below are the two halves that answer has to satisfy:

* ON A CLOSED SET NOTHING MOVES.  The tables it returns are BIT-IDENTICAL
  to what ``compute_centroid_sym_perm(..., extend_trs=True)`` returned
  before the consolidation, on all three orbit-closed sets in the tree,
  including both TRS-active decks.  This is the "no behavior change on the
  closed path" claim, measured rather than asserted in prose.
* ON A NON-CLOSED SET THE FALLBACK IS THE ANSWER, not an exception, and it
  carries the numbers — the worst op and its residual — plus the
  consequence sentence, so the announcement the monorepo prints is composed
  from a measurement and not from a template.

THE FIXTURE TABLE, MEASURED 2026-08-08 on the in-tree decks::

    gnppm_debug     centroids_frac_399.txt          2 ops  CLOSED   5.6e-17
    bispinor_debug  centroids_frac_209_current.txt  2 ops  CLOSED   1.1e-16
    si_cohsex_debug centroids_frac_144.txt         48 ops  CLOSED   1.0e-06
    bispinor_debug  centroids_frac_256.txt          2 ops  open     1.436e-01
    si_cohsex_debug centroids_frac_960.txt         48 ops  open     1.318e-01
    si_bse_debug    centroids_frac_480.txt         48 ops  open     1.718e-01

Both closed TRS decks are in the first group and both are exercised, which
matters here for the same reason it matters in the q_irr gates: Si carries
zero TRS rows and structurally cannot reach the antiunitary branch, so a
table verified only on Si proves nothing about ``extend_trs``.

FIXTURES ARE READ-ONLY.  Every open below is ``'r'`` and no
``centroids_frac_*.txt`` is written, regenerated or reordered here.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import _deck_stub
from symmetry_maps import (FULL_BZ_CONSEQUENCE, CentroidClosureVerdict,
                           QgridSymmetryResolution,
                           compute_centroid_sym_perm, resolve_qgrid_symmetry)

#: ``(deck, centroid file, n_sym, closed)`` — the measured table above.
_CLOSED = [
    ("gnppm_debug", "centroids_frac_399.txt", 2),
    ("bispinor_debug", "centroids_frac_209_current.txt", 2),
    ("si_cohsex_debug", "centroids_frac_144.txt", 48),
]

#: ``(deck, centroid file, n_violating, worst)`` for the open sets.
_OPEN = [
    ("bispinor_debug", "centroids_frac_256.txt", 1, 1.436e-01),
    ("si_cohsex_debug", "centroids_frac_960.txt", 47, 1.318e-01),
    ("si_bse_debug", "centroids_frac_480.txt", 47, 1.718e-01),
]


def _ids(rows):
    return [f"{d}-{n.split('_')[2].split('.')[0]}" for d, n, *_ in rows]


def _deck_header(deck):
    """``(sym_matrices, tnp, fft_grid)`` out of a deck's ``mf_header``.

    Read with h5py in ``'r'``, the same access ``_deck_stub`` makes, so the
    cell is a property of ``symmetry_maps`` and not of whichever loader
    usually feeds it.  ``si_bse_debug`` is deliberately not in
    ``_deck_stub.DECKS`` (that tuple parametrises the star-table acceptance
    gate) so its header is read here by the same route.
    """
    import h5py

    wfn = _deck_stub.DECK_WFN.get(deck, "WFN.h5")
    with h5py.File(_deck_stub.deck_path(deck, wfn), "r") as f:
        g = f["mf_header"]["symmetry"]
        n = int(g["ntran"][()])
        fft = np.asarray(f["mf_header"]["gspace"]["FFTgrid"][:],
                         dtype=np.int64)
        return g["mtrx"][:n], g["tnp"][:n], fft


def _centroid_idx(deck, name, fft):
    """The centroid file snapped to integer FFT-grid indices.

    MEASURED worst off-grid offset over every set used here: 4.0e-05 of a
    grid step (the files' own six-decimal rounding).  ``np.rint`` is
    therefore exact, and the assertion below is what says so rather than
    assuming it.
    """
    frac = np.loadtxt(os.path.join(_deck_stub.regression_dir(), deck, name))
    scaled = frac * fft[None, :]
    off = float(np.abs(scaled - np.rint(scaled)).max())
    assert off < 1e-3, (
        f"{deck}/{name} is {off:.2e} of a grid step off the {fft.tolist()} "
        f"FFT grid — snapping it to integers would move the points, so this "
        f"fixture cannot stand in for a centroid index table")
    return (np.rint(scaled).astype(np.int64) % fft[None, :]).astype(np.int32)


def _have(deck, name):
    wfn = _deck_stub.DECK_WFN.get(deck, "WFN.h5")
    return (os.path.isfile(_deck_stub.deck_path(deck, wfn))
            and os.path.isfile(os.path.join(_deck_stub.regression_dir(),
                                            deck, name)))


def _resolve(deck, name, **kw):
    S, tnp, fft = _deck_header(deck)
    idx = _centroid_idx(deck, name, fft)
    return resolve_qgrid_symmetry(idx, S, tnp=tnp, fft_grid=fft, **kw), \
        (idx, S, tnp, fft)


# ---------------------------------------------------------------------------
# The closed path: bit-identical, and silent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck,name,n_sym", _CLOSED, ids=_ids(_CLOSED))
def test_a_closed_set_resolves_to_ibz_with_bit_identical_tables(
        deck, name, n_sym):
    """NO BEHAVIOR CHANGE ON THE CLOSED PATH — measured to the byte.

    The consolidation is only safe if the resolution hands back exactly
    what the old ``try: compute_centroid_sym_perm(...)`` handed back.  Not
    "equal to tolerance" — the tables are integer gather indices and an
    integer lattice wrap, so anything but bit-equality is a different
    table and a different unfold.
    """
    if not _have(deck, name):
        pytest.skip(f"{deck}/{name} not in this tree")
    res, (idx, S, tnp, fft) = _resolve(deck, name)
    assert isinstance(res, QgridSymmetryResolution)
    assert res.mode == "ibz" and res.use_ibz, res.reason
    assert res.reason == ""
    assert res.n_sym_spatial == n_sym
    assert isinstance(res.verdict, CentroidClosureVerdict)
    assert res.verdict.closed, res.verdict.describe()

    want_perm, want_L = compute_centroid_sym_perm(
        idx, sym_matrices=S, translations=tnp, fft_grid=fft,
        extend_trs=True)
    got_perm, got_L = res.tables()
    assert got_perm.dtype == want_perm.dtype
    assert got_L.dtype == want_L.dtype
    assert np.array_equal(got_perm, want_perm), (
        f"{deck}/{name}: sym_perm differs from the pre-consolidation "
        f"builder in {int((got_perm != want_perm).sum())} entries")
    assert np.array_equal(got_L, want_L), (
        f"{deck}/{name}: L_table differs from the pre-consolidation "
        f"builder in {int((got_L != want_L).sum())} entries")
    # extend_trs=True is the default here for the same reason every q-axis
    # consumer in the tree passes it: ``sym_idx_q`` ranges over
    # ``[0, 2·ntran)`` and a half-height table clips silently.
    assert got_perm.shape == (2 * n_sym, idx.shape[0])
    assert got_L.shape == (2 * n_sym, idx.shape[0], 3)


@pytest.mark.parametrize("deck,name,n_sym", _CLOSED, ids=_ids(_CLOSED))
def test_a_closed_set_announces_nothing(deck, name, n_sym):
    """RED TWIN of the loud path: a healthy deck stays quiet.

    An announcement that fires on every run is an announcement nobody
    reads, and the whole value of the fallback line is that its presence
    means something.
    """
    if not _have(deck, name):
        pytest.skip(f"{deck}/{name} not in this tree")
    res, _ = _resolve(deck, name, context="unit test")
    assert res.announcement() is None


# ---------------------------------------------------------------------------
# The open path: an answer, not an exception
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck,name,n_viol,worst", _OPEN, ids=_ids(_OPEN))
def test_a_non_closed_set_resolves_to_full_bz_and_says_why(
        deck, name, n_viol, worst):
    """The production case.  It RESOLVES; it does not raise.

    47 of 48 ops on the Si decks, 1 of 2 on the bispinor 256-set.  The
    counts are carried because they are the diagnosis: 47 with the
    identity clean says the set was simply not generated orbit-aware,
    while one violating op would mean something else entirely.
    """
    if not _have(deck, name):
        pytest.skip(f"{deck}/{name} not in this tree")
    res, _ = _resolve(deck, name)
    assert res.mode == "full_bz" and not res.use_ibz
    assert res.sym_perm is None and res.L_table is None
    assert not res.verdict.closed
    assert res.verdict.n_violating == n_viol, res.verdict.describe()
    assert abs(res.verdict.worst_residual - worst) <= 0.005 * worst
    # The reason must carry the numbers, not just the word "failed".
    assert f"op {res.verdict.worst_op}" in res.reason, res.reason
    assert f"{res.verdict.worst_residual:.3e}" in res.reason, res.reason
    assert "NOT CLOSED" in res.reason, res.reason


@pytest.mark.parametrize("deck,name,n_viol,worst", _OPEN, ids=_ids(_OPEN))
def test_asking_a_full_bz_resolution_for_tables_refuses_by_name(
        deck, name, n_viol, worst):
    """A caller that branched wrong gets told WHICH decision it ignored.

    ``None`` propagating into a gather several frames later is the shape
    of the bug this whole consolidation exists to kill, so the accessor
    refuses at the seam and repeats the reason.
    """
    if not _have(deck, name):
        pytest.skip(f"{deck}/{name} not in this tree")
    res, _ = _resolve(deck, name)
    with pytest.raises(RuntimeError, match="full_bz"):
        res.tables()


def test_the_announcement_carries_the_numbers_and_the_consequence():
    """The loud line is composed FROM the measurement.

    A fallback line that said only "orbit closure failed" would be the
    same silence with extra steps: the operator cannot tell a set that
    misses by a rounding floor from one that misses by three grid steps,
    and cannot tell which op to look at.  So the announcement is required
    to carry the worst op, its residual, the call-site context, and the
    single shared consequence sentence.
    """
    deck, name = "si_cohsex_debug", "centroids_frac_960.txt"
    if not _have(deck, name):
        pytest.skip(f"{deck}/{name} not in this tree")
    res, _ = _resolve(deck, name, context="W[static] Dyson solve")
    msg = res.announcement()
    assert msg is not None
    assert "W[static] Dyson solve" in msg
    assert f"op {res.verdict.worst_op}" in msg
    assert f"{res.verdict.worst_residual:.3e}" in msg
    assert FULL_BZ_CONSEQUENCE in msg
    # The consequence names BOTH halves of the cost — the solve and the
    # artifact — because a reader who only hears "full BZ" will read it as
    # a speed note and miss that the restart tensors stay 8× larger.
    assert "full BZ" in FULL_BZ_CONSEQUENCE
    assert "restart tensors stay full-BZ" in FULL_BZ_CONSEQUENCE
    assert "verify_centroid_orbit_closure" in FULL_BZ_CONSEQUENCE


def test_the_announce_key_is_the_centroid_set_not_the_call_site():
    """Dedup is keyed on the SET, and that is the load-bearing choice.

    A scalar run resolves the same centroid set at the ζ̃ write, at the
    V_q pass, at the W Dyson solve and once per self-consistency
    iteration; keyed on the call site that would be four lines saying one
    thing.  A bispinor run resolves TWO different sets whose closure can
    genuinely differ, and those are two facts.
    """
    deck, name = "si_cohsex_debug", "centroids_frac_960.txt"
    other = "centroids_frac_144.txt"
    if not (_have(deck, name) and _have(deck, other)):
        pytest.skip(f"{deck} centroid files not in this tree")
    a, _ = _resolve(deck, name, context="V_q")
    b, _ = _resolve(deck, name, context="W[static] Dyson solve")
    c, _ = _resolve(deck, other, context="V_q")
    assert a.announce_key == b.announce_key
    assert a.announce_key != c.announce_key


# ---------------------------------------------------------------------------
# The second way to land on full_bz, and why it is a different diagnosis
# ---------------------------------------------------------------------------

def test_the_table_builders_own_refusal_is_caught_here_and_named_apart():
    """Arm 2: verdict CLOSED, table builder still refuses.

    The closure measurement scores a minimum-image distance in fractional
    coordinates.  The table builder additionally needs the image to land
    on THIS FFT grid.  Loosening the tolerance until the 960-set reads as
    closed drives exactly that wedge — a real construction, no mock — and
    the resolution must still return ``full_bz`` while saying something
    OTHER than "your set is not closed", because at that tolerance it is
    and the operator would go looking in the wrong place.

    This is also the one place in the tree that catches
    ``compute_centroid_sym_perm``'s ``RuntimeError``.  If it stopped
    catching, this cell turns from a resolution into a traceback.
    """
    deck, name = "si_cohsex_debug", "centroids_frac_960.txt"
    if not _have(deck, name):
        pytest.skip(f"{deck}/{name} not in this tree")
    res, _ = _resolve(deck, name, tol=1.0)
    assert res.verdict.closed, (
        "at tol=1.0 the 960-set's 1.3e-01 worst residual must read as "
        "closed; if it does not, the residual moved and this construction "
        "no longer drives arm 2")
    assert res.mode == "full_bz" and not res.use_ibz
    assert "refused" in res.reason, res.reason
    assert "CLOSED" in res.reason, res.reason
    assert res.sym_perm is None and res.L_table is None
    # And the announcement still works — a caller must not have to know
    # which arm it landed on to print something true.
    assert FULL_BZ_CONSEQUENCE in res.announcement()


# ---------------------------------------------------------------------------
# What must still RAISE — resolving a programming error to "slower" is how
# a convention bug reaches production
# ---------------------------------------------------------------------------

def test_the_2pi_contract_is_forwarded_not_reinvented():
    """Neither/both of ``tnp=``/``tau=`` is a refusal, here as there.

    ``verify_centroid_orbit_closure`` is the ONE place in the service
    where ``tnp = 2π·τ`` is divided, and it has no positional slot for the
    translations so that no caller can pass the wrong convention
    silently.  A resolution wrapper that grew its own positional
    translation argument would reopen exactly that hole, so the contract
    is forwarded and this cell is what says it still is.
    """
    idx = np.zeros((4, 3), dtype=np.int32)
    S = np.eye(3, dtype=np.int64)[None, :, :]
    with pytest.raises(ValueError, match="exactly one of tnp"):
        resolve_qgrid_symmetry(idx, S, fft_grid=(4, 4, 4))
    with pytest.raises(ValueError, match="exactly one of tnp"):
        resolve_qgrid_symmetry(idx, S, tnp=np.zeros((1, 3)),
                               tau=np.zeros((1, 3)), fft_grid=(4, 4, 4))
    with pytest.raises(ValueError, match="pass them as tnp"):
        resolve_qgrid_symmetry(idx, S, tau=np.full((1, 3), 2 * np.pi * 0.5),
                               fft_grid=(4, 4, 4))


def test_a_malformed_centroid_table_or_grid_raises_rather_than_degrading():
    """Shape errors are NOT resolved to ``full_bz``.

    "Fall back to the slower mode" is the right answer to "these
    centroids are not orbit-closed" and the wrong answer to "this array
    is the wrong shape": one is a property of the deck, the other is a
    bug, and a mode that swallows both is how the bug ships.
    """
    S = np.eye(3, dtype=np.int64)[None, :, :]
    tnp = np.zeros((1, 3))
    with pytest.raises(ValueError, match="r_mu_fft_idx must be"):
        resolve_qgrid_symmetry(np.zeros((4, 2), dtype=np.int32), S,
                               tnp=tnp, fft_grid=(4, 4, 4))
    with pytest.raises(ValueError, match="fft_grid must be positive"):
        resolve_qgrid_symmetry(np.zeros((4, 3), dtype=np.int32), S,
                               tnp=tnp, fft_grid=(4, 0, 4))


def test_the_verdict_is_present_on_both_paths():
    """The numbers survive the decision.

    A resolution that dropped the verdict on the ``ibz`` path would force
    the q_irr writer — which stamps the closure verdict into the file — to
    take a SECOND measurement of the same question, and two answers to one
    question is how they drift.
    """
    for deck, name, _ in _CLOSED[:1] + [(_OPEN[1][0], _OPEN[1][1], None)]:
        if not _have(deck, name):
            continue
        res, _ = _resolve(deck, name)
        assert isinstance(res.verdict, CentroidClosureVerdict)
        assert res.verdict.n_centroids > 0
        assert res.verdict.residual_by_op.shape == (res.n_sym_spatial,)
