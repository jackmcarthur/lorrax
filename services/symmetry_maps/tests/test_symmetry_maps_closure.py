"""``verify_centroid_orbit_closure`` — the door diagnostic, and its twins.

WHY THIS EXISTS AT ALL.  Reconstructing ``W(Sq)`` from ``W(q)`` in the ISDF
basis is a permutation of the (μ, ν) indices, and that permutation exists
ONLY if every symmetry maps the centroid set into itself.  A q_irr restart
file written against a non-closed set is silently unrecoverable — there is
no α to invert with — so closure is a hard prerequisite and the writer
refuses on it (``test_symmetry_maps_qirr_store.py``).  This module pins the
measurement the refusal is taken from.

THE ACCEPTANCE TEST IS THREE MEASURED ROWS, NOT A PROPERTY.
``SPEC_qirr_restart_tensors.md`` §2 measured closure on the committed
centroid files and wrote the table down::

    si_bse_debug     centroids_frac_480.txt   worst 1.718e-01   47/48 ops
    si_cohsex_debug  centroids_frac_960.txt   worst 1.318e-01   47/48 ops
    si_cohsex_debug  centroids_frac_144.txt   worst 1.000e-06    0/48  CLOSED

Those three rows are :func:`test_the_spec_table_reproduces` below, to three
significant figures, and they are the reason this function can be believed
about a file nobody has measured.  The 144-set's 1.000e-06 is the text
file's six-decimal rounding floor and not a defect — which is exactly why
the tolerance is 1e-5 and not 1e-9.

THE FILES ARE READ, NEVER WRITTEN.  ``tests/regression/*`` is chmod'd
read-only on purpose and no ``centroids_frac_*.txt`` is regenerated here or
anywhere on this branch (survey §8(a): regenerating the production sets as
orbit-closed is an owner decision and a measured accuracy trade, not a
cleanup).  Every open below is ``'r'``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import _deck_stub
from symmetry_maps import (CLOSURE_TOL_DEFAULT, CentroidClosureVerdict,
                           verify_centroid_orbit_closure)

#: The spec's §2 table, verbatim: (deck, centroid file, worst, n_violating).
#: ``si_bse_debug`` is not one of ``_deck_stub.DECKS`` — that tuple is the
#: star-table acceptance gate's parametrisation and pinning a fifth deck
#: into it would break a different test — so its header is read here.
_SPEC_ROWS = [
    ("si_cohsex_debug", "centroids_frac_960.txt", 1.318e-01, 47),
    ("si_cohsex_debug", "centroids_frac_144.txt", 1.000e-06, 0),
    ("si_bse_debug", "centroids_frac_480.txt", 1.718e-01, 47),
]

#: The FFT grid all three of these sets were snapped to, from their own
#: file headers ("snapped to FFT grid (24, 24, 24)").
_SI_FFT = (24, 24, 24)


def _deck_syms(deck):
    """``(sym_matrices, tnp)`` out of a deck's ``mf_header/symmetry``.

    Read with h5py directly and in ``'r'``: this is the same access the
    service's ``_deck_stub`` makes, and it keeps the cell a property of
    ``symmetry_maps`` rather than of whatever loader usually feeds it.
    """
    import h5py

    with h5py.File(_deck_stub.deck_path(deck, "WFN.h5"), "r") as f:
        g = f["mf_header"]["symmetry"]
        n = int(g["ntran"][()])
        return g["mtrx"][:n], g["tnp"][:n]


def _centroids(deck, name):
    return np.loadtxt(os.path.join(_deck_stub.regression_dir(), deck, name))


def _have(deck, name):
    return (os.path.isfile(_deck_stub.deck_path(deck, "WFN.h5"))
            and os.path.isfile(os.path.join(_deck_stub.regression_dir(),
                                            deck, name)))


# ---------------------------------------------------------------------------
# The acceptance test: three measured rows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck,name,worst,n_viol", _SPEC_ROWS,
                         ids=[f"{d}-{n.split('_')[-1][:-4]}"
                              for d, n, _, _ in _SPEC_ROWS])
def test_the_spec_table_reproduces(deck, name, worst, n_viol):
    """The spec's §2 rows, to three significant figures.

    Both halves matter and they fail differently.  The COUNT (47/48, or
    0/48) says which ops are implicated: 47 with the identity clean is the
    signature of a set that simply was not generated orbit-aware, whereas a
    single violating op would be a different diagnosis entirely.  The
    RESIDUAL says by how much — 0.13 in fractional coordinates is ~3 grid
    steps on the 24³ grid these centroids live on, i.e. nowhere near a
    centroid, not a rounding question.
    """
    if not _have(deck, name):
        pytest.skip(f"{deck}/{name} not in this tree")
    S, tnp = _deck_syms(deck)
    v = verify_centroid_orbit_closure(_centroids(deck, name), S, tnp=tnp)
    assert isinstance(v, CentroidClosureVerdict)
    assert v.n_sym == 48, v.n_sym
    assert v.n_violating == n_viol, (
        f"{deck}/{name}: {v.n_violating}/{v.n_sym} ops violating, spec "
        f"says {n_viol}/48")
    assert v.closed == (n_viol == 0), v.describe()
    assert abs(v.worst_residual - worst) <= 0.005 * worst, (
        f"{deck}/{name}: worst residual {v.worst_residual:.4e}, spec says "
        f"{worst:.4e}")


def test_the_closed_row_is_closed_only_because_the_tolerance_admits_rounding():
    """WHY 1e-5, stated as a measurement instead of a comment.

    The 144-set's residual is 1.000e-06 — the six decimals its text file is
    written to, not a defect in the set.  A tolerance below that would call
    the one orbit-closed set in the tree unclosed and the feature would
    have no vehicle at all; a tolerance above ~1e-1 would admit the 960
    set.  This pins the gap both ways so nobody tightens the default into
    the rounding floor.
    """
    if not _have("si_cohsex_debug", "centroids_frac_144.txt"):
        pytest.skip("si_cohsex_debug not in this tree")
    S, tnp = _deck_syms("si_cohsex_debug")
    c144 = _centroids("si_cohsex_debug", "centroids_frac_144.txt")
    assert verify_centroid_orbit_closure(c144, S, tnp=tnp).closed
    tight = verify_centroid_orbit_closure(c144, S, tnp=tnp, tol=1e-9)
    assert not tight.closed, (
        "at tol=1e-9 the six-decimal rounding must read as a violation; if "
        "it does not, the files gained precision and this comment is stale")
    assert 1e-6 < CLOSURE_TOL_DEFAULT < 1e-1


# ---------------------------------------------------------------------------
# Red twins — both directions, constructed
# ---------------------------------------------------------------------------

def test_a_rotated_closed_set_is_still_closed():
    """RED TWIN (positive direction): closure is a property of the SET.

    Push every member of the orbit-closed 144-set through one group
    element.  Because the set is a union of orbits the image is the same
    set — reordered, and re-derived through different arithmetic — so the
    verdict must be unchanged.  A checker that had accidentally keyed on
    row order, on the first centroid, or on the file's own float
    representation would move here and nowhere else.
    """
    if not _have("si_cohsex_debug", "centroids_frac_144.txt"):
        pytest.skip("si_cohsex_debug not in this tree")
    S, tnp = _deck_syms("si_cohsex_debug")
    c = _centroids("si_cohsex_debug", "centroids_frac_144.txt")
    base = verify_centroid_orbit_closure(c, S, tnp=tnp)
    assert base.closed, base.describe()

    # BGW r-action r' = mtrx⁻¹·r + τ, one non-identity op.
    s = 7
    Rinv = np.rint(np.linalg.inv(np.asarray(S[s], dtype=np.float64)))
    tau = np.asarray(tnp, dtype=np.float64)[s] / (2.0 * np.pi)
    rot = (c @ Rinv.T + tau) % 1.0

    got = verify_centroid_orbit_closure(rot, S, tnp=tnp)
    assert got.closed, got.describe()
    assert got.n_centroids == base.n_centroids
    # Both sit on the six-decimal rounding floor, not on the same number:
    # the rotation mixes axes, so a per-axis 1e-6 becomes √2·1e-6 in the
    # Euclidean metric.  MEASURED 1.000e-06 -> 1.414e-06.  What has to hold
    # is that both are floor-level and orders below the tolerance.
    assert base.worst_residual < 2e-6 and got.worst_residual < 2e-6
    assert got.worst_residual < CLOSURE_TOL_DEFAULT

    # And the same set REORDERED must hash DIFFERENTLY.  That is not a
    # weakness of the hash, it is the requirement: ``sym_perm`` addresses
    # centroids by ROW POSITION, so a permuted set is a different table and
    # a q_irr file stamped with one and read against the other would gather
    # the wrong μ.  MEASURED here: the point sets are identical as sets
    # (verified below) and the digests differ.
    h_base = verify_centroid_orbit_closure(c, S, tnp=tnp, fft_grid=_SI_FFT)
    h_rot = verify_centroid_orbit_closure(rot, S, tnp=tnp, fft_grid=_SI_FFT)
    grid = np.asarray(_SI_FFT)
    as_set = {tuple(r) for r in
              (np.rint(c * grid).astype(int) % grid).tolist()}
    rot_set = {tuple(r) for r in
               (np.rint(rot * grid).astype(int) % grid).tolist()}
    assert as_set == rot_set, "the op did not map the set onto itself"
    assert h_rot.centroid_hash != h_base.centroid_hash, (
        "an ORDER-INSENSITIVE centroid hash would let a permuted table "
        "pass the reader's identity check; sym_perm addresses by row")


def test_dropping_one_centroid_breaks_closure():
    """RED TWIN (negative direction): one missing point, and it must show.

    The whole value of the check is that it notices a set that is *nearly*
    closed.  Removing a single member of a 144-point set leaves 143 of 144
    images landing perfectly and one op-orbit missing its target — a
    failure a coarse "does the set look symmetric" heuristic would round
    away.  The residual must also be a real distance (a whole grid step on
    a 24³ grid is 1/24 ≈ 4.2e-02), not an epsilon.
    """
    if not _have("si_cohsex_debug", "centroids_frac_144.txt"):
        pytest.skip("si_cohsex_debug not in this tree")
    S, tnp = _deck_syms("si_cohsex_debug")
    c = _centroids("si_cohsex_debug", "centroids_frac_144.txt")
    assert verify_centroid_orbit_closure(c, S, tnp=tnp).closed

    for drop in (0, 5, 71, 143):
        short = np.delete(c, drop, axis=0)
        v = verify_centroid_orbit_closure(short, S, tnp=tnp)
        assert not v.closed, (
            f"dropped centroid {drop} and the set still read as closed:\n"
            f"{v.describe()}")
        assert v.n_centroids == 143
        assert v.n_violating >= 1
        assert v.worst_residual > 1.0 / 24.0 * 0.9, (
            f"residual {v.worst_residual:.3e} is smaller than one grid "
            f"step; that is not a missing centroid")
        assert v.worst_op in v.violating_ops


def test_the_verdict_names_every_violating_op_not_just_the_first():
    """Spec §6 gate 1 asks for "the offending op index and residual".

    Plural.  The production failure is 47 ops at once, and a refusal that
    named only the first would read as one bad operation and send the
    reader looking for it.  Both the object and the rendered message carry
    all of them.
    """
    if not _have("si_cohsex_debug", "centroids_frac_960.txt"):
        pytest.skip("si_cohsex_debug not in this tree")
    S, tnp = _deck_syms("si_cohsex_debug")
    v = verify_centroid_orbit_closure(
        _centroids("si_cohsex_debug", "centroids_frac_960.txt"), S, tnp=tnp)
    assert len(v.violating_ops) == 47
    assert 0 not in v.violating_ops, "the identity op cannot violate closure"
    text = v.describe()
    for s in v.violating_ops:
        assert f"s={s}:" in text, f"op {s} missing from the refusal text"
    assert "NOT CLOSED" in text
    assert f"{v.worst_residual:.3e}" in text

    with pytest.raises(RuntimeError) as exc:
        v.raise_if_not_closed("write_qirr_tensor")
    assert "write_qirr_tensor" in str(exc.value)
    assert "s=1:" in str(exc.value)


def test_raise_if_not_closed_is_silent_on_a_closed_set():
    """RED TWIN for the refusal above: the pass path must not raise."""
    if not _have("si_cohsex_debug", "centroids_frac_144.txt"):
        pytest.skip("si_cohsex_debug not in this tree")
    S, tnp = _deck_syms("si_cohsex_debug")
    v = verify_centroid_orbit_closure(
        _centroids("si_cohsex_debug", "centroids_frac_144.txt"), S, tnp=tnp)
    v.raise_if_not_closed("write_qirr_tensor")
    assert "CLOSED" in v.describe() and "NOT CLOSED" not in v.describe()
    assert v.as_attr().startswith("closed ")


# ---------------------------------------------------------------------------
# The 2π that cost an hour
# ---------------------------------------------------------------------------

def test_the_translation_convention_cannot_be_passed_silently():
    """``tnp`` is ``2π·τ``, and this is the ONE place it gets divided.

    The spec records that mixing the two makes every centroid set look
    unclosed *including the one that is fine*, and that the mistake
    produced a wrong first answer.  The defence is structural, not
    documentary: there is no positional slot for the translations, so a
    caller must name which convention they hold.

    Three refusals and one agreement:

    * neither keyword — refused;
    * both keywords — refused (there is no "and they agree" path; the
      caller who passes both does not know which one is right);
    * ``tau=`` carrying a ``tnp`` — refused BY NAME, because a fractional
      translation lives in [0, 1) and 2π·τ generally does not;
    * ``tnp=t`` and ``tau=t/2π`` — the same verdict, bit for bit.

    The fourth refusal is not available and the docstring says so: a
    ``tnp=`` that is really a τ is undetectable, because a symmorphic deck
    has ``tnp = 0 = τ`` and everything between is legal.
    """
    if not _have("si_cohsex_debug", "centroids_frac_960.txt"):
        pytest.skip("si_cohsex_debug not in this tree")
    S, tnp = _deck_syms("si_cohsex_debug")
    c = _centroids("si_cohsex_debug", "centroids_frac_960.txt")

    with pytest.raises(ValueError, match="exactly one of tnp= or tau="):
        verify_centroid_orbit_closure(c, S)
    with pytest.raises(ValueError, match="exactly one of tnp= or tau="):
        verify_centroid_orbit_closure(c, S, tnp=tnp, tau=tnp / (2 * np.pi))
    with pytest.raises(ValueError, match="look like BGW ``tnp``"):
        verify_centroid_orbit_closure(c, S, tau=tnp)

    a = verify_centroid_orbit_closure(c, S, tnp=tnp)
    b = verify_centroid_orbit_closure(c, S, tau=np.asarray(tnp) / (2 * np.pi))
    assert a.worst_residual == b.worst_residual
    assert a.violating_ops == b.violating_ops


def test_the_undivided_convention_would_have_condemned_the_closed_set():
    """RED TWIN, and the measurement behind the spec's warning.

    Feed the CLOSED 144-set the undivided ``tnp`` as if it were τ — the
    exact mistake — through the one door that will accept it, ``tnp=``
    holding ``2π·tnp``.  A set that is orbit-closed reads as violating.
    This is the failure mode the keyword split exists to make unreachable,
    written down rather than asserted about.
    """
    if not _have("si_cohsex_debug", "centroids_frac_144.txt"):
        pytest.skip("si_cohsex_debug not in this tree")
    S, tnp = _deck_syms("si_cohsex_debug")
    c = _centroids("si_cohsex_debug", "centroids_frac_144.txt")
    assert verify_centroid_orbit_closure(c, S, tnp=tnp).closed
    wrong = verify_centroid_orbit_closure(
        c, S, tnp=np.asarray(tnp) * (2.0 * np.pi))
    assert not wrong.closed, (
        "the 2π error must condemn the closed set; if it no longer does, "
        "this deck's τ went to zero and the test needs a non-symmorphic "
        "vehicle")
    assert wrong.n_violating > 0


# ---------------------------------------------------------------------------
# fft_grid: hash canonicalisation and the separate grid diagnosis
# ---------------------------------------------------------------------------

def test_the_grid_does_not_move_the_residual_but_does_move_the_hash():
    """``fft_grid`` is a hash rule and a guard, never part of the metric.

    If passing a grid snapped the coordinates, the 144-set's residual would
    collapse from 1.000e-06 to 0 and the spec's table would stop
    reproducing depending on an optional argument.  It does not.  What the
    grid changes is the hash: an integer-index digest, prefixed ``g:`` so
    it can never compare equal to the ``f:`` fractional digest of the same
    points.
    """
    if not _have("si_cohsex_debug", "centroids_frac_144.txt"):
        pytest.skip("si_cohsex_debug not in this tree")
    S, tnp = _deck_syms("si_cohsex_debug")
    c = _centroids("si_cohsex_debug", "centroids_frac_144.txt")
    no_grid = verify_centroid_orbit_closure(c, S, tnp=tnp)
    on_grid = verify_centroid_orbit_closure(c, S, tnp=tnp, fft_grid=_SI_FFT)
    assert on_grid.worst_residual == no_grid.worst_residual
    assert on_grid.closed == no_grid.closed
    assert no_grid.centroid_hash.startswith("f:")
    assert on_grid.centroid_hash.startswith("g:")
    assert on_grid.centroid_hash != no_grid.centroid_hash
    # Deterministic, and it separates sets.
    assert on_grid.centroid_hash == verify_centroid_orbit_closure(
        c, S, tnp=tnp, fft_grid=_SI_FFT).centroid_hash
    other = verify_centroid_orbit_closure(np.delete(c, 3, axis=0), S,
                                          tnp=tnp, fft_grid=_SI_FFT)
    assert other.centroid_hash != on_grid.centroid_hash


def test_an_incommensurate_grid_is_a_grid_diagnosis_not_a_closure_verdict():
    """A wrong ``fft_grid`` must not come back as "your set is not closed".

    Centroids that do not sit on the grid cannot have their images rounded
    onto grid points, so the closure test would fail for a reason that has
    nothing to do with the group — and the reader would go regenerate a set
    that was fine.  Separate refusal, separate message.

    The TWIN is the tolerance itself: the committed files carry six
    decimals, so ``0.083333 × 24 = 1.999992`` is off-grid by 8.000e-06 of a
    step (MEASURED).  That is the file's precision and must be accepted.
    """
    if not _have("si_cohsex_debug", "centroids_frac_144.txt"):
        pytest.skip("si_cohsex_debug not in this tree")
    S, tnp = _deck_syms("si_cohsex_debug")
    c = _centroids("si_cohsex_debug", "centroids_frac_144.txt")
    # TWIN: the real files, at their real precision, are accepted.
    verify_centroid_orbit_closure(c, S, tnp=tnp, fft_grid=_SI_FFT)
    scaled = c * np.asarray(_SI_FFT)
    assert float(np.abs(scaled - np.rint(scaled)).max()) < 1e-4

    with pytest.raises(ValueError, match="not commensurate with fft_grid"):
        verify_centroid_orbit_closure(c, S, tnp=tnp, fft_grid=(23, 23, 23))


def test_the_shape_refusals_are_shape_refusals():
    """Malformed input must name the shape, not produce a verdict."""
    S = np.stack([np.eye(3, dtype=np.int64)])
    tnp = np.zeros((1, 3))
    good = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    verify_centroid_orbit_closure(good, S, tnp=tnp)          # twin: accepted
    with pytest.raises(ValueError, match=r"centroids_frac must be"):
        verify_centroid_orbit_closure(good[:, :2], S, tnp=tnp)
    with pytest.raises(ValueError, match="empty centroid set"):
        verify_centroid_orbit_closure(np.zeros((0, 3)), S, tnp=tnp)
    with pytest.raises(ValueError, match=r"sym_matrices must be"):
        verify_centroid_orbit_closure(good, S[0], tnp=tnp)
    with pytest.raises(ValueError, match="translations must be"):
        verify_centroid_orbit_closure(good, S, tnp=np.zeros(3))
    with pytest.raises(ValueError, match="index the same"):
        verify_centroid_orbit_closure(good, S, tnp=np.zeros((2, 3)))
