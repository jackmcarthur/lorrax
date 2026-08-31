"""The Cartesian transpose contract: ``R_cart`` vs ``R_cart_forward`` (G5).

THE GAP THIS CLOSES.  Survey §7.2 G5: "The ``R_cart`` transpose contract's
only test is ``extra``-marked AND fixture-skipped" — ``tests/test_R_proper_
cri3.py`` carries ``pytestmark = pytest.mark.extra`` (deselected by the
suite's own ``addopts``) on top of a ``skipif`` against two hard-coded
``/pscratch/…/lorrax_sandbox/`` paths.  Doubly invisible, and it was the
only executable transpose contract in the tree.  Everything here runs on
in-tree deck headers and needs no CrI3 fixture (design decision 7).

WHAT THE PROPERTY IS.  ``R_cart`` is the Cartesian image of ``mtrx``, and
``mtrx`` is the INVERSE real-space rotation, so ``R_cart`` is the inverse
of the rotation carrying ``k_irr`` to ``S·k_irr``.  The spinor sandwich
wants it that way and is provably unaffected (two inversions cancel in the
transposed Shepperd form).  Anything rotating a Cartesian INDEX wants the
other one.  ``SymMaps.R_cart_forward`` is that other one, named — decision
5: an ADDITIVE property, no rename, consumer math untouched this wave.

WHY IT NEEDED A NAME AT ALL — the measurement, not the argument.  For an
ORTHOGONAL R the transpose is the inverse, so the wrong spelling is
invisible to every cheap check.  MEASURED on ``si_cohsex_debug`` (2026-08-07,
this file's own numbers, seed-0 vector):

    non-symmetric ops                       56 of 96
    rank-1  |R v| vs |Rᵀ v|                  delta 0.000e+00   (identical)
    rank-1  worst rel |R v − Rᵀ v|           2.000e+00        (a sign flip)
    rank-2  trace(R M Rᵀ) vs trace(Rᵀ M R)   delta 0.000e+00
    rank-2  Frobenius norm delta             0.000e+00
    rank-2  worst rel |A − B| at op 6        1.401e+00

Norm, trace and Frobenius norm are EXACTLY equal and the answer is
completely different.  That is the failure shape 061f8a3's caution
describes, and :func:`test_the_wrong_spelling_is_invisible_to_every_cheap_
check` is where it is made visible.

THE DECK MATTERS.  On ``gnppm_debug`` and ``bispinor_debug`` every op is
SYMMETRIC (0 of 4 non-symmetric — MoS2 3x3 is symmorphic with σ_h and the
Cartesian images come out symmetric), so ``R_cart_forward == R_cart`` there
and a cell parametrized over all four decks would be green and vacuous on
half of them.  ``test_two_of_the_four_decks_cannot_tell_the_spellings_apart``
writes that down.
"""

from __future__ import annotations

import numpy as np
import pytest

from _deck_stub import DECKS, deck_available, read_deck
from symmetry_maps import SymMaps

#: The decks whose ops are non-symmetric, i.e. where the two spellings are
#: different matrices.  MEASURED: si 56/96, cohsex 8/24, gnppm 0/4,
#: bispinor 0/4.
_DISCRIMINATING = ("si_cohsex_debug", "cohsex_debug")


def _sym(deck):
    pytest.importorskip("h5py", reason="h5py is not importable")
    if not deck_available(deck):
        pytest.skip(f"no {deck} WFN in this checkout (fixture blobs absent)")
    d = read_deck(deck)
    return d, SymMaps(d)


def _asymmetric_ops(R):
    return [i for i in range(R.shape[0])
            if float(np.abs(R[i] - R[i].T).max()) > 1e-8]


class _TrivialDeck:
    """``nosym``: ntran = 1, identity only.  The other __init__ branch."""

    def __init__(self):
        ax = [np.arange(n) / n for n in (2, 2, 1)]
        self.kpoints = np.stack(np.meshgrid(*ax, indexing="ij"),
                                -1).reshape(-1, 3)
        self.kgrid = np.asarray([2, 2, 1], dtype=np.int32)
        self.shift = np.zeros(3)
        self.nkpts = int(self.kpoints.shape[0])
        self.ntran = 1
        self.sym_matrices = np.eye(3, dtype=np.int32)[None]
        self.translations = np.zeros((1, 3))
        self.avec = np.eye(3)
        self.atom_types = np.array([1])
        self.atom_crys = np.zeros((1, 3))
        self.trs_holds = True


# ---------------------------------------------------------------------------
# The property itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck", DECKS)
def test_the_forward_rotation_is_the_transpose_per_op(deck):
    """Decision 5's defining equation: ``forward[s] == R_cart[s].T``.

    ``np.array_equal`` and not ``allclose``: a transpose is a view, there
    is no arithmetic, and a tolerance here would hide the one way this can
    go wrong (swapping the wrong pair of axes on a stacked array, which
    ``allclose`` would report as a difference of exactly the same size).
    """
    _, sym = _sym(deck)
    R = np.asarray(sym.R_cart)
    F = np.asarray(sym.R_cart_forward)
    assert F.shape == R.shape
    assert np.array_equal(F, np.swapaxes(R, -1, -2))
    for s in range(R.shape[0]):
        assert np.array_equal(F[s], R[s].T), f"{deck}: op {s}"


@pytest.mark.parametrize("deck", DECKS)
def test_every_op_is_orthogonal_and_the_transpose_is_the_inverse(deck):
    """``R Rᵀ = I`` per op — which is WHY the two spellings are confusable.

    The bar is the deck's own header precision, not a chosen tolerance:
    MEASURED 0.000e+00 on si_cohsex_debug, gnppm_debug and bispinor_debug
    (integer-orthogonal Cartesian images) and 2.798e-09 on cohsex_debug,
    whose hexagonal ``avec`` is stored to single-precision-ish accuracy in
    the header.  Both are asserted against 1e-6, and the exact-zero decks
    are asserted exact so a regression that introduced arithmetic where
    there was none shows up.
    """
    _, sym = _sym(deck)
    R = np.asarray(sym.R_cart)
    F = np.asarray(sym.R_cart_forward)
    worst = max(float(np.abs(R[s] @ R[s].T - np.eye(3)).max())
                for s in range(R.shape[0]))
    assert worst < 1e-6, f"{deck}: worst |R Rᵀ − I| = {worst:.3e}"
    if deck != "cohsex_debug":
        assert worst == 0.0, (
            f"{deck}'s Cartesian images used to be integer-exact "
            f"(measured 0.000e+00); now {worst:.3e}")
    # The transpose IS the inverse, which is the whole trap.
    for s in range(R.shape[0]):
        assert float(np.abs(F[s] @ R[s] - np.eye(3)).max()) < 1e-6


@pytest.mark.parametrize("deck", DECKS)
def test_the_determinants_follow_the_documented_row_convention(deck):
    """``(2·ntran, 3, 3)``, and the TRS half is ``−`` the spatial half.

    ``syms_crystal_to_cartesian`` builds ``concat([R_spatial, −R_spatial])``
    (:1505-1507), so negating a 3x3 flips the determinant and
    ``det[ntran:] == −det[:ntran]`` op for op.  It is NOT "proper first
    half, improper second": each half carries BOTH — MEASURED on Si, 24
    proper and 24 improper in each of the two 48-op halves — and a test
    written to the "first half is proper" reading would pass on a deck with
    no improper spatial ops and be wrong everywhere else.

    ``R_proper`` (:1188-1193) is the one with the det-flip applied, and it
    duplicates the SPATIAL half rather than negating it: that difference is
    asserted here so the two tables cannot silently converge.
    """
    d, sym = _sym(deck)
    R = np.asarray(sym.R_cart)
    n = int(d.ntran)
    assert R.shape == (2 * n, 3, 3)
    assert np.array_equal(R[n:], -R[:n])
    dets = np.rint(np.linalg.det(R.astype(float))).astype(int)
    assert set(dets.tolist()) <= {1, -1}, sorted(set(dets.tolist()))
    assert np.array_equal(dets[n:], -dets[:n])
    P = np.asarray(sym.R_proper)
    assert P.shape == (2 * n, 3, 3)
    assert np.array_equal(P[n:], P[:n]), (
        "R_proper's TRS half DUPLICATES the spatial half (:1192); negating "
        "it is the pre-2026-05-14 bug")
    assert np.allclose(np.linalg.det(P.astype(float)), 1.0, atol=1e-6)


def test_the_no_symmetry_branch_publishes_a_forward_rotation_too():
    """``ntran <= 1`` sets ``R_cart`` on a different line (:983).

    A property that read an attribute only the general branch defines would
    raise ``AttributeError`` on every ``nosym`` deck — the branch a caller
    reaches for exactly when they are debugging something else.  The
    identity is its own transpose, so the assertion is equality.
    """
    sym = SymMaps(_TrivialDeck())
    R = np.asarray(sym.R_cart)
    F = np.asarray(sym.R_cart_forward)
    assert R.shape == (2, 3, 3)
    np.testing.assert_array_equal(R[0], np.eye(3))
    np.testing.assert_array_equal(R[1], -np.eye(3))
    np.testing.assert_array_equal(F, R)


# ---------------------------------------------------------------------------
# G5 — the visibility test.  The point of the whole property.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck", _DISCRIMINATING)
def test_the_wrong_spelling_is_invisible_to_every_cheap_check(deck):
    """G5.  Same norm, same trace, same Frobenius norm, DIFFERENT answer.

    061f8a3's caution says the error "leaves norms, hermiticity and traces
    intact, so the error is invisible to the obvious checks".  Stated that
    way it is a warning; here it is a measurement, on a real deck, with
    both halves asserted:

    * the invariants agree to the DECK'S OWN orthogonality residual;
    * the answer differs by O(1) relative.

    MEASURED 2026-08-07, seed-0 vector, both discriminating decks::

        deck            asym    |R Rᵀ−I|   norm delta   worst rank-1 rel
        si_cohsex_debug 56/96   0.000e+00  0.000e+00    2.000e+00
        cohsex_debug     8/24   2.798e-09  6.840e-12    3.573e-01

        rank-2, A = R M Rᵀ vs B = Rᵀ M R
        si   op 6   rel 1.401e+00   trace delta 0.000e+00  fro delta 0.000e+00
        cohsex op 2 rel 1.051e+00   trace delta 9.870e-09  fro delta 1.096e-08

        si op 6:  R  v = [-0.13210486 -0.12573022  0.64042265]
                  Rᵀ v = [ 0.13210486  0.12573022  0.64042265]
                  |R v| = |Rᵀ v| = 0.665883589377457

    THE BAR IS THE DECK'S OWN PRECISION, NOT A NUMBER SOMEBODY CHOSE.  Si's
    Cartesian images are integer-exact and the invariants agree to 0.000e+00
    EXACTLY; cohsex's hexagonal ``avec`` is stored to ~3e-09 in the header
    and everything downstream inherits it.  So the invariant bar is derived
    from the measured ``|R Rᵀ − I|`` per deck, and what is asserted as the
    CLAIM is the RATIO — at least six orders between "the invariant moved"
    and "the answer moved".  A fixed 1e-13 would have been a tolerance
    tuned until cohsex passed, which is the thing this whole suite is
    against; it also fails, and did (6.840e-12 > 1e-13).

    A rank-1 Cartesian index is a dipole matrix element's vector index; the
    rank-2 arm is the shape a conductivity or a Lorentz-mixed 3-vertex
    block has.  Both are done here the way a consumer would do them, so the
    cell fails if the property ever stops being the other spelling.
    """
    _, sym = _sym(deck)
    R = np.asarray(sym.R_cart)
    F = np.asarray(sym.R_cart_forward)
    asym = _asymmetric_ops(R)
    assert asym, (
        f"PRECONDITION: {deck} must have at least one NON-SYMMETRIC op or "
        f"R_cart_forward == R_cart and this cell proves nothing")

    orth = max(float(np.abs(R[s] @ R[s].T - np.eye(3)).max())
               for s in range(R.shape[0]))
    rng = np.random.default_rng(0)
    v = rng.standard_normal(3)
    #: What an orthogonality residual of ``orth`` can move a norm by, with
    #: a decade of slack; exactly zero when the ops are integer-exact.
    bar = 1e3 * orth * float(np.linalg.norm(v))

    worst, invariant = 0.0, 0.0
    for s in asym:
        a, b = R[s] @ v, F[s] @ v
        invariant = max(invariant,
                        abs(np.linalg.norm(a) - np.linalg.norm(b)))
        worst = max(worst, float(np.abs(a - b).max() / np.abs(a).max()))
    assert invariant <= bar, (
        f"{deck} rank-1: the two spellings must agree on the NORM to the "
        f"deck's own orthogonality residual ({orth:.3e}); moved "
        f"{invariant:.3e} against a bar of {bar:.3e}.  That agreement is "
        f"what makes this whole class invisible.")
    if orth == 0.0:
        assert invariant == 0.0, (
            f"{deck}'s ops are integer-exact, so the norms must be EQUAL, "
            f"not close; moved {invariant:.3e}")
    assert worst > 0.1, (
        f"{deck}: rotating a rank-1 index with R_cart untransposed must give "
        f"a visibly different vector; worst rel {worst:.3e}")
    assert worst > 1e6 * max(invariant, 1e-16), (
        f"{deck}: the discriminating RATIO is the claim — the invariant "
        f"moved {invariant:.3e} and the answer moved {worst:.3e}, which is "
        f"not the orders-apart separation this cell is about")

    s = asym[0]
    M = rng.standard_normal((3, 3))
    M = 0.5 * (M + M.T)
    A = R[s] @ M @ R[s].T
    B = F[s] @ M @ F[s].T
    scale = float(np.abs(A).max())
    tr = abs(np.trace(A) - np.trace(B))
    fro = abs(np.linalg.norm(A) - np.linalg.norm(B))
    rel = float(np.abs(A - B).max() / scale)
    assert tr <= 1e3 * orth * scale + 1e-13, f"traces must agree: {tr:.3e}"
    assert fro <= 1e3 * orth * scale + 1e-13, (
        f"Frobenius norms must agree: {fro:.3e}")
    assert np.allclose(A, A.T) and np.allclose(B, B.T), (
        "hermiticity survives both spellings — that is the point")
    assert rel > 0.1, (
        f"{deck} op {s}: the rank-2 conjugation must differ visibly; "
        f"got {rel:.3e}")
    assert rel > 1e6 * max(tr, fro, 1e-16), (
        f"{deck} op {s}: invariants moved {max(tr, fro):.3e} and the answer "
        f"moved {rel:.3e}; the separation is the claim")


def test_two_of_the_four_decks_cannot_tell_the_spellings_apart():
    """RED TWIN for the deck choice — why the cell above is not a loop.

    On gnppm_debug and bispinor_debug every Cartesian op is SYMMETRIC, so
    ``R_cart_forward`` IS ``R_cart`` and the visibility test would be a
    green no-op.  MEASURED: 0 of 4 non-symmetric on both, against 56 of 96
    on Si and 8 of 24 on cohsex.  Writing this down is what stops a later
    "why not parametrize over all four?" from deleting the contract while
    adding coverage.
    """
    pytest.importorskip("h5py", reason="h5py is not importable")
    vacuous, sharp = [], []
    for deck in DECKS:
        if not deck_available(deck):
            continue
        _, sym = _sym(deck)
        R = np.asarray(sym.R_cart)
        (sharp if _asymmetric_ops(R) else vacuous).append(deck)
    if not vacuous and not sharp:
        pytest.skip("no decks in this checkout")
    assert set(vacuous) <= {"gnppm_debug", "bispinor_debug"}, vacuous
    assert set(sharp) <= set(_DISCRIMINATING), sharp
    assert sharp, (
        "no in-tree deck distinguishes R_cart from its transpose any more; "
        "the G5 visibility test has nowhere to run and decision 5's "
        "property is untestable in this tree")


def test_the_forward_property_is_not_a_stored_attribute():
    """It is DERIVED, so it cannot go stale against ``R_cart``.

    A cached copy set in ``__init__`` would be a second source of truth for
    the same three numbers, and the failure mode is the one this service
    already has a receipt for: two tables encoding one policy and drifting.
    Asserted structurally (it is a ``property`` on the class) and
    behaviourally (mutating ``R_cart`` moves it).
    """
    sym = SymMaps(_TrivialDeck())
    assert isinstance(getattr(type(sym), "R_cart_forward", None), property)
    assert "R_cart_forward" not in vars(sym)
    sym.R_cart = np.array([[[0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0]]])
    assert np.array_equal(np.asarray(sym.R_cart_forward)[0],
                          np.asarray(sym.R_cart)[0].T)
