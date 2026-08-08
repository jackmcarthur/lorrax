"""The off-diagonal symmetry gate the diagonal one is structurally blind to.

F3.  ``tests/harness.py``'s ``compare_to_bgw`` reports a ``_star_spread``
computed from the REAL DIAGONAL ``sigTOT`` values of a star's members.  It
is the right check for what it checks — agreement with the BerkeleyGW
anchor — and it CANNOT see the conjugation class at all, because
conjugating a Hermitian block leaves its real diagonal exactly intact.
27cc885 measured that class at **183.61 eV** with "the DIAGONAL left
exactly intact"; the electron count, hermiticity, the spectrum and the
eqp.dat V_H column all survived it, and nothing in the suite turned red for
a month.

So this file asks the off-diagonal question where it is answerable: on the
committed FULL MATRICES in ``tests/regression/cohsex_debug/sigma_mnk.h5``
(``hartree_kij_ev`` and ``sigma_sx_kij_ev``, both ``(9, 30, 30)``
complex128, k axis in the ``SymMaps`` table order — the ``Eo``-vector
partition from ``eqp_ref.dat`` is ``[[0], [1,2,3,5,6,7], [4,8]]``, which is
the star partition exactly).

THE PAIRS ARE DERIVED, NEVER HAND-LISTED.  Θ is antiunitary, so two members
of one star are related by a conjugation iff their ``_star_conj_flags`` XOR
flags DIFFER.  Two things follow and only the first is gated here:

* **pure time-reversal partners** — same spatial op, one with time reversal
  and one without, i.e. ``sym_idx[i] % ntran == sym_idx[j] % ntran`` — carry
  the relation with NO spatial rotation in between.  Derived, they are
  ``(1,2)``, ``(3,6)``, ``(5,7)``.  These are the gate.
* **spatial partners** — anything with a genuine rotation between them —
  are BROKEN on this deck at 1.8e-01 to 4.0e-01, and that is the §8.2
  register item, not a defect in the symmetry code: ``cohsex_debug``'s
  60-centroid ISDF quadrature is not orbit-closed, so the quadrature itself
  breaks the spatial group.  NOT GATED.  Asserted as a measured fact
  instead, so the hole is a named one.

MEASURED 2026-08-07 (rel Frobenius against the first member's norm)::

    pair    kind      hartree conj   hartree plain   sx conj    sx plain
    (1,2)   pure TRS   7.514e-07      3.577e-01     6.980e-04   3.992e-01
    (3,6)   pure TRS   2.629e-04      3.478e-01     5.707e-04   4.041e-01
    (5,7)   pure TRS   2.649e-04      3.437e-01     6.112e-04   4.004e-01
    (1,3)   spatial    1.848e-01      3.522e-01     2.209e-01   3.988e-01
    (4,8)   spatial    2.735e-01      2.469e-01     3.890e-01   3.735e-01

Three to six orders between the relation and its removal on the gated
pairs, with the anti-tautology arm built in.  The whole-star ``star_spread``
on this fixture is 113.08 (hartree) and 8.89 (sigma_sx) — dominated by the
broken spatial relations — so it is NOT a usable gate on this deck either,
which is the second reason this file exists.

FIXTURES ARE READ-ONLY.  ``tests/regression/*`` is chmod'd read-only on
purpose and ``harness.protect_fixtures()`` re-applies it at every session
start.  The corruption twin copies the file to ``tmp_path`` with
``shutil.copyfile`` — a COPY, never a symlink: a symlinked stage destroyed
this very ``sigma_mnk.h5`` on 2026-07-25 with no error and no test failure
(``cohsex_debug/README.md:29-42``) — and asserts the original's size and
mtime are unchanged afterwards.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from ffi import _services

_services.ensure_on_path()

from symmetry_maps import SymMaps                               # noqa: E402
from symmetry_maps.maps import _star_conj_flags                 # noqa: E402

h5py = pytest.importorskip("h5py")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DECK = os.path.join(_REPO, "tests", "regression", "cohsex_debug")
_SIGMA = os.path.join(_DECK, "sigma_mnk.h5")
_WFN = os.path.join(_DECK, "WFNsmall.h5")

#: The two datasets the relation is gated on.  ``sigma_xc_qsgw_kij_ev`` is
#: the same shape and is deliberately NOT gated: it is a derived
#: combination, so a failure there would not say which input moved.
_GATED = ("hartree_kij_ev", "sigma_sx_kij_ev")

#: Ceiling for the conjugation relation on a pure-TRS pair, and floor for
#: its removal.  MEASURED 7.5e-07 .. 7.0e-04 and 3.4e-01 .. 4.0e-01 — the
#: bar sits an order above the worst relation and an order below the
#: weakest anti-tautology signal, so neither is tuned to a measurement.
_REL_MAX = 5e-3
_PLAIN_MIN = 0.1


def _need_fixtures():
    for p in (_SIGMA, _WFN):
        if not os.path.isfile(p):
            pytest.skip(f"no {os.path.relpath(p, _REPO)} in this checkout "
                        f"(fixture blobs absent)")


class _Header:
    """The eleven attributes ``SymMaps`` reads, out of ``mf_header``."""

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            g = f["mf_header"]
            avec = g["crystal/avec"][:]
            apos = g["crystal/apos"][:]
            self.kpoints = g["kpoints/rk"][:]
            self.kgrid = g["kpoints/kgrid"][:]
            self.shift = g["kpoints/shift"][:]
            self.nkpts = int(g["kpoints/nrk"][()])
            self.ntran = int(g["symmetry/ntran"][()])
            self.sym_matrices = g["symmetry/mtrx"][:]
            self.translations = g["symmetry/tnp"][:]
            self.avec = avec
            self.atom_types = g["crystal/atyp"][:]
            self.atom_crys = np.einsum("ij,kj->ki",
                                       np.linalg.inv(avec).T, apos)
            self.trs_holds = True


def _star_pairs():
    """``(pure_trs, spatial)`` — every within-star pair, classified.

    Both lists come out of ``SymMaps`` plus ``_star_conj_flags`` through
    the service door.  ``conj`` is the XOR of the two members' flags, which
    is the antiunitary rule; ``pure_trs`` additionally requires the two
    rows to share a spatial op, so that the relation between them involves
    NO rotation and therefore no ISDF quadrature.
    """
    sym = SymMaps(_Header(_WFN))
    irr = np.asarray(sym.irr_idx_k)
    sidx = np.asarray(sym.sym_idx_k)
    nss = int(np.asarray(sym.sym_mats_k).shape[0]) // 2
    _, xor = _star_conj_flags(irr, sidx, nss)
    pure, spatial = [], []
    for i in range(len(irr)):
        for j in range(i + 1, len(irr)):
            if irr[i] != irr[j]:
                continue
            conj = bool(xor[i]) != bool(xor[j])
            if int(sidx[i]) % nss == int(sidx[j]) % nss:
                pure.append((i, j, conj))
            else:
                spatial.append((i, j, conj))
    return pure, spatial, (irr, sidx, nss)


def _rel(a, b):
    """``‖a − b‖_F / ‖a‖_F`` — relative to the FIRST member's norm."""
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)).ravel())
                 / np.linalg.norm(np.asarray(a).ravel()))


def _read(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][:] for k in _GATED}


def _diagonal_star_metric(M, irr):
    """``compare_to_bgw``'s metric, recomputed here: real diag, per-band
    max−min within a star, worst over stars and bands.

    Written out rather than imported because the harness computes it from
    a parsed ``.dat``; the claim being pinned is about the METRIC's shape,
    so the shape is what is reproduced.
    """
    worst = 0.0
    for lab in sorted({int(v) for v in irr}):
        mem = [k for k in range(len(irr)) if int(irr[k]) == lab]
        if len(mem) < 2:
            continue
        blk = np.stack([np.real(np.diag(M[k])) for k in mem])
        worst = max(worst, float((blk.max(0) - blk.min(0)).max()))
    return worst


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_the_pairs_are_the_three_the_tables_imply():
    """PRECONDITION.  The gate is only worth running on these pairs.

    Derived, not typed: ``(1,2)``, ``(3,6)``, ``(5,7)``.  If the
    op-selection policy moves (survey §8.1, the 15.9 eV fork) this set
    changes and the gate below is measuring something else — so it FAILS
    here rather than quietly gating a different claim.  Every pure-TRS pair
    must also be a CONJUGATION pair: two rows sharing a spatial op and
    differing only by time reversal cannot have equal XOR flags.
    """
    _need_fixtures()
    pure, spatial, (irr, sidx, nss) = _star_pairs()
    assert nss == 12
    assert [(i, j) for i, j, _ in pure] == [(1, 2), (3, 6), (5, 7)], (
        f"the pure-TRS pair set moved to {[(i, j) for i, j, _ in pure]}; "
        f"the op-selection policy is register-don't-touch (survey §8.1)")
    assert all(c for _, _, c in pure), (
        "a pure time-reversal partner must be a conjugation pair; if it is "
        "not, _star_conj_flags' XOR and sym_idx disagree about which rows "
        "are time-reversed")
    assert spatial, "no spatial pairs at all; the §8.2 arm has nothing to say"


@pytest.mark.parametrize("dataset", _GATED)
def test_the_conjugation_relation_holds_off_the_diagonal(dataset):
    """THE GATE.  ``M[j] == conj(M[i])`` on every pure-TRS pair.

    OFF the diagonal is where this lives: the same assertion restricted to
    real diagonals is satisfied by ANY conjugation state and is exactly
    what ``compare_to_bgw`` already does.  So the residual is taken over
    the whole matrix, and the anti-tautology arm — the same comparison
    without the conjugation — is asserted LARGE in the same cell, because a
    "small residual" that would also be small without the conj proves
    nothing.

    MEASURED: relation 7.514e-07 / 2.629e-04 / 2.649e-04 (hartree) and
    6.980e-04 / 5.707e-04 / 6.112e-04 (sigma_sx); removal 3.44e-01 to
    4.04e-01.  Three to six orders.
    """
    _need_fixtures()
    pure, _, _ = _star_pairs()
    M = _read(_SIGMA)[dataset]
    for i, j, conj in pure:
        assert conj
        rel = _rel(M[i], np.conj(M[j]))
        plain = _rel(M[i], M[j])
        assert rel < _REL_MAX, (
            f"{dataset} pair ({i},{j}): the time-reversal conjugation "
            f"relation is broken at {rel:.3e} (measured 7.5e-07..7.0e-04 "
            f"when this gate was written).  This is the 183.61 eV class and "
            f"it is invisible to every diagonal observable.")
        assert plain > _PLAIN_MIN, (
            f"{dataset} pair ({i},{j}): removing the conjugation must be a "
            f"LARGE signal or the assertion above is a tautology; got "
            f"{plain:.3e}")
        assert plain > 100 * rel, (
            f"{dataset} pair ({i},{j}): relation {rel:.3e} vs its removal "
            f"{plain:.3e} — not the orders-apart separation this gates")


@pytest.mark.parametrize("dataset", _GATED)
def test_the_relation_is_genuinely_off_diagonal(dataset):
    """Why the gate is on the full matrix and not on the diagonal.

    Restrict the SAME comparison to the real diagonal and the "removal"
    signal collapses to zero: ``conj`` is the identity on a real number.
    That is the blindness ``compare_to_bgw``'s docstring now states, shown
    here on the same data the gate above uses, so the two statements cannot
    drift apart.
    """
    _need_fixtures()
    pure, _, _ = _star_pairs()
    M = _read(_SIGMA)[dataset]
    for i, j, _c in pure:
        di, dj = np.real(np.diag(M[i])), np.real(np.diag(M[j]))
        with_conj = _rel(di, dj)          # conj is the identity here
        assert _rel(di, np.real(np.diag(np.conj(M[j])))) == with_conj, (
            "conjugating changed a REAL diagonal; then the 183.61 eV class "
            "would not have been invisible and this whole file is moot")


@pytest.mark.parametrize("dataset", _GATED)
def test_the_spatial_pairs_are_broken_and_that_is_the_register_item(dataset):
    """§8.2, measured on cohsex_debug — REPORTED, not gated.

    ``cohsex_debug``'s ``centroids_frac_60.txt`` is not orbit-closed, so
    the ISDF quadrature does not respect the 12-op spatial group and every
    pair with a genuine rotation between it disagrees at O(0.1-0.4)
    whichever conjugation you apply.  MEASURED: (1,3) 1.848e-01 / (4,8)
    2.735e-01 on hartree, 2.209e-01 / 3.890e-01 on sigma_sx.

    This is asserted as a FACT rather than left out, for two reasons.  A
    silently ungated pair is indistinguishable from a forgotten one; and if
    the production centroids are ever regenerated orbit-closed (an OWNER
    row — it means re-freezing the BerkeleyGW anchor) this cell fails and
    tells whoever did it that the spatial arm can now be gated too.
    """
    _need_fixtures()
    _, spatial, _ = _star_pairs()
    M = _read(_SIGMA)[dataset]
    worst_best = 0.0
    for i, j, conj in spatial:
        best = min(_rel(M[i], np.conj(M[j])), _rel(M[i], M[j]))
        worst_best = max(worst_best, best)
    assert worst_best > 0.1, (
        f"{dataset}: the spatial relations now hold to {worst_best:.3e}.  "
        f"They were BROKEN at 1.8e-01..3.9e-01 by the non-orbit-closed "
        f"60-centroid ISDF quadrature (survey §8.2).  If the centroids were "
        f"regenerated orbit-closed, gate the spatial pairs here too.")


# ---------------------------------------------------------------------------
# The red twin — and the blindness, pinned
# ---------------------------------------------------------------------------

def test_a_conjugated_member_fires_the_gate_and_moves_the_old_metric_by_zero(
        tmp_path):
    """THE TWIN.  One corrupted row: new gate fires, old metric does not.

    Both halves are the point.  Conjugating member row 2 — a pure-TRS
    partner of row 1 — is exactly the corruption 27cc885's bug produced,
    and on the committed fixture it:

    * takes the conj-pair relation from 6.980e-04 to 3.992e-01, five
      orders, so the new gate refuses;
    * moves ``compare_to_bgw``'s diagonal star metric by **EXACTLY 0.0** —
      1.2130460739135742 before and after on ``sigma_sx_kij_ev``,
      61.512237548828125 on ``hartree_kij_ev``.  Asserted with ``==``, not
      a tolerance, because "exactly 0.0" is the claim.

    The corruption happens on a COPY under ``tmp_path``
    (``shutil.copyfile`` — a copy, never a symlink; a symlinked stage
    destroyed this file on 2026-07-25 with no error and no test failure),
    and the original's size and mtime are checked afterwards.
    """
    _need_fixtures()
    pure, _, (irr, sidx, nss) = _star_pairs()
    i, j, _c = pure[0]
    assert (i, j) == (1, 2)

    before_stat = os.stat(_SIGMA)
    dst = tmp_path / "sigma_mnk.h5"
    shutil.copyfile(_SIGMA, dst)
    os.chmod(dst, 0o644)
    with h5py.File(dst, "r+") as f:
        for name in _GATED:
            f[name][j] = np.conj(f[name][j][:])

    clean = _read(_SIGMA)
    broken = _read(str(dst))
    for name in _GATED:
        was = _rel(clean[name][i], np.conj(clean[name][j]))
        now = _rel(broken[name][i], np.conj(broken[name][j]))
        assert was < _REL_MAX, f"{name}: the clean fixture must pass first"
        assert now > _PLAIN_MIN, (
            f"{name}: the gate did not fire on a conjugated member "
            f"({was:.3e} -> {now:.3e})")
        assert now > 100 * was

        # ...and THE BLINDNESS.  Exactly unchanged.
        m_clean = _diagonal_star_metric(clean[name], irr)
        m_broken = _diagonal_star_metric(broken[name], irr)
        assert m_clean > 0.0, (
            f"{name}: the diagonal star metric is 0 on the clean fixture, "
            f"so 'unchanged' would be vacuous")
        assert m_broken == m_clean, (
            f"{name}: the diagonal star metric moved from {m_clean!r} to "
            f"{m_broken!r} under a conjugation.  It is supposed to be "
            f"EXACTLY blind to this class — that blindness is why 183.61 eV "
            f"survived a month, and it is what compare_to_bgw's docstring "
            f"now states.")

    after_stat = os.stat(_SIGMA)
    assert (after_stat.st_size, after_stat.st_mtime) == (
        before_stat.st_size, before_stat.st_mtime), (
        "the committed fixture was modified; corruptions belong on the "
        "tmp_path copy")


def test_the_diagonal_metric_is_not_blind_to_everything(tmp_path):
    """RED TWIN FOR THE BLINDNESS ITSELF.

    "Moved by exactly 0.0" would also be true of a metric that always
    returns a constant.  Perturb a real DIAGONAL entry of the same member
    and the same metric must move — which is what says the ``== 0.0`` above
    is about the conjugation class and not about the metric being inert.
    """
    _need_fixtures()
    _pure, _sp, (irr, sidx, nss) = _star_pairs()
    clean = _read(_SIGMA)
    name = "sigma_sx_kij_ev"
    base = _diagonal_star_metric(clean[name], irr)
    bumped = np.array(clean[name])
    bumped[2, 0, 0] += 10.0
    assert _diagonal_star_metric(bumped, irr) > base + 1.0, (
        "the diagonal star metric did not notice a 10 eV shift on a real "
        "diagonal entry; it is inert, not merely blind to conjugation")
