"""G1: ``broadcast_ibz_to_full_bz`` pinned end-to-end on the committed decks.

THE GAP.  Survey §7.2 ranked this CRITICAL and first: *"``gw.kin_ion_io.
broadcast_ibz_to_full_bz`` has ZERO test references anywhere in the tree"*
— grep found only ``kin_ion_io.py:193, :749, :1090`` and one docstring
cross-reference.  It is the function whose two-commit interaction (3e002f2
undoing f7ef931's antiunitary unfold, then 27cc885 restoring it through an
explicit ``trs_reference``) cost 183.61 eV in a single night, and nothing
collected exercised it.

WHAT THE REFERENCES ARE.  ``tests/regression/<deck>/kin_ion.h5`` holds one
dataset, ``kin_ion``, of shape ``(nk_full, nb, nb)`` complex128, with the
k axis in ``SymMaps``' full-grid order.  Read in place and never written:
``tests/regression/*`` is chmod'd read-only on purpose and
``harness.protect_fixtures()`` re-applies it at every session start
(``cohsex_debug/README.md:44-52``), and a symlinked stage destroyed
``sigma_mnk.h5`` on 2026-07-25 with no error and no test failure (:29-42).
Nothing here copies, chmods or symlinks anything.

THE NORM IS FROBENIUS-RELATIVE, ``‖Δ‖_F / ‖A‖_F``, throughout, because
that is the convention the measurement dossier's §3 table was taken in and
every number below reproduces it to four digits.  It also happens to be the
norm that DISTINGUISHES the two wrong answers: under a max-abs norm
``star_row`` and ``plain`` both come back 7.050e-01 on gnppm (their worst
single element is the same element), and the fact that ``star_row`` is
wrong on eight rows where ``plain`` is wrong on four — a factor of exactly
√2 in Frobenius, 5.553e-01 / 3.927e-01 = 1.4141 — is visible in one and
invisible in the other.

MEASURED 2026-08-07 on this tree, and these are the assertions' provenance::

    deck              ibz_slab     star_row      plain     sorted-stack
    gnppm_debug       3.717e-16    5.553e-01   3.927e-01     4.501e-01
    bispinor_debug    1.232e-03    2.317e-01   1.638e-01     1.729e-01
    si_cohsex_debug   5.130e-04    5.130e-04   5.130e-04     5.130e-04

Si is the NEGATIVE CONTROL: it has zero time-reversed rows, so all four
columns are the same number and no conjugation happens anywhere.  Its
5.130e-04 is the committed file's own independent computation, not an
unfold error — see :func:`test_si_wants_no_conjugation_at_all`.

AND SINCE THE BROADCAST MOVED TO READ TIME, the second half of this file
gates the FORMAT that move created (``§ THE STORED FORMAT`` below): a file
written on the IBZ reads back bit-identical to the full write it replaces,
a file that lies about being one refuses, and — the assertion that protects
every fixture above — a file with no ``k_storage`` attr is still read
verbatim.  That last one is not politeness.  Four of the six committed
``kin_ion.h5`` were computed independently at every full-BZ k and their
rows do NOT satisfy the star relation (measured ``unfold(select(A))``
against ``A``: 3.557e-14, 2.001e-03, 1.726e-01 and 7.782e+00 Ry on
gnppm / si_cohsex / bispinor / cohsex).  Reinterpreting one of those as
compressible would move physics by up to 7.8 Ry, and the only thing
standing between the reader and that is a default.

MEASURED on the two fixtures this generator actually wrote —
``si_bse_debug`` (nk 64, nrk 8) and ``hbn_cohsex_debug`` (nk 18, nrk 18) —
``unfold(select(A))`` is bit-identical to the committed array on BOTH
``kin_ion`` and ``v_hartree``, max|Δ| exactly 0.000e+00, and si_bse_debug's
payload goes 7.3728 MB → 0.9216 MB, 8.00×.
"""

from __future__ import annotations

import ast
import os
import pathlib

import numpy as np
import pytest

from ffi import _services

_services.ensure_on_path()

import symmetry_maps                                            # noqa: E402
from symmetry_maps import SymMaps                               # noqa: E402
from symmetry_maps.maps import _star_row_order                  # noqa: E402

h5py = pytest.importorskip("h5py")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REG = os.path.join(_REPO, "tests", "regression")
_KIN_ION_IO = os.path.join(_REPO, "src", "gw", "kin_ion_io.py")
#: Where the adapter — and therefore the predicate — lives NOW.  It moved
#: out of ``src/gw/kin_ion_io.py`` when the broadcast moved from write time
#: to read time; :func:`test_the_call_site_passes_ibz_slab_as_a_literal`
#: followed it, and gained a second arm asserting the writer has no
#: ``star_broadcast`` call left.
_FILE_IO_KIN_ION = os.path.join(_REPO, "src", "file_io", "kin_ion.py")


def _adapter_or_skip():
    """``gw.kin_ion_io.broadcast_ibz_to_full_bz``, imported LAZILY.

    Importing ``gw.kin_ion_io`` runs ``runtime.initialize_communicator_
    stack``, which REFUSES without a built FFI ``.so`` (docs/architecture/
    decisions.md, 2026-08-01 — the FFI layer is required).  So the import
    cannot sit at module scope: on a machine with no library it would turn
    this whole file into a collection error, which is how
    ``test_kin_ion_padded_gvectors.py`` earned its one entry in the
    baseline red set.  ``tests/test_kin_ion_padded_gvectors.py:314``
    records the same constraint and solves it the same way.

    THE PREDICATE ITSELF IS PINNED WITHOUT THE IMPORT.  The call site's
    ``trs_reference="ibz_slab"`` is checked by parsing the FILE
    (:func:`test_the_call_site_passes_ibz_slab_as_a_literal`), which is
    the assertion 27cc885's regression would have failed, and it runs on
    every machine.  What skips here is only the behavioural arm.
    """
    try:
        from gw.kin_ion_io import broadcast_ibz_to_full_bz
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(
            f"gw.kin_ion_io needs a built FFI .so at import "
            f"({type(exc).__name__}) — covered by any leg with the library "
            f"(Perlmutter/Frontera); the call-site literal and the whole "
            f"star algebra are pinned here without it")
    return broadcast_ibz_to_full_bz


#: The decks whose ``kin_ion.h5`` k axis is the FULL grid.  cohsex_debug
#: ships one too, but its IBZ is a genuine reduction (nrk 4 vs nk_tot 9)
#: and that file's k axis is the IBZ one, so the rebuild-from-``sym_idx==0``
#: recipe below does not apply to it; all three here have nrk == nk_tot.
_DECKS = ("gnppm_debug", "bispinor_debug", "si_cohsex_debug")

#: The dossier's §3 table, as the assertion CLASSES.  Quoted values are
#: what this tree measures today; the assertions are order-of-magnitude
#: separations, because the bispinor floor (1.2e-03) is a property of how
#: that file was written and not of the unfold.
_EXPECT = {
    # deck: (ibz_slab measured, ceiling, wrong-flavour floor)
    "gnppm_debug": (3.717e-16, 1e-12, 0.3),
    "bispinor_debug": (1.232e-03, 5e-3, 1e-1),
}


def _deck_or_skip(deck):
    wfn = os.path.join(_REG, deck, "WFN.h5")
    kin = os.path.join(_REG, deck, "kin_ion.h5")
    for p in (wfn, kin):
        if not os.path.isfile(p):
            pytest.skip(f"no {os.path.relpath(p, _REPO)} in this checkout "
                        f"(fixture blobs absent)")
    return wfn, kin


class _Header:
    """The eleven attributes ``SymMaps`` reads, straight out of mf_header.

    A header stub rather than ``WfnLoader`` so this cell costs milliseconds
    on a 44 MiB file and depends on nothing but h5py.  The service suite's
    ``services/symmetry_maps/tests/test_symmetry_maps_deck_tables.py``
    carries the parity arm proving the stub and the production loader build
    the same tables; this file leans on that rather than repeating it.
    """

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


def _tables(deck):
    """``(A_full, sym, irr, sidx, n_sym_spatial, labels, rep_rows)``.

    ``rep_rows[j]`` is the full-BZ row the file holds UNTRANSFORMED for
    star ``labels[j]`` — the first row whose ``sym_idx`` is 0.  Those rows
    are the IBZ SLAB: read verbatim with no symmetry operation applied, so
    their TRS flag is False by construction, which is exactly the case
    ``trs_reference="ibz_slab"`` is for.
    """
    wfn_path, kin_path = _deck_or_skip(deck)
    with h5py.File(kin_path, "r") as f:
        assert list(f.keys()) == ["kin_ion"], list(f.keys())
        A_full = f["kin_ion"][:]
    sym = SymMaps(_Header(wfn_path))
    irr = np.asarray(sym.irr_idx_k)
    sidx = np.asarray(sym.sym_idx_k)
    nss = int(np.asarray(sym.sym_mats_k).shape[0]) // 2
    _, labels = _star_row_order(irr)
    reps = {}
    for k in range(len(irr)):
        if int(sidx[k]) == 0 and int(irr[k]) not in reps:
            reps[int(irr[k])] = k
    assert set(reps) == {int(v) for v in labels}, (
        f"{deck}: some star has no untransformed (sym_idx == 0) row, so the "
        f"file carries no IBZ slab to rebuild A_irr from")
    return A_full, sym, irr, sidx, nss, labels, reps


def _rel(got, ref):
    """``‖got − ref‖_F / ‖ref‖_F`` — the dossier's norm."""
    return float(np.linalg.norm((np.asarray(got) - ref).ravel())
                 / np.linalg.norm(ref.ravel()))


# ---------------------------------------------------------------------------
# I10 — the predicate, on the real files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck", ["gnppm_debug", "bispinor_debug"])
def test_only_the_ibz_slab_predicate_reproduces_the_committed_file(deck):
    """I10.  Three unfolds of one operand; exactly one is the file.

    ``A_irr`` is rebuilt from the file's own ``sym_idx == 0`` rows, so it
    IS the IBZ slab and ``trs_reference="ibz_slab"`` is the correct
    predicate by construction.  The other two are the two ways to get it
    wrong, and both are asserted LARGE rather than merely different:

    * ``"star_row"`` — 27cc885's bug.  Conjugates by the XOR against the
      star's kept row, which is the right rule for a ``star_select``
      output and the wrong one here.
    * plain gather — 3e002f2's regression of f7ef931.  No conjugation at
      all.

    MEASURED: gnppm 3.717e-16 / 5.553e-01 / 3.927e-01; bispinor
    1.232e-03 / 2.317e-01 / 1.638e-01.  The bispinor FLOOR is 1.2e-03 and
    not 1e-15 because that file's rows were not written by this broadcast;
    the claim is the class ordering, which is three orders wide either way.
    """
    A_full, sym, irr, sidx, nss, labels, reps = _tables(deck)
    A_irr = A_full[[reps[int(v)] for v in labels]]
    quoted, ceiling, floor = _EXPECT[deck]

    slab = symmetry_maps.star_broadcast(A_irr, irr, sidx, nss,
                                        irr_labels=labels,
                                        trs_reference="ibz_slab")
    row = symmetry_maps.star_broadcast(A_irr, irr, sidx, nss,
                                       irr_labels=labels,
                                       trs_reference="star_row")
    pos = {int(v): i for i, v in enumerate(labels)}
    plain = A_irr[[pos[int(v)] for v in irr]]

    got = _rel(slab, A_full)
    assert got < ceiling, (
        f"{deck}: the ibz_slab unfold must reproduce the committed file "
        f"(measured {quoted:.3e} when this test was written); got "
        f"{got:.3e} against a ceiling of {ceiling:.0e}")
    assert _rel(row, A_full) > floor, (
        f"{deck}: the star_row predicate must be VISIBLY wrong here — that "
        f"is 27cc885's 183.61 eV; got {_rel(row, A_full):.3e}")
    assert _rel(plain, A_full) > floor, (
        f"{deck}: a plain gather must be VISIBLY wrong — that is 3e002f2's "
        f"regression of f7ef931; got {_rel(plain, A_full):.3e}")
    # And the two wrong answers are DIFFERENT wrongs: star_row conjugates
    # eight rows where plain conjugates none, so in Frobenius it is worse
    # by exactly sqrt(2) (5.553e-01 / 3.927e-01 = 1.4141).  Under a max-abs
    # norm both are 7.050e-01 and this distinction disappears.
    assert _rel(row, A_full) > _rel(plain, A_full)


@pytest.mark.parametrize("deck", ["gnppm_debug", "bispinor_debug"])
def test_the_error_lands_on_exactly_the_time_reversed_rows(deck):
    """I11, the anti-tautology.  Not "some conjugation happened" — WHERE.

    With a plain gather the residual must be nonzero on EXACTLY the rows
    whose ``sym_idx >= n_sym_spatial`` and exactly zero on the others.  A
    test that only measured a norm would pass on a unfold that conjugated
    the wrong four rows and got a similar number.

    MEASURED on both decks: the error rows are ``{1, 3, 4, 5}`` and the
    TRS rows are ``{1, 3, 4, 5}``.  On gnppm the file's TRS rows are
    EXACTLY ``conj(rep)`` — per-row residual 4.0e-16 to 7.6e-16.
    """
    A_full, sym, irr, sidx, nss, labels, reps = _tables(deck)
    pos = {int(v): i for i, v in enumerate(labels)}
    A_irr = A_full[[reps[int(v)] for v in labels]]
    plain = A_irr[[pos[int(v)] for v in irr]]
    scale = float(np.abs(A_full).max())
    err = np.array([float(np.abs(plain[k] - A_full[k]).max()) / scale
                    for k in range(len(irr))])
    bad = {int(k) for k in np.where(err > 1e-10)[0]}
    trs = {int(k) for k in np.where(sidx >= nss)[0]}
    assert trs == {1, 3, 4, 5}, (
        f"PRECONDITION: {deck}'s time-reversed rows moved to {sorted(trs)}; "
        f"the op-selection policy is register-don't-touch (survey §8.1)")
    assert bad == trs, (
        f"{deck}: the plain gather's error must land on exactly the "
        f"time-reversed rows.  error rows {sorted(bad)}, TRS rows "
        f"{sorted(trs)}")


def test_the_gnppm_file_holds_conj_of_its_representative_exactly():
    """The mechanism behind I11, read straight off the file.

    Θ is antiunitary, so a time-reversed row's value is ``conj`` of the row
    it was reached from.  On gnppm that is EXACT — the file was written by
    this broadcast — and measuring it per row is what makes the norm-level
    assertions above statements about the RULE rather than about a total.

    MEASURED: 4.0e-16 to 7.6e-16 relative, on all four TRS rows.
    """
    A_full, sym, irr, sidx, nss, labels, reps = _tables("gnppm_debug")
    scale = float(np.abs(A_full).max())
    worst = 0.0
    for k in np.where(sidx >= nss)[0]:
        rep = A_full[reps[int(irr[k])]]
        worst = max(worst,
                    float(np.abs(A_full[k] - np.conj(rep)).max()) / scale)
    assert worst < 1e-14, f"per-row conj relation is only {worst:.3e}"


# ---------------------------------------------------------------------------
# I2 — the stacking order, and the trap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck", ["gnppm_debug", "bispinor_debug"])
def test_stacking_a_irr_in_sorted_label_order_is_wrong(deck):
    """RED TWIN for the reconstruction, and a trap that is easy to fall in.

    ``A_irr``'s rows are addressed by POSITION in the label array
    ``star_broadcast`` uses, and that array is first-occurrence order —
    ``_star_row_order``, not ``np.unique``.  On gnppm the two differ:
    ``[0, 2, 6, 8, 7]`` against ``[0, 2, 6, 7, 8]``, positions 3 and 4
    swapped.  Build ``A_irr`` with ``np.unique`` (the obvious way) and let
    ``star_broadcast`` derive the labels (the default), and every k in
    stars 7 and 8 gets the OTHER star's matrix — right shape, right
    hermiticity, right norm class.

    MEASURED: 4.501e-01 on gnppm, 1.729e-01 on bispinor, against
    3.717e-16 and 1.232e-03 for the same operand stacked correctly.

    Self-consistent SORTED stacking is fine — pass sorted labels with
    sorted rows and it round-trips exactly.  The bug is the MISMATCH, so
    that half is asserted too, or this cell would read as "sorting is
    forbidden" and somebody would correctly disprove it.
    """
    A_full, sym, irr, sidx, nss, labels, reps = _tables(deck)
    assert labels.tolist() == [0, 2, 6, 8, 7], (
        f"PRECONDITION: {deck}'s first-occurrence labels must be "
        f"non-monotone for this trap to exist; got {labels.tolist()}")
    srt = np.asarray(sorted(labels.tolist()), dtype=np.int32)
    assert srt.tolist() != labels.tolist()

    A_srt = A_full[[reps[int(v)] for v in srt]]
    trap = symmetry_maps.star_broadcast(A_srt, irr, sidx, nss,
                                        trs_reference="ibz_slab")
    assert _rel(trap, A_full) > 0.1, (
        f"{deck}: sorted stacking against derived (first-occurrence) labels "
        f"must be visibly wrong; got {_rel(trap, A_full):.3e}")

    consistent = symmetry_maps.star_broadcast(A_srt, irr, sidx, nss,
                                              irr_labels=srt,
                                              trs_reference="ibz_slab")
    good = symmetry_maps.star_broadcast(
        A_full[[reps[int(v)] for v in labels]], irr, sidx, nss,
        irr_labels=labels, trs_reference="ibz_slab")
    assert _rel(consistent, A_full) == _rel(good, A_full), (
        "sorted rows with sorted labels is self-consistent and must give "
        "the same answer; the bug is the mismatch, not the sort")


# ---------------------------------------------------------------------------
# I12 / I13 — the two ways a broadcast can look right and be vacuous
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck", _DECKS)
def test_every_full_bz_row_is_distinct(deck):
    """I12.  9 of 9 (64 of 64 on Si), not 1.

    ``kin_ion_io.py:174-179`` says it outright: a broadcast that wrote ONE
    matrix everywhere would score perfectly on any within-star spread test.
    The row-distinctness count is the check that CAN fail, and it is the
    reason a spread test is NOT used as this unfold's correctness check
    (I14).
    """
    A_full, *_ = _tables(deck)
    n = A_full.shape[0]
    distinct = len({A_full[k].tobytes() for k in range(n)})
    assert distinct == n, f"{deck}: only {distinct} of {n} rows are distinct"


def test_si_wants_no_conjugation_at_all():
    """I13/I9.  THE NEGATIVE CONTROL, and it is a FUNCTION property.

    Si has ZERO time-reversed rows, so ``star_broadcast`` must reduce to a
    plain gather — asserted with ``np.array_equal``, bit for bit, which is
    a statement about the code and needs no reference file at all.

    The file self-consistency is a SEPARATE and weaker claim, because the
    committed Si ``kin_ion.h5`` rows were computed independently rather
    than written by this broadcast: the whole-array residual is 5.130e-04.
    What the file DOES pin is that no conjugation is wanted, and that is
    the second assertion.  MEASURED per row (Frobenius, row-relative):

        plain residual   0.0 on 8 rows, 8.109e-05 .. 9.647e-04 on 46
        conj  residual   7.959e-02 .. 9.286e-02 on all 64
        per-row conj/plain ratio  87.7 .. 1136.0

    ⚠ DISCREPANCY WITH THE MEASUREMENT DOSSIER, recorded rather than
    smoothed over.  MEASUREMENTS_step2.md §3 quotes "8.2-9.3e-05 per-row
    plain residual ... and conj is 8.2-9.3e-02".  The CONJ range
    reproduces (7.959e-02..9.286e-02); the PLAIN range does not — its top
    is 9.647e-04, an order above the quoted 9.3e-05, and the two quoted
    ranges share their mantissas exactly, which is the fingerprint of a
    transcription that reused one range's digits with a shifted exponent.
    Every other number in that table reproduces to four digits under
    ``‖Δ‖_F/‖A‖_F``, so the convention is not in doubt.  The assertion
    below is on the SEPARATION, which is what the row is for and which
    reproduces with two orders to spare.
    """
    A_full, sym, irr, sidx, nss, labels, reps = _tables("si_cohsex_debug")
    assert int((sidx >= nss).sum()) == 0, (
        "PRECONDITION: Si must have ZERO time-reversed rows, or it is not "
        "the negative control")

    pos = {int(v): i for i, v in enumerate(labels)}
    A_irr = A_full[[reps[int(v)] for v in labels]]
    plain = A_irr[[pos[int(v)] for v in irr]]
    out = symmetry_maps.star_broadcast(A_irr, irr, sidx, nss,
                                       irr_labels=labels,
                                       trs_reference="ibz_slab")
    assert np.array_equal(np.asarray(out), plain), (
        "with no time-reversed row star_broadcast must BE a plain gather, "
        "bit for bit")

    def rowrel(x, k):
        return float(np.linalg.norm((x[k] - A_full[k]).ravel())
                     / np.linalg.norm(A_full[k].ravel()))

    pr = [rowrel(plain, k) for k in range(len(irr))]
    cj = [rowrel(np.conj(plain), k) for k in range(len(irr))]
    assert max(pr) < 1e-3, f"worst plain per-row residual {max(pr):.3e}"
    assert min(cj) > 1e-2, f"best conj per-row residual {min(cj):.3e}"
    assert min(cj) > 50 * max(pr), (
        f"plain must beat conj by orders on Si, or the file does not pin "
        f"that no conjugation is wanted: {max(pr):.3e} vs {min(cj):.3e}")


# ---------------------------------------------------------------------------
# The adapter itself, and I8
# ---------------------------------------------------------------------------

def test_the_adapter_agrees_with_the_raw_star_broadcast():
    """``broadcast_ibz_to_full_bz`` is a THIN adapter — measured, not said.

    It passes identity labels because its ``A_irr`` rows are raw IBZ
    indices, so its gather is ``A[irr_idx_k]`` with the ibz_slab
    conjugation.  On these three decks ``nrk == nk_tot``, so the whole
    full-BZ array IS a legal input and the adapter's answer must equal the
    free function's with identity labels.  Bit-equal: both are one gather.
    """
    adapter = _adapter_or_skip()
    for deck in _DECKS:
        A_full, sym, irr, sidx, nss, labels, reps = _tables(deck)
        assert A_full.shape[0] == int(sym.nk_tot)
        got = np.asarray(adapter(A_full, sym))
        ref = np.asarray(symmetry_maps.star_broadcast(
            A_full, irr, sidx, nss,
            irr_labels=np.arange(A_full.shape[0], dtype=np.int32),
            trs_reference="ibz_slab"))
        assert np.array_equal(got, ref), deck


def test_the_adapter_refuses_a_table_the_parent_map_overruns():
    """RED TWIN.  ``irr_idx_k`` reaching past ``A_irr`` is the sweep having
    run on the wrong k-set, and it must raise rather than clip."""
    adapter = _adapter_or_skip()
    A_full, sym, *_ = _tables("gnppm_debug")
    with pytest.raises(ValueError, match="did not run on the IBZ k-set"):
        adapter(A_full[:2], sym)


def test_none_in_none_out():
    """The peers gather with ``owner_only=True`` and hold no table."""
    adapter = _adapter_or_skip()
    A_full, sym, *_ = _tables("gnppm_debug")
    assert adapter(None, sym) is None


def test_the_call_site_passes_ibz_slab_as_a_literal():
    """I8.  The predicate is chosen AT THE CALL SITE, in the source.

    ``star_broadcast``'s default is ``"star_row"`` and 27cc885's whole
    content is that this caller needs the other one.  A default that
    changed, or a variable that resolved to the wrong string at runtime,
    would put 183.61 eV back with no test failing — so what is checked is
    that the argument is a string CONSTANT with the value ``"ibz_slab"``,
    parsed out of the file.

    Mirrors ``tests/test_sc_kstar_spread.py:124``'s wiring check, but with
    ``ast`` rather than a substring: a substring test passes on
    ``trs_reference=("ibz_slab" if x else "star_row")`` and on the string
    appearing anywhere in a docstring, and both of those are exactly the
    shapes this needs to reject.

    THE PIN MOVED, DELIBERATELY, AND GOT STRONGER.  It used to parse
    ``src/gw/kin_ion_io.py``, because that is where the broadcast ran while
    it happened at WRITE time.  Storing kin_ion on the IBZ moved the
    broadcast to the read boundary, so the adapter — and the only
    ``star_broadcast`` call for this predicate — is now in
    ``src/file_io/kin_ion.py``.  A pin that had been *mechanically* updated
    would just follow the file; this one keeps the old location as a second
    assertion instead, because the failure it guards against is precisely
    somebody re-adding a broadcast at the writer with the default
    predicate, which would put the file back on the full BZ AND put
    27cc885 back with it.
    """
    # Visited by the 2026-08-08 rename sweep and deliberately unchanged:
    # ``star_broadcast`` is a KEEP name (already mathematical).  The
    # matcher below compares an AST attribute against the STRING
    # "star_broadcast", so a rename tool would have moved the call site
    # in file_io/kin_ion.py and left this matcher looking for a name that
    # no longer occurs — the assert would then find zero calls and this
    # cell would fail loudly rather than pass vacuously, which is the right
    # failure mode but only because the count is asserted to be exactly 1.
    src = pathlib.Path(_FILE_IO_KIN_ION).read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "broadcast_ibz_to_full_bz")
    calls = [c for c in ast.walk(fn)
             if isinstance(c, ast.Call)
             and getattr(c.func, "attr", None) == "star_broadcast"]
    assert len(calls) == 1, (
        f"expected exactly one star_broadcast call in "
        f"file_io.kin_ion.broadcast_ibz_to_full_bz; found {len(calls)}")
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "trs_reference" in kw, (
        "broadcast_ibz_to_full_bz no longer passes trs_reference "
        "explicitly; star_broadcast's default is 'star_row' and that is "
        "27cc885's 183.61 eV")
    node = kw["trs_reference"]
    assert isinstance(node, ast.Constant) and node.value == "ibz_slab", (
        f"trs_reference must be the LITERAL 'ibz_slab' at this call site, "
        f"not {ast.dump(node)} — a conditional or a name is a runtime "
        f"question and this is not a runtime property")


def test_the_writer_holds_no_star_broadcast_of_its_own():
    """The other half of I8: the broadcast did not come BACK to the writer.

    ``gw.kin_ion_io`` still owns a ``broadcast_ibz_to_full_bz``, but it is
    a wrapper that unpacks a live ``SymMaps`` and forwards; the rule has
    ONE implementation and one call.  A second ``star_broadcast`` appearing
    in the writer is either a duplicated predicate (two places to get
    ``trs_reference`` wrong) or a re-instated write-time unfold that
    silently undoes the compression — so it is asserted absent rather than
    left to review.
    """
    tree = ast.parse(pathlib.Path(_KIN_ION_IO).read_text())
    calls = [c for c in ast.walk(tree)
             if isinstance(c, ast.Call)
             and getattr(c.func, "attr", None) == "star_broadcast"]
    assert calls == [], (
        f"src/gw/kin_ion_io.py has {len(calls)} star_broadcast call(s) at "
        f"line(s) {[c.lineno for c in calls]}; the adapter lives in "
        f"src/file_io/kin_ion.py and the writer forwards to it")


# ===========================================================================
# § THE STORED FORMAT — the broadcast moved to read time, and what gates it
# ===========================================================================
# ``gw.kin_ion_io`` used to broadcast one statement after the sweep and write
# the full-BZ table; it now writes the PRE-broadcast block and
# ``file_io.kin_ion`` unfolds on read.  That is a pure storage change and the
# cells below are what make "pure" checkable:
#
#   * the compressed file reads back BIT-IDENTICAL to the full write it
#     replaces, on both datasets, on the two decks the generator wrote;
#   * a file that CLAIMS IBZ storage and cannot back the claim refuses, in
#     each of the six ways it can fail to back it;
#   * a file with no attr is read verbatim, which is what keeps the four
#     older fixtures — whose rows do NOT satisfy the star relation — meaning
#     what they have always meant.
#
# None of this needs the FFI: the writer is h5py and the reader under test is
# the pure-host ``read_full_bz_dataset``, so unlike the adapter cells above
# these run on any machine.  ``SlabIO``'s collective path is the same unfold
# behind the same predicate and is covered by any leg with the library.

from file_io.kin_ion import (                                   # noqa: E402
    IRR_IDX_DATASET, K_STORAGE_ATTR, K_STORAGE_FULL, K_STORAGE_IBZ,
    K_STORAGE_VERSION, K_STORAGE_VERSION_ATTR, N_SYM_SPATIAL_ATTR,
    SYM_IDX_DATASET, read_full_bz_dataset, read_kin_ion_provenance,
    read_star_map, validate_kin_ion_against_run,
)
from file_io.kin_ion import (                                   # noqa: E402
    broadcast_ibz_to_full_bz as _read_side_adapter,
)

#: The decks whose committed ``kin_ion.h5`` THIS generator wrote — they are
#: the ones stamped ``k_set_computed='ibz'``, and the only ones whose
#: full-BZ rows are the star broadcast of their own IBZ slab.  The round
#: trip is asserted bit-identical on these and on no others, deliberately.
_GENERATOR_WRITTEN = ("si_bse_debug", "hbn_cohsex_debug")


def _slab_of(A_full, irr, sidx, nss):
    """Invert the broadcast: the IBZ slab ``A_full`` was unfolded from.

    One row per IBZ index, un-conjugated — ``conj`` is its own inverse, so
    this is exact whichever row of the star is picked.  It reconstructs what
    the generator HELD; the generator itself never inverts anything, it just
    stops broadcasting.
    """
    nrk = int(np.asarray(irr).max()) + 1
    out = np.empty((nrk,) + A_full.shape[1:], dtype=A_full.dtype)
    seen = set()
    for k in range(len(irr)):
        j = int(irr[k])
        if j in seen:
            continue
        seen.add(j)
        out[j] = np.conj(A_full[k]) if sidx[k] >= nss else A_full[k]
    return out


def _write_kin_ion(path, arrays, *, irr, sidx, nss, storage=K_STORAGE_IBZ,
                   version=K_STORAGE_VERSION, tables=True,
                   stamp_nss=True):
    """A ``kin_ion.h5`` written the way the generator writes one.

    Every knob exists because a red twin turns it: ``storage`` to mislabel,
    ``version`` to age, ``tables``/``stamp_nss`` to omit.  The arrays are
    passed in already sliced so a twin can truncate one.
    """
    with h5py.File(path, "w") as f:
        if tables:
            f.create_dataset(IRR_IDX_DATASET, data=np.asarray(irr, np.int32))
            f.create_dataset(SYM_IDX_DATASET, data=np.asarray(sidx, np.int32))
        for name, arr in arrays.items():
            ds = f.create_dataset(name, data=arr, dtype=np.complex128)
            ds.attrs[K_STORAGE_ATTR] = storage
            if storage == K_STORAGE_IBZ:
                ds.attrs[K_STORAGE_VERSION_ATTR] = int(version)
                if stamp_nss:
                    ds.attrs[N_SYM_SPATIAL_ATTR] = int(nss)
    return path


def _generator_deck(deck):
    """``(committed arrays, irr, sidx, nss)`` for a generator-written deck."""
    wfn = os.path.join(_REG, deck, "WFN.h5")
    kin = os.path.join(_REG, deck, "kin_ion.h5")
    for p in (wfn, kin):
        if not os.path.isfile(p):
            pytest.skip(f"no {os.path.relpath(p, _REPO)} in this checkout "
                        f"(fixture blobs absent)")
    sym = SymMaps(_Header(wfn))
    irr = np.asarray(sym.irr_idx_k)
    sidx = np.asarray(sym.sym_idx_k)
    nss = int(np.asarray(sym.sym_mats_k).shape[0]) // 2
    with h5py.File(kin, "r") as f:
        assert f["kin_ion"].attrs.get("k_set_computed") == "ibz", (
            f"PRECONDITION: {deck}'s kin_ion.h5 must be one THIS generator "
            f"wrote, or its full-BZ rows are not the broadcast of anything")
        arrays = {n: f[n][:] for n in f
                  if f[n].ndim == 3 and f[n].shape[0] == len(irr)}
    assert arrays, f"{deck}: no full-BZ 3-D dataset in kin_ion.h5"
    return arrays, irr, sidx, nss


@pytest.mark.parametrize("deck", _GENERATOR_WRITTEN)
def test_a_compressed_file_reads_back_the_old_full_write_bit_for_bit(
        deck, tmp_path):
    """THE gate.  Store the slab, read it back, compare to the full write.

    Not "agrees to a tolerance" — ``np.array_equal`` on complex128, on
    every dataset the file carries, because the claim being made is that
    this is a storage change and a storage change that moves a bit is not
    one.  It holds by construction (the reader unfolds exactly what the
    writer stopped broadcasting), and the point of measuring it is that
    "by construction" has been wrong twice in this area, both times
    diagonal-preserving.

    MEASURED: max|Δ| exactly 0.000e+00 on ``kin_ion`` AND ``v_hartree``,
    both decks; si_bse_debug 7.3728 MB → 0.9216 MB payload (8.00×),
    hbn_cohsex_debug 1.00× (nrk == nk — the honest answer on a deck where
    every k is its own star, and the reason the size claim is asserted as
    the star ratio rather than as a constant).
    """
    arrays, irr, sidx, nss = _generator_deck(deck)
    nrk = int(irr.max()) + 1
    slabs = {n: _slab_of(a, irr, sidx, nss) for n, a in arrays.items()}
    path = _write_kin_ion(tmp_path / "kin_ion.h5", slabs,
                          irr=irr, sidx=sidx, nss=nss)

    for name, ref in arrays.items():
        got = read_full_bz_dataset(str(path), name)
        assert got.shape == ref.shape, (deck, name, got.shape, ref.shape)
        assert np.array_equal(got, ref), (
            f"{deck}/{name}: compressed read differs from the full write by "
            f"max|d|={float(np.abs(got - ref).max()):.3e}; a storage change "
            f"must move nothing")

    stored = sum(a.nbytes for a in slabs.values())
    full = sum(a.nbytes for a in arrays.values())
    assert stored * len(irr) == full * nrk, (
        f"{deck}: stored payload {stored} B against {full} B is not the "
        f"star ratio {len(irr)}/{nrk}")


@pytest.mark.parametrize(
    "deck", ["gnppm_debug", "bispinor_debug", "si_cohsex_debug",
             "cohsex_debug", "si_bse_debug", "hbn_cohsex_debug"])
def test_a_file_with_no_attr_is_read_verbatim(deck):
    """NO-ATTR-MEANS-FULL, asserted on every committed fixture.

    The default is what keeps the four older files meaning what they mean.
    They were computed independently at every full-BZ k, so their rows do
    NOT satisfy the star relation — MEASURED ``unfold(select(A))`` against
    ``A``: 3.557e-14 (gnppm), 2.001e-03 (si_cohsex), 1.726e-01 (bispinor)
    and 7.782e+00 Ry (cohsex).  If the reader ever inferred compressibility
    from shape, or from the ``k_set_computed`` attr, or from the presence
    of the tables, cohsex_debug would move by 7.8 Ry and every diagonal
    observable would survive it.  So the discriminator is an attribute the
    old writer never wrote, and this cell is the proof it stays that way.
    """
    kin = os.path.join(_REG, deck, "kin_ion.h5")
    if not os.path.isfile(kin):
        pytest.skip(f"no {os.path.relpath(kin, _REPO)} in this checkout")
    assert read_star_map(kin, "kin_ion") is None, (
        f"{deck}: a committed fixture must read as full-BZ storage")
    with h5py.File(kin, "r") as f:
        ref = f["kin_ion"][:]
    assert np.array_equal(read_full_bz_dataset(kin, "kin_ion"), ref)


def _twin(kind, arrays, irr, sidx, nss, tmp_path):
    """One way to write a file that lies, and the message it must raise."""
    slabs = {n: _slab_of(a, irr, sidx, nss) for n, a in arrays.items()}
    p = tmp_path / "kin_ion.h5"
    if kind == "truncated":
        # The slab lost a star.  This is the corruption that CANNOT be seen
        # by looking at the array — every row is a valid matrix.
        return (_write_kin_ion(p, {n: a[:-1] for n, a in slabs.items()},
                               irr=irr, sidx=sidx, nss=nss),
                "do not describe the same calculation")
    if kind == "mislabelled_full":
        # A full-BZ array stamped as an IBZ slab: the shape and the tables
        # disagree, which the spec makes a refusal rather than a preference.
        return (_write_kin_ion(p, arrays, irr=irr, sidx=sidx, nss=nss),
                "do not describe the same calculation")
    if kind == "no_tables":
        return (_write_kin_ion(p, slabs, irr=irr, sidx=sidx, nss=nss,
                               tables=False),
                "cannot be unfolded at all")
    if kind == "wrong_version":
        return (_write_kin_ion(p, slabs, irr=irr, sidx=sidx, nss=nss,
                               version=K_STORAGE_VERSION + 1),
                "format version")
    if kind == "no_n_sym_spatial":
        return (_write_kin_ion(p, slabs, irr=irr, sidx=sidx, nss=nss,
                               stamp_nss=False),
                "no threshold to test against")
    if kind == "illegal_storage":
        return (_write_kin_ion(p, slabs, irr=irr, sidx=sidx, nss=nss,
                               storage="wedge"),
                "which is neither")
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["truncated", "mislabelled_full",
                                  "no_tables", "wrong_version",
                                  "no_n_sym_spatial", "illegal_storage"])
def test_a_file_that_lies_about_ibz_storage_refuses(kind, tmp_path):
    """RED TWINS.  Six ways to claim IBZ storage without backing it.

    Each is a file that is perfectly well-formed HDF5 and perfectly
    plausible to a reader that only looked at shapes.  ``truncated`` and
    ``mislabelled_full`` are the two that matter most and they are the same
    check from opposite sides: the stored k extent must equal the number of
    stars the tables describe, so a slab short by one star and a full-BZ
    array wearing the IBZ stamp both fail it.  Neither is visible in the
    values — every row of both is a valid Hermitian block.
    """
    arrays, irr, sidx, nss = _generator_deck("si_bse_debug")
    path, message = _twin(kind, arrays, irr, sidx, nss, tmp_path)
    with pytest.raises(ValueError, match=message):
        read_full_bz_dataset(str(path), "kin_ion")


def test_the_read_side_adapter_is_the_star_algebra_itself():
    """``file_io.kin_ion.broadcast_ibz_to_full_bz`` IS ``star_broadcast``.

    The behavioural arm the FFI-gated adapter cells above cannot run on a
    machine with no library: this adapter lives under ``file_io``, imports
    the service lazily and needs no ``.so``, so the identity is asserted
    bit-for-bit here on all three star decks.  Both sides are one gather.
    """
    for deck in _DECKS:
        A_full, sym, irr, sidx, nss, labels, reps = _tables(deck)
        got = np.asarray(_read_side_adapter(A_full, irr, sidx, nss))
        ref = np.asarray(symmetry_maps.star_broadcast(
            A_full, irr, sidx, nss,
            irr_labels=np.arange(A_full.shape[0], dtype=np.int32),
            trs_reference="ibz_slab"))
        assert np.array_equal(got, ref), deck
    assert _read_side_adapter(None, irr, sidx, nss) is None
    with pytest.raises(ValueError, match="did not run on the IBZ k-set"):
        _read_side_adapter(A_full[:2], irr, sidx, nss)


def test_provenance_reports_the_logical_k_count_not_the_stored_one(tmp_path):
    """``_nk_logical`` is what a run means by nk; ``_shape`` is the disk.

    ``validate_kin_ion_against_run`` compares the file's nk against the
    run's, and it is the ONE check standing between a run and a kin_ion.h5
    from a different k-grid.  Reading the stored extent there would refuse
    every compressed file on any deck with symmetry — and the natural
    "fix", dropping the check, would let a genuinely wrong file through.
    So the two counts are separate keys and the check names which it uses.
    """
    arrays, irr, sidx, nss = _generator_deck("si_bse_debug")
    nrk = int(irr.max()) + 1
    assert nrk < len(irr), "PRECONDITION: si_bse_debug must actually reduce"
    slabs = {n: _slab_of(a, irr, sidx, nss) for n, a in arrays.items()}
    path = str(_write_kin_ion(tmp_path / "kin_ion.h5", slabs,
                              irr=irr, sidx=sidx, nss=nss))

    attrs = read_kin_ion_provenance(path)
    assert attrs["_k_storage"] == K_STORAGE_IBZ
    assert attrs["_shape"][0] == nrk
    assert attrs["_nk_logical"] == len(irr)

    validate_kin_ion_against_run(path, nk=len(irr), print_fn=lambda *a: None)
    with pytest.raises(ValueError, match="but the run has nk"):
        validate_kin_ion_against_run(path, nk=nrk, print_fn=lambda *a: None)


def test_the_full_storage_stamp_round_trips_too(tmp_path):
    """``k_storage="full"`` written explicitly must read as full.

    The generator stamps it under ``--fold-hartree``, whose whole job is
    reproducing the legacy layout, so "absent" and "``full``" have to be
    the same answer and not merely both non-``ibz``.
    """
    arrays, irr, sidx, nss = _generator_deck("si_bse_debug")
    path = str(_write_kin_ion(tmp_path / "kin_ion.h5", arrays,
                              irr=irr, sidx=sidx, nss=nss,
                              storage=K_STORAGE_FULL))
    assert read_star_map(path, "kin_ion") is None
    assert np.array_equal(read_full_bz_dataset(path, "kin_ion"),
                          arrays["kin_ion"])
