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
    """
    src = pathlib.Path(_KIN_ION_IO).read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "broadcast_ibz_to_full_bz")
    calls = [c for c in ast.walk(fn)
             if isinstance(c, ast.Call)
             and getattr(c.func, "attr", None) == "star_broadcast"]
    assert len(calls) == 1, (
        f"expected exactly one star_broadcast call in "
        f"broadcast_ibz_to_full_bz; found {len(calls)}")
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
