"""Every IBZ→full-BZ unfold asks the symmetry service; none is hand-rolled.

The rule: ``services/symmetry_maps`` owns symmetry unfolding, and a driver
that wants a full-BZ quantity from a wedge one calls it — it does not
open-code a star expansion, an index map, or a k-matching loop.

Two drivers reached the full BZ their own way and both are fixed on this
branch, so both are pinned here:

``bandstructure.htransform.read_eqp_energies``
    required a PRE-UNFOLDED full-BZ text file and paired its
    ``k-point N:`` blocks to k BY POSITION, checking only the count.  The
    unfold happened one hop upstream in an out-of-tree converter.

``bse.bse_window.apply_eqp_corrections``
    matched each full-BZ k to a wedge block by comparing MEAN-FIELD
    ENERGIES to 0.01 eV.  Right by accident (E_DFT is constant over a
    star) and silently wrong the moment two stars are degenerate to
    10 meV across the compared window.

The cells below are in two layers.  The AST layer fails if either
function grows its own unfold back.  The behavioural layer fails if an
unfold drops, duplicates or mis-parents a k-point — checked against the
star tables directly, with k-tagged values so a wrong parent NAMES the k
it came from rather than merely differing.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest


_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

#: The names a driver may reach the unfold through.  All route to ONE
#: backend (``symmetry_maps.star_broadcast``);
#: ``tests/test_kin_ion_star_broadcast.py`` pins that the file-table
#: adapter holds exactly one ``star_broadcast`` call with the ``ibz_slab``
#: predicate.  These cells pin that the drivers reach the unfold THROUGH
#: one of them rather than around it.
#:
#: The two ``unfold_*_wedge_to_full_bz`` names exist because there are two
#: different IBZs and they are NOT the same size — file wedge
#: (``wfn.kpoints``) vs star wedge (``star_select``): 8 = 8 on
#: si_cohsex_debug but 4 vs 3 on cohsex_debug and 9 vs 5 on gnppm_debug.
_ADAPTERS = {
    "broadcast_ibz_to_full_bz",          # file-table path (kin_ion read)
    "unfold_file_wedge_to_full_bz",      # live SymMaps, file wedge
    "unfold_star_wedge_to_full_bz",      # live SymMaps, star wedge
}

_UNFOLDERS = [
    ("bandstructure/htransform.py", "read_eqp_energies"),
    ("bse/bse_window.py", "apply_eqp_corrections"),
]


def _code_of(relpath: str, name: str) -> str:
    """A function's CODE, with its docstring removed.

    The docstrings on these functions deliberately NAME the constructs
    that were deleted, so a substring search over the raw source finds
    them and reports the explanation as the defect.  Stripping the
    docstring node and unparsing leaves only what actually runs.
    """
    fn = _function(relpath, name)
    body = fn.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return "\n".join(ast.unparse(node) for node in body)


def _function(relpath: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((_SRC / relpath).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{relpath} has no function {name}")


# ===========================================================================
#  AST layer — the unfold is delegated, and no second one grew back
# ===========================================================================

@pytest.mark.parametrize("relpath,name", _UNFOLDERS)
def test_the_unfold_is_delegated_to_the_service_adapter(relpath, name):
    """The function reaches the full BZ by CALLING the adapter."""
    fn = _function(relpath, name)
    called = {
        (c.func.id if isinstance(c.func, ast.Name) else
         getattr(c.func, "attr", None))
        for c in ast.walk(fn) if isinstance(c, ast.Call)
    }
    assert called & _ADAPTERS, (
        f"{relpath}::{name} reaches the full BZ without any of {_ADAPTERS}. "
        f"If it now does so some other way, that way is a second unfold: "
        f"the time-reversal rule and the star row order live in the "
        f"service, and a driver-local copy is what this cell prevents.")


@pytest.mark.parametrize("relpath,name", _UNFOLDERS)
def test_no_hand_rolled_star_loop_came_back(relpath, name):
    """No nested ``for`` — the shape both bespoke unfolds had.

    Deliberately structural rather than a substring search.  The two
    implementations removed here were a ``for ik_full: for ik_ibz:``
    nearest-match search and a ``for ik: for band:`` positional fill;
    both are loop-over-k-inside-loop-over-k.  A delegated unfold is one
    call and needs no nesting, so "contains a nested loop" is a cheap
    and honest proxy for "started doing symmetry by hand again".
    """
    fn = _function(relpath, name)
    for outer in ast.walk(fn):
        if not isinstance(outer, (ast.For, ast.While)):
            continue
        for inner in ast.walk(outer):
            if inner is not outer and isinstance(inner, (ast.For, ast.While)):
                raise AssertionError(
                    f"{relpath}::{name} has a nested loop at line "
                    f"{inner.lineno}.  Both bespoke unfolds this branch "
                    f"removed had exactly that shape; if this one is "
                    f"legitimate, it still does not belong in a function "
                    f"whose k-mapping is the service's job.")


def test_no_energy_fingerprint_matching_survives_in_the_bse_path():
    """The 0.01 eV mean-field nearest-match is gone, not merely bypassed.

    It was reachable only through ``input_file=None``, so deleting the
    call site would have left the code in place for the next caller to
    rediscover.  Pinned on the source: no tolerance constant and no
    'match by energy' machinery in the function.
    """
    seg = _code_of("bse/bse_window.py", "apply_eqp_corrections")
    for token in ("tol_ev", "best_ibz", "best_err"):
        assert token not in seg, (
            f"apply_eqp_corrections still contains {token!r} — the "
            f"energy-fingerprint unfold is back")


# ===========================================================================
#  Behavioural layer — the unfold covers the BZ exactly once
# ===========================================================================

def _star_geometry():
    """A 12-point BZ over a 4-point wedge, no time-reversed rows.

    ``irr_idx_k[ik_full]`` is the wedge row that ``ik_full`` reduces to —
    the same convention ``SymMaps`` publishes.  Star sizes are deliberately
    UNEQUAL (4, 2, 5, 1) so a bug that assumes a uniform multiplicity, or
    that reconstructs stars by dividing nk_full by nk_irr, cannot pass.
    """
    irr_idx_k = np.array([0, 1, 2, 0, 2, 0, 3, 1, 2, 0, 2, 2], dtype=np.int32)
    sym_idx_k = np.zeros_like(irr_idx_k)          # all spatial ⇒ no conj
    n_sym_spatial = 2
    return irr_idx_k, sym_idx_k, n_sym_spatial


def _adapter():
    from file_io.kin_ion import broadcast_ibz_to_full_bz
    return broadcast_ibz_to_full_bz


def _read_eqp_energies_or_skip():
    """``htransform.read_eqp_energies``, imported LAZILY.

    Importing ``bandstructure.htransform`` runs
    ``runtime.initialize_communicator_stack`` at module scope, which
    REFUSES without a built FFI ``.so`` for the resolved platform — on a
    CPU-platform leg that is ``liblorrax_ffi_host.so``, which not every
    environment builds.  Same constraint and same remedy as
    ``tests/test_kin_ion_star_broadcast.py:92`` and
    ``tests/test_kin_ion_padded_gvectors.py:314``.

    WHAT SKIPS HERE IS ONLY THE END-TO-END ARM.  The property that
    matters — that this driver delegates its unfold instead of rolling
    one — is pinned by parsing the FILE in the AST cells above, which
    need no import and run on every machine.
    """
    try:
        from bandstructure.htransform import read_eqp_energies
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(
            f"bandstructure.htransform needs a built FFI .so for the "
            f"resolved platform at import ({type(exc).__name__}) — runs on "
            f"any leg with the library (measured green on Perlmutter with "
            f"JAX_PLATFORMS=cuda); the delegation itself is pinned above "
            f"without importing anything")
    return read_eqp_energies


def test_the_unfold_covers_every_k_exactly_once_and_parents_them_right():
    """No k dropped, none duplicated, each carrying ITS OWN parent's row.

    The wedge values are k-tagged (row j is ``100*j``), so a row that
    lands under the wrong parent reports which wedge row it actually got.
    """
    irr, sidx, nss = _star_geometry()
    nk_full, nk_irr = irr.shape[0], int(irr.max()) + 1
    wedge = (100.0 * np.arange(nk_irr)[:, None]
             + np.arange(3)[None, :]).astype(np.float64)

    full = np.asarray(_adapter()(wedge, irr, sidx, nss))

    assert full.shape == (nk_full, 3), (
        f"unfold returned {full.shape[0]} k-points for a {nk_full}-point BZ "
        f"— a k was dropped or duplicated")
    parent_of_row = np.rint(full[:, 0] / 100.0).astype(np.int64)
    np.testing.assert_array_equal(
        parent_of_row, irr,
        err_msg=("a full-BZ row carries a different wedge row than the one "
                 "irr_idx_k assigns it — the unfold mis-parented a k"))
    # Nothing dropped: every wedge row is used, with the multiplicity the
    # star tables specify.
    got = np.bincount(parent_of_row, minlength=nk_irr)
    want = np.bincount(irr, minlength=nk_irr)
    np.testing.assert_array_equal(got, want, err_msg="star sizes changed")
    assert int(got.sum()) == nk_full


def test_a_dropped_or_duplicated_k_would_be_caught():
    """THE RED TWIN — the cell above must be able to fail.

    Feeds the same checker a deliberately corrupted unfold (one k
    re-parented onto its neighbour's star) and asserts it is rejected.
    Without this, a broadcast that silently returned a constant array
    would satisfy the shape and the bincount.
    """
    irr, _sidx, _nss = _star_geometry()
    nk_irr = int(irr.max()) + 1
    wedge = (100.0 * np.arange(nk_irr)[:, None]
             + np.arange(3)[None, :]).astype(np.float64)

    corrupt_idx = irr.copy()
    corrupt_idx[3] = 1                       # was 0 — one k stolen by star 1
    corrupted = wedge[corrupt_idx]
    parent_of_row = np.rint(corrupted[:, 0] / 100.0).astype(np.int64)
    assert not np.array_equal(parent_of_row, irr), (
        "the corruption was not detectable, so the cell above proves nothing")
    got = np.bincount(parent_of_row, minlength=nk_irr)
    want = np.bincount(irr, minlength=nk_irr)
    assert not np.array_equal(got, want), (
        "a re-parented k left the star sizes unchanged — the multiplicity "
        "check cannot see this class")


def _fake_sym(irr, sidx, n_sym_spatial, wedge_kpts):
    """The SymMaps surface ``read_eqp_energies`` touches, and only that.

    ``unfolded_kpts[kirr_fullids] == wfn.kpoints`` is the service's own
    documented identity (``maps.py:1293``), so the wedge is placed at the
    first occurrence of each parent — the rows ``star_select`` keeps.
    """
    from types import SimpleNamespace

    nk_full = irr.shape[0]
    first = {}
    for i, lab in enumerate(int(v) for v in irr):
        first.setdefault(lab, i)
    kirr_fullids = np.array([first[j] for j in range(len(first))],
                            dtype=np.int32)
    unfolded = np.zeros((nk_full, 3), dtype=np.float64)
    for ik in range(nk_full):
        unfolded[ik] = wedge_kpts[int(irr[ik])] + np.array([0.0, 0.0, 0.0])
    # Give the members distinct coordinates; only the wedge rows are
    # compared, and those must reproduce ``wedge_kpts`` exactly.
    for j, kf in enumerate(kirr_fullids):
        unfolded[kf] = wedge_kpts[j]
    return SimpleNamespace(
        unfolded_kpts=unfolded, kirr_fullids=kirr_fullids,
        irr_idx_k=irr, sym_idx_k=sidx, nk_tot=nk_full, nk_red=len(first),
        sym_mats_k=np.zeros((2 * n_sym_spatial, 3, 3)))


def test_htransform_reads_the_wedge_file_and_unfolds_it(tmp_path):
    """End to end: a real ``eqp1.dat`` in, full-BZ Ry out, right parents.

    The QP column is k-tagged, so a k that ends up under the wrong star
    names the wedge row it actually received.
    """
    read_eqp_energies = _read_eqp_energies_or_skip()
    from common.units import RYD_TO_EV
    from gw.eqp_bgw import write_bgw_eqp

    irr, sidx, nss = _star_geometry()
    nk_irr = int(irr.max()) + 1
    wedge_kpts = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0],
                           [0.25, 0.25, 0.0], [0.5, 0.25, 0.25]])
    sym = _fake_sym(irr, sidx, nss, wedge_kpts)

    band_offset, nb_file = 3, 4
    e_dft = np.zeros((nk_irr, nb_file))
    e_qp = (100.0 * np.arange(nk_irr)[:, None]
            + np.arange(nb_file)[None, :]).astype(np.float64)
    path = tmp_path / "eqp1.dat"
    write_bgw_eqp(str(path), wedge_kpts, e_dft, e_qp,
                  band_offset=band_offset)

    # Ask for an interior window, so the absolute-band arithmetic matters.
    got = np.asarray(read_eqp_energies(str(path), sym, (4, 6)))
    assert got.shape == (2, irr.shape[0]), (
        f"expected (nb=2, nk_full={irr.shape[0]}), got {got.shape}")

    want_ev = e_qp[:, 1:3][irr]              # columns 4,5 abs -> 1,2 local
    np.testing.assert_allclose(got.T * RYD_TO_EV, want_ev, atol=1e-6,
                               err_msg="wrong parent, or wrong band columns")
    # ...and the unit conversion really happened.
    assert not np.allclose(got.T, want_ev), "eV was not converted to Ry"


def test_htransform_refuses_a_file_from_another_deck(tmp_path):
    """Coordinates are CHECKED, not trusted — the point of writing them.

    A file with the right block count but the wrong k must not be read
    positionally; that is the failure this whole branch exists to remove.
    """
    read_eqp_energies = _read_eqp_energies_or_skip()
    from gw.eqp_bgw import write_bgw_eqp

    irr, sidx, nss = _star_geometry()
    nk_irr = int(irr.max()) + 1
    wedge_kpts = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0],
                           [0.25, 0.25, 0.0], [0.5, 0.25, 0.25]])
    sym = _fake_sym(irr, sidx, nss, wedge_kpts)

    wrong = wedge_kpts.copy()
    wrong[2] = [0.125, 0.375, 0.0]           # same count, different k
    e = np.zeros((nk_irr, 4))
    path = tmp_path / "eqp1_other_deck.dat"
    write_bgw_eqp(str(path), wrong, e, e, band_offset=0)

    with pytest.raises(ValueError, match="does not belong to this"):
        read_eqp_energies(str(path), sym, (0, 4))


def test_htransform_refuses_a_pre_unfolded_full_bz_file(tmp_path):
    """The old input shape is refused, not silently re-interpreted."""
    read_eqp_energies = _read_eqp_energies_or_skip()
    from gw.eqp_bgw import write_bgw_eqp

    irr, sidx, nss = _star_geometry()
    wedge_kpts = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0],
                           [0.25, 0.25, 0.0], [0.5, 0.25, 0.25]])
    sym = _fake_sym(irr, sidx, nss, wedge_kpts)

    full_kpts = wedge_kpts[irr]              # nk_full blocks — the old form
    e = np.zeros((irr.shape[0], 4))
    path = tmp_path / "eqp1_fullbz.dat"
    write_bgw_eqp(str(path), full_kpts, e, e, band_offset=0)

    with pytest.raises(ValueError, match="irreducible wedge"):
        read_eqp_energies(str(path), sym, (0, 4))


def test_the_adapter_is_a_plain_gather_when_no_row_is_time_reversed():
    """With every row spatial, the unfold IS ``wedge[irr_idx_k]``.

    Pins that the delegated call does not quietly do something else to
    the values — the reason a driver may hand it energies at all.
    """
    irr, sidx, nss = _star_geometry()
    rng = np.random.default_rng(0)
    wedge = rng.standard_normal((int(irr.max()) + 1, 5))
    full = np.asarray(_adapter()(wedge, irr, sidx, nss))
    np.testing.assert_allclose(full, wedge[irr], atol=0.0, rtol=0.0)


# ===========================================================================
#  The other two sites fixed on this branch — the patterns must not return
# ===========================================================================

#: (file, function, substrings that MUST NOT reappear).  Each token is the
#: literal name of a construct this branch deleted, so the cell names the
#: defect rather than a shape that might legitimately recur.
_BANNED = [
    ("postprocess/rotate_wfn_to_qp.py", "read_kirr_to_kfull",
     ("argmin", "np.argmin", "tol=1e-6")),
    ("file_io/qe_save_reader.py", "_reduce_mp_to_ibz",
     ("_EPS", "equiv", "wkk")),
    ("bse/bse_window.py", "apply_eqp_corrections",
     ("tol_ev", "best_ibz", "best_err")),
]


@pytest.mark.parametrize("relpath,name,banned", _BANNED)
def test_the_deleted_k_matching_constructs_do_not_come_back(
        relpath, name, banned):
    """No argmin over k, no tolerance snapping, no hand-built orbit map.

    Source-level and named, because each of these was a DIFFERENT wrong
    way to answer "which k is this": a nearest-coordinate search
    (rotate_wfn_to_qp), a 1e-5 grid snap with a hand-rolled equivalence
    array (qe_save_reader), and a mean-field-energy fingerprint
    (bse_window).  A shape-based check would miss at least one of them.
    """
    body = _code_of(relpath, name)
    for token in banned:
        assert token not in body, (
            f"{relpath}::{name} contains {token!r} again — one of the "
            f"hand-rolled k-matching constructs this branch removed")


def test_rotate_wfn_reads_the_services_table_instead_of_rebuilding_it():
    """``kirr_to_kfull`` is READ from the rotation file, not re-derived.

    The file already carries it, written from ``sym.kirr_fullids`` by
    both producers.  Rebuilding it here by nearest-coordinate search is
    what let the rotated WFN and ``eqp{0,1}.dat`` — which reads the same
    dataset in ``gw.eqp_bgw`` — disagree about a k.
    """
    src = (_SRC / "postprocess/rotate_wfn_to_qp.py").read_text()
    assert "f_rot['kirr_to_kfull']" in src or 'f_rot["kirr_to_kfull"]' in src, (
        "rotate_wfn_to_qp no longer reads kirr_to_kfull from the rotation "
        "file — if it derives the map again, it is a second answer to a "
        "question the symmetry service already answered")
    assert "def add_kpoint_mapping_to_rotation_file" not in src, (
        "the --add-mapping path is back; it OVERWROTE the service's "
        "kirr_to_kfull with a nearest-coordinate approximation of itself")


def test_qe_kgrid_reduction_delegates_and_keeps_qe_s_transpose_convention():
    """``_reduce_mp_to_ibz`` calls the service, on the AS-STORED matrices.

    The convention is the load-bearing part and was MEASURED, not
    assumed: on ``si_cohsex_debug`` (4x4x4, 48 ops) the as-stored
    matrices reproduce the previous implementation exactly while the
    transposed ones move k by 2.5e-01; on a 2-op deck both agree, so
    only a high-symmetry deck can tell them apart.  A silent transpose
    here changes every QE-derived k-grid in the tree.
    """
    fn = _function("file_io/qe_save_reader.py", "_reduce_mp_to_ibz")
    called = {
        (c.func.id if isinstance(c.func, ast.Name) else
         getattr(c.func, "attr", None))
        for c in ast.walk(fn) if isinstance(c, ast.Call)
    }
    assert "find_irreducible_bz_points" in called, (
        "_reduce_mp_to_ibz no longer delegates to the symmetry service")
    assert ".transpose(" not in _code_of(
            "file_io/qe_save_reader.py", "_reduce_mp_to_ibz"), (
        "a transpose appeared in _reduce_mp_to_ibz.  The measurement says "
        "this grid reduces with the AS-STORED QE matrices; transposing "
        "them silently changes every QE-derived k-grid")


# ===========================================================================
#  TRS: the two predicates, and the blindness that let them diverge
# ===========================================================================

def _symmetry_maps():
    """``symmetry_maps``, with the service path bootstrapped first.

    ``from symmetry_maps import ...`` at cell scope only works if some
    EARLIER test already called ``ffi._services.ensure_on_path()`` — so a
    cell that does it bare passes in a full run and fails when selected
    alone.  One accessor removes the ordering dependence.
    """
    from ffi import _services
    _services.ensure_on_path()
    import symmetry_maps
    return symmetry_maps


def _star_with_a_time_reversed_first_member():
    """A 4-k BZ over 2 stars where star 1's FIRST member is a TRS row.

    That is the only configuration in which the two conjugation
    predicates disagree, and it is a property of the op-selection policy
    rather than of the physics — which is why it can lie dormant on
    every deck in a tree and then appear.

    ``irr_idx_k`` labels the star; ``sym_idx_k >= n_sym_spatial`` marks a
    time-reversal row.  Rows 2 and 3 form star 1, and row 2 — the one
    ``star_select`` keeps — is time-reversed.
    """
    irr = np.array([0, 0, 1, 1], dtype=np.int32)
    sym = np.array([0, 2, 2, 0], dtype=np.int32)      # n_sym_spatial = 2
    return irr, sym, 2


def _hermitian_stack(rng, nk, nb):
    a = rng.standard_normal((nk, nb, nb)) + 1j * rng.standard_normal((nk, nb, nb))
    return a + np.conj(np.swapaxes(a, 1, 2))


def test_the_two_trs_predicates_really_do_disagree():
    """Both branches run; on this star they give DIFFERENT matrices.

    If they ever agree here the fixture has stopped modelling the case,
    and every cell below would pass without testing anything.
    """
    star_broadcast = _symmetry_maps().star_broadcast
    irr, sym, nss = _star_with_a_time_reversed_first_member()
    rng = np.random.default_rng(0)
    wedge = _hermitian_stack(rng, 2, 3)

    a = np.asarray(star_broadcast(wedge, irr, sym, nss,
                                  trs_reference="star_row"))
    b = np.asarray(star_broadcast(wedge, irr, sym, nss,
                                  trs_reference="ibz_slab"))
    assert not np.allclose(a, b), (
        "the two trs_reference branches agree on a star whose first member "
        "is time-reversed — that is the ONLY case they differ on, so this "
        "fixture no longer exercises the 183.61 eV mix-up")


def test_a_diagonal_check_is_structurally_blind_to_the_mix_up():
    """THE POINT.  The wrong predicate is invisible to the cheap checks.

    Conjugating a Hermitian block leaves its REAL DIAGONAL exactly
    intact, so every diagonal observable — the electron count,
    hermiticity, the spectrum, the eqp.dat V_H column, and the
    diagonal star-spread metric ``compare_to_bgw`` reports — survives a
    wrong ``trs_reference`` unchanged.  That is why 27cc885's 183.61 eV
    error sat undetected for a month.

    This cell asserts the blindness as a PROPERTY rather than leaving it
    as a warning in a docstring: a future reader who adds a diagonal
    assertion and believes it covers the conjugation class has this cell
    telling them, in numbers, that it does not.
    """
    star_broadcast = _symmetry_maps().star_broadcast
    irr, sym, nss = _star_with_a_time_reversed_first_member()
    rng = np.random.default_rng(1)
    wedge = _hermitian_stack(rng, 2, 3)

    a = np.asarray(star_broadcast(wedge, irr, sym, nss,
                                  trs_reference="star_row"))
    b = np.asarray(star_broadcast(wedge, irr, sym, nss,
                                  trs_reference="ibz_slab"))

    # The real diagonals are EXACTLY equal — not close, equal.
    da = np.real(np.diagonal(a, axis1=1, axis2=2))
    db = np.real(np.diagonal(b, axis1=1, axis2=2))
    np.testing.assert_array_equal(
        da, db,
        err_msg="the real diagonals differ, so a diagonal check WOULD see "
                "the mix-up and this cell's premise is wrong")
    # ...while the off-diagonals differ by O(the matrix itself).
    off = np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)
    assert off > 0.1, (
        f"the two predicates differ by only {off:.2e} relative off-diagonal; "
        f"the recorded disagreement is O(1) (183.61 eV on a real deck)")


def test_even_the_self_consistency_spread_is_blind_to_it():
    """AND SO IS ``star_spread`` — which is the part that makes this bite.

    The natural assumption is that the diagonal metric is blind but the
    full-matrix one catches it.  IT DOES NOT, and the reason is worth
    stating: the wrong predicate conjugates an ENTIRE STAR uniformly, and
    a uniformly conjugated star is still perfectly self-consistent under
    the star relation.  ``star_spread`` measures self-consistency, so it
    reports ~0 on both.

    What the mix-up changes is the star's relation to the DATA IT CAME
    FROM, not its internal consistency.  So the only thing that can see
    it is a comparison against independently computed full-BZ values —
    which is exactly how 27cc885 was caught (183.61 eV "against an
    independently computed V_H") and exactly what
    ``tests/test_star_offdiag_gate.py`` does on the committed full
    matrices in ``cohsex_debug/sigma_mnk.h5``.

    This cell exists so nobody adds ``star_spread`` as the guard against
    this class and believes they are covered.
    """
    _sm = _symmetry_maps()
    star_broadcast, star_spread = _sm.star_broadcast, _sm.star_spread
    irr, sym, nss = _star_with_a_time_reversed_first_member()
    rng = np.random.default_rng(2)
    wedge = _hermitian_stack(rng, 2, 3)

    right = np.asarray(star_broadcast(wedge, irr, sym, nss,
                                      trs_reference="star_row"))
    wrong = np.asarray(star_broadcast(wedge, irr, sym, nss,
                                      trs_reference="ibz_slab"))
    scale = max(float(np.abs(right).max()), 1e-300)

    # BOTH are self-consistent.  This is the claim.
    assert float(star_spread(right, irr, sym, nss)) / scale < 1e-12
    assert float(star_spread(wrong, irr, sym, nss)) / scale < 1e-12, (
        "star_spread DID see the mix-up — if that is now true the comment "
        "above is wrong and this class has a cheaper guard than believed")

    # ...and only comparison against the known-correct full-BZ array does.
    assert float(np.abs(right - wrong).max()) / scale > 0.1, (
        "the two broadcasts agree, so nothing can distinguish them here")


def test_star_broadcast_refuses_to_guess_the_predicate():
    """No default: the wrong branch is invisible, so it cannot be implicit.

    ``trs_reference`` used to default to ``"star_row"``, which is right
    for a ``star_select`` operand and wrong for a file slab.  A caller who
    had not thought about it got a plausible wrong matrix; now they get a
    TypeError.
    """
    star_broadcast = _symmetry_maps().star_broadcast
    irr, sym, nss = _star_with_a_time_reversed_first_member()
    wedge = _hermitian_stack(np.random.default_rng(3), 2, 3)
    with pytest.raises(TypeError):
        star_broadcast(wedge, irr, sym, nss)  # trs-reference-exempt: the point


def test_every_call_site_in_the_tree_states_its_flavour():
    """AST sweep: no ``star_broadcast`` call may omit ``trs_reference``.

    The signature enforces this at run time; this enforces it at read
    time, across the whole tree including tools and multi-device gates,
    so the answer is visible at every call site rather than only when one
    executes.
    """
    root = _SRC.parent
    missing = []
    for path in root.rglob("*.py"):
        if ".git" in str(path) or "jax_cache" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (getattr(node.func, "attr", None)
                    or getattr(node.func, "id", None))
            if name != "star_broadcast":
                continue
            kw = {k.arg for k in node.keywords}
            if "trs_reference" in kw:
                continue
            # A call may opt out ONLY by saying so on its own line, which
            # is greppable and shows up in review.  The one user is the
            # refusal cell above, whose whole point is the omission.
            line = path.read_text().splitlines()[node.lineno - 1]
            if "trs-reference-exempt" in line:
                continue
            missing.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not missing, (
        "star_broadcast called without an explicit trs_reference at: "
        + ", ".join(missing)
        + ".  Which predicate applies depends on where the operand came "
          "from, and the wrong one is invisible to every diagonal check.")


# ===========================================================================
#  The two wedges — why there are two named unfolds and not one
# ===========================================================================

_DECK_WEDGES = [
    # (deck, WFN name, file wedge nk_red, star wedge n_orbits)
    ("si_cohsex_debug", "WFN.h5", 8, 8),      # COINCIDE — the deck most gates run
    ("cohsex_debug", "WFNsmall.h5", 4, 3),    # diverge
    ("gnppm_debug", "WFN.h5", 9, 5),          # diverge
]


@pytest.mark.parametrize("deck,wfn_name,nk_red,n_orbits", _DECK_WEDGES)
def test_the_file_wedge_and_the_star_wedge_are_different_objects(
        deck, wfn_name, nk_red, n_orbits):
    """MEASURED sizes, pinned per deck — the reason for two named unfolds.

    The FILE wedge is ``wfn.kpoints`` (``sym.nk_red``), what every ``.dat``
    is indexed by and what BerkeleyGW means by the IBZ.  The STAR wedge is
    what ``star_select`` keeps, one row per orbit.

    They COINCIDE on ``si_cohsex_debug`` and diverge on the other two.  A
    single ``unfold_ibz_to_full_bz`` would therefore have been correct on
    the deck most gates run and silently wrong elsewhere — where the
    lengths differ a mistake raises, where they coincide it does not.
    This cell fails if that stops being true, i.e. if the fixtures stop
    covering both cases.
    """
    import os
    reg = _SRC.parent / "tests" / "regression" / deck
    wfn_path = reg / wfn_name
    if not wfn_path.exists():
        pytest.skip(f"{deck}/{wfn_name} not in this tree")
    try:
        from ffi import _services
        _services.ensure_on_path()
        from symmetry_maps import SymMaps
        from wfn_loader import WfnLoader
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"loader/service unavailable here ({type(exc).__name__})")

    sym = SymMaps(WfnLoader(str(wfn_path)))
    got_file = int(sym.nk_red)
    got_star = len(set(np.asarray(sym.irr_idx_k).tolist()))
    assert (got_file, got_star) == (nk_red, n_orbits), (
        f"{deck}: file wedge {got_file} (expected {nk_red}), star wedge "
        f"{got_star} (expected {n_orbits}).  These two sizes are what make "
        f"two named unfolds necessary; if they moved, re-read the register "
        f"before collapsing them.")


def test_at_least_one_fixture_has_the_two_wedges_coinciding_and_one_not():
    """The fixture set must cover BOTH cases or the distinction is untested.

    If every deck diverged, a conflating bug would always raise and nobody
    would need two names.  If every deck coincided, it would never raise
    and the bug would be permanently silent.  The tree has both, and that
    is what makes this testable at all.
    """
    coincide = [d for d, _, a, b in _DECK_WEDGES if a == b]
    diverge = [d for d, _, a, b in _DECK_WEDGES if a != b]
    assert coincide and diverge, (
        f"fixtures cover only one case: coincide={coincide} diverge={diverge}")


def test_the_reduce_and_the_file_wedge_unfold_are_not_inverses():
    """They are not, and the asymmetry is real rather than an oversight.

    ``reduce_full_bz_to_file_wedge`` keeps one row per STORED k.
    ``unfold_file_wedge_to_full_bz`` rebuilds every full-BZ k from its
    ORBIT PARENT.  Where the WFN carries two k in the same orbit — which
    it does on ``cohsex_debug``, where file wedge row 1 is the
    time-reverse of row 2 — the round trip replaces row 1's own stored
    values with ``conj`` of row 2's.

    Pinned so nobody writes ``unfold(reduce(x)) == x`` into a gate and is
    surprised on the two decks where it is false.  On ``si_cohsex_debug``
    it happens to hold, which is exactly why it needs pinning.
    """
    _sm = _symmetry_maps()
    reg = _SRC.parent / "tests" / "regression" / "cohsex_debug" / "WFNsmall.h5"
    if not reg.exists():
        pytest.skip("cohsex_debug not in this tree")
    try:
        from wfn_loader import WfnLoader
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"loader unavailable ({type(exc).__name__})")

    sym = _sm.SymMaps(WfnLoader(str(reg)))
    rng = np.random.default_rng(7)
    nk_full = int(sym.nk_tot)
    full = (rng.standard_normal((nk_full, 2, 2))
            + 1j * rng.standard_normal((nk_full, 2, 2)))

    wedge = np.asarray(_sm.reduce_full_bz_to_file_wedge(sym, full))
    assert wedge.shape[0] == int(sym.nk_red)
    back = np.asarray(_sm.unfold_file_wedge_to_full_bz(sym, wedge))
    assert back.shape[0] == nk_full

    # On THIS deck the round trip is not the identity, because one stored
    # k is the time-reverse of another and the unfold rebuilds it from the
    # orbit parent rather than from its own row.
    assert not np.allclose(back, full), (
        "reduce->unfold round-tripped exactly on cohsex_debug — if that is "
        "now true, the orbit structure of this fixture changed and the "
        "asymmetry documented on reduce_full_bz_to_file_wedge needs "
        "re-measuring")
