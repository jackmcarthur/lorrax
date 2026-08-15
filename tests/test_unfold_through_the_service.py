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

#: The service's single adapter over ``symmetry_maps.star_broadcast``.
#: ``tests/test_kin_ion_star_broadcast.py`` pins that the adapter itself
#: holds exactly one ``star_broadcast`` call with the ``ibz_slab``
#: predicate; these cells pin that the drivers reach the unfold THROUGH
#: it rather than around it.
_ADAPTER = "broadcast_ibz_to_full_bz"

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
    assert _ADAPTER in called, (
        f"{relpath}::{name} no longer calls {_ADAPTER}.  If it now reaches "
        f"the full BZ some other way, that way is a second unfold: the "
        f"time-reversal rule and the star row order live in the service, "
        f"and a driver-local copy is what this cell exists to prevent.")


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
