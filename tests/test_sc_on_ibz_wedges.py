"""The SC loop runs on the STAR wedge; every ``.dat`` writer gets the FILE wedge.

Those are two different k-sets and they are not the same size (4 vs 3 on
``cohsex_debug``, 9 vs 5 on ``gnppm_debug``, 8 = 8 on ``si_cohsex_debug``).
Handing the loop's rows straight to the eqp writer was::

    sc_iteration.py:1397  _write_sc_eqp_snapshot
      -> eqp_bgw.py:145   write_bgw_eqp
    ValueError: e_qp shape (5, 46) does not match e_dft (9, 46)

and it would NOT have raised on ``si_cohsex_debug``, where the two
coincide.  It went unnoticed for as long as it did because **no committed
deck ran the SC path at all** — every one was ``qp_solver =
one_shot_dft``, so the flag's default could have been flipped without a
single suite result changing.

Three layers here, and they fail for different reasons:

* **AST** — the two boundary functions reach the wedge through the named
  service calls and hold no index table.  Fails if either grows its own
  ``kirr_fullids`` gather back.
* **Behavioural, no driver** — the composite the fix relies on
  (star wedge → full BZ → file wedge) has the right length on every
  committed deck, and the star round trip inside it is exact.
* **End to end** — the SC deck runs under the DEFAULT, and its outputs
  land on the file wedge with the file wedge's own coordinates.  This is
  the layer that was missing; the other two cannot catch a driver that
  never runs.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest


_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_REG = _ROOT / "tests" / "regression"

#: The named service calls the SC boundary is allowed to reach the wedge
#: through.  Both route to ONE backend (``symmetry_maps.star_broadcast``
#: for the unfold, a row take for the reduce); the point of naming them at
#: the call site is that a reader sees WHICH wedge and WHICH direction
#: without reading an index expression.
_REDUCE = "reduce_full_bz_to_file_wedge"
_UNFOLD_STAR = "unfold_star_wedge_to_full_bz"

#: The service's index tables.  A driver may still hand one to a WRITER as
#: file PAYLOAD — ``qp_wfn_rotations.h5`` stores ``kirr_to_kfull`` on
#: purpose, so a consumer can map without re-deriving it, and the
#: canonical writer at ``gw_output.py:1361`` passes exactly the same array.
#: What a driver may not do is SUBSCRIPT by one.  That is the index
#: arithmetic this branch removed, and the distinction is the rule: the
#: table as data is fine, the table as a gather is not.
_FORBIDDEN_TABLES = ("kirr_fullids", "irr_idx_k", "sym_idx_k")


def _subscripts_by_a_symmetry_table(node: ast.AST) -> list[str]:
    """Every ``x[<table>]`` inside ``node``, including through a local.

    The construct being banned reached the table through a local first::

        irr = np.asarray(inputs.sym.kirr_fullids, dtype=np.int64)
        e_dft = e_dft_full[irr]

    so a check that only looked at the subscript expression would have
    passed the very code that crashed.  Any name assigned from an
    expression mentioning a table is therefore tainted too.
    """
    tainted = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Assign):
            continue
        value = ast.unparse(sub.value)
        if any(t in value for t in _FORBIDDEN_TABLES):
            for tgt in sub.targets:
                for nm in ast.walk(tgt):
                    if isinstance(nm, ast.Name):
                        tainted.add(nm.id)
    bad = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Subscript):
            continue
        index = ast.unparse(sub.slice)
        names = {n.id for n in ast.walk(sub.slice) if isinstance(n, ast.Name)}
        if any(t in index for t in _FORBIDDEN_TABLES) or (names & tainted):
            bad.append(ast.unparse(sub))
    return bad


def _function(relpath: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((_SRC / relpath).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{relpath}: no function {name}")


def _code_of(relpath: str, name: str) -> str:
    """A function's CODE — docstring AND imports removed.

    The docstrings on these two deliberately NAME the tables and the
    crash, so a raw substring search over the source finds the
    explanation and reports it as the defect.

    Imports go too, and that is not cosmetic: a mutation that deletes the
    CALL to ``unfold_star_wedge_to_full_bz`` leaves the ``from
    symmetry_maps import ...`` line behind, so a check that searched the
    whole body would still find the name and pass.  Measured — the
    pre-fix body was re-applied on 2026-08-15 and this cell went green on
    the import alone until imports were stripped.  Only calls count.
    """
    body = _function(relpath, name).body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return "\n".join(
        ast.unparse(node) for node in body
        if not isinstance(node, (ast.Import, ast.ImportFrom)))


# ---------------------------------------------------------------------------
# Layer 1 — AST.  The boundary names its wedge and holds no table.
# ---------------------------------------------------------------------------

_BOUNDARIES = [
    ("gw/sc_iteration.py", "_write_sc_eqp_snapshot"),
    ("gw/sc_iteration.py", "dump_qp_wfn_artifacts"),
]


@pytest.mark.parametrize("relpath,name", _BOUNDARIES)
def test_the_sc_boundary_reduces_through_the_named_service_call(relpath, name):
    """Both writers reach the file wedge by name, not by index."""
    code = _code_of(relpath, name)
    assert _REDUCE in code, (
        f"{relpath}:{name} no longer calls {_REDUCE}.  Its writers are "
        f"indexed by wfn.kpoints — the FILE wedge — and that call is how "
        f"the rows are selected without the driver holding kirr_fullids.")


@pytest.mark.parametrize("relpath,name", _BOUNDARIES)
def test_the_sc_boundary_never_gathers_by_a_symmetry_table(relpath, name):
    """No ``array[kirr_fullids]`` in the driver, directly or via a local."""
    bad = _subscripts_by_a_symmetry_table(_function(relpath, name))
    assert not bad, (
        f"{relpath}:{name} gathers by a symmetry index table: {bad}.  Index "
        f"tables must not cross the service boundary into a driver — that is "
        f"the rule this branch exists to enforce.  Use {_REDUCE} / "
        f"{_UNFOLD_STAR}, which name the wedge and the direction instead.")


def test_the_star_wedge_operand_goes_through_the_full_bz():
    """There is no star-wedge → file-wedge hop, and none is invented here.

    The two wedges are different objects; the only route between them that
    is right on every deck is star → full BZ → file.  This pins that
    ``_write_sc_eqp_snapshot``'s ``kstar`` branch takes it, and pins the
    ORDER — structurally, by nesting, not by where the two names happen to
    appear in the text.

    An earlier version of this cell compared string offsets.  That passed
    for a spurious reason (the ``from symmetry_maps import`` line put both
    names near the top) and kept passing under a mutation that deleted the
    call, which is how the weakness was found.  Nesting is the property
    that actually holds: the unfold's RESULT must be the reduce's ARGUMENT.
    """
    fn = _function("gw/sc_iteration.py", "_write_sc_eqp_snapshot")

    # Local helpers that ARE the reduce — the reduce is normally factored
    # into one, so accept either the service name or such an alias.
    reducers = {_REDUCE}
    for node in ast.walk(fn):
        if isinstance(node, ast.FunctionDef) and node is not fn:
            if _REDUCE in "\n".join(ast.unparse(b) for b in node.body):
                reducers.add(node.name)

    parent = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    unfolds = [n for n in ast.walk(fn)
               if isinstance(n, ast.Call) and _UNFOLD_STAR in ast.unparse(n.func)]
    assert unfolds, (
        "_write_sc_eqp_snapshot no longer CALLS unfold_star_wedge_to_full_bz. "
        "Under sc_on_ibz the loop's rows are one per ORBIT and this writer "
        "wants one per STORED k; skipping the full-BZ hop is the "
        "(5, 46) vs (9, 46) crash.")

    def _wrapped_by_a_reduce(call):
        node = parent.get(call)
        while node is not None:
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func).split(".")[-1]
                return name in reducers
            node = parent.get(node)
        return False

    assert any(_wrapped_by_a_reduce(c) for c in unfolds), (
        "the star unfold's result is not passed to the file-wedge reduce. "
        "star -> full BZ -> file wedge is the only route that is right on "
        f"every deck; reducers seen here: {sorted(reducers)}")


def test_the_default_runs_the_loop_reduced():
    """``sc_on_ibz`` defaults True — the directive, and now the default.

    Flipping it changed no suite result on its own, which is exactly why
    the deck below had to be added in the same breath.
    """
    import sys
    sys.path.insert(0, str(_SRC))
    from gw.gw_config import _DEFAULTS
    assert _DEFAULTS["sc_on_ibz"] is True, (
        "sc_on_ibz is no longer True by default.  H^QP is then built and "
        "eigh'd at every full-BZ k — 8x the diagonalisation on the Si deck "
        "— against the owner's standing directive.")


# ---------------------------------------------------------------------------
# Layer 2 — the composite, on real decks, no driver.
# ---------------------------------------------------------------------------

#: MEASURED 2026-08-15 over every committed fixture: (file wedge, star
#: wedge).  Three coincide and three do not, and BOTH cases have to stay
#: represented — see ``test_unfold_through_the_service.py``'s corpus cell.
_DECK_WEDGES = [
    ("si_cohsex_debug", "WFN.h5", 8, 8),
    ("si_bse_debug", "WFN.h5", 8, 8),
    ("hbn_cohsex_debug", "WFN.h5", 18, 18),
    ("cohsex_debug", "WFNsmall.h5", 4, 3),
    ("gnppm_debug", "WFN.h5", 9, 5),
    ("bispinor_debug", "WFN.h5", 9, 5),
]


def _sym_for(deck: str, wfn_name: str):
    path = _REG / deck / wfn_name
    if not path.exists():
        pytest.skip(f"{deck}/{wfn_name} not in this tree")
    try:
        from ffi import _services
        _services.ensure_on_path()
        import symmetry_maps as sm
        from wfn_loader import WfnLoader
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"loader/service unavailable here ({type(exc).__name__})")
    return sm, WfnLoader(str(path))


@pytest.mark.parametrize("deck,wfn_name,nk_red,n_orbits", _DECK_WEDGES)
def test_star_to_full_to_file_lands_on_wfn_kpoints(
        deck, wfn_name, nk_red, n_orbits):
    """The composite the SC boundary relies on, on every committed deck.

    Its output length is ``wfn.nkpts`` whatever the two wedge sizes are —
    which is the property the replaced size-matching did not have.
    """
    sm, wfn = _sym_for(deck, wfn_name)
    sym = sm.SymMaps(wfn)
    assert (int(sym.nk_red), int(sm.KStarMap.from_sym(sym, int(wfn.ntran)).nk_irr)) \
        == (nk_red, n_orbits), (
        f"{deck}: the two wedge sizes moved; re-read the register before "
        f"trusting anything below.")

    rng = np.random.default_rng(19)
    star = (rng.standard_normal((n_orbits, 4))
            + 1j * rng.standard_normal((n_orbits, 4)))
    full = np.asarray(sm.unfold_star_wedge_to_full_bz(sym, star))
    assert full.shape[0] == int(sym.nk_tot)
    got = np.asarray(sm.reduce_full_bz_to_file_wedge(sym, full))
    assert got.shape[0] == int(wfn.nkpts) == nk_red, (
        f"{deck}: star({n_orbits}) -> full({sym.nk_tot}) -> file gave "
        f"{got.shape[0]} rows, not wfn.nkpts={int(wfn.nkpts)}.")


@pytest.mark.parametrize("deck,wfn_name,nk_red,n_orbits", _DECK_WEDGES)
def test_the_star_round_trip_inside_the_composite_is_exact(
        deck, wfn_name, nk_red, n_orbits):
    """star → full → star is the identity, so the composite loses nothing.

    Distinct from the file-wedge round trip, which is deliberately NOT the
    identity (``reduce_full_bz_to_file_wedge``'s docstring, and
    ``test_unfold_through_the_service.py``).  Pinned separately because the
    SC boundary depends on THIS one.
    """
    sm, wfn = _sym_for(deck, wfn_name)
    sym = sm.SymMaps(wfn)
    rng = np.random.default_rng(23)
    star = (rng.standard_normal((n_orbits, 4))
            + 1j * rng.standard_normal((n_orbits, 4)))
    full = np.asarray(sm.unfold_star_wedge_to_full_bz(sym, star))
    back = np.asarray(sm.star_select(full, np.asarray(sym.irr_idx_k))[0])
    assert np.array_equal(back, star), (
        f"{deck}: star round trip is not exact — max|d| = "
        f"{np.abs(back - star).max():.3e}")


# ---------------------------------------------------------------------------
# Layer 3 — the driver.  The layer that was missing.
# ---------------------------------------------------------------------------

def _eqp_blocks(path: pathlib.Path):
    """``(kpts (nk, 3), rows (nk, nb, 2))`` out of a BGW eqp file."""
    ks, rows, cur = [], [], None
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        tok = line.split()
        if len(tok) == 4 and "." in tok[0]:
            cur = []
            ks.append([float(tok[0]), float(tok[1]), float(tok[2])])
            rows.append(cur)
        elif len(tok) == 4:
            cur.append([float(tok[2]), float(tok[3])])
    return np.array(ks), np.array(rows)


@pytest.mark.regression
def test_sc_loop_runs_on_the_star_wedge_and_writes_the_file_wedge(
        gnppm_sc_session):
    """The deck runs, and the log states both k-sets by name.

    ``gnppm_debug`` has file wedge 9 and star wedge 5, so the two numbers
    below cannot both be right by coincidence.
    """
    out = gnppm_sc_session.stdout
    assert "nk_irr=5" in out and "H/E/U on the IBZ" in out, (
        "the SC loop did not reduce to the star wedge — sc_on_ibz's "
        f"default may have moved.  stdout tail:\n{out[-3000:]}")
    assert "QP dump k-sets: WFN_qp 9 (file wedge)" in out, (
        "the QP dump did not land on the file wedge (9 on this deck); the "
        f"star wedge is 5 and the two must not be confused.  stdout tail:"
        f"\n{out[-3000:]}")
    assert "loop 5 (star wedge)" in out


@pytest.mark.regression
def test_sc_eqp_outputs_are_on_the_file_wedge_with_its_own_coordinates(
        gnppm_sc_session):
    """9 blocks, and the coordinates ARE ``wfn.kpoints``.

    A count alone would pass on this deck for the wrong reason — 9 is also
    ``nk_tot`` here — so the coordinates are checked against the WFN's own
    k-list, in order.  That is what distinguishes the file wedge from the
    full BZ when the two happen to be the same length.
    """
    try:
        from ffi import _services
        _services.ensure_on_path()
        from wfn_loader import WfnLoader
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"loader unavailable ({type(exc).__name__})")

    wfn = WfnLoader(str(gnppm_sc_session.run_dir / "WFN.h5"))
    want = np.asarray(wfn.kpoints, dtype=np.float64)
    for name in ("eqp0.dat", "eqp1.dat"):
        path = gnppm_sc_session.run_dir / name
        assert path.exists(), f"SC run wrote no {name}"
        got_k, rows = _eqp_blocks(path)
        assert got_k.shape[0] == int(wfn.nkpts) == 9, (
            f"{name}: {got_k.shape[0]} blocks, expected wfn.nkpts=9")
        assert rows.shape[1] == 46, f"{name}: {rows.shape[1]} bands, expected 46"
        assert np.abs(got_k - want).max() < 1e-6, (
            f"{name}: block coordinates are not wfn.kpoints — max|d| = "
            f"{np.abs(got_k - want).max():.3e}.  The rows are on some other "
            f"k-set than the one the header claims.")
        assert np.isfinite(rows).all(), f"{name}: non-finite energies"


@pytest.mark.regression
def test_sc_per_map_snapshots_are_on_the_file_wedge(gnppm_sc_session):
    """``eqp0_iter####.dat`` too — this is the writer that crashed.

    ``_write_sc_eqp_snapshot`` is called from inside the mixing loop, so
    it is the first thing a reduced loop reaches; the crash was here and
    not in the final writers.
    """
    snaps = sorted(gnppm_sc_session.run_dir.glob("eqp0_iter[0-9][0-9][0-9][0-9].dat"))
    assert snaps, "the SC run wrote no per-map eqp snapshots"
    for path in snaps:
        got_k, rows = _eqp_blocks(path)
        assert got_k.shape == (9, 3), (
            f"{path.name}: {got_k.shape[0]} blocks, expected the file "
            f"wedge (9).  The star wedge is 5 — a 5 here is the loop's "
            f"own k-set reaching a writer indexed by wfn.kpoints.")
        assert rows.shape == (9, 46, 2)
