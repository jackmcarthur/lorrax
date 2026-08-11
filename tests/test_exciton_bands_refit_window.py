"""The BSE-window exchange refit, and the gate that replaces its tile null.

WHAT THIS FILE IS ABOUT.  ``bse.exciton_bands --vq-mode=refit`` used to demand
that the deck's band window BE the producer's ζ-fit window, because that is
the window in which the refit reproduces the stored ``V_qmunu`` tiles and can
therefore be certified against them.  On the Si 4×4×4 / 960-centroid lineage
that window is 60 bands, whose htransform Galerkin rank bound is
nk·nb = 64·60 = 3840 against a parent ISDF basis of n_μ·n_s = 960·2 = 1920 —
so ``build_fH_R`` refuses and the refit is unreachable
(``tests/known_failures/2026-08-10-exciton-bands-offgrid-Q-is-slab-only.md``,
third wall).

``--refit-window=bse`` fits ζ' on the deck's window instead: the BSE window
plus its conduction guards, which is every pair density the exchange kernel
can ask for, and a bound of 64·20 = 1280 the parent basis spans comfortably.
The price is that ζ' ≠ ζ, so the TILE identity is gone.  The cells below are
about the two halves of that trade: the slice must be located by the restart's
own band-window stamp and must not cut a multiplet, and the certification must
MOVE UP to the contracted eigenvalues rather than be relaxed in place.

The last two cells are the ones that would catch a silent regression: the
tolerance must stay a module constant with no CLI route to it, and the driver
must hand ``refit_vq`` the SLICED ζ view — passing the unsliced bundle fits ζ'
on the wrong band count with no shape error anywhere, because the pair-density
band axes are contracted away before any shape is checked.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pytest

SRC_DRIVER = os.path.join(os.path.dirname(__file__), "..", "src", "bse",
                          "exciton_bands.py")

RY2EV = 13.6056980659


def _driver():
    pytest.importorskip("jax")
    from bse import exciton_bands
    return exciton_bands


def _vq():
    pytest.importorskip("jax")
    from bse import vq_interp
    return vq_interp


def _zx(nb_zeta=60, b0=0, nk=64, ns=2, n_mu=191, enk=None):
    """A ζ bundle stub carrying only what the window view reads."""
    if enk is None:
        # Kramers pairs: bands 2m and 2m+1 exactly degenerate, pairs split by
        # 1 Ry.  Every EVEN boundary is safe, every odd one cuts a pair.
        enk = np.repeat(np.arange(nb_zeta // 2, dtype=np.float64), 2)
        enk = np.tile(enk, (nk, 1))
    return {
        "restart_file": "/nowhere/isdf_tensors.h5",
        "_h5_restart": {"band_window": np.array([b0, b0, b0 + 8,
                                                 b0 + nb_zeta, b0 + nb_zeta])},
        "psi": np.arange(nk * nb_zeta * ns * n_mu, dtype=np.complex128
                         ).reshape(nk, nb_zeta, ns, n_mu),
        "enk": enk,
        "nk": nk, "nb": nb_zeta, "ns": ns, "n_mu": n_mu,
    }


# ---------------------------------------------------------------------------
# (1) the CLI contract
# ---------------------------------------------------------------------------

def test_refit_window_defaults_to_the_producers_zeta_window():
    """Default is ``zeta``: every run that certified on the TILE null before
    this branch still certifies on the tile null after it."""
    actions = {a.dest: a for a in _driver().build_parser()._actions}
    assert "refit_window" in actions, "--refit-window is gone from the CLI"
    assert actions["refit_window"].default == "zeta", (
        "the windowed refit became the default; every existing --vq-mode="
        "refit run would silently switch from the tile identity to the "
        "contracted gate")
    assert set(actions["refit_window"].choices) == {"zeta", "bse"}


def test_help_says_the_gate_moves_rather_than_relaxes():
    """A reader of ``--help`` must learn that ``bse`` changes WHICH gate runs,
    not how loose it is."""
    actions = {a.dest: a for a in _driver().build_parser()._actions}
    h = (actions["refit_window"].help or "").lower()
    for token in ("zeta'", "contracted", "stored", "meV".lower()):
        assert token in h, f"--refit-window help never mentions {token!r}"


def test_bse_window_is_refused_outside_pure_refit():
    """``--vq-mode=both`` scores interp against the refit as GROUND TRUTH; a
    re-fitted ζ' would move that reference under the comparison."""
    src = ast.parse(open(SRC_DRIVER, encoding="utf8").read())
    txt = open(SRC_DRIVER, encoding="utf8").read()
    assert 'args.refit_window == "bse" and not pure_refit' in txt, (
        "nothing refuses --refit-window=bse outside --vq-mode=refit")
    assert src is not None


# ---------------------------------------------------------------------------
# (2) the band-axis view: located by the stamp, never by an assumed origin
# ---------------------------------------------------------------------------

def test_the_identity_slice_returns_the_bundle_itself():
    """A deck window equal to the ζ-fit window is not a windowed refit and
    must not become one — the caller then still runs the tile null."""
    vq = _vq()
    zx = _zx(nb_zeta=60, b0=0)
    out = vq.refit_window_view(zx, (0, 60), log_fn=lambda *_: None)
    assert out is zx


def test_the_sub_window_is_sliced_at_the_stamped_offset():
    """The whole point of reading ``band_window``: the ζ-fit window's absolute
    origin is b0, and the deck's absolute range must be mapped through it."""
    vq = _vq()
    b0 = 12                     # a ζ-fit window that does NOT start at band 0
    zx = _zx(nb_zeta=60, b0=b0)
    out = vq.refit_window_view(zx, (b0 + 8, b0 + 28),
                               log_fn=lambda *_: None)
    assert out is not zx
    assert out["nb"] == 20
    assert out["psi"].shape[1] == 20
    np.testing.assert_array_equal(out["psi"], zx["psi"][:, 8:28])
    np.testing.assert_array_equal(out["enk"], np.asarray(zx["enk"])[:, 8:28])
    assert out["_refit_window_abs"] == (b0 + 8, b0 + 28)
    # the untouched bundle is still the untouched bundle (a VIEW, not a
    # mutation — the caller keeps ``zx`` for the stored-tile route)
    assert zx["nb"] == 60 and zx["psi"].shape[1] == 60


def test_the_galerkin_bound_is_what_the_slice_buys():
    """The arithmetic the third wall is about, asserted rather than asserted
    about: 64·60 = 3840 > 1920 ≥ 64·20 = 1280."""
    vq = _vq()
    zx = _zx(nb_zeta=60, b0=0)
    parent_basis = 960 * 2
    assert zx["nk"] * zx["nb"] > parent_basis
    out = vq.refit_window_view(zx, (0, 20), log_fn=lambda *_: None)
    assert out["nk"] * out["nb"] <= parent_basis


def test_a_window_outside_the_stored_one_refuses():
    """The refit fits ζ' FROM the stored ψ, so it can narrow that window and
    never reach outside it."""
    vq = _vq()
    zx = _zx(nb_zeta=60, b0=0)
    with pytest.raises(SystemExit) as e:
        vq.refit_window_view(zx, (0, 72), log_fn=lambda *_: None)
    assert "not contained" in str(e.value)


def test_a_bundle_with_no_band_window_stamp_refuses():
    """Without the stamp the offset would have to be assumed, and an assumed
    offset slices the wrong bands with every shape still matching."""
    vq = _vq()
    zx = _zx(nb_zeta=60, b0=0)
    zx["_h5_restart"] = {}
    with pytest.raises(SystemExit) as e:
        vq.refit_window_view(zx, (0, 20), log_fn=lambda *_: None)
    assert "band_window" in str(e.value)


def test_a_stamp_that_disagrees_with_the_tensor_refuses():
    vq = _vq()
    zx = _zx(nb_zeta=60, b0=0)
    zx["_h5_restart"]["band_window"] = np.array([0, 0, 8, 40, 40])
    with pytest.raises(SystemExit) as e:
        vq.refit_window_view(zx, (0, 20), log_fn=lambda *_: None)
    assert "disagree" in str(e.value)


# ---------------------------------------------------------------------------
# (3) degeneracy-strict edges on the refit window
# ---------------------------------------------------------------------------

def test_a_refit_window_that_cuts_a_kramers_pair_refuses_by_default():
    """ζ' fitted on half a multiplet is not a subspace of anything, and the
    owner's 2026-08-10 ruling is that strict REFUSES rather than snapping."""
    vq = _vq()
    from common.band_degeneracy import BandWindowDegeneracyError
    zx = _zx(nb_zeta=60, b0=0)
    with pytest.raises(BandWindowDegeneracyError):
        vq.refit_window_view(zx, (0, 21), log_fn=lambda *_: None)


def test_the_even_boundary_next_to_it_is_accepted():
    """The control: the same spectrum, the boundary moved off the pair."""
    vq = _vq()
    zx = _zx(nb_zeta=60, b0=0)
    out = vq.refit_window_view(zx, (0, 20), log_fn=lambda *_: None)
    assert out["nb"] == 20


def test_snap_is_reachable_but_is_not_the_default():
    """``snap`` degrades to a warning here (there is nothing to widen at a
    committed window), and it has to be asked for by name."""
    vq = _vq()
    zx = _zx(nb_zeta=60, b0=0)
    said = []
    out = vq.refit_window_view(zx, (0, 21), log_fn=said.append,
                               degeneracy_mode="snap")
    assert out["nb"] == 21
    assert any("degenerate multiplet" in s for s in said), said


# ---------------------------------------------------------------------------
# (4) the tile null refuses a windowed refit instead of being re-tuned
# ---------------------------------------------------------------------------

def test_the_tile_null_refuses_a_windowed_refit_state():
    """``refit_ongrid_null`` computes a DIFFERENT object under a windowed ζ'.
    It must say so, not be handed a bigger ``tol``."""
    vq = _vq()
    rst = {"window_mode": "bse", "window_abs": (0, 20)}
    with pytest.raises(SystemExit) as e:
        vq.refit_ongrid_null({}, rst, None, [4, 4, 4], None,
                             log_fn=lambda *_: None)
    msg = str(e.value)
    assert "TILE-level" in msg
    assert "do not widen" in msg.lower()


# ---------------------------------------------------------------------------
# (5) the contracted certification
# ---------------------------------------------------------------------------

def _cert(delta_ry, n_eig=8):
    """Refit route vs stored route differing by ``delta_ry`` at one level."""
    eb = _driver()
    nQ = 6
    cert_idx = [2, 4]
    Qpath = np.zeros((nQ, 3))
    Qpath[2] = [0.0, 0.5, 0.5]          # X, on the 4x4x4 tile grid
    Qpath[4] = [0.5, 0.5, 0.5]          # L
    refit = np.tile(np.linspace(2.0, 3.0, n_eig) / RY2EV, (nQ, 1))
    stored = np.stack([refit[i].copy() for i in cert_idx])
    stored[1, 3] += delta_ry
    return eb._certify_refit_against_stored(
        cert_idx, Qpath, refit, stored, np.array([4, 4, 4]),
        log=lambda *_: None)


def test_the_certification_passes_when_the_routes_agree():
    rows = _cert(0.0)
    assert len(rows) == 2
    assert all(r[2] == pytest.approx(0.0, abs=1e-12) for r in rows)
    # the rows carry the TILE index, which is what makes them auditable
    assert rows[0][1] == (0, 2, 2) and rows[1][1] == (2, 2, 2)


def test_the_certification_refuses_above_the_gate():
    """0.02 meV against a 0.01 meV gate: refuse, and say the number."""
    eb = _driver()
    delta_ry = 0.02e-3 / RY2EV
    with pytest.raises(SystemExit) as e:
        _cert(delta_ry)
    msg = str(e.value)
    assert "0.020" in msg, msg
    assert f"{eb.REFIT_CERT_TOL_MEV:g} meV gate" in msg
    assert "not a tolerance to widen" in msg


def test_the_certification_accepts_just_under_the_gate():
    """The bracket is real in both directions — a gate that never passes is
    not a gate."""
    rows = _cert(0.008e-3 / RY2EV)
    assert max(r[2] for r in rows) == pytest.approx(0.008, rel=1e-3)


def test_the_refusal_names_the_window_and_the_basis_not_the_tolerance():
    """What a failure MEANS: the window or the basis, never 'raise the cap'."""
    with pytest.raises(SystemExit) as e:
        _cert(1.0e-3 / RY2EV)
    msg = str(e.value)
    assert "nval/ncond" in msg
    assert "orthonormality" in msg
    assert "Do not raise REFIT_CERT_TOL_MEV" in msg


# ---------------------------------------------------------------------------
# (6) the two regressions that would be silent
# ---------------------------------------------------------------------------

def test_the_gate_tolerance_has_no_cli_route():
    """A flag on this number is a flag for making a failed certification pass,
    and the failure it guards is silent by construction."""
    eb = _driver()
    assert eb.REFIT_CERT_TOL_MEV == 0.01
    dests = {a.dest for a in eb.build_parser()._actions}
    for bad in ("refit_cert_tol", "refit_cert_tol_mev", "cert_tol",
                "refit_tol"):
        assert bad not in dests, (
            f"--{bad.replace('_', '-')} exists: the contracted certification "
            f"can now be widened from the command line")
    src = open(SRC_DRIVER, encoding="utf8").read()
    assert "LORRAX_REFIT_CERT" not in src, (
        "an environment override for the certification tolerance appeared")


def test_there_are_exactly_two_grades_and_both_are_module_constants():
    """THE SHAPE OF THE CONCESSION, and the reason it is not a dial.

    ``--cert-grade`` was added so a deliverable that is a PICTURE can be drawn
    on a route whose own floor is ~0.9 meV.  What makes that a grade rather
    than a relaxation is that the tolerance surface is closed: two names, two
    module constants, and argparse ``choices`` refusing everything else.  A
    third number reachable from anywhere — a float flag, an env var, a deck
    key — turns the whole thing back into the knob the reference constant's
    docstring refuses, so this cell checks the closure and not merely the
    values.
    """
    eb = _driver()
    assert eb.CERT_TOL_BY_GRADE == {"reference": 0.01, "visualization": 1.0}
    assert eb.CERT_TOL_BY_GRADE["reference"] is eb.REFIT_CERT_TOL_MEV
    assert (eb.CERT_TOL_BY_GRADE["visualization"]
            is eb.CERT_TOL_VISUALIZATION_MEV)
    act = {a.dest: a for a in eb.build_parser()._actions}["cert_grade"]
    assert act.default == "reference", (
        "the default grade moved off 'reference' — every existing invocation "
        "would silently start certifying to 1 meV")
    assert set(act.choices) == set(eb.CERT_TOL_BY_GRADE), (
        "--cert-grade's choices and CERT_TOL_BY_GRADE disagree, so a grade "
        "exists that has no constant or a constant that cannot be selected")
    assert act.type is None, "--cert-grade takes a NAME, never a number"
    src = open(SRC_DRIVER, encoding="utf8").read()
    for bad in ("LORRAX_CERT_TOL", "LORRAX_CERT_GRADE", "cert_tol_mev="):
        assert bad not in src, f"a third route to the tolerance appeared: {bad}"


def _cert_args(dmax_mev, n_q=2):
    """Synthetic dual-solve rows whose worst |ΔE_S| is ``dmax_mev``."""
    Qpath = np.array([[0.0, 0.5, 0.5], [0.5, 0.5, 0.5]] * n_q)[:n_q]
    ry = dmax_mev / (RY2EV * 1e3)
    refit = [np.array([1.0, 2.0]) for _ in range(n_q)]
    stored = [np.array([1.0, 2.0 - ry]) for _ in range(n_q)]
    return list(range(n_q)), Qpath, refit, stored, np.array([4, 4, 4])


@pytest.mark.parametrize("grade,tol", [("reference", 0.01),
                                       ("visualization", 1.0)])
def test_every_grade_still_refuses_above_its_own_line(grade, tol):
    """A GRADE IS STILL A GATE.  The concession is the number, never the
    refusal: the dual solve runs at both grades and raises at both.  Checked
    on either side of each line so a grade cannot become a pass-through.
    """
    eb = _driver()
    idx, Qp, refit, stored, kg = _cert_args(tol * 0.5)
    rows, worst = eb._certify_refit_against_stored(
        idx, Qp, refit, stored, kg, log=lambda *_: None, grade=grade)
    assert len(rows) == len(idx)
    assert worst == pytest.approx(tol * 0.5, rel=1e-6)

    idx, Qp, refit, stored, kg = _cert_args(tol * 2.0)
    with pytest.raises(SystemExit) as exc:
        eb._certify_refit_against_stored(
            idx, Qp, refit, stored, kg, log=lambda *_: None, grade=grade)
    assert f"{tol:g} meV {grade} gate" in str(exc.value)


def test_an_unknown_grade_is_refused_rather_than_defaulted():
    """The one failure mode a dict lookup invites."""
    eb = _driver()
    idx, Qp, refit, stored, kg = _cert_args(0.0)
    with pytest.raises(SystemExit, match="is not one of"):
        eb._certify_refit_against_stored(
            idx, Qp, refit, stored, kg, log=lambda *_: None, grade="loose")


def test_a_visualization_pass_stamps_the_grade_and_the_number_together():
    """WHAT KEEPS A PICTURE FROM BEING QUOTED AS A NUMBER.

    The stamp is one string used by the provenance line, the ``.dat`` header
    and the ``.png``; it must carry BOTH the words and the certified value,
    and the driver must write it into all three.  Checked at the source level
    for the three sinks, because a run that only logs it has stamped nothing
    a reader of the figure can see.
    """
    eb = _driver()
    stamp = eb.cert_grade_stamp("visualization", 0.85783)
    assert "visualization grade" in stamp
    assert "0.85783" in stamp and "1 meV" in stamp
    assert "reference grade" in eb.cert_grade_stamp("reference", 0.001)

    src = open(SRC_DRIVER, encoding="utf8").read()
    assert src.count("cert_grade_stamp(args.cert_grade") >= 3, (
        "the certification stamp does not reach all three sinks — it must be "
        "written into the provenance line, the .dat header AND the plot, and "
        "the plot is the one that ends up in a talk")
    assert "fig.text(" in src, "the .png carries no stamp"


def test_the_driver_refits_against_the_sliced_zeta_view():
    """THE SILENT ONE.  ``refit_vq`` contracts the band axes away, so handing
    it the unsliced bundle while ``rst`` carries the narrow window fits ζ' on
    a mismatched band count with no shape error anywhere.  The driver must
    pass ``rst["zx_fit"]``.
    """
    tree = ast.parse(open(SRC_DRIVER, encoding="utf8").read())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "refit_vq"]
    assert calls, "exciton_bands no longer calls vq_interp.refit_vq at all"
    first = [c.args[0] for c in calls if c.args]
    names = [a.id for a in first if isinstance(a, ast.Name)]
    assert "zx_fit" in names, (
        "no refit_vq call in exciton_bands takes zx_fit — the per-Q refit is "
        "fitting against the UNSLICED zeta bundle, which under "
        "--refit-window=bse is the wrong band window and raises nothing")
    assert len(names) == len(first), (
        "a refit_vq call takes a non-Name first argument; check it is the "
        "sliced view and update this cell")


def test_the_tile_null_runs_only_on_the_zeta_branch():
    """The modes certify DIFFERENT objects.  Running the TILE null on a
    windowed ζ' would compare against tiles ζ' never claimed to reproduce, so
    the call has to sit inside the ``refit_window == "zeta"`` branch — not
    merely somewhere after it.  Structural, via AST, because the source also
    NAMES both functions in its docstring and a substring search reads those.
    """
    tree = ast.parse(open(SRC_DRIVER, encoding="utf8").read())

    def _calls(node, attr):
        return [n for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == attr]

    guards = [n for n in ast.walk(tree)
              if isinstance(n, ast.If)
              and isinstance(n.test, ast.Compare)
              and isinstance(n.test.left, ast.Attribute)
              and n.test.left.attr == "refit_window"
              and isinstance(n.test.comparators[0], ast.Constant)
              and n.test.comparators[0].value == "zeta"]
    assert len(guards) == 1, (
        f"expected exactly one `args.refit_window == \"zeta\"` branch, found "
        f"{len(guards)}")
    guard = guards[0]
    in_branch = _calls(guard, "refit_ongrid_null")
    assert len(in_branch) == 1, (
        "the tile null is not inside the refit-window branch — a windowed ζ' "
        "would be checked against tiles it is not computing")
    assert len(_calls(tree, "refit_ongrid_null")) == 1, (
        "exciton_bands calls refit_ongrid_null more than once; only the "
        "guarded call may exist")
    # ...and the contracted gate is the else side, not an unconditional extra.
    assert _calls(guard, "_certify_refit_against_stored") == []
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "_certify_refit_against_stored"
               for n in ast.walk(tree)), (
        "the contracted certification is never called")
