"""The PRODUCER side of q_irr storage: the capture, the coupling, the stamp.

WHAT IS MEASURED, AND WHAT HONESTLY CANNOT BE HERE.  The bytes are written
by ``file_io.slab_io.SlabIO``, which needs the phdf5 FFI this checkout does
not build — every restart-writer cell in the tree is red here for that
reason and this file does not pretend otherwise.  So the producer is
measured at its SEAMS, which are the parts that carry the decisions:

* the CAPTURE — that the pre-unfold block reaches the writer, that taking it
  REMOVES it, and that the writer REFUSES the combination "resolved ibz,
  nothing captured" rather than falling back to slicing the unfolded tensor;
* the CROSS-CHECK — that a capture whose tables are not the writer's own
  resolution's tables is refused, on the permutation AND on the umklapp
  wrap, separately;
* the COUPLING — that V and the W0 placeholder are sized from ONE decision,
  asserted on the source, because that is the trap ``tagged_arrays``'s own
  comment has been describing since dbe3b4ec;
* the STAMP — through ``stamp_qirr_tensor`` against a dataset written by
  plain h5py, which is what SlabIO's output looks like to the stamp, and
  read back through the real consumer seam bit-identically.

THE SLICE IS NOT THE WEDGE, and that is the reason the capture exists at
all.  Slicing the unfolded tensor at the IBZ parent rows gives the same
numbers only when ``sym_idx_q[q_irr_full_idx] == 0`` — a consequence of an
op-selection policy nobody agreed to freeze for this purpose.  A cell here
CONSTRUCTS a q-star ordering where it does not hold and shows the two
arrays differ, so the refusal above is guarding something real.
"""

from __future__ import annotations

import ast
import pathlib

import h5py
import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_TAGGED = _ROOT / "src" / "file_io" / "tagged_arrays.py"
_GW_INIT = _ROOT / "src" / "gw" / "gw_init.py"
_GW_OUTPUT = _ROOT / "src" / "gw" / "gw_output.py"
_V_Q = _ROOT / "src" / "gw" / "v_q_g_flat.py"
_SCREENING = _ROOT / "src" / "gw" / "screening.py"

from tests.test_restart_qirr_consumers import (       # noqa: E402
    _Arm, _assert_offdiag_elementwise)


@pytest.fixture()
def arm():
    return _Arm("gnppm_debug")


def _capture_from(arm, name="V_qmunu"):
    """Deposit ``arm``'s wedge exactly as the producer's unfold site does."""
    from gw.restart_q_storage import deposit_pre_unfold, take_pre_unfold
    t = arm.tables
    deposit_pre_unfold(
        name, arm.X_ibz, n_rmu_logical=arm.n_mu,
        q_irr_frac=t.q_irr_frac, irr_idx_q=t.irr_idx_q,
        sym_idx_q=t.sym_idx_q, sym_perm=t.sym_perm, L_table=t.L_table,
        n_sym_spatial=t.n_sym_spatial)
    return take_pre_unfold(name)


class _FakeResolution:
    """The three things the writer reads off a ``QgridSymmetryResolution``."""

    def __init__(self, arm, *, perm=None, L=None):
        self._perm = arm.tables.sym_perm if perm is None else perm
        self._L = arm.tables.L_table if L is None else L
        self.verdict = arm.verdict
        self.mode = "ibz"
        self.reason = ""
        self.n_sym_spatial = int(arm.tables.n_sym_spatial)

    @property
    def use_ibz(self):
        return True

    def tables(self):
        return self._perm, self._L


# ---------------------------------------------------------------------------
# 1. The capture hand-off
# ---------------------------------------------------------------------------

def test_the_capture_carries_the_block_and_taking_it_removes_it(arm):
    """TAKE REMOVES — the rule that keeps two writers off one wedge.

    A capture read twice is the W0 writer storing V's block under W0's
    name, which reconstructs to plausible, wrong numbers at every q while
    every shape check stays green.
    """
    from gw.restart_q_storage import peek_pre_unfold, take_pre_unfold

    cap = _capture_from(arm)
    assert cap is not None
    assert np.array_equal(np.asarray(cap.X_ibz), arm.X_ibz)
    assert cap.n_rmu_logical == arm.n_mu
    assert peek_pre_unfold("V_qmunu") is None
    assert take_pre_unfold("V_qmunu") is None


def test_the_capture_scope_isolates(arm):
    """RED TWIN of the slot's lifetime: a scope's deposits do not escape it."""
    from gw.restart_q_storage import (capture_scope, deposit_pre_unfold,
                                      take_pre_unfold)
    t = arm.tables
    with capture_scope():
        deposit_pre_unfold(
            "V_qmunu", arm.X_ibz, n_rmu_logical=arm.n_mu,
            q_irr_frac=t.q_irr_frac, irr_idx_q=t.irr_idx_q,
            sym_idx_q=t.sym_idx_q, sym_perm=t.sym_perm, L_table=t.L_table,
            n_sym_spatial=t.n_sym_spatial)
        assert take_pre_unfold("V_qmunu") is not None
    assert take_pre_unfold("V_qmunu") is None


def test_the_capture_builds_the_tables_the_unfold_used(arm):
    """PADDED tables plus the logical extent — not the unpadded pair.

    ``QgridSymmetryResolution.tables()`` returns the tables BEFORE the
    producer bakes in the μ pad, and the producer unfolds with the padded
    form.  Storing the unpadded pair beside a padded tensor would be a file
    whose tables and tensor describe different extents; storing the padded
    pair lets the writer STRIP it and run 652b731e's three invariants on
    the tail, which is a free corruption check.
    """
    cap = _capture_from(arm)
    tables = cap.tables()
    assert np.array_equal(np.asarray(tables.sym_perm),
                          np.asarray(arm.tables.sym_perm))
    assert int(tables.n_sym_spatial) == int(arm.tables.n_sym_spatial)


# ---------------------------------------------------------------------------
# 2. The cross-check
# ---------------------------------------------------------------------------

def test_matching_tables_pass_the_cross_check(arm):
    """The control arm: the real pairing must not refuse."""
    from gw.restart_q_storage import assert_capture_matches

    cap = _capture_from(arm)
    assert_capture_matches(cap, _FakeResolution(arm), context="gate")


def test_a_crossed_permutation_refuses(arm):
    """RED TWIN: tables that are not this tensor's are REFUSED.

    A stale slot, a bispinor channel crossed, a resolution taken against
    the wrong ``sym`` — all arrive as this, and none of them is visible
    downstream: shapes agree, Hermiticity agrees, the spectrum is
    plausible.  Same failure shape as the two conjugation bugs, reached
    through the plumbing instead of through the algebra.
    """
    from gw.restart_q_storage import assert_capture_matches

    cap = _capture_from(arm)
    perm = np.array(arm.tables.sym_perm, copy=True)
    perm[0, 0], perm[0, 1] = perm[0, 1], perm[0, 0]
    with pytest.raises(ValueError, match=r"permutation the writer"):
        assert_capture_matches(cap, _FakeResolution(arm, perm=perm),
                               context="gate")


def test_a_crossed_umklapp_wrap_refuses_SEPARATELY(arm):
    """RED TWIN 2, and it is its own cell because L fails DIFFERENTLY.

    A wrong ``L_table`` is wrong only on the q's that wrap, so it is
    diagonal-preserving and off-diagonal-destroying — the shape this area
    has shipped twice — while a wrong permutation is wrong everywhere.  A
    single message for both would describe the wrong symptom half the time.
    """
    from gw.restart_q_storage import assert_capture_matches

    cap = _capture_from(arm)
    L = np.array(arm.tables.L_table, copy=True)
    L[0, 0, 0] += 1
    with pytest.raises(ValueError, match=r"umklapp"):
        assert_capture_matches(cap, _FakeResolution(arm, L=L),
                               context="gate")


# ---------------------------------------------------------------------------
# 3. The stamp, and the round trip through the real consumer
# ---------------------------------------------------------------------------

def test_a_slabio_style_write_stamps_and_reads_back_bit_identically(
        arm, tmp_path):
    """WHAT SlabIO'S OUTPUT LOOKS LIKE TO THE STAMP, end to end.

    The dataset is created by plain h5py at the LOGICAL μ extent — which is
    exactly what SlabIO leaves behind, since ``_mu_logical_shape`` clips the
    μ axes on the way out — then stamped, then read back through the REAL
    consumer seam and compared element-wise on the off-diagonals against
    ``unfold_isdf_operator`` on the same wedge.  BIT equality: the stored
    block is the pre-unfold array, so the reader and the producing run
    evaluate one function on one argument list.
    """
    from bse.bse_io import restart_munu_full_bz
    from symmetry_maps import stamp_qirr_tensor

    cap = _capture_from(arm)
    path = str(tmp_path / "restart.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("V_qmunu", data=np.asarray(cap.X_ibz))
    stamp_qirr_tensor(path, "V_qmunu", tables=cap.tables(),
                      closure_verdict=arm.verdict,
                      n_rmu_logical=cap.n_rmu_logical)
    with h5py.File(path, "r") as f:
        assert f["V_qmunu"].attrs["q_storage"] == "ibz"
        got = restart_munu_full_bz(f["V_qmunu"], "V_qmunu", path)
    _assert_offdiag_elementwise(got, arm.kernel(), "producer stamp round trip")
    assert np.array_equal(got, arm.kernel())


def test_the_stamp_refuses_a_tensor_at_the_wrong_mu_extent(arm, tmp_path):
    """RED TWIN: a half-clipped write must not be stamped as whole.

    The tables are stripped to ``n_rmu_logical`` and the tensor must
    already be at it.  A disagreement means the file's tables and its
    tensor describe different centroid sets — recoverable-looking and
    unrecoverable.
    """
    from symmetry_maps import stamp_qirr_tensor

    cap = _capture_from(arm)
    path = str(tmp_path / "short.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("V_qmunu",
                         data=np.asarray(cap.X_ibz)[:, :-1, :-1])
    with pytest.raises(ValueError, match=r"μ extent"):
        stamp_qirr_tensor(path, "V_qmunu", tables=cap.tables(),
                          closure_verdict=arm.verdict,
                          n_rmu_logical=cap.n_rmu_logical)


def test_the_stamp_refuses_a_non_closed_set(arm, tmp_path):
    """RED TWIN: the closure refusal is on THIS path too, not only the other.

    ``stamp_qirr_tensor`` is a second door into the format and a door
    without the refusal is a door production walks through.
    """
    import dataclasses
    from symmetry_maps import stamp_qirr_tensor

    cap = _capture_from(arm)
    path = str(tmp_path / "open.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("V_qmunu", data=np.asarray(cap.X_ibz))
    bad = dataclasses.replace(arm.verdict, closed=False,
                              violating_ops=(1,), worst_residual=0.13)
    with pytest.raises(Exception, match=r"(?i)clos"):
        stamp_qirr_tensor(path, "V_qmunu", tables=cap.tables(),
                          closure_verdict=bad,
                          n_rmu_logical=cap.n_rmu_logical)


def test_write_qirr_tensor_and_the_stamp_produce_the_SAME_attrs(arm,
                                                                tmp_path):
    """ONE stamp implementation, measured rather than asserted in prose.

    The producer stamps a SlabIO-written dataset and the format's own
    writer stamps one it created; if those were two code paths they would
    be two claims about what a file says, differing on the day one of them
    gained an attr.  Every attr except the timestamp is compared.
    """
    from symmetry_maps import stamp_qirr_tensor, write_qirr_tensor

    cap = _capture_from(arm)
    a = str(tmp_path / "a.h5")
    b = str(tmp_path / "b.h5")
    write_qirr_tensor(a, "V_qmunu", arm.X_ibz, tables=arm.tables,
                      closure_verdict=arm.verdict)
    with h5py.File(b, "w") as f:
        f.create_dataset("V_qmunu", data=np.asarray(cap.X_ibz))
    stamp_qirr_tensor(b, "V_qmunu", tables=cap.tables(),
                      closure_verdict=arm.verdict,
                      n_rmu_logical=cap.n_rmu_logical)
    skip = {"qirr_written_utc"}
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        ka = {k: v for k, v in fa["V_qmunu"].attrs.items() if k not in skip}
        kb = {k: v for k, v in fb["V_qmunu"].attrs.items() if k not in skip}
        assert set(ka) == set(kb)
        for k in ka:
            assert np.all(np.asarray(ka[k]) == np.asarray(kb[k])), k


# ---------------------------------------------------------------------------
# 4. Why the capture exists: the slice is NOT the wedge
# ---------------------------------------------------------------------------

def test_slicing_the_unfolded_tensor_is_a_DIFFERENT_array(arm):
    """THE MEASUREMENT BEHIND THE DESIGN DECISION, not an appeal to it.

    Slicing the unfolded tensor at the IBZ parent rows equals the wedge
    only when each parent q's own row uses the identity op.  ``gnppm``'s
    committed star tables put a non-identity op on parent rows, so the
    slice and the wedge genuinely differ here — which is what makes
    ``write_restart_state_to_h5``'s "resolved ibz, nothing captured"
    refusal a guard on something real rather than a formality.
    """
    full = arm.kernel()
    irr = np.asarray(arm.tables.irr_idx_q)
    sym = np.asarray(arm.tables.sym_idx_q)
    # First full-BZ row for each parent, in parent order — the slice a
    # writer without a capture would be forced to take.
    rows = [int(np.nonzero(irr == p)[0][0])
            for p in range(int(arm.tables.n_q_ibz))]
    sliced = full[rows]
    identity_rows = all(int(sym[r]) == 0 for r in rows)
    if identity_rows:
        pytest.skip("this deck's parent rows all use the identity op; the "
                    "slice happens to coincide and this cell measures "
                    "nothing here")
    assert not np.array_equal(sliced, arm.X_ibz), (
        "the slice and the pre-unfold wedge are supposed to differ on this "
        "deck; if they no longer do, the op-selection policy changed and "
        "the design note about it needs re-checking")


# ---------------------------------------------------------------------------
# 5. Source ratchets: the decision reaches both writers, once
# ---------------------------------------------------------------------------

def test_the_placeholder_is_sized_from_the_resolved_tensor():
    """THE COUPLING dbe3b4ec NAMED, closed and pinned.

    ``init_W0`` sized the W0 placeholder from ``V_qmunu.shape``, so V's
    storage decision silently became W0's.  It must now read the RESOLVED
    array.  Asserted on the source because the behavioural path needs
    SlabIO and the FFI this box does not build.
    """
    src = _TAGGED.read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "write_restart_state_to_h5")
    body = ast.unparse(fn)
    assert "_mu_logical_shape(V_on_disk.shape" in body, (
        "the W0 placeholder no longer takes its shape from the resolved "
        "tensor; V's q-storage decision would silently become W0's again")
    assert "_write('V_qmunu', V_on_disk" in body


def test_both_writers_take_a_resolution_and_a_capture():
    """One decision object, two writers, and each takes its OWN capture."""
    src = _TAGGED.read_text()
    for fn_name in ("write_restart_state_to_h5", "write_w0_qmunu_to_h5"):
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == fn_name)
        names = {a.arg for a in fn.args.kwonlyargs}
        assert "qirr" in names, f"{fn_name} does not take the resolution"
    for path, tensor in ((_GW_INIT, "V_qmunu"), (_GW_OUTPUT, "W0_qmunu")):
        calls = [n for n in ast.walk(ast.parse(path.read_text()))
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "take_pre_unfold"]
        assert len(calls) == 1, f"{path.name}: {len(calls)} take calls"
        assert ast.unparse(calls[0].args[0]) == f"'{tensor}'", (
            f"{path.name} takes the wrong tensor's capture")


def test_both_unfold_sites_deposit_before_they_unfold():
    """The offer must precede the unfold, in both producers.

    Depositing AFTER would hand the writer the unfolded tensor under the
    wedge's name — the same shape mistake as slicing, with no refusal to
    catch it because the object would be present.
    """
    for path in (_V_Q, _SCREENING):
        src = path.read_text()
        dep = src.index("deposit_pre_unfold(")
        # the FIRST unfold call after the deposit's own import line
        unf = src.index("unfold_isdf_operator(", dep)
        assert dep < unf, f"{path.name}: deposit does not precede the unfold"


def test_only_the_charge_tile_is_captured():
    """The bispinor CT/TT tiles are not restart tensors and must not deposit.

    ``_compute_V_q_g_flat_one_tile`` runs once per tile; a deposit on every
    one would leave the LAST tile's block in the slot for the writer, which
    is a transverse-channel wedge stored as V_qmunu.
    """
    tree = ast.parse(_V_Q.read_text())
    deposits = [n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "deposit_pre_unfold"]
    assert len(deposits) == 1
    guards = [n for n in ast.walk(tree)
              if isinstance(n, ast.If)
              and "deposit_pre_unfold" in ast.unparse(n)]
    assert guards, "the deposit is unguarded; every tile would deposit"
    assert any("timing_label" in ast.unparse(g.test) for g in guards), (
        "the deposit guard no longer selects a tile by name")
