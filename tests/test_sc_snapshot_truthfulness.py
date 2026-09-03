"""The per-map ``eqp0_iterNNNN.dat`` snapshot must not lie about itself.

Three separate ways this file has misdescribed its own contents, all of
which downstream residual scripts read as convergence:

1. **An all-zero map-output column.**  MEASURED 2026-08-15 on the sodium
   48b one-shot metallic arms
   (``runs/Na/02_soc48b_qsgw_mpa/01_lorrax_metal_mpa/r4_np*/eqp0_iter0000.dat``,
   4 arms x 1392 rows, byte-identical apart from the timestamp): the column
   documented as ``eigvalsh(F(H_in))`` was written as zeros, and the
   header's own ``map_output_RMS_dE_prev_output = 2.467681671e+01 eV`` is
   simply ``RMS|E_DFT|`` -- the arithmetic signature of subtracting zero.
   The campaign's ``r6_residual.py`` computes ``d = e_qp - prev`` from
   exactly these files and would have reported a FALSE converged from call
   1 onward; ``r4_grid_floor.py`` returned a ``0.000000000e+00`` "floor".

2. **A stamp measured against a TRIAL neighbour.**  Under rCROP the
   preceding map call alternates trial / accepted, and a trial step sits
   near its accepted neighbour by construction.  Re-analysis of an accepted
   MPA QSGW run's snapshots (2026-08-14) put the ledger's 2.6571 meV figure
   at ~19x below the accepted-to-accepted 50.87 meV max\\|dE\\| over the
   protected bands -- so a run reported converged at 2 meV was 25x off.

3. **Post-mix stamps on a pre-mix column.**  Under ``mixing != 1`` the file
   holds the UNMIXED candidate while the stamp was computed from the mixed
   accepted state: the self-description is wrong exactly when mixing is on.

These cells are numpy-only; no WFN, no GPU, no jit.
"""
from __future__ import annotations

import ast
import os
from types import SimpleNamespace

import numpy as np
import pytest

from gw import sc_iteration
from gw.sc_iteration import _refuse_empty_map_output


_SRC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src")


def _func(name):
    tree = ast.parse(open(os.path.join(_SRC, "gw", "sc_iteration.py"),
                          encoding="utf-8").read())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _block(name):
    return ast.get_source_segment(
        open(os.path.join(_SRC, "gw", "sc_iteration.py"),
             encoding="utf-8").read(), _func(name))


# ---------------------------------------------------------------------------
# 1.  The all-zero / non-finite map output is REFUSED, not written
# ---------------------------------------------------------------------------

def test_a_healthy_map_output_is_accepted():
    """NOT-VOID control: the guard must pass on the ordinary case, or every
    refusal below is about the call rather than about the values."""
    rng = np.random.default_rng(20260822)
    _refuse_empty_map_output(rng.normal(size=(8, 12)) * 5.0,
                             call_index=0, role="one_shot")
    # A spectrum containing SOME exact zeros is fine -- only an identically
    # zero one is impossible.
    e = np.zeros((4, 6))
    e[2, 3] = -1.25
    _refuse_empty_map_output(e, call_index=3, role="linear")


def test_an_all_zero_map_output_is_refused():
    with pytest.raises(ValueError, match="identically ZERO"):
        _refuse_empty_map_output(np.zeros((29, 48)),
                                 call_index=0, role="one_shot")


def test_a_non_finite_map_output_is_refused():
    e = np.ones((3, 4))
    e[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _refuse_empty_map_output(e, call_index=2, role="trial")


def test_an_empty_map_output_is_refused():
    with pytest.raises(ValueError, match="EMPTY"):
        _refuse_empty_map_output(np.zeros((0, 4)),
                                 call_index=0, role="one_shot")


def test_the_zero_signature_is_exactly_what_was_measured():
    """The 24.68 eV header value was RMS|E_DFT|, and that is why it looked
    like a number rather than like an absence.  Reproduce the arithmetic so
    the failure signature stays legible to whoever meets it next."""
    rng = np.random.default_rng(0)
    e_dft = rng.normal(size=(29, 48)) * 20.0
    e_out = np.zeros_like(e_dft)
    rms = float(np.sqrt(np.mean((e_out - e_dft) ** 2)))
    assert rms == pytest.approx(float(np.sqrt(np.mean(e_dft ** 2))))
    with pytest.raises(ValueError):
        _refuse_empty_map_output(e_out, call_index=0, role="one_shot")


def test_the_guard_runs_before_the_rank_gate():
    """A rank-0-only refusal leaves P-1 peers in the next collective.

    The check is over a replicated host array, so it is bit-identical on
    every rank; placing it above ``process_rank()`` is what makes the
    refusal collective.
    """
    body = _block("_write_sc_eqp_snapshot")
    assert "_refuse_empty_map_output" in body
    assert body.index("_refuse_empty_map_output") < body.index(
        "if process_rank() != 0"), (
        "the map-output refusal sits below the rank gate, so only rank 0 "
        "would raise and the peers would hang")


# ---------------------------------------------------------------------------
# 2 + 3.  The stamps name which pair they measured
# ---------------------------------------------------------------------------

def test_the_snapshot_stamps_the_convergence_criterion():
    """The file must carry the pair the driver actually stops on: this
    call's output against this call's own input, over the non-scissored
    set, max-abs first."""
    body = _block("_write_sc_eqp_snapshot")
    for key in ("map_fixedpoint_max_abs_dE_protected_ev",
                "map_fixedpoint_RMS_dE_protected_ev",
                "verdict.max_abs_ev", "verdict.rms_protected_ev",
                "verdict.cutoff_ev", "verdict.converged"):
        assert key in body, f"the snapshot does not stamp {key}"
    assert "THIS is the convergence criterion" in body


def test_snapshot_writes_eqp1_but_never_uses_z_to_drive_the_map():
    body = _block("_write_sc_eqp_snapshot")
    assert "eqp1_iter" in body
    assert "e_eval + z_factor * (e_output - e_eval)" in body
    assert "Z is output-only" in body
    assert "pathological_z_factor_mask" not in body
    assert "z_factor_iter" not in body

    clear = _block("_clear_sc_eqp_snapshots")
    assert "eqp1" in clear and "z_factor" in clear

    gw_output = open(os.path.join(_SRC, "gw", "gw_output.py"),
                     encoding="utf-8").read()
    assert "guard_pathological_z=not results.self_consistent" in gw_output


def test_the_legacy_stamp_names_the_previous_calls_role():
    """``map_output_RMS_dE_prev_output`` is kept -- it is a real number and
    something parses it -- but it is measured against whatever call came
    before, and under rCROP that alternates trial / accepted.  Naming the
    role is what stops it being quoted as an accepted-iterate residual."""
    body = _block("_write_sc_eqp_snapshot")
    assert "prev call role=" in body
    assert "prev_output_role" in body


def test_every_call_site_supplies_both_new_stamps():
    """Three call sites; a defaulted kwarg would let one of them keep
    shipping an unstamped file, so the parameters carry no defaults and
    this cell checks the sites rather than the signature."""
    sig = _func("_write_sc_eqp_snapshot")
    kwonly = {a.arg for a in sig.args.kwonlyargs}
    assert {"verdict", "prev_output_role"} <= kwonly
    assert all(d is None for d in sig.args.kw_defaults), (
        "a default on verdict/prev_output_role would let a call site ship "
        "an unstamped snapshot silently")

    src = open(os.path.join(_SRC, "gw", "sc_iteration.py"),
               encoding="utf-8").read()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "_write_sc_eqp_snapshot"]
    assert len(calls) == 3, f"expected 3 snapshot writers, found {len(calls)}"
    for call in calls:
        names = {kw.arg for kw in call.keywords}
        assert {"verdict", "prev_output_role"} <= names, (
            f"snapshot call at line {call.lineno} does not stamp the "
            f"criterion")


def test_linear_mixing_stamps_from_the_map_output_history():
    """Under ``mixing != 1`` the column is the PRE-mix candidate, so its
    stamp must come from the candidate sequence and not from the mixed
    accepted one."""
    body = _block("_run_linear_mixing")
    assert "_out_history" in body, (
        "the linear path has no map-output history, so its snapshot stamp "
        "is computed from the mixed iterate that is NOT in the file")
    # The candidate, not E_new_ev (post-mix), feeds the stamp.
    assert "cand_rms" in body and "E_candidate_ev - _out_history[-1]" in body
    idx_write = body.index("_write_sc_eqp_snapshot")
    assert body.index("cand_rms = ") < idx_write
    assert "rms_ev=cand_rms" in body


def test_the_verdict_dataclass_still_carries_what_the_stamp_reads():
    """A required-kwarg / attribute rename on ConvergenceVerdict would break
    the stamp at runtime and nowhere else (the 2026-08-17 API-shape lesson),
    so the fields the writer reads are asserted here."""
    v = sc_iteration.protected_band_convergence(
        np.array([[1.0, 2.0]]), np.array([[1.0, 2.001]]),
        np.array([True, True]), np.array([True, True]), 5.0e-3)
    for field in ("max_abs_ev", "rms_protected_ev", "n_protected",
                  "n_total", "cutoff_ev", "converged"):
        assert hasattr(v, field), field


def test_sc_dump_persists_the_exact_map_rotation(tmp_path, monkeypatch):
    """The mechanism audit needs U for every map, not only final energies."""
    rotation = np.array(
        [[[0.0, 1.0j], [1.0, 0.0]]], dtype=np.complex128)
    inputs = SimpleNamespace(
        config=SimpleNamespace(sc=SimpleNamespace(dump_dir=str(tmp_path))),
        print_fn=lambda *_args, **_kwargs: None,
    )
    state = SimpleNamespace(outputs=SimpleNamespace(sigma_basis_U=rotation))
    monkeypatch.setattr(sc_iteration, "barrier", lambda *_a, **_k: None)

    path = sc_iteration._dump_sc_rotation(inputs, state, call_index=7)

    assert path == str(tmp_path / "rotation_iter0007.npy")
    np.testing.assert_array_equal(np.load(path), rotation)
