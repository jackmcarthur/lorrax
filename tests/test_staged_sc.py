"""Unit tests for staged self-consistency (2026-08-14).

Four things are pinned here, in the order they can go wrong:

1. **Stage parsing** — ``sc_stage_N_type`` / ``_cutoff`` / ``_max_iter``
   are strings + floats + ints on the deck, ``none`` is a hole rather
   than a terminator, and an unknown type is refused BY NAME rather than
   resolved to a neighbouring mode.
2. **The refusal path** — ``self_energy_eval_type = linearized`` under
   ``qp_solver = self_consistent`` refuses.  It must NOT be silently
   coerced to ``hermitianized``: reporting at-DFT Newton numbers under a
   self-consistent label is the defect the refusal exists to prevent.
3. **Default ladder selection** — self-consistency alone runs one stage
   of the deck's own mode to 2 meV; ``compute_mode = mpa`` runs GN-PPM
   to 5 meV and then MPA to 2 meV.
4. **Stage plumbing** — a stage must apply its mode to BOTH
   ``SCInputs.compute_mode`` and ``config.compute_mode``, and the driver
   must refuse to report convergence its own predicate never measured.
   The PREDICATE itself is tested in
   ``tests/test_sc_convergence_predicate.py``, beside the fix it gates.

Everything here runs on a throwaway input file or a numpy array — no
WFN, no GPU, no jit.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from gw.gw_config import (
    ComputeMode, LorraxConfig, QPSolver, SelfEnergyEvalType,
    SC_DEFAULT_CUTOFF_EV, SC_DEFAULT_HANDOFF_CUTOFF_EV,
    default_sc_ladder, resolve_self_energy_eval_type,
)


BASE_INPUT = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""

SC = "qp_solver = self_consistent\n"


def _config(tmp_path, extra: str = "", name: str = "staged_sc.in"):
    """Build a config from ``BASE_INPUT + extra``, mode included.

    ``compute_mode`` is REQUIRED for 0.1.0 — no default and no ``auto`` —
    so every deck here has to state one.  The cells that care which mode
    it is (the default-ladder rows, the two-reader plumbing row) write
    their own ``compute_mode =`` line into ``extra``, and this default
    steps aside for them rather than emitting a duplicate key.
    """
    body = BASE_INPUT + extra
    if not re.search(r"(?m)^\s*compute_mode\s*=", body):
        body += "compute_mode = cohsex\n"
    path = tmp_path / name
    path.write_text(body)
    return LorraxConfig.from_input_file(str(path), print_fn=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# 1. stage parsing
# ---------------------------------------------------------------------------

def test_stage_types_are_strings_not_integers(tmp_path):
    """The owner's spelling: ``gnppm``/``mpa``, matching ``compute_mode``."""
    cfg = _config(tmp_path, SC + (
        "sc_stage_1_type = gnppm\n"
        "sc_stage_1_cutoff = 5.0e-3\n"
        "sc_stage_2_type = mpa\n"
        "sc_stage_2_cutoff = 2.0e-3\n"
        "sc_stage_3_type = none\n"))
    stages = cfg.sc.stages
    assert [s.mode for s in stages] == [ComputeMode.GN_PPM, ComputeMode.MPA]
    assert [s.cutoff_ev for s in stages] == [5.0e-3, 2.0e-3]


def test_none_is_a_hole_not_a_terminator(tmp_path):
    """``1=gnppm, 2=none, 3=mpa`` keeps the mpa stage.

    A terminator reading would silently drop stage 3 and run half the
    ladder the deck asked for, which is the kind of quiet omission that
    only shows up as a wrong number much later.
    """
    cfg = _config(tmp_path, SC + (
        "sc_stage_1_type = gnppm\n"
        "sc_stage_2_type = none\n"
        "sc_stage_3_type = mpa\n"))
    assert [s.mode for s in cfg.sc.stages] == [
        ComputeMode.GN_PPM, ComputeMode.MPA]


def test_unknown_stage_type_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="sc_stage_1_type"):
        _config(tmp_path, SC + "sc_stage_1_type = gn_ppm_v2\n")


def test_all_none_stages_refuse(tmp_path):
    with pytest.raises(ValueError, match="no scheme to iterate"):
        _config(tmp_path, SC + (
            "sc_stage_1_type = none\n"
            "sc_stage_2_type = none\n"
            "sc_stage_3_type = none\n"))


def test_stated_stage_without_cutoff_takes_the_5mev_default(tmp_path):
    cfg = _config(tmp_path, SC + "sc_stage_1_type = cohsex\n")
    assert cfg.sc.stages[0].cutoff_ev == SC_DEFAULT_HANDOFF_CUTOFF_EV


def test_per_stage_max_iter_defaults_to_sc_max_iter(tmp_path):
    """``sc_max_iter`` is a PER-STAGE ceiling now, not a global budget.

    This is the point of the knob: a global budget lets stage 1 consume
    the whole allowance and stage 2 never run.
    """
    cfg = _config(tmp_path, SC + (
        "sc_max_iter = 4\n"
        "sc_stage_1_type = gnppm\n"
        "sc_stage_2_type = mpa\n"
        "sc_stage_2_max_iter = 2\n"))
    assert [s.max_iter for s in cfg.sc.stages] == [4, 2]


# ---------------------------------------------------------------------------
# 2. the refusal path
# ---------------------------------------------------------------------------

def test_self_consistency_with_linearized_refuses(tmp_path):
    with pytest.raises(ValueError, match="incompatible"):
        _config(tmp_path, SC + "self_energy_eval_type = linearized\n")


def test_the_refusal_is_not_a_silent_coercion():
    """The bad pairing raises; it does not quietly return hermitianized."""
    with pytest.raises(ValueError):
        resolve_self_energy_eval_type(
            "linearized", QPSolver.SELF_CONSISTENT)


def test_self_consistency_defaults_to_hermitianized(tmp_path):
    """Unset resolves from qp_solver, so no existing deck changes meaning."""
    assert _config(tmp_path, SC).sc.eval_type is (
        SelfEnergyEvalType.HERMITIANIZED)


def test_one_shot_defaults_to_linearized(tmp_path):
    assert _config(tmp_path).sc.eval_type is SelfEnergyEvalType.LINEARIZED


def test_explicit_hermitianized_under_self_consistency_is_accepted(tmp_path):
    cfg = _config(tmp_path, SC + "self_energy_eval_type = hermitianized\n")
    assert cfg.sc.eval_type is SelfEnergyEvalType.HERMITIANIZED


def test_linearized_is_fine_without_self_consistency(tmp_path):
    cfg = _config(tmp_path, "self_energy_eval_type = linearized\n")
    assert cfg.sc.eval_type is SelfEnergyEvalType.LINEARIZED


def test_unknown_eval_type_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="self_energy_eval_type"):
        _config(tmp_path, "self_energy_eval_type = quasiparticle\n")


# ---------------------------------------------------------------------------
# 3. default ladder selection
# ---------------------------------------------------------------------------

def test_default_ladder_for_mpa_is_two_stages():
    ladder = default_sc_ladder(ComputeMode.MPA)
    assert ladder == ((ComputeMode.GN_PPM, SC_DEFAULT_HANDOFF_CUTOFF_EV),
                      (ComputeMode.MPA, SC_DEFAULT_CUTOFF_EV))


def test_default_ladder_for_gnppm_is_one_stage_at_2mev():
    assert default_sc_ladder(ComputeMode.GN_PPM) == (
        (ComputeMode.GN_PPM, SC_DEFAULT_CUTOFF_EV),)


def test_default_ladder_does_not_substitute_the_decks_mode():
    """A COHSEX deck runs COHSEX, not a silently-substituted GN-PPM.

    The owner's table says "self-consistency enabled -> GN-PPM to 2 meV",
    which is what the GN-PPM deck gets.  Hard-wiring GN-PPM for EVERY
    non-MPA deck would make ``compute_mode = cohsex`` run a different
    self-energy than the one it named -- the ``screening_method = ctsp``
    defect class.
    """
    assert default_sc_ladder(ComputeMode.COHSEX) == (
        (ComputeMode.COHSEX, SC_DEFAULT_CUTOFF_EV),)


def test_deck_with_mpa_and_no_stage_keys_gets_the_two_stage_ladder(tmp_path):
    cfg = _config(tmp_path, SC + "compute_mode = mpa\n")
    assert [s.mode for s in cfg.sc.stages] == [
        ComputeMode.GN_PPM, ComputeMode.MPA]
    assert [s.cutoff_ev for s in cfg.sc.stages] == [
        SC_DEFAULT_HANDOFF_CUTOFF_EV, SC_DEFAULT_CUTOFF_EV]


def test_explicit_stages_override_the_default_ladder(tmp_path):
    """An MPA deck that states one cohsex stage gets exactly that."""
    cfg = _config(tmp_path, SC + (
        "compute_mode = mpa\nsc_stage_1_type = cohsex\n"))
    assert [s.mode for s in cfg.sc.stages] == [ComputeMode.COHSEX]


# ---------------------------------------------------------------------------
# 4. the convergence predicate
# ---------------------------------------------------------------------------
# Lives in tests/test_sc_convergence_predicate.py, beside the fix it
# gates (fix/rcrop-convergence-2026-08-15).  The predicate is not a
# staging feature -- staging only chooses WHICH cutoff to hand it --
# so testing it here as well would pin the same behaviour in two
# places and make the standalone commit look optional.


# ---------------------------------------------------------------------------
# 5. a stage must apply its mode to BOTH readers
# ---------------------------------------------------------------------------

def test_stage_overrides_the_mode_on_inputs_AND_on_config(tmp_path, monkeypatch):
    """A stage's mode must reach the kernels, not just the dispatch call.

    THE REGRESSION THIS EXISTS FOR.  ``compute_sigma_xc`` is handed the
    stage's mode explicitly, but the kernels underneath it re-read
    ``config.compute_mode`` for themselves -- ``ppm_pipeline`` does, and
    refuses a non-plasmon-pole mode.  Setting only ``SCInputs.compute_mode``
    therefore half-applies the stage, and it fails in exactly the
    configuration staging exists for: MEASURED on a gn_ppm -> mpa ladder,
    stage 1 dispatched to the PPM pipeline and then died inside it with
    "compute_mode = mpa is not a plasmon-pole model".

    A SINGLE-SCHEME ladder cannot catch this -- there the deck's mode and
    the stage's mode agree and both readers are silently consistent -- so
    this cell asserts on a ladder whose stage mode DIFFERS from the deck's.
    """
    from gw import sc_iteration

    cfg = _config(tmp_path, SC + (
        "compute_mode = mpa\n"
        "sc_stage_1_type = gnppm\nsc_stage_2_type = mpa\n"))
    assert cfg.compute_mode is ComputeMode.MPA, "deck mode is the mpa arm"

    seen = []

    def _fake_run_self_consistency(state, inputs, **kw):
        seen.append((inputs.compute_mode, inputs.config.compute_mode))
        return state, []


    monkeypatch.setattr(
        sc_iteration, "run_self_consistency", _fake_run_self_consistency)
    from gw.band_partition import BandPartition
    ones = np.ones(4, dtype=bool)
    inputs = sc_iteration.SCInputs(
        wfns_dft=None, V_q=None, kin_ion_dft=None, head_channel=None,
        quad=None, e_ref=0.0, static_head_terms=None, head_resolver=None,
        config=cfg, meta=None, mesh_xy=None, sym=None, wfn=None,
        centroid_indices=None, band_slices=None, input_dir=str(tmp_path),
        partition=BandPartition(protected_mask=ones, in_range_mask=ones),
        e_dft_active_kn_ry=None,
        valence_mask_active_kn=None, print_fn=lambda *a, **k: None,
    )
    sc_iteration.run_staged_self_consistency(
        sc_iteration.SCState(H_qp_dft=None, iteration=0), inputs,
        stages=cfg.sc.stages)

    assert seen == [
        (ComputeMode.GN_PPM, ComputeMode.GN_PPM),
        (ComputeMode.MPA, ComputeMode.MPA),
    ], (
        "each stage must present ITS mode on both SCInputs.compute_mode "
        f"and config.compute_mode; got {seen}")


def test_empty_stage_ladder_refuses():
    from gw import sc_iteration
    with pytest.raises(ValueError, match="ladder is empty"):
        sc_iteration.run_staged_self_consistency(
            sc_iteration.SCState(H_qp_dft=None, iteration=0),
            object.__new__(sc_iteration.SCInputs), stages=())


# ---------------------------------------------------------------------------
# 6. sc_accelerator retirement
# ---------------------------------------------------------------------------

def test_sc_accelerator_linear_is_refused(tmp_path):
    """``linear`` named a different iteration; running rCROP under that
    name would be a mode substitution."""
    with pytest.raises(ValueError, match="sc_accelerator = linear"):
        _config(tmp_path, "sc_accelerator = linear\n")


def test_sc_accelerator_rcrop_is_retired_and_ignored(tmp_path):
    with pytest.deprecated_call(match="sc_accelerator"):
        cfg = _config(tmp_path, "sc_accelerator = rcrop\n")
    assert not hasattr(cfg.sc, "accelerator")


# ---------------------------------------------------------------------------
# 7. the convergence POSTCONDITION on the real run path
# ---------------------------------------------------------------------------

def _stub_inputs(cfg, tmp_path, nb=4):
    from gw import sc_iteration
    from gw.band_partition import BandPartition
    ones = np.ones(nb, dtype=bool)
    return sc_iteration.SCInputs(
        wfns_dft=None, V_q=None, kin_ion_dft=None, head_channel=None,
        quad=None, e_ref=0.0, static_head_terms=None, head_resolver=None,
        config=cfg, meta=None, mesh_xy=None, sym=None, wfn=None,
        centroid_indices=None, band_slices=None, input_dir=str(tmp_path),
        partition=BandPartition(protected_mask=ones, in_range_mask=ones),
        e_dft_active_kn_ry=None, valence_mask_active_kn=None,
        print_fn=lambda *a, **k: None,
    )


def test_reporting_converged_without_a_verdict_is_refused(tmp_path, monkeypatch):
    """THE CELL FOR THE DEFECT THAT ACTUALLY HAPPENED.

    The rCROP bug was not a wrong predicate -- the predicate was never
    consulted.  rcrop_nojit declared convergence on its own L2 residual,
    the driver relayed it, and nothing connected the word "converged" to
    a measured max|dE|.  A unit test on the predicate cannot catch that.
    This asserts the DRIVER refuses to report convergence it cannot
    substantiate.
    """
    from gw import sc_iteration

    cfg = _config(tmp_path, SC + "compute_mode = gn_ppm\n")

    # A stage that "succeeds" but never fills verdict_out -- i.e. some
    # other stopping rule decided, exactly as rCROP used to.
    def _silent_success(state, inputs, **kw):
        return state, [1.0]

    monkeypatch.setattr(
        sc_iteration, "run_self_consistency", _silent_success)
    monkeypatch.setattr(
        sc_iteration, "_kshard_eigh_kernels",
        lambda mesh, *a, **k: (None, lambda H: np.zeros((2, 4))))

    with pytest.raises(AssertionError, match="NO verdict"):
        sc_iteration.run_staged_self_consistency(
            sc_iteration.SCState(H_qp_dft=None, iteration=0),
            _stub_inputs(cfg, tmp_path), stages=cfg.sc.stages)


def test_converged_flag_must_agree_with_its_own_verdict(tmp_path, monkeypatch):
    """A verdict that does not clear the cutoff cannot be reported converged."""
    from gw import sc_iteration

    cfg = _config(tmp_path, SC + "compute_mode = gn_ppm\n")
    cutoff = cfg.sc.stages[0].cutoff_ev

    def _lying_success(state, inputs, **kw):
        v = kw.get("verdict_out")
        if v is not None:
            # max_abs 10x the cutoff, but the record claims converged.
            v.append(sc_iteration.ConvergenceVerdict(
                converged=True, max_abs_ev=10.0 * cutoff,
                rms_protected_ev=0.0, rms_all_ev=0.0, n_protected=4,
                n_total=4, worst_k=0, worst_band=0, cutoff_ev=cutoff))
        return state, [1.0]

    monkeypatch.setattr(sc_iteration, "run_self_consistency", _lying_success)
    monkeypatch.setattr(
        sc_iteration, "_kshard_eigh_kernels",
        lambda mesh, *a, **k: (None, lambda H: np.zeros((2, 4))))

    with pytest.raises(AssertionError, match="does not support it"):
        sc_iteration.run_staged_self_consistency(
            sc_iteration.SCState(H_qp_dft=None, iteration=0),
            _stub_inputs(cfg, tmp_path), stages=cfg.sc.stages)
