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
4. **The convergence predicate** — max-abs over the NON-SCISSORED bands,
   i.e. ``protected_mask | in_range_mask``.  Two load-bearing cells:
   ``test_rms_passes_where_max_abs_correctly_fails`` (a synthetic case
   built so the RMS sits under the cutoff while one band is far above it
   — different tests, and RMS is the looser one), and
   ``test_non_protected_in_range_bands_still_block_convergence`` (the
   test set is "not scissored", not "protected"; those coincide only
   because ``run_sc_driver`` happens to build both masks equal).

Everything here runs on a throwaway input file or a numpy array — no
WFN, no GPU, no jit.
"""
from __future__ import annotations

import numpy as np
import pytest

from gw.gw_config import (
    ComputeMode, LorraxConfig, QPSolver, SelfEnergyEvalType,
    SC_DEFAULT_CUTOFF_EV, SC_DEFAULT_HANDOFF_CUTOFF_EV,
    default_sc_ladder, resolve_self_energy_eval_type,
)
from gw.sc_iteration import protected_band_convergence


BASE_INPUT = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""

SC = "qp_solver = self_consistent\n"


def _config(tmp_path, extra: str = "", name: str = "staged_sc.in"):
    path = tmp_path / name
    path.write_text(BASE_INPUT + extra)
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

def _energies(nk=3, nb=6):
    rng = np.random.default_rng(0)
    return rng.normal(size=(nk, nb)) * 5.0


def test_max_abs_over_protected_bands_is_the_criterion():
    e_prev = _energies()
    e_new = e_prev.copy()
    protected = np.array([True] * 3 + [False] * 3)
    e_new[0, 0] += 4.0e-3            # protected, under a 5 meV cutoff
    v = protected_band_convergence(e_new, e_prev, protected, protected, 5.0e-3)
    assert v.converged
    assert v.max_abs_ev == pytest.approx(4.0e-3)
    assert v.worst_k == 0 and v.worst_band == 0


def test_scissored_bands_are_excluded_from_the_test():
    """A huge move on a NON-protected band must not block convergence.

    Scissored bands take a refitted alpha*E+beta law rather than Sigma,
    so their motion measures the scissor fit, not the fixed point.
    """
    e_prev = _energies()
    e_new = e_prev.copy()
    protected = np.array([True] * 3 + [False] * 3)
    e_new[1, 4] += 10.0              # non-protected: 10 eV, ignored
    v = protected_band_convergence(e_new, e_prev, protected, protected, 5.0e-3)
    assert v.converged
    assert v.max_abs_ev == 0.0
    # ...but the all-band RMS still SEES it, which is why that number is
    # reported separately and is not the criterion.
    assert v.rms_all_ev > 1.0


def test_rms_passes_where_max_abs_correctly_fails():
    """THE cell this change exists for.

    One protected band moves 40 meV; the other 59 protected entries do
    not move at all.  Against a 5 meV cutoff:

      * RMS over the protected set = 40 meV / sqrt(60) = 5.16 meV ...
        which is the same order as the cutoff and, with one more clean
        band, would slip under it;
      * max-abs = 40 meV, which is 8x the cutoff and correctly fails.

    The construction below is chosen so the RMS is UNDER the cutoff
    while max-abs is 8x over it, i.e. the two tests give opposite
    answers on the same data.
    """
    nk, nb = 10, 20
    protected = np.ones(nb, dtype=bool)
    e_prev = np.zeros((nk, nb))
    e_new = np.zeros((nk, nb))
    e_new[0, 0] = 40.0e-3            # one band, 40 meV -- 8x the cutoff

    cutoff = 5.0e-3
    v = protected_band_convergence(e_new, e_prev, protected, protected, cutoff)

    # The RMS would PASS: 40 meV spread over 200 protected entries.
    assert v.rms_protected_ev < cutoff, (
        "the synthetic case must be one where RMS passes")
    assert v.rms_protected_ev == pytest.approx(
        40.0e-3 / np.sqrt(nk * nb), rel=1e-12)
    # The max-abs correctly FAILS.
    assert v.max_abs_ev == pytest.approx(40.0e-3)
    assert not v.converged, (
        "max-abs must fail where a single protected band is 8x the cutoff")


def test_non_protected_in_range_bands_still_block_convergence():
    """The test set is NOT ``protected_mask`` alone -- it is "not scissored".

    The partition is THREE-way and only the third category is scissored:
    ``apply_band_partition`` substitutes alpha*E_DFT+beta exactly where
    ``in_range_mask`` is False.  A band that is in range but NOT protected
    keeps its own Sigma-derived diagonal and merely loses its off-diagonal
    mixing, so it is a genuine independent degree of freedom.

    Here bands 0-1 are protected, bands 2-3 are in range but NOT protected,
    and bands 4-5 are scissored.  The ONLY band still moving is band 3 --
    in range, not protected.  Testing ``protected_mask`` alone would
    silently declare convergence; the union correctly refuses.

    Today the production partition sets both masks equal
    (``run_sc_driver``), so this configuration does not arise in a current
    run -- this pins the PREDICATE rather than reporting a live defect.
    """
    protected = np.array([True, True, False, False, False, False])
    in_range = np.array([True, True, True, True, False, False])

    e_prev = np.zeros((4, 6))
    e_new = e_prev.copy()
    e_new[2, 3] = 40.0e-3            # in range, NOT protected, 8x cutoff

    v = protected_band_convergence(e_new, e_prev, protected, in_range, 5.0e-3)
    assert v.n_protected == 4, "test set is protected | in_range"
    assert v.max_abs_ev == pytest.approx(40.0e-3)
    assert v.worst_band == 3
    assert not v.converged, (
        "a non-protected IN-RANGE band carries a Sigma-derived diagonal and "
        "must block convergence; testing protected_mask alone would miss it")

    # Control: the same motion on a SCISSORED band is correctly ignored,
    # because its energy is alpha*E_DFT+beta and moves only when the fit does.
    e_scissored = e_prev.copy()
    e_scissored[2, 5] = 40.0e-3
    v2 = protected_band_convergence(
        e_scissored, e_prev, protected, in_range, 5.0e-3)
    assert v2.converged and v2.max_abs_ev == 0.0


def test_verdict_summary_labels_which_number_is_the_criterion():
    """The log line must not let RMS be mistaken for the test.

    Adjacent to KNOWN_LORRAX_ISSUES' snapshot-RMS-mislabel row: a number
    printed beside a cutoff reads as the thing being compared to it.
    """
    protected = np.ones(4, dtype=bool)
    v = protected_band_convergence(
        np.zeros((2, 4)), np.zeros((2, 4)), protected, protected, 5.0e-3)
    text = v.summary()
    assert "CRITERION" in text
    assert "NOT the criterion" in text


def test_protected_mask_length_must_match_the_active_window():
    """A frozen band window against moved energies is refused, not padded."""
    with pytest.raises(ValueError, match="protected_mask"):
        protected_band_convergence(
            np.zeros((2, 6)), np.zeros((2, 6)),
            np.ones(4, dtype=bool), np.ones(4, dtype=bool), 5.0e-3)


def test_zero_protected_bands_refuses_rather_than_declaring_victory():
    """``max`` over the empty set is vacuously true -- refuse instead."""
    with pytest.raises(ValueError, match="ZERO"):
        protected_band_convergence(
            np.zeros((2, 6)), np.zeros((2, 6)),
            np.zeros(6, dtype=bool), np.zeros(6, dtype=bool), 5.0e-3)


def test_mismatched_energy_shapes_refuse():
    with pytest.raises(ValueError, match="shapes disagree"):
        protected_band_convergence(
            np.zeros((2, 6)), np.zeros((3, 6)),
            np.ones(6, dtype=bool), np.ones(6, dtype=bool), 5.0e-3)


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
