"""Unit tests for the ``qp_solver`` config axis and the ``SCConfig`` group.

Covers the auto-resolution contract of ``LorraxConfig.qp_solver``
(G0W0_SC_TOGGLE_DESIGN.md §2a):

1. Default → ``ONE_SHOT_DFT`` (one-shot full-matrix effective H at
   E_DFT, distinct from fixed-state diagonal G0W0).
2. Legacy ``self_consistent = true`` → ``SELF_CONSISTENT`` (deprecated,
   honored).
3. Legacy ``sigma_at_dft_energies = true`` → ``ONE_SHOT_DFT`` (the orphan
   flag's intended meaning IS the default; deprecated, honored).
4. Explicit ``qp_solver`` overrides the legacy flags.
5. Validation: ``fixed_point`` × static mode → error; the removed
   ``sigma_omega_accumulation = kij_stream`` spelling → parse-time error.
6. ``SCConfig``: sc_* input keys parsed; ``LORRAX_SC_*`` envs honored as
   deprecated overrides; knob validation.

All tests run on a throwaway input file — no WFN, no GPU, no jit.
"""
from __future__ import annotations

import pathlib
import pytest

from gw.gw_config import (
    BispinorGWMode, LorraxConfig, QPSolver,
    qp_solver_semantics,
    uses_four_spinor_finite_q_charge,
)
from common.four_current_model import resolve_four_current_representation


BASE_INPUT = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""


def _config(tmp_path, extra: str = "", name: str = "cohsex_qp.in"):
    path = tmp_path / name
    path.write_text(BASE_INPUT + extra)
    return LorraxConfig.from_input_file(str(path), print_fn=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# auto-resolution
# ---------------------------------------------------------------------------

def test_default_is_one_shot_dft(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.qp_solver is QPSolver.ONE_SHOT_DFT


def test_one_shot_semantics_name_the_full_matrix_not_textbook_g0w0():
    semantics = qp_solver_semantics(QPSolver.ONE_SHOT_DFT)
    assert "one-shot full-matrix effective Hamiltonian" in semantics.description
    assert "fixed-DFT-state diagonal G0W0" in semantics.description
    assert "textbook" not in semantics.description
    assert semantics.energy_definition == (
        "full_matrix_effective_hamiltonian_sigma_at_e_dft")
    assert semantics.sigma_evaluation_provenance == "at_e_dft"


def test_legacy_self_consistent_resolves_and_warns(tmp_path):
    with pytest.warns(DeprecationWarning, match="self_consistent"):
        cfg = _config(tmp_path, "self_consistent = true\n")
    assert cfg.qp_solver is QPSolver.SELF_CONSISTENT


def test_legacy_sigma_at_dft_energies_resolves_and_warns(tmp_path):
    with pytest.warns(DeprecationWarning, match="sigma_at_dft_energies"):
        cfg = _config(tmp_path, "sigma_at_dft_energies = true\n")
    assert cfg.qp_solver is QPSolver.ONE_SHOT_DFT


def test_explicit_overrides_legacy(tmp_path):
    with pytest.warns(DeprecationWarning):
        cfg = _config(
            tmp_path, "self_consistent = true\nqp_solver = one_shot_dft\n")
    assert cfg.qp_solver is QPSolver.ONE_SHOT_DFT


def test_explicit_fixed_point_dynamic(tmp_path):
    cfg = _config(
        tmp_path, "compute_mode = gn_ppm\nqp_solver = fixed_point\n")
    assert cfg.qp_solver is QPSolver.FIXED_POINT


def test_unknown_value_raises(tmp_path):
    with pytest.raises(ValueError, match="qp_solver"):
        _config(tmp_path, "qp_solver = bogus\n")


# ---------------------------------------------------------------------------
# validation of inconsistent axis combinations
# ---------------------------------------------------------------------------

def test_fixed_point_static_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="fixed_point"):
        _config(
            tmp_path, "compute_mode = cohsex\nqp_solver = fixed_point\n")


def test_kij_stream_refused_at_parse(tmp_path):
    # kij_stream was REMOVED (2026-07-31).  A removed VALUE of a known
    # key must refuse at parse with the removal named — never silently
    # reroute an old deck (announce-or-refuse).
    with pytest.raises(ValueError, match="REMOVED"):
        _config(
            tmp_path,
            "compute_mode = gn_ppm\n"
            "sigma_omega_accumulation = kij_stream\n")


def test_kij_stream_refused_even_in_static_modes(tmp_path):
    # The removed value refuses regardless of compute_mode — a removed
    # spelling must not survive in decks just because the knob would
    # not have been read.
    with pytest.raises(ValueError, match="REMOVED"):
        _config(
            tmp_path,
            "compute_mode = cohsex\nqp_solver = self_consistent\n"
            "sigma_omega_accumulation = kij_stream\n")


# ---------------------------------------------------------------------------
# bispinor_gw: TWO values, and the retired spellings refuse BY NAME
# (owner ruling 2026-09-01; lane J architecture review section 2)
# ---------------------------------------------------------------------------

def test_bispinor_gw_grammar_is_exactly_two_values():
    assert [m.value for m in BispinorGWMode] == [
        "bare_transverse", "full_static_cohsex"]


@pytest.mark.parametrize("spelling,gate,replacement", [
    ("charge_hall_cubature",
     "bispinor_gw_charge_hall_cubature_retired", "full_static_cohsex"),
    ("pauli_reference_bare_transverse",
     "bispinor_gw_pauli_reference_retired", "bare_transverse"),
    ("isometric_kinetic_balance_bare_transverse",
     "bispinor_gw_isometric_kinetic_balance_retired", "bare_transverse"),
])
def test_retired_bispinor_gw_spellings_refuse_and_name_the_replacement(
        tmp_path, spelling, gate, replacement):
    """A retired MODE VALUE stops the run and says what replaced it.

    Never an alias: the mode value is the one deck word that decides which
    physics runs, so a stale deck must not be silently re-pointed at a
    different calculation.
    """
    with pytest.raises(ValueError) as exc:
        _config(tmp_path, f"bispinor = true\nbispinor_gw = {spelling}\n")
    text = str(exc.value)
    assert gate in text
    assert f"bispinor_gw = {replacement}" in text
    assert "docs/input_reference.md" in text


def test_both_shipped_modes_ride_the_one_raw_carrier():
    """``bispinor_gw`` picks which Lorentz blocks are screened, not a lift."""
    reps = [resolve_four_current_representation(True, mode)
            for mode in ("bare_transverse", "full_static_cohsex")]
    assert reps[0] == reps[1]
    assert reps[0].charge_lift == reps[0].current_lift == "raw"
    assert reps[0].scalar_head_bispinor
    assert uses_four_spinor_finite_q_charge(True, "full_static_cohsex")


def test_default_bispinor_self_consistency_uses_live_four_current(tmp_path):
    lines: list[str] = []
    path = tmp_path / "cohsex_bispinor_sc.in"
    path.write_text(
        BASE_INPUT
        + "bispinor = true\n"
        + "qp_solver = self_consistent\n")
    cfg = LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert cfg.qp_solver is QPSolver.SELF_CONSISTENT
    assert cfg.density_self_consistent
    assert "density_self_consistent" not in cfg.raw_input_keys
    assert any("required live (rho, J) Hartree rebuild" in line
               for line in lines)


def test_bispinor_explicit_frozen_density_is_refused(tmp_path):
    with pytest.raises(
            ValueError,
            match="bispinor_self_consistency_requires_live_four_current"):
        _config(
            tmp_path,
            "bispinor = true\n"
            "bispinor_gw = bare_transverse\n"
            "qp_solver = self_consistent\n"
            "density_self_consistent = false\n")


def test_bispinor_accepts_live_four_current_self_consistency(tmp_path):
    cfg = _config(
        tmp_path,
        "bispinor = true\n"
        "bispinor_gw = bare_transverse\n"
        "qp_solver = self_consistent\n"
        "density_self_consistent = true\n")
    assert cfg.qp_solver is QPSolver.SELF_CONSISTENT
    assert cfg.density_self_consistent


def test_scalar_self_consistency_keeps_fixed_density_default(tmp_path):
    cfg = _config(tmp_path, "qp_solver = self_consistent\n")
    assert cfg.qp_solver is QPSolver.SELF_CONSISTENT
    assert not cfg.density_self_consistent


def test_legacy_bispinor_self_consistency_gets_same_live_default(tmp_path):
    with pytest.warns(DeprecationWarning, match="self_consistent"):
        cfg = _config(
            tmp_path,
            "bispinor = true\n"
            "bispinor_gw = bare_transverse\n"
            "self_consistent = true\n")
    assert cfg.qp_solver is QPSolver.SELF_CONSISTENT
    assert cfg.density_self_consistent


def test_packed_mode_requires_four_current_enablement(tmp_path):
    with pytest.raises(ValueError, match="bispinor_gw_requires_bispinor"):
        _config(
            tmp_path,
            "bispinor = false\nbispinor_gw = full_static_cohsex\n")


def test_bispinor_refuses_no_local_fields_head(tmp_path):
    """PHYSICS/POLICY row 20: two head values on a bispinor deck.

    ``no_local_fields`` is a scalar diagnostic; on the bare-transverse
    route it also silently moved the deck off the packed path, which is a
    head dial choosing a route.
    """
    with pytest.raises(
            ValueError,
            match="bispinor_head_correction_no_local_fields_unavailable"):
        _config(
            tmp_path,
            "bispinor = true\nhead_correction = no_local_fields\n")


def test_scalar_deck_keeps_no_local_fields(tmp_path):
    cfg = _config(tmp_path, "head_correction = no_local_fields\n")
    assert cfg.head.correction.value == "no_local_fields"


def test_bispinor_tt_head_correction_key_is_tombstoned(tmp_path):
    """The deleted key refuses BY NAME, at either value."""
    for value in ("true", "false"):
        with pytest.raises(ValueError) as exc:
            _config(
                tmp_path,
                f"bispinor = true\nbispinor_tt_head_correction = {value}\n")
        text = str(exc.value)
        assert "'bispinor_tt_head_correction' has been REMOVED" in text
        assert "Gamma-cell" in text


def test_static_gauge_hall_file_is_unset_by_default(tmp_path):
    """An unnamed optional artifact must stay EMPTY, not become the deck dir.

    ``resolve_input_paths`` used to join "" onto the input directory, and
    ``os.path.exists`` reports a directory as present -- an unset Hall file
    would then have looked like a Hall artifact to the completion.
    """
    assert _config(tmp_path).paths.static_gauge_hall_file == ""


# ---------------------------------------------------------------------------
# SCConfig
# ---------------------------------------------------------------------------

def test_sc_defaults(tmp_path):
    sc = _config(tmp_path).sc
    assert (sc.max_iter, sc.tol_ev, sc.accelerator, sc.history_depth,
            sc.mixing, sc.dump_dir, sc.exact_degeneracy_tol_ev,
            sc.tail_fit, sc.buffer_nbands, sc.buffer_mode) == (
                20, 1.0e-4, "rcrop", 5, 1.0, None, 1.0e-4, "frontier",
                0, "diagonal")


def test_sc_input_keys(tmp_path):
    sc = _config(
        tmp_path,
        "sc_max_iter = 7\nsc_tol_ev = 1e-6\nsc_accelerator = linear\n"
        "sc_history_depth = 3\nsc_mixing = 0.5\nsc_dump_dir = sc_hist\n"
        "sc_exact_degeneracy_tol_ev = 5e-5\n"
        "sc_tail_fit = buffer_edges\nsc_buffer_nbands = 3\n"
        "sc_buffer_mode = one_sided\n").sc
    assert (sc.max_iter, sc.tol_ev, sc.accelerator, sc.history_depth,
            sc.mixing, sc.dump_dir, sc.exact_degeneracy_tol_ev,
            sc.tail_fit, sc.buffer_nbands, sc.buffer_mode) == (
                7, 1.0e-6, "linear", 3, 0.5, "sc_hist", 5.0e-5,
                "buffer_edges", 3, "one_sided")


def test_sc_exact_degeneracy_tolerance_refuses_mev_pair_averaging(tmp_path):
    with pytest.raises(ValueError, match="sc_exact_degeneracy_tol_ev"):
        _config(tmp_path, "sc_exact_degeneracy_tol_ev = 0.0017\n")


def test_sc_tail_fit_refuses_unknown_policy(tmp_path):
    with pytest.raises(ValueError, match="sc_tail_fit"):
        _config(tmp_path, "sc_tail_fit = nearest\n")


def test_sc_buffer_controls_refuse_invalid_values(tmp_path):
    with pytest.raises(ValueError, match="sc_buffer_nbands"):
        _config(tmp_path, "sc_buffer_nbands = -1\n")
    with pytest.raises(ValueError, match="sc_buffer_mode"):
        _config(tmp_path, "sc_buffer_mode = full\n")
    with pytest.raises(ValueError, match="requires sc_buffer_nbands"):
        _config(tmp_path, "sc_tail_fit = buffer_edges\n")


def test_sc_env_overrides_deprecated(tmp_path, monkeypatch):
    monkeypatch.setenv("LORRAX_SC_MAX_ITER", "3")
    monkeypatch.setenv("LORRAX_SC_TOL_EV", "1e-10")
    monkeypatch.setenv("LORRAX_SC_ACCEL", "linear")
    monkeypatch.setenv("LORRAX_SC_DEPTH", "2")
    monkeypatch.setenv("LORRAX_SC_MIXING", "0.25")
    monkeypatch.setenv("LORRAX_SC_DUMP_DIR", "/tmp/sc_dump")
    lines: list[str] = []
    path = tmp_path / "cohsex_env.in"
    path.write_text(BASE_INPUT + "sc_max_iter = 99\n")
    sc = LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: lines.append(" ".join(map(str, a)))
    ).sc
    assert (sc.max_iter, sc.tol_ev, sc.accelerator, sc.history_depth,
            sc.mixing, sc.dump_dir) == (3, 1.0e-10, "linear", 2, 0.25,
                                        "/tmp/sc_dump")
    assert any("deprecated env override" in l for l in lines)


def test_sc_bad_accelerator_rejected(tmp_path):
    with pytest.raises(ValueError, match="sc_accelerator"):
        _config(tmp_path, "sc_accelerator = bogus\n")


# ---------------------------------------------------------------------------
# Optional eqp2 fixed-Sigma eigenvalue consistency
# ---------------------------------------------------------------------------

def test_eqp2_defaults_off_with_one_mev_cutoff(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.eqp2.enabled is False
    assert cfg.eqp2.tol_ev == pytest.approx(1.0e-3)
    assert cfg.eqp2.max_iter == 20
    assert cfg.eqp2.accelerator == "rcrop"
    assert cfg.eqp2.history_depth == 5
    assert cfg.paths.eqp2_file.endswith("eqp2.dat")


def test_eqp2_dynamic_one_shot_keys_parse(tmp_path):
    cfg = _config(
        tmp_path,
        "compute_mode = gn_ppm\n"
        "write_eqp2 = true\n"
        "eqp2_tol_ev = 0.002\n"
        "eqp2_max_iter = 7\n"
        "eqp2_accelerator = linear\n"
        "eqp2_history_depth = 3\n"
        "eqp2_file = custom.eqp2\n")
    assert cfg.eqp2.enabled is True
    assert cfg.eqp2.tol_ev == pytest.approx(0.002)
    assert cfg.eqp2.max_iter == 7
    assert cfg.eqp2.accelerator == "linear"
    assert cfg.eqp2.history_depth == 3
    assert cfg.paths.eqp2_file.endswith("custom.eqp2")


@pytest.mark.parametrize("line,match", [
    ("eqp2_tol_ev = 0\n", "eqp2_tol_ev"),
    ("eqp2_max_iter = 0\n", "eqp2_max_iter"),
    ("eqp2_accelerator = bogus\n", "eqp2_accelerator"),
    ("eqp2_history_depth = 0\n", "eqp2_history_depth"),
])
def test_eqp2_invalid_iteration_knobs_refuse(tmp_path, line, match):
    with pytest.raises(ValueError, match=match):
        _config(tmp_path, line)


def test_eqp2_static_mode_refuses(tmp_path):
    with pytest.raises(ValueError, match="dynamic full-matrix"):
        _config(tmp_path, "compute_mode = cohsex\nwrite_eqp2 = true\n")


@pytest.mark.parametrize("solver", ["fixed_point", "self_consistent"])
def test_eqp2_is_additional_to_one_shot_only(tmp_path, solver):
    with pytest.raises(ValueError, match="additional fixed-Sigma result"):
        _config(
            tmp_path,
            "compute_mode = gn_ppm\nwrite_eqp2 = true\n"
            f"qp_solver = {solver}\n")


# ---------------------------------------------------------------------------
# Distributed-linalg axes: distributed_cholesky / distributed_lu
# (portable backend names; the legacy cusolvermp_charge/cusolvermp_lu
# aliases were REMOVED 2026-07-31 — unknown deck keys are ignored)
#
# The parse outcome is JAX-backend-dependent (explicit cusolvermp REFUSES
# on a CPU backend — doctrine 3, audit fix/zq 2026-07-28; 'auto' LU
# demotes there with an announcement), so each cell PINS the backend via
# monkeypatch instead of inheriting whatever the test box runs, and the
# parse time no longer runs any transport capability probe (2026-08-06).
# ---------------------------------------------------------------------------

def _pin_backend(monkeypatch, backend_name: str):
    import jax
    monkeypatch.setattr(jax, "default_backend", lambda: backend_name)
    # The two slab_io router stubs that used to live here are gone with the
    # routers (2026-08-06).  Parse time no longer runs any transport
    # capability probe or MPI init, so there is nothing left to stub.


def test_distributed_linalg_defaults(tmp_path, monkeypatch):
    _pin_backend(monkeypatch, "gpu")
    cfg = _config(tmp_path)
    assert cfg.backend.distributed_cholesky == "auto"
    assert cfg.backend.distributed_lu == "auto"


def test_distributed_cholesky_slate_accepted(tmp_path):
    cfg = _config(tmp_path, "distributed_cholesky = slate\n")
    assert cfg.backend.distributed_cholesky == "slate"


def test_distributed_lu_slate_rejected(tmp_path):
    # No SLATE getrf wrapper exists yet — the value must fail loudly.
    with pytest.raises(ValueError, match="distributed_lu"):
        _config(tmp_path, "distributed_lu = slate\n")


def test_removed_legacy_aliases_are_inert(tmp_path, monkeypatch):
    # cusolvermp_charge / cusolvermp_lu were REMOVED (2026-07-31): they
    # are unknown keys now and must neither DeprecationWarning nor steer
    # the portable keys (the pre-removal alias honored 'on' → cusolvermp
    # here).  They DO appear in the aggregated unknown-key stdout report
    # like any other unknown key — that is the point of the removal.
    _pin_backend(monkeypatch, "gpu")
    cfg = _config(tmp_path, "cusolvermp_charge = on\ncusolvermp_lu = off\n")
    assert cfg.backend.distributed_cholesky == "auto"
    assert cfg.backend.distributed_lu == "auto"


def test_distributed_cholesky_invalid_value_raises(tmp_path):
    with pytest.raises(ValueError, match="distributed_cholesky"):
        _config(tmp_path, "distributed_cholesky = scalapack\n")


@pytest.mark.parametrize("key", ["distributed_cholesky", "distributed_lu"])
def test_explicit_cusolvermp_on_cpu_refuses(tmp_path, monkeypatch, key):
    # Doctrine 3: an explicit CUDA-only backend on a CPU JAX backend
    # REFUSES at parse time — it is not rewritten to 'off' (which silently
    # ran a different solver than the input file names).  Substring
    # contract, not exact text.
    _pin_backend(monkeypatch, "cpu")
    with pytest.raises(ValueError, match="CUDA-only"):
        _config(tmp_path, f"{key} = cusolvermp\n")


# ---------------------------------------------------------------------------
# Unknown-key check (read_lorrax_input): a deck key that is neither in
# _DEFAULTS nor covered by a legacy branch is reported in ONE aggregated
# rank-0 stdout warning and ignored; ``strict_keys = true`` upgrades the
# report to a ValueError naming every unknown key.  These cases call
# ``read_lorrax_input`` directly — the parsing path needs no jax.
# ---------------------------------------------------------------------------

def test_unknown_key_warns_on_stdout_and_still_parses(tmp_path, capsys):
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_unknown.in"
    p.write_text(BASE_INPUT + "x_only = true\nuse_chunked_isdf = false\n")
    params = read_lorrax_input(str(p))
    out = capsys.readouterr().out
    # One aggregated report naming each key with its line number.
    assert "unrecognized deck key" in out
    assert "x_only (line 6)" in out
    assert "use_chunked_isdf (line 7)" in out
    assert "ignored" in out
    # ...and the deck still parses: known keys land, unknown ones don't.
    assert params["nval"] == 2
    assert "x_only" not in params


def test_strict_keys_true_raises_naming_every_unknown_key(tmp_path):
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_strict.in"
    p.write_text(BASE_INPUT + "strict_keys = true\n"
                 "x_only = true\nbogus_knob = 3\n")
    with pytest.raises(ValueError, match="strict_keys") as exc:
        read_lorrax_input(str(p))
    # ALL unknown keys named at once, not just the first.
    assert "x_only" in str(exc.value)
    assert "bogus_knob" in str(exc.value)


def test_mixed_case_key_is_recognised_not_reported_unknown(tmp_path, capsys):
    """``do_G0`` is honoured, so it must not also be called unrecognised.

    configparser folds option names to lower case; ``do_G0`` is the only
    key in _DEFAULTS that is not already lower case.  Before the fold was
    applied to the recognition test, this deck BOTH steered the run
    (``params["do_G0"] is False``) and printed ``do_g0 ... not a
    recognized deck key`` -- and ``strict_keys = true`` refused it.
    Checked for the failure signature (reported-unknown) rather than for
    a success marker.
    """
    from gw.gw_config import read_lorrax_input
    for spelling in ("do_G0", "do_g0"):
        p = tmp_path / f"deck_{spelling}.in"
        p.write_text(BASE_INPUT + f"{spelling} = false\n")
        params = read_lorrax_input(str(p))
        out = capsys.readouterr().out
        assert params["do_G0"] is False, (
            f"{spelling}: value not honoured -- test premise is wrong")
        assert "not a recognized deck key" not in out, (
            f"{spelling}: honoured key reported as unrecognised:\n{out}")


def test_mixed_case_key_survives_strict_keys(tmp_path):
    """The same key must not be REFUSED by the strict gate."""
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_strict_case.in"
    p.write_text(BASE_INPUT + "strict_keys = true\ndo_G0 = false\n")
    params = read_lorrax_input(str(p))     # must not raise
    assert params["do_G0"] is False


def test_screening_method_ctsp_refuses_and_names_minimax(tmp_path):
    """``ctsp`` must REFUSE, not resolve to minimax in silence.

    It parsed and ran minimax for months because nothing reads
    ``ScreeningConfig.method``.  The refusal has to name the supported
    method, per the explicit-request-fails-loudly convention.
    """
    with pytest.raises(ValueError, match="minimax") as exc:
        _config(tmp_path, "screening_method = ctsp\n")
    assert "ctsp" in str(exc.value)


def test_screening_method_minimax_and_default_still_parse(tmp_path):
    """The supported value, and omitting the key, both stay clean."""
    assert (_config(tmp_path, "screening_method = minimax\n")
            .screening.method == "minimax")
    assert _config(tmp_path, "").screening.method == "minimax"


def test_every_tracked_fixture_deck_has_no_dead_keys(tmp_path):
    """No tracked ``*.in`` fixture may carry a key nothing reads.

    The guard for a cleanup that would otherwise silently rot: 7 of 7
    fixtures once carried keys absent from _DEFAULTS (``x_only`` in six of
    them -- a VALUE of ``compute_mode``, never a key -- plus
    ``use_chunked_isdf``, ``sigma_debug_split_contrib``, ``max_r_chunks``,
    ``profile_qloop``, ``profile_trace_dir``).  Every one parsed clean and
    steered nothing.

    ``strict_keys`` is forced ON here regardless of the shipped default,
    so this stays a real gate if that default never flips.  The knob is
    injected right after the ``[cohsex]`` header: appending at EOF can
    land past the section end when a K_POINTS block follows, which would
    silently test nothing.
    """
    import re
    from gw.gw_config import read_lorrax_input

    repo = pathlib.Path(__file__).resolve().parent.parent
    decks = sorted(p for p in repo.glob("tests/**/*.in")
                   if re.search(r"^\s*\[cohsex\]", p.read_text(errors="replace"),
                                re.I | re.M))
    assert decks, "no [cohsex] fixture decks found -- glob is wrong"

    offenders = {}
    for deck in decks:
        out, injected = [], False
        for ln in deck.read_text().splitlines(keepends=True):
            if re.match(r"\s*strict_keys\s*[=:]", ln, re.I):
                continue
            out.append(ln)
            if not injected and ln.strip().lower().startswith("[cohsex]"):
                out.append("strict_keys = true\n")
                injected = True
        probe = tmp_path / deck.name
        probe.write_text("".join(out))
        try:
            read_lorrax_input(str(probe))
        except ValueError as exc:
            if "strict_keys" in str(exc):
                offenders[deck.relative_to(repo)] = str(exc).splitlines()[1:]
            else:
                raise
    assert not offenders, (
        "tracked fixture decks carry keys that nothing reads:\n"
        + "\n".join(f"  {k}: {v}" for k, v in offenders.items()))


def test_legacy_keys_keep_their_dedicated_messages(tmp_path, capsys):
    # Keys with an explicit legacy branch (dedicated DeprecationWarning or
    # refusal) must NOT also be reported as unknown — one key, one message.
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_legacy.in"
    p.write_text(BASE_INPUT + "chunk_size = 8\noutput_file = out.dat\n")
    with pytest.warns(DeprecationWarning) as rec:
        read_lorrax_input(str(p))
    messages = [str(w.message) for w in rec]
    assert any("chunk_size" in m for m in messages)
    assert any("output_file" in m for m in messages)
    assert "unrecognized deck key" not in capsys.readouterr().out
    # The refusal branch also stays dedicated (no unknown-key rewording).
    q = tmp_path / "deck_refused.in"
    q.write_text(BASE_INPUT + "use_shipped_minimax_tables = true\n")
    with pytest.raises(ValueError, match="regenerate_minimax_tables"):
        read_lorrax_input(str(q))


def test_auto_on_cpu_chol_passes_lu_demotes_announced(tmp_path, monkeypatch):
    # 'auto' MAY demote, announced: distributed_cholesky=auto passes
    # through (it carries the replicated rank-truncation route on CPU);
    # distributed_lu=auto demotes to 'off' and says so.
    _pin_backend(monkeypatch, "cpu")
    lines: list[str] = []
    path = tmp_path / "cohsex_auto_cpu.in"
    path.write_text(BASE_INPUT)
    cfg = LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert cfg.backend.distributed_cholesky == "auto"
    assert cfg.backend.distributed_lu == "off"
    assert any("distributed_lu=auto" in l and "off" in l for l in lines)


def test_the_dynamic_self_consistent_regression_deck_is_supported():
    """The shipped dynamic SC deck remains a live supported combination."""
    deck = (pathlib.Path(__file__).resolve().parent
            / "regression" / "gnppm_debug" / "gnppm_sc.in")
    assert deck.is_file(), f"the one SC deck is missing: {deck}"
    cfg = LorraxConfig.from_input_file(
        str(deck), print_fn=lambda *a, **k: None)
    assert cfg.qp_solver is QPSolver.SELF_CONSISTENT
    assert cfg.compute_mode.is_dynamic
